"""Small ViT training with validation selection and resumable run artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision
import yaml

from vit_lab.models.vision_transformer import VisionTransformer
from vit_lab.training.data import cifar10_loaders


def learning_rate_for_epoch(
    epoch: int, *, base_rate: float, epochs: int, warmup_epochs: int
) -> float:
    """Linear warmup followed by cosine decay, indexed from epoch zero."""
    if not (0 <= epoch < epochs) or not (0 <= warmup_epochs < epochs):
        raise ValueError("Invalid epoch or warmup length")
    if epoch < warmup_epochs:
        return base_rate * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    return base_rate * 0.5 * (1 + math.cos(math.pi * progress))


def _device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    return requested


def _run_epoch(
    model: VisionTransformer,
    loader: torch.utils.data.DataLoader,
    *,
    device: str,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> tuple[float, float, torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_items = 0
    predictions = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, (images, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device)
            targets = targets.flatten().to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in batch {batch_index}")
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * targets.numel()
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_items += targets.numel()
            if not training:
                predictions.append(logits.detach().cpu())
    if total_items == 0:
        raise ValueError("No batches were processed")
    return total_loss / total_items, total_correct / total_items, torch.cat(predictions) if predictions else torch.empty(0)


def train_cifar10(
    config: dict[str, Any],
    *,
    run_dir: Path,
    requested_device: str = "auto",
    epochs_override: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Train a small ViT and leave a complete, restartable run record."""
    config = json.loads(json.dumps(config))
    training = config["training"]
    if training["dataset"] != "cifar10":
        raise ValueError("This trainer currently supports CIFAR-10 only")
    if epochs_override is not None:
        training["epochs"] = epochs_override
    if training["epochs"] <= 0:
        raise ValueError("epochs must be positive")
    if training["warmup_epochs"] >= training["epochs"]:
        training["warmup_epochs"] = max(0, training["epochs"] - 1)
    if max_train_batches is not None and max_train_batches <= 0:
        raise ValueError("max_train_batches must be positive")
    if max_val_batches is not None and max_val_batches <= 0:
        raise ValueError("max_val_batches must be positive")
    device = _device(requested_device)
    seed = int(training["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        # Deterministic kernels so the reloaded checkpoint can reproduce validation logits.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.cuda.manual_seed_all(seed)
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    generator = torch.Generator().manual_seed(seed)
    run_dir = Path(run_dir)
    if run_dir.exists() and not resume:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, validation_loader, split = cifar10_loaders(
        root="data",
        batch_size=int(training["batch_size"]),
        seed=seed,
        generator=generator,
        device=device,
        image_size=int(config["model"]["image_size"]),
    )
    config_path = run_dir / "config.yaml"
    split_path = run_dir / "split_indices.json"
    if resume:
        if yaml.safe_load(config_path.read_text()) != config:
            raise ValueError("Resume config differs from the saved run")
        if json.loads(split_path.read_text()) != split:
            raise ValueError("Resume data split differs from the saved run")
    else:
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        split_path.write_text(json.dumps(split) + "\n", encoding="utf-8")

    model = VisionTransformer(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    first_epoch = 0
    best_accuracy = -1.0
    if resume:
        checkpoint = torch.load(run_dir / "last.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        first_epoch = checkpoint["epoch"] + 1
        best_accuracy = checkpoint["best_accuracy"]
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        generator.set_state(checkpoint["loader_rng_state"].cpu())

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": device,
        **({"gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda}
           if device == "cuda" else {}),
        "git_commit": git_commit,
        "seed": seed,
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
        "smoke_run": max_train_batches is not None or max_val_batches is not None,
    }
    (run_dir / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("a" if resume else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "learning_rate", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "seconds"],
        )
        if not resume:
            writer.writeheader()
        for epoch in range(first_epoch, int(training["epochs"])):
            learning_rate = learning_rate_for_epoch(
                epoch,
                base_rate=float(training["learning_rate"]),
                epochs=int(training["epochs"]),
                warmup_epochs=int(training["warmup_epochs"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            start = time.perf_counter()
            train_loss, train_accuracy, _ = _run_epoch(
                model, train_loader, device=device, optimizer=optimizer,
                max_batches=max_train_batches,
            )
            val_loss, val_accuracy, val_logits = _run_epoch(
                model, validation_loader, device=device, optimizer=None,
                max_batches=max_val_batches,
            )
            seconds = time.perf_counter() - start
            row = {
                "epoch": epoch + 1,
                "learning_rate": learning_rate,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "seconds": seconds,
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"epoch {epoch + 1}/{training['epochs']} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_accuracy={val_accuracy:.4f} seconds={seconds:.1f}",
                flush=True,
            )
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best = {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_accuracy": best_accuracy,
                    "validation_logits": val_logits,
                }
                torch.save(best, run_dir / "checkpoint.tmp")
                (run_dir / "checkpoint.tmp").replace(run_dir / "checkpoint.pt")
            last = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_accuracy": best_accuracy,
                "torch_rng_state": torch.get_rng_state(),
                "loader_rng_state": generator.get_state(),
            }
            torch.save(last, run_dir / "last.tmp")
            (run_dir / "last.tmp").replace(run_dir / "last.pt")

    best = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=True)
    reloaded = VisionTransformer(**config["model"]).to(device)
    reloaded.load_state_dict(best["model"], strict=True)
    _, reloaded_accuracy, reloaded_logits = _run_epoch(
        reloaded, validation_loader, device=device, optimizer=None,
        max_batches=max_val_batches,
    )
    saved_logits = best["validation_logits"]
    # CPU must reproduce bit for bit; accelerators get a tiny documented tolerance.
    torch.testing.assert_close(reloaded_logits, saved_logits, rtol=0,
                               atol=0 if device == "cpu" else 1e-4)
    summary = {
        "best_epoch": best["epoch"] + 1,
        "best_validation_accuracy": best_accuracy,
        "reloaded_validation_accuracy": reloaded_accuracy,
        "validation_predictions_identical_after_reload": bool(
            torch.equal(reloaded_logits.argmax(1), saved_logits.argmax(1))),
        "reload_max_abs_logit_difference": (reloaded_logits - saved_logits).abs().max().item(),
        "smoke_run": environment["smoke_run"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
