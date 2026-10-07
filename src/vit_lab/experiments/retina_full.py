"""Full-backbone RetinaMNIST fine-tuning from the verified ViT-Base (GPU protocol).

Same splits, normalization, validation-QWK selection, single test evaluation, and
prediction-derived metrics as the partial study in `retina.py`; here all blocks train
with augmentation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torchvision import transforms

from vit_lab.experiments.retina import (
    SPLITS, _environment, _md5, _qwk, load_split, plot, summarize, verified_backbone,
    write_predictions,
)
from vit_lab.models.vision_transformer import VisionTransformer
from vit_lab.training.trainer import learning_rate_for_epoch

RESULT_DIR = Path("results/transfer/retina_full")


def _normalize_nchw(images: torch.Tensor, data_cfg: dict[str, Any]) -> torch.Tensor:
    mean = torch.tensor(data_cfg["normalization_mean"]).view(1, 3, 1, 1)
    std = torch.tensor(data_cfg["normalization_std"]).view(1, 3, 1, 1)
    return (images.float() / 255 - mean) / std


def _autocast(device: str, enabled: bool):
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device, dtype=dtype, enabled=enabled)


@torch.inference_mode()
def predict(model: nn.Module, images: torch.Tensor, data_cfg: dict[str, Any], *,
            device: str, amp: bool, batch_size: int = 64) -> np.ndarray:
    """Softmax probabilities for uint8 NHWC images, without augmentation."""
    model.eval()
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = _normalize_nchw(images[start:start + batch_size].permute(0, 3, 1, 2),
                                data_cfg).to(device)
        with _autocast(device, amp):
            logits = model(batch)
        outputs.append(F.softmax(logits.float(), dim=1).cpu())
    return torch.cat(outputs).numpy()


def finetune_full(backbone: VisionTransformer, images: dict[str, torch.Tensor],
                  labels: dict[str, np.ndarray], cfg: dict[str, Any],
                  data_cfg: dict[str, Any], *, seed: int, num_classes: int, device: str,
                  max_train_batches: int | None = None) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = copy.deepcopy(backbone)
    model.head = nn.Linear(model.head.in_features, num_classes)
    model.to(device)
    amp = bool(cfg["mixed_precision"]) and device == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler(device, enabled=amp)
    augmentation = cfg["augmentation"]
    augment = transforms.Compose([
        transforms.RandomResizedCrop(data_cfg["size"],
                                     scale=tuple(augmentation["random_resized_crop_scale"]),
                                     antialias=True),
        *([transforms.RandomHorizontalFlip()] if augmentation["horizontal_flip"] else []),
    ])
    generator = torch.Generator().manual_seed(seed)
    train_images, train_labels = images["train"], torch.from_numpy(labels["train"])
    history, best = [], None
    for epoch in range(cfg["epochs"]):
        rate = learning_rate_for_epoch(epoch, base_rate=cfg["learning_rate"],
                                       epochs=cfg["epochs"], warmup_epochs=cfg["warmup_epochs"])
        for group in optimizer.param_groups:
            group["lr"] = rate
        start = time.perf_counter()
        model.train()
        order = torch.randperm(len(train_images), generator=generator)
        total, seen = 0.0, 0
        for batch_index, i in enumerate(range(0, len(order), cfg["batch_size"])):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            index = order[i:i + cfg["batch_size"]]
            batch = torch.stack([augment(train_images[j].permute(2, 0, 1)) for j in index])
            batch = _normalize_nchw(batch, data_cfg).to(device)
            targets = train_labels[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                logits = model(batch)
            loss = F.cross_entropy(logits.float(), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(index)
            seen += len(index)
        val_probs = predict(model, images["val"], data_cfg, device=device, amp=amp)
        val_qwk = _qwk(labels["val"], val_probs.argmax(1), num_classes)
        history.append({
            "epoch": epoch + 1, "learning_rate": rate, "train_loss": total / seen,
            "val_loss": float(F.nll_loss(torch.log(torch.from_numpy(val_probs).clamp_min(1e-12)),
                                         torch.from_numpy(labels["val"]))),
            "val_accuracy": float((val_probs.argmax(1) == labels["val"]).mean()),
            "val_quadratic_weighted_kappa": val_qwk,
            "seconds": time.perf_counter() - start,
        })
        print(f"full seed {seed} epoch {epoch + 1}/{cfg['epochs']} "
              f"train_loss={history[-1]['train_loss']:.4f} val_qwk={val_qwk:.4f} "
              f"seconds={history[-1]['seconds']:.1f}", flush=True)
        if best is None or val_qwk > best[0]:  # Ties keep the earlier epoch.
            best = (val_qwk, epoch + 1,
                    {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    # Single final test evaluation of the validation-selected epoch.
    rows = {s: (labels[s], predict(model, images[s], data_cfg, device=device, amp=amp))
            for s in ("val", "test")}
    return {"best_epoch": best[1], "history": history, "rows": rows, "mixed_precision": amp,
            "trainable_parameters": sum(p.numel() for p in model.parameters())}


def run(config_path: Path, *, device: str, offline: bool, result_dir: Path = RESULT_DIR,
        smoke: bool = False) -> list[dict[str, Any]]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg, ft_cfg = cfg["data"], dict(cfg["finetune"])
    if smoke:  # Code-path check only: tiny subsets, one epoch, one seed.
        ft_cfg.update(epochs=1, warmup_epochs=0, seeds=ft_cfg["seeds"][:1])
    images, labels = {}, {}
    for split in SPLITS:
        images[split], labels[split] = load_split(split, data_cfg)
        if smoke:
            images[split], labels[split] = images[split][:16], labels[split][:16]
    backbone, checkpoint_sha = verified_backbone(Path(cfg["reference_config"]), offline=offline)
    num_classes = cfg["num_classes"]
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = result_dir / "manifest.json"
    runs = (json.loads(manifest_path.read_text())["finetune"]
            if manifest_path.exists() else {})
    methods = []
    for seed in ft_cfg["seeds"]:
        name = f"full_finetune_seed{seed}"
        methods.append(name)
        if (result_dir / f"predictions_{name}.csv").exists() and name in runs:
            print(f"{name}: already complete, skipping", flush=True)
            continue
        result = finetune_full(backbone, images, labels, ft_cfg, data_cfg, seed=seed,
                               num_classes=num_classes, device=device,
                               max_train_batches=1 if smoke else None)
        write_predictions(result_dir / f"predictions_{name}.csv", result["rows"])
        with (result_dir / f"{name}_metrics.csv").open("w", newline="", encoding="utf-8") as h:
            writer = csv.DictWriter(h, fieldnames=list(result["history"][0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(result["history"])
        runs[name] = {"seed": seed, "best_epoch": result["best_epoch"],
                      "mixed_precision_fp16": result["mixed_precision"],
                      "trainable_parameters": result["trainable_parameters"],
                      "total_epoch_seconds": sum(r["seconds"] for r in result["history"])}
        environment = _environment()
        environment.update(device=device, threads=torch.get_num_threads(), **(
            {"gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda}
            if device == "cuda" else {}))
        manifest = {
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "smoke_run": smoke,
            "dataset_md5": _md5(Path(data_cfg["root"]) / f"retinamnist_{data_cfg['size']}.npz"),
            "class_counts": {s: np.bincount(labels[s], minlength=num_classes).tolist()
                             for s in SPLITS},
            "checkpoint_sha256": checkpoint_sha,
            "finetune": runs,
            "environment": environment,
        }
        # Written after every seed so an interrupted session keeps finished seeds.
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return summarize(result_dir, cfg, methods=methods)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/experiments/retina_full.yaml"))
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    args = parser.parse_args()
    run(args.config, device=args.device, offline=args.offline,
        result_dir=args.result_dir, smoke=args.smoke)
    for suffix in ("svg", "png"):
        plot(args.result_dir, args.result_dir / f"retina_full_transfer.{suffix}")


if __name__ == "__main__":
    main()
