"""RetinaMNIST transfer from the verified ViT-Base backbone on official splits.

Run stages: cache frozen features once, fit baselines and partial fine-tuning with
validation-only selection, write val/test predictions, then derive every metric
from the saved prediction files.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from vit_lab.models.classification_head import pool_tokens
from vit_lab.models.vision_transformer import VisionTransformer
from vit_lab.training.trainer import learning_rate_for_epoch

SPLITS = ("train", "val", "test")
RESULT_DIR = Path("results/transfer/retina")


class TransferHead(nn.Module):
    """The last ViT blocks, final norm, and a new classifier over cached tokens."""

    def __init__(self, blocks: nn.ModuleList, norm: nn.Module, head: nn.Linear) -> None:
        super().__init__()
        self.blocks, self.norm, self.head = blocks, norm, head

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(pool_tokens(self.norm(tokens), "cls"))


def load_split(split: str, data_cfg: dict[str, Any]) -> tuple[torch.Tensor, np.ndarray]:
    from medmnist import RetinaMNIST

    dataset = RetinaMNIST(split=split, size=data_cfg["size"], root=data_cfg["root"],
                          download=False)
    return torch.from_numpy(dataset.imgs), dataset.labels.flatten().astype(np.int64)


def normalize(images: torch.Tensor, data_cfg: dict[str, Any]) -> torch.Tensor:
    """uint8 NHWC to normalized float NCHW."""
    mean = torch.tensor(data_cfg["normalization_mean"]).view(1, 3, 1, 1)
    std = torch.tensor(data_cfg["normalization_std"]).view(1, 3, 1, 1)
    return (images.permute(0, 3, 1, 2).float() / 255 - mean) / std


def verified_backbone(reference_config: Path, *, offline: bool) -> tuple[VisionTransformer, str]:
    from vit_lab.reference.loader import load_reference_model
    from vit_lab.reference.weight_map import load_exact_reference_weights

    config = yaml.safe_load(reference_config.read_text(encoding="utf-8"))
    reference, _, checkpoint_sha = load_reference_model(
        config["reference"], local_files_only=offline
    )
    model = VisionTransformer(**config["model"])
    load_exact_reference_weights(model, reference)
    return model.eval(), checkpoint_sha


@torch.inference_mode()
def cache_features(model: VisionTransformer, images: torch.Tensor, *, frozen_blocks: int,
                   data_cfg: dict[str, Any], batch_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokens after `frozen_blocks` blocks and the final normalized CLS features."""
    tokens, features = [], []
    for start in range(0, len(images), batch_size):
        output = model(normalize(images[start:start + batch_size], data_cfg),
                       return_hidden_states=True)
        tokens.append(output.hidden_states[frozen_blocks - 1].clone())
        features.append(output.final_features.clone())
    return torch.cat(tokens), torch.cat(features)


def classification_metrics(labels: np.ndarray, predictions: np.ndarray,
                           num_classes: int) -> dict[str, Any]:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                                 confusion_matrix, f1_score)

    classes = list(range(num_classes))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Bootstrap resamples can miss a class.
        return {
            "accuracy": accuracy_score(labels, predictions),
            "balanced_accuracy": balanced_accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, labels=classes, average="macro",
                                 zero_division=0),
            "quadratic_weighted_kappa": cohen_kappa_score(
                labels, predictions, labels=classes, weights="quadratic"),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
        }


def bootstrap_intervals(labels: np.ndarray, predictions: np.ndarray, num_classes: int,
                        *, resamples: int, seed: int) -> dict[str, list[float]]:
    """Percentile 95% intervals from resampling evaluated images with replacement."""
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(resamples):
        index = rng.integers(0, len(labels), len(labels))
        for name, value in classification_metrics(labels[index], predictions[index],
                                                  num_classes).items():
            if name != "confusion_matrix":
                draws.setdefault(name, []).append(float(np.nan_to_num(value)))
    return {name: np.percentile(values, [2.5, 97.5]).tolist() for name, values in draws.items()}


def _qwk(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> float:
    return classification_metrics(labels, predictions, num_classes)["quadratic_weighted_kappa"]


def write_predictions(path: Path, rows: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """rows: split -> (labels, probabilities [N, C])."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        num_classes = next(iter(rows.values()))[1].shape[1]
        writer.writerow(["split", "index", "label", "prediction",
                         *[f"prob_{c}" for c in range(num_classes)]])
        for split, (labels, probabilities) in rows.items():
            for index, (label, probs) in enumerate(zip(labels, probabilities)):
                writer.writerow([split, index, int(label), int(np.argmax(probs)),
                                 *[f"{p:.6f}" for p in probs]])


def read_predictions(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for split in sorted({row["split"] for row in rows}):
        selected = [row for row in rows if row["split"] == split]
        result[split] = (np.array([int(r["label"]) for r in selected]),
                         np.array([int(r["prediction"]) for r in selected]))
    return result


def majority_baseline(labels: dict[str, np.ndarray], num_classes: int) -> dict[str, Any]:
    majority = int(np.bincount(labels["train"], minlength=num_classes).argmax())
    onehot = np.eye(num_classes)[majority]
    return {"majority_class": majority,
            "rows": {s: (labels[s], np.tile(onehot, (len(labels[s]), 1))) for s in ("val", "test")}}


def linear_probe(features: dict[str, np.ndarray], labels: dict[str, np.ndarray],
                 cfg: dict[str, Any], num_classes: int) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    selection = []
    best = None
    for c in cfg["c_grid"]:
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=c, max_iter=cfg["max_iter"]))
        model.fit(features["train"], labels["train"])
        qwk = _qwk(labels["val"], model.predict(features["val"]), num_classes)
        selection.append({"C": c, "validation_quadratic_weighted_kappa": qwk})
        if best is None or qwk > best[0]:  # Ties keep the stronger regularization.
            best = (qwk, c, model)
    _, c, model = best
    # Test probabilities are computed once, for the validation-selected C only.
    return {"selected_C": c, "selection": selection,
            "rows": {s: (labels[s], model.predict_proba(features[s])) for s in ("val", "test")}}


@torch.inference_mode()
def _probabilities(model: nn.Module, tokens: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    return torch.cat([F.softmax(model(tokens[i:i + batch_size]), dim=1)
                      for i in range(0, len(tokens), batch_size)]).numpy()


def finetune(backbone: VisionTransformer, tokens: dict[str, torch.Tensor],
             labels: dict[str, np.ndarray], cfg: dict[str, Any], *, seed: int,
             num_classes: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    first_trainable = len(backbone.blocks) - cfg["trainable_blocks"]
    model = TransferHead(copy.deepcopy(backbone.blocks[first_trainable:]),
                         copy.deepcopy(backbone.norm),
                         nn.Linear(backbone.head.in_features, num_classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    train_x, train_y = tokens["train"], torch.from_numpy(labels["train"])
    history, best = [], None
    for epoch in range(cfg["epochs"]):
        rate = learning_rate_for_epoch(epoch, base_rate=cfg["learning_rate"],
                                       epochs=cfg["epochs"], warmup_epochs=cfg["warmup_epochs"])
        for group in optimizer.param_groups:
            group["lr"] = rate
        start = time.perf_counter()
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        total = 0.0
        for i in range(0, len(order), cfg["batch_size"]):
            index = order[i:i + cfg["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(train_x[index]), train_y[index])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(index)
        val_probs = _probabilities(model, tokens["val"], cfg["batch_size"])
        val_qwk = _qwk(labels["val"], val_probs.argmax(1), num_classes)
        history.append({
            "epoch": epoch + 1, "learning_rate": rate, "train_loss": total / len(order),
            "val_loss": float(F.nll_loss(torch.log(torch.from_numpy(val_probs).clamp_min(1e-12)),
                                         torch.from_numpy(labels["val"]))),
            "val_accuracy": float((val_probs.argmax(1) == labels["val"]).mean()),
            "val_quadratic_weighted_kappa": val_qwk,
            "seconds": time.perf_counter() - start,
        })
        print(f"seed {seed} epoch {epoch + 1}/{cfg['epochs']} "
              f"train_loss={history[-1]['train_loss']:.4f} val_qwk={val_qwk:.4f}", flush=True)
        if best is None or val_qwk > best[0]:  # Ties keep the earlier epoch.
            best = (val_qwk, epoch + 1, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[2])
    # Single final test evaluation of the validation-selected epoch.
    rows = {s: (labels[s], _probabilities(model, tokens[s], cfg["batch_size"]))
            for s in ("val", "test")}
    trainable = sum(p.numel() for p in model.parameters())
    return {"best_epoch": best[1], "history": history, "rows": rows,
            "trainable_parameters": trainable}


def _environment() -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "timm": version("timm"),
        "scikit-learn": version("scikit-learn"), "medmnist": version("medmnist"),
        "device": "cpu", "threads": torch.get_num_threads(), "git_commit": commit,
    }


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def run(config_path: Path, *, offline: bool, result_dir: Path = RESULT_DIR) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg, ft_cfg = cfg["data"], cfg["finetune"]
    torch.set_num_threads(ft_cfg["threads"])
    images, labels = {}, {}
    for split in SPLITS:
        images[split], labels[split] = load_split(split, data_cfg)
    backbone, checkpoint_sha = verified_backbone(Path(cfg["reference_config"]), offline=offline)
    frozen_blocks = len(backbone.blocks) - ft_cfg["trainable_blocks"]
    cache_path = Path(data_cfg["root"]) / f"retina_cache_b{frozen_blocks}.pt"
    cache_key = {"checkpoint_sha256": checkpoint_sha, "frozen_blocks": frozen_blocks,
                 "data": data_cfg}
    start = time.perf_counter()
    cached = torch.load(cache_path, weights_only=True) if cache_path.exists() else None
    if cached is None or cached["key"] != json.loads(json.dumps(cache_key)):
        cached = {"key": cache_key, "tokens": {}, "features": {}}
        for split in SPLITS:
            cached["tokens"][split], cached["features"][split] = cache_features(
                backbone, images[split], frozen_blocks=frozen_blocks, data_cfg=data_cfg)
            print(f"cached {split}: {tuple(cached['tokens'][split].shape)}", flush=True)
        torch.save(cached, cache_path)
    cache_seconds = time.perf_counter() - start
    del images

    num_classes = cfg["num_classes"]
    result_dir.mkdir(parents=True, exist_ok=True)
    majority = majority_baseline(labels, num_classes)
    write_predictions(result_dir / "predictions_majority.csv", majority["rows"])
    features = {s: cached["features"][s].numpy() for s in SPLITS}
    probe = linear_probe(features, labels, cfg["linear_probe"], num_classes)
    write_predictions(result_dir / "predictions_linear_probe.csv", probe["rows"])
    finetune_runs = {}
    for seed in ft_cfg["seeds"]:
        result = finetune(backbone, cached["tokens"], labels, ft_cfg, seed=seed,
                          num_classes=num_classes)
        write_predictions(result_dir / f"predictions_finetune_seed{seed}.csv", result["rows"])
        with (result_dir / f"finetune_seed{seed}_metrics.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["history"][0]),
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(result["history"])
        finetune_runs[f"finetune_seed{seed}"] = {
            "seed": seed, "best_epoch": result["best_epoch"],
            "trainable_parameters": result["trainable_parameters"],
            "total_epoch_seconds": sum(row["seconds"] for row in result["history"]),
        }
    manifest = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "dataset_file": f"retinamnist_{data_cfg['size']}.npz",
        "dataset_md5": _md5(Path(data_cfg["root"]) / f"retinamnist_{data_cfg['size']}.npz"),
        "class_counts": {s: np.bincount(labels[s], minlength=num_classes).tolist()
                         for s in SPLITS},
        "checkpoint_sha256": checkpoint_sha,
        "frozen_blocks": frozen_blocks,
        "feature_cache_seconds": cache_seconds,
        "majority_class": majority["majority_class"],
        "linear_probe": {"selected_C": probe["selected_C"], "selection": probe["selection"]},
        "finetune": finetune_runs,
        "environment": _environment(),
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summarize(result_dir, cfg)


def summarize(result_dir: Path, cfg: dict[str, Any],
              methods: list[str] | None = None) -> list[dict[str, Any]]:
    """Recompute all reported metrics from saved predictions and labels."""
    num_classes, boot = cfg["num_classes"], cfg["bootstrap"]
    methods = methods or ["majority", "linear_probe",
                          *[f"finetune_seed{seed}" for seed in cfg["finetune"]["seeds"]]]
    rows, details = [], {}
    for method in methods:
        predictions = read_predictions(result_dir / f"predictions_{method}.csv")
        details[method] = {}
        row = {"method": method}
        for split in ("val", "test"):
            y, p = predictions[split]
            metrics = classification_metrics(y, p, num_classes)
            details[method][split] = metrics
            for name, value in metrics.items():
                if name != "confusion_matrix":
                    row[f"{split}_{name}"] = round(float(value), 4)
        y, p = predictions["test"]
        intervals = bootstrap_intervals(y, p, num_classes, resamples=boot["resamples"],
                                        seed=boot["seed"])
        details[method]["test_bootstrap_95ci"] = intervals
        for name, (low, high) in intervals.items():
            row[f"test_{name}_ci_low"] = round(low, 4)
            row[f"test_{name}_ci_high"] = round(high, 4)
        rows.append(row)
    seeds = [row for row in rows if "finetune_seed" in row["method"]]
    aggregate = {
        name: {"mean": float(np.mean([r[name] for r in seeds])),
               "sample_std": float(np.std([r[name] for r in seeds], ddof=1))}
        for name in rows[0] if name.startswith(("val_", "test_")) and "_ci_" not in name
    }
    with (result_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (result_dir / "metrics.json").write_text(json.dumps(
        {"per_method": details, "finetune_across_seeds": aggregate}, indent=2) + "\n")
    return rows


def plot(result_dir: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (result_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    names = [row["method"].replace("_", " ") for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    for ax, metric, title in zip(axes, ("quadratic_weighted_kappa", "balanced_accuracy"),
                                 ("Quadratic weighted kappa", "Balanced accuracy")):
        values = [float(r[f"test_{metric}"]) for r in rows]
        low = [v - float(r[f"test_{metric}_ci_low"]) for v, r in zip(values, rows)]
        high = [float(r[f"test_{metric}_ci_high"]) - v for v, r in zip(values, rows)]
        ax.bar(range(len(rows)), values, yerr=[low, high], capsize=4, color="#4C78A8")
        ax.set_xticks(range(len(rows)), names, rotation=30, ha="right")
        ax.set_title(f"Test {title} (95% bootstrap CI)")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("RetinaMNIST-224 transfer from the verified ViT-Base backbone (n_test=400)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/retina.yaml"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Recompute metrics and figures from saved predictions")
    args = parser.parse_args()
    if args.summarize_only:
        summarize(RESULT_DIR, yaml.safe_load(args.config.read_text(encoding="utf-8")))
    else:
        run(args.config, offline=args.offline)
    for suffix in ("svg", "png"):
        plot(RESULT_DIR, Path(f"results/figures/retina_transfer.{suffix}"))


if __name__ == "__main__":
    main()
