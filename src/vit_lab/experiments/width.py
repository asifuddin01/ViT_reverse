"""Serial, resumable CIFAR-10 width-96-versus-192 ablation at fixed patch size 8."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


WIDTHS = (96, 192)
SEEDS = (7, 11, 19)
CASE_ORDER = tuple((width, seed) for seed in SEEDS for width in WIDTHS)


def variant_config(base: dict[str, Any], *, width: int, seed: int) -> dict[str, Any]:
    """Change only embedding width and paired model/split seed from frozen P=8 runs."""
    if width not in WIDTHS or seed not in SEEDS:
        raise ValueError("Variant is outside the preregistered width matrix")
    config = json.loads(json.dumps(base))
    config["model"]["embed_dim"] = width
    config["training"]["seed"] = seed
    if width % config["model"]["num_heads"]:
        raise ValueError("Head count must divide embedding width")
    return config


def _name(width: int, seed: int) -> str:
    return f"width_d{width:03d}_seed{seed}"


def _existing_config(seed: int) -> Path:
    return Path(f"configs/experiments/patch_size/patch_p08_seed{seed}.yaml")


def _existing_run(seed: int) -> Path:
    return Path(f"results/runs/patch_size/patch_p08_seed{seed}")


def prepare_configs(config_dir: Path) -> list[dict[str, Any]]:
    base = yaml.safe_load(_existing_config(7).read_text(encoding="utf-8"))
    config_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for width, seed in CASE_ORDER:
        config = variant_config(base, width=width, seed=seed)
        path = (_existing_config(seed) if width == 96 else
                config_dir / f"{_name(width, seed)}.yaml")
        if path.exists():
            if yaml.safe_load(path.read_text(encoding="utf-8")) != config:
                raise ValueError(f"Frozen width config differs: {path}")
        else:
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        cases.append({"width": width, "seed": seed, "config_path": str(path)})
    return cases


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_case(case: dict[str, Any], *, run_dir: Path, result_dir: Path) -> dict[str, Any]:
    name = _name(case["width"], case["seed"])
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    run_config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    frozen_config = yaml.safe_load(Path(case["config_path"]).read_text(encoding="utf-8"))
    if run_config != frozen_config:
        raise ValueError(f"Saved run config differs from frozen width variant: {name}")
    if environment["seed"] != case["seed"] or environment["smoke_run"]:
        raise ValueError(f"Run has wrong seed or is only a smoke test: {name}")
    if not summary["validation_predictions_identical_after_reload"] or summary["smoke_run"]:
        raise ValueError(f"Best checkpoint reload proof is missing: {name}")
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != frozen_config["training"]["epochs"] or int(rows[-1]["epoch"]) != len(rows):
        raise ValueError(f"Incomplete epoch history: {name}")
    if max(float(row["val_accuracy"]) for row in rows) != summary["best_validation_accuracy"]:
        raise ValueError(f"Best validation accuracy differs from per-epoch metrics: {name}")
    split = json.loads((run_dir / "split_indices.json").read_text(encoding="utf-8"))
    if split["seed"] != case["seed"] or len(split["train_indices"]) != 45000 or len(split["validation_indices"]) != 5000:
        raise ValueError(f"Saved split is inconsistent: {name}")
    split_hash = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    result_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = result_dir / f"{name}_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    exported = {
        "embed_dim": case["width"],
        "seed": case["seed"],
        "run_dir": str(run_dir),
        "config_path": case["config_path"],
        "config_sha256": _sha256(Path(case["config_path"])),
        "split_sha256": split_hash,
        "best_epoch": summary["best_epoch"],
        "best_validation_accuracy": summary["best_validation_accuracy"],
        "reloaded_validation_accuracy": summary["reloaded_validation_accuracy"],
        "validation_predictions_identical_after_reload": True,
        "total_epoch_seconds": sum(float(row["seconds"]) for row in rows),
        "environment": environment,
        "metrics_path": str(metrics_path),
    }
    (result_dir / f"{name}_summary.json").write_text(
        json.dumps(exported, indent=2) + "\n", encoding="utf-8"
    )
    return exported


def _write_progress(cases: list[dict[str, Any]], result_dir: Path,
                    completed: list[dict[str, Any]]) -> None:
    done = {(row["embed_dim"], row["seed"]) for row in completed}
    pending = [case for case in cases if (case["width"], case["seed"]) not in done]
    progress = {
        "completed": [_name(row["embed_dim"], row["seed"]) for row in completed],
        "pending": [_name(row["width"], row["seed"]) for row in pending],
        "next_case": pending[0] if pending else None,
        "resume_command": "python scripts/run_width_ablation.py --run",
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")


def _write_aggregate(completed: list[dict[str, Any]], result_dir: Path) -> None:
    if len(completed) != len(CASE_ORDER):
        return
    rows = []
    for width in WIDTHS:
        selected = sorted((row for row in completed if row["embed_dim"] == width),
                          key=lambda row: row["seed"])
        values = [row["best_validation_accuracy"] for row in selected]
        rows.append({
            "embed_dim": width,
            "seeds": ",".join(str(row["seed"]) for row in selected),
            "validation_accuracy_mean": statistics.mean(values),
            "validation_accuracy_sample_std": statistics.stdev(values),
            "total_epoch_hours": sum(row["total_epoch_seconds"] for row in selected) / 3600,
        })
    with (result_dir / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_matrix(*, config_dir: Path, run_root: Path, result_dir: Path,
               max_runs: int | None = None) -> None:
    cases = prepare_configs(config_dir)
    completed: list[dict[str, Any]] = []
    started = 0
    for case in cases:
        width, seed = case["width"], case["seed"]
        run_dir = _existing_run(seed) if width == 96 else run_root / _name(width, seed)
        if not (run_dir / "summary.json").exists():
            if max_runs is not None and started >= max_runs:
                continue
            command = [sys.executable, "-m", "vit_lab.cli", "train",
                       "--config", case["config_path"], "--run-dir", str(run_dir),
                       "--device", "cpu"]
            if run_dir.exists():
                command.append("--resume")
            print(f"Starting {_name(width, seed)}: {' '.join(command)}", flush=True)
            subprocess.run(command, check=True)
            started += 1
        completed.append(export_case(case, run_dir=run_dir, result_dir=result_dir))
        if len({row["split_sha256"] for row in completed if row["seed"] == seed}) != 1:
            raise ValueError(f"Paired seed {seed} did not use an identical split")
        _write_progress(cases, result_dir, completed)
    _write_aggregate(completed, result_dir)
    _write_progress(cases, result_dir, completed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-runs", type=int)
    args = parser.parse_args()
    if not args.prepare_only and not args.run:
        parser.error("Choose --prepare-only or --run")
    if args.max_runs is not None and args.max_runs < 1:
        parser.error("--max-runs must be positive")
    paths = {
        "config_dir": Path("configs/experiments/width"),
        "run_root": Path("results/runs/width"),
        "result_dir": Path("results/ablations/width"),
    }
    if args.run:
        run_matrix(**paths, max_runs=args.max_runs)
    else:
        cases = prepare_configs(paths["config_dir"])
        print(f"Prepared {len(cases)} frozen width cases")


if __name__ == "__main__":
    main()
