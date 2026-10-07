"""Measure inference cost of the three trained seed-7 patch-size variants."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("results/ablations/patch_size/cost.csv"))
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    results = []
    for patch_size in (4, 8, 16):
        config_path = (
            Path("configs/vit_tiny_cifar.yaml") if patch_size == 4 else
            Path(f"configs/experiments/patch_size/patch_p{patch_size:02d}_seed7.yaml")
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        run_dir = (
            Path("results/runs/cifar_tiny_seed7") if patch_size == 4 else
            Path(f"results/runs/patch_size/patch_p{patch_size:02d}_seed7")
        )
        checkpoint = run_dir / "checkpoint.pt"
        if not checkpoint.exists() or not (run_dir / "summary.json").exists():
            raise FileNotFoundError(f"Completed seed-7 checkpoint missing: {checkpoint}")
        payload = {
            "config": config,
            "implementation": "trained_patch_model",
            "checkpoint_path": str(checkpoint),
            "resolution": 32,
            "batch_size": 1,
            "warmups": args.warmups,
            "trials": args.trials,
            "threads": args.threads,
            "offline": True,
        }
        completed = subprocess.run(
            [sys.executable, "-m", "vit_lab.benchmarking.runner", "--worker",
             json.dumps(payload)], capture_output=True, text=True, check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"P={patch_size} cost benchmark failed:\n{completed.stderr}")
        result = json.loads(completed.stdout)
        result["patch_size"] = patch_size
        result["seed_checkpoint"] = 7
        results.append(result)
        print(f"P={patch_size}: {result['median_latency_ms']:.3f} ms; "
              f"{result['macs_per_image'] / 1e6:.2f} MMAC", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = [{key: value for key, value in row.items() if key != "trial_latencies_ms"}
                for row in results]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    args.output.with_suffix(".json").write_text(
        json.dumps({
            "method": "Fresh process per architecture; seed-7 best checkpoint; "
                      "CPU FP32; batch 1; random 32×32 image; no preprocessing; "
                      "warmups then perf_counter_ns forward trials",
            "memory_method": "psutil process RSS sampled every 5 ms; runtime and allocator included",
            "cases": results,
        }, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
