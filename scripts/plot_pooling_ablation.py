"""Plot the paired pooling experiment as a standalone SVG or PNG."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_csv", type=Path)
    parser.add_argument("output", type=Path, help="SVG or PNG output path")
    args = parser.parse_args()
    with args.aggregate_csv.open(newline="", encoding="utf-8") as source:
        aggregate = {row["pooling"]: row for row in csv.DictReader(source)}
    if set(aggregate) != {"cls", "mean"} or any(
        row["seeds"] != "7,11,19" for row in aggregate.values()
    ):
        raise ValueError("Expected complete CLS/mean aggregates for paired seeds")
    scores = {}
    for seed in (7, 11, 19):
        scores[seed] = [
            json.loads((args.aggregate_csv.parent /
                        f"pooling_{mode}_seed{seed}_summary.json").read_text())[
                            "best_validation_accuracy"] * 100
            for mode in ("cls", "mean")
        ]
    means = [float(aggregate[mode]["validation_accuracy_mean"]) * 100
             for mode in ("cls", "mean")]
    deviations = [float(aggregate[mode]["validation_accuracy_sample_std"]) * 100
                  for mode in ("cls", "mean")]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.24, top=0.87)
    for seed, values in scores.items():
        ax.plot((0, 1), values, "o-", color="#888888", alpha=0.6,
                linewidth=1.5, markersize=4, label=f"Seed {seed}")
    ax.errorbar((0, 1), means, yerr=deviations, fmt="o-", capsize=7,
                color="#1767b3", ecolor="#dd6b20", linewidth=2.5, markersize=8,
                label="Mean ± sample SD")
    ax.set_xticks((0, 1), ("CLS token", "Mean of patch tokens"))
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(min(v - d for v, d in zip(means, deviations)) - 4,
                max(v + d for v, d in zip(means, deviations)) + 4)
    ax.set_ylabel("Best validation accuracy (%)")
    ax.set_title("Pooling: three paired CIFAR-10 seeds")
    ax.grid(axis="y", alpha=0.25)
    for x, mean, deviation in zip((0, 1), means, deviations):
        ax.annotate(f"{mean:.2f} ± {deviation:.2f}", (x, mean),
                    xytext=(0, 14), textcoords="offset points", ha="center")
    ax.legend(loc="lower right", fontsize=8)
    fig.text(0.5, 0.045, "20 epochs · patch size 8 · width 96 · validation only",
             ha="center", fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    if args.output.suffix.lower() == ".svg":
        cleaned = "\n".join(line.rstrip() for line in args.output.read_text().splitlines())
        args.output.write_text(cleaned + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
