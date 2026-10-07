"""Plot the completed head-count experiment as a standalone SVG or PNG."""

from __future__ import annotations

import argparse
import csv
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
        rows = sorted(csv.DictReader(source), key=lambda row: int(row["heads"]))
    if [int(row["heads"]) for row in rows] != [4, 8, 12, 16]:
        raise ValueError("Expected the complete 4, 8, 12, 16-head aggregate")
    if any(row["seeds"] != "7,11,19" for row in rows):
        raise ValueError("Expected paired seeds 7, 11, and 19 for every head count")
    counts = [int(row["heads"]) for row in rows]
    means = [float(row["validation_accuracy_mean"]) * 100 for row in rows]
    deviations = [float(row["validation_accuracy_sample_std"]) * 100 for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.25, top=0.87)
    ax.errorbar(
        counts, means, yerr=deviations, fmt="o-", capsize=6,
        color="#1767b3", ecolor="#dd6b20", linewidth=2, markersize=7,
    )
    ax.set_xticks(counts)
    ax.set_xlabel("Attention heads at width 96 and patch size 8")
    ax.set_ylabel("Best validation accuracy (%)")
    ax.set_title("Head-count ablation: mean ± sample SD over three paired seeds")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(2, 18)
    ax.set_ylim(max(0, min(mean - sd for mean, sd in zip(means, deviations)) - 5),
                min(100, max(mean + sd for mean, sd in zip(means, deviations)) + 5))
    for heads, mean, deviation in zip(counts, means, deviations):
        ax.annotate(f"{mean:.2f} ± {deviation:.2f}", (heads, mean),
                    xytext=(0, 12), textcoords="offset points", ha="center")
    fig.text(0.5, 0.045, "20 epochs · patch size 8 · seeds 7, 11, 19 · validation only",
             ha="center", fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    if args.output.suffix.lower() == ".svg":
        cleaned = "\n".join(line.rstrip() for line in args.output.read_text().splitlines())
        args.output.write_text(cleaned + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
