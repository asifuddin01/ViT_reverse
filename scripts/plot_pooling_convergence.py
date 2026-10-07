"""Plot mean validation accuracy by epoch for paired pooling variants."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _history(result_dir: Path, mode: str) -> tuple[list[int], list[float], list[float]]:
    histories = []
    for seed in (7, 11, 19):
        with (result_dir / f"pooling_{mode}_seed{seed}_metrics.csv").open(
            newline="", encoding="utf-8"
        ) as source:
            rows = list(csv.DictReader(source))
        if [int(row["epoch"]) for row in rows] != list(range(1, 21)):
            raise ValueError(f"Incomplete 20-epoch history for {mode}, seed {seed}")
        histories.append([float(row["val_accuracy"]) * 100 for row in rows])
    epochs = list(range(1, 21))
    means = [statistics.mean(values) for values in zip(*histories)]
    deviations = [statistics.stdev(values) for values in zip(*histories)]
    return epochs, means, deviations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("output", type=Path, help="SVG or PNG output path")
    args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.22, top=0.88)
    for mode, label, color in (("cls", "CLS token", "#1767b3"),
                               ("mean", "Mean patch tokens", "#dd6b20")):
        epochs, means, deviations = _history(args.result_dir, mode)
        ax.plot(epochs, means, color=color, linewidth=2.2, label=label)
        ax.fill_between(epochs,
                        [mean - deviation for mean, deviation in zip(means, deviations)],
                        [mean + deviation for mean, deviation in zip(means, deviations)],
                        color=color, alpha=0.15)
    ax.set_xticks((1, 5, 10, 15, 20))
    ax.set_xlim(1, 20)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Pooling ablation: validation accuracy through training")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.text(0.5, 0.045, "Mean ± sample SD across seeds 7, 11, 19 · validation only",
             ha="center", fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    if args.output.suffix.lower() == ".svg":
        cleaned = "\n".join(line.rstrip() for line in args.output.read_text().splitlines())
        args.output.write_text(cleaned + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
