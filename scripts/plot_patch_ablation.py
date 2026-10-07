"""Plot paired-seed patch-size validation accuracy with sample SD bars."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.aggregate_csv.open(newline="", encoding="utf-8") as source:
        rows = sorted(csv.DictReader(source), key=lambda row: int(row["patch_size"]))
    if len(rows) != 3 or [int(row["patch_size"]) for row in rows] != [4, 8, 16]:
        raise ValueError("Expected the complete P=4,8,16 aggregate")
    left, top, width, height = 85, 90, 620, 340
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="550" viewBox="0 0 800 550">',
        '<style>text{font-family:Arial,sans-serif;fill:#243247}.title{font-size:23px;font-weight:700}'
        '.tick{font-size:15px}.axis{fill:none;stroke:#63758b;stroke-width:2}'
        '.grid{fill:none;stroke:#e3e9f0;stroke-width:1}.bar{stroke:#1767b3;stroke-width:3}'
        '</style>',
        '<rect width="800" height="550" fill="white"/>',
        '<text x="85" y="43" class="title">CIFAR-10 patch-size ablation</text>',
        '<text x="85" y="66" class="tick">Validation accuracy, mean ± sample SD across three paired seeds</text>',
        f'<path d="M{left} {top}V{top + height}H{left + width}" class="axis"/>',
    ]
    for step in range(6):
        accuracy = step * 0.2
        y = top + height - accuracy * height
        elements.extend([
            f'<path d="M{left} {y:.1f}H{left + width}" class="grid"/>',
            f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" '
            f'class="tick">{accuracy:.1f}</text>',
        ])
    for index, row in enumerate(rows):
        x = left + width * (index + 1) / 4
        mean = float(row["validation_accuracy_mean"])
        sd = float(row["validation_accuracy_sample_std"])
        y = top + height - mean * height
        upper = top + height - min(1.0, mean + sd) * height
        lower = top + height - max(0.0, mean - sd) * height
        elements.extend([
            f'<path d="M{x:.1f} {upper:.1f}V{lower:.1f} '
            f'M{x - 12:.1f} {upper:.1f}H{x + 12:.1f} '
            f'M{x - 12:.1f} {lower:.1f}H{x + 12:.1f}" class="bar"/>',
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#dd6b20"/>',
            f'<text x="{x:.1f}" y="{top + height + 28}" text-anchor="middle" '
            f'class="tick">P={row["patch_size"]}</text>',
            f'<text x="{x:.1f}" y="{upper - 13:.1f}" text-anchor="middle" '
            f'class="tick">{mean:.3f} ± {sd:.3f}</text>',
        ])
    elements.extend([
        '<text x="400" y="508" text-anchor="middle" class="tick">Patch size on native 32×32 images</text>',
        '<text x="400" y="535" text-anchor="middle" class="tick">'
        '20 epochs; width 96; validation only; paired seeds 7, 11, 19</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(elements) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
