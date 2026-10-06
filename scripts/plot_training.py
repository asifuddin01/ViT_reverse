"""Render the saved training history as a self-contained SVG figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def polyline(rows: list[dict[str, float]], key: str, x0: int, y0: int, width: int,
             height: int, lower: float, upper: float) -> str:
    count = len(rows)
    return " ".join(
        f"{x0 + (index * width / max(count - 1, 1)):.1f},"
        f"{y0 + height - (row[key] - lower) * height / (upper - lower):.1f}"
        for index, row in enumerate(rows)
    )


def panel(rows: list[dict[str, float]], x0: int, title: str,
          series: tuple[tuple[str, str, str], ...], lower: float, upper: float) -> str:
    left, top, width, height = x0 + 58, 75, 450, 300
    elements = [
        f'<text x="{left}" y="35" class="title">{title}</text>',
        f'<path d="M{left} {top}V{top + height}H{left + width}" class="axis"/>',
    ]
    for step in range(5):
        value = lower + (upper - lower) * step / 4
        y = top + height - step * height / 4
        elements.extend([
            f'<path d="M{left} {y:.1f}H{left + width}" class="grid"/>',
            f'<text x="{left - 8}" y="{y + 5:.1f}" text-anchor="end" class="tick">{value:.2f}</text>',
        ])
    for index in (0, len(rows) // 2, len(rows) - 1):
        x = left + index * width / max(len(rows) - 1, 1)
        elements.append(
            f'<text x="{x:.1f}" y="{top + height + 24}" text-anchor="middle" '
            f'class="tick">{int(rows[index]["epoch"])}</text>'
        )
    for position, (key, label, color) in enumerate(series):
        points = polyline(rows, key, left, top, width, height, lower, upper)
        legend_x = left + position * 130
        elements.extend([
            f'<polyline points="{points}" stroke="{color}" class="line"/>',
            f'<path d="M{legend_x} {top + height + 52}h22" stroke="{color}" class="line"/>',
            f'<text x="{legend_x + 29}" y="{top + height + 57}" class="legend">{label}</text>',
        ])
    return "\n".join(elements)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="metrics.csv from a completed training run")
    parser.add_argument("output", type=Path, help="SVG output path")
    args = parser.parse_args()
    with args.metrics.open(newline="") as source:
        rows = [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(source)]
    if not rows:
        raise ValueError("metrics.csv contains no epochs")
    losses = [row[key] for row in rows for key in ("train_loss", "val_loss")]
    upper_loss = max(losses) * 1.05
    svg = "\n".join([
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="465" viewBox="0 0 1120 465">',
        '<style>text{font-family:Arial,sans-serif;fill:#243247}.title{font-size:22px;font-weight:700}'
        '.tick{font-size:13px}.legend{font-size:14px}.axis{fill:none;stroke:#63758b;stroke-width:2}'
        '.grid{fill:none;stroke:#e3e9f0;stroke-width:1}.line{fill:none;stroke-width:3;stroke-linejoin:round}'
        '</style>',
        '<rect width="1120" height="465" fill="white"/>',
        panel(rows, 0, "Cross-entropy loss", (("train_loss", "Train", "#1767b3"),
                                             ("val_loss", "Validation", "#dd6b20")), 0, upper_loss),
        panel(rows, 560, "Accuracy", (("train_accuracy", "Train", "#1767b3"),
                                       ("val_accuracy", "Validation", "#dd6b20")), 0, 1),
        '<text x="560" y="453" text-anchor="middle" class="tick">Epoch</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg + "\n")


if __name__ == "__main__":
    main()
