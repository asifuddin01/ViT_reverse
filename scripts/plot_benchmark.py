"""Plot measured resolution scaling from the versioned benchmark CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def panel(rows: list[dict[str, str]], x0: int, title: str, key: str,
          divisor: float, unit: str, color: str) -> str:
    left, top, width, height = x0 + 65, 70, 420, 275
    x_values = [int(row["tokens"]) for row in rows]
    y_values = [float(row[key]) / divisor for row in rows]
    maximum = max(y_values) * 1.12
    x_min, x_max = min(x_values), max(x_values)

    def point(x: int, y: float) -> tuple[float, float]:
        return (left + (x - x_min) * width / max(x_max - x_min, 1),
                top + height - y * height / maximum)

    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in
                      (point(x, y) for x, y in zip(x_values, y_values)))
    output = [
        f'<text x="{left}" y="35" class="title">{title}</text>',
        f'<path d="M{left} {top}V{top + height}H{left + width}" class="axis"/>',
    ]
    for step in range(5):
        value = maximum * step / 4
        y = top + height - step * height / 4
        output.extend([
            f'<path d="M{left} {y:.1f}H{left + width}" class="grid"/>',
            f'<text x="{left - 8}" y="{y + 5:.1f}" text-anchor="end" class="tick">{value:.0f}</text>',
        ])
    output.append(f'<polyline points="{points}" stroke="{color}" class="line"/>')
    for tokens, value in zip(x_values, y_values):
        x, y = point(tokens, value)
        output.extend([
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>',
            f'<text x="{x:.1f}" y="{top + height + 24}" text-anchor="middle" '
            f'class="tick">{tokens}</text>',
            f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" '
            f'class="tick">{value:.1f}</text>',
        ])
    output.append(f'<text x="{left + width / 2:.1f}" y="{top + height + 53}" '
                  f'text-anchor="middle" class="tick">Tokens including CLS ({unit})</text>')
    return "\n".join(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.benchmark_csv.open(newline="") as source:
        rows = [row for row in csv.DictReader(source)
                if row["implementation"] == "educational" and row["batch_size"] == "1"]
    rows.sort(key=lambda row: int(row["tokens"]))
    if len(rows) < 2:
        raise ValueError("Need at least two educational batch-1 resolutions")
    svg = "\n".join([
        '<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="440" viewBox="0 0 1050 440">',
        '<style>text{font-family:Arial,sans-serif;fill:#243247}.title{font-size:21px;font-weight:700}'
        '.tick{font-size:13px}.axis{fill:none;stroke:#63758b;stroke-width:2}'
        '.grid{fill:none;stroke:#e3e9f0;stroke-width:1}.line{fill:none;stroke-width:3}'
        '</style>',
        '<rect width="1050" height="440" fill="white"/>',
        panel(rows, 0, "Median latency (ms)", "median_latency_ms", 1, "batch 1", "#1767b3"),
        panel(rows, 525, "Sampled process RSS (MB)", "sampled_peak_rss_bytes",
              1_000_000, "batch 1", "#dd6b20"),
        '<text x="525" y="425" text-anchor="middle" class="tick">'
        'CPU FP32; process RSS is sampled during inference, not device allocation.</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
