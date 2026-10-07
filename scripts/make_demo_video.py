"""Render docs/demo/vit_lab_demo.mp4 from captured terminal logs, figures, and screenshots.

Terminal scenes replay real outputs saved in docs/demo/logs/ (regenerate them with the
commands shown in each scene). Requires `pip install imageio-ffmpeg`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 10
BG, FG, DIM, ACCENT, GREEN = (13, 17, 23), (230, 237, 243), (139, 148, 158), (88, 166, 255), (63, 185, 80)


def _font(names: list[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


MONO = _font(["/System/Library/Fonts/Menlo.ttc", "DejaVuSansMono.ttf"], 19)
SANS = _font(["/System/Library/Fonts/Helvetica.ttc", "DejaVuSans.ttf"], 26)
TITLE = _font(["/System/Library/Fonts/Helvetica.ttc", "DejaVuSans.ttf"], 52)


def card(lines: list[tuple[str, ImageFont.ImageFont, tuple[int, int, int]]]) -> Image.Image:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    heights = [draw.textbbox((0, 0), text, font=font)[3] + 18 for text, font, _ in lines]
    y = (H - sum(heights)) // 2
    for (text, font, color), height in zip(lines, heights):
        width = draw.textlength(text, font=font)
        draw.text(((W - width) / 2, y), text, font=font, fill=color)
        y += height
    return image


def terminal(command: str, output: list[str], caption: str) -> list[Image.Image]:
    """Type the command, then reveal output lines; the last frame is held by the caller."""
    frames = []

    def render(typed: str, shown: list[str], cursor: bool) -> Image.Image:
        image = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, W, 34), fill=(22, 27, 34))
        for i, color in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
            draw.ellipse((16 + 22 * i, 11, 28 + 22 * i, 23), fill=color)
        draw.text((W / 2 - 120, 8), "ViT_reverse — zsh", font=MONO, fill=DIM)
        y = 52
        draw.text((24, y), "$", font=MONO, fill=GREEN)
        draw.text((44, y), typed + ("▌" if cursor else ""), font=MONO, fill=FG)
        y += 30
        for line in shown[-(int((H - 150) / 22)):]:
            color = GREEN if ("passed" in line or "Failed comparisons: 0" in line) else FG
            draw.text((24, y), line[:110], font=MONO, fill=color)
            y += 22
        draw.rectangle((0, H - 64, W, H), fill=(22, 27, 34))
        draw.text((24, H - 48), caption, font=SANS, fill=ACCENT)
        return image

    step = max(1, len(command) // 12)
    for end in range(0, len(command) + step, step):
        frames.append(render(command[:end], [], True))
    for count in range(1, len(output) + 1, 2):
        frames.append(render(command, output[:count], False))
    frames.append(render(command, output, False))
    return frames


def picture(path: Path, caption: str) -> Image.Image:
    image = Image.new("RGB", (W, H), (255, 255, 255))
    source = Image.open(path).convert("RGB")
    source.thumbnail((W - 40, H - 100), Image.LANCZOS)
    image.paste(source, ((W - source.width) // 2, 20 + (H - 100 - source.height) // 2))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, H - 64, W, H), fill=BG)
    draw.text((24, H - 48), caption, font=SANS, fill=FG)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/demo/vit_lab_demo.mp4"))
    args = parser.parse_args()
    logs = Path("docs/demo/logs")
    shots = Path("docs/demo/screenshots")

    def read(name: str) -> list[str]:
        return (logs / name).read_text(encoding="utf-8").rstrip("\n").splitlines()

    scenes: list[tuple[list[Image.Image], float]] = [
        ([card([("Inside the Vision Transformer", TITLE, FG),
                ("A from-scratch PyTorch ViT, verified layer by layer", SANS, DIM),
                ("against the pinned timm ViT-Base/16 checkpoint", SANS, DIM),
                ("github.com/asifuddin01/ViT_reverse", SANS, ACCENT)])], 3.5),
        (terminal("pytest -q", read("pytest.txt"), "34 tests: components, gradients, timm oracle, ablation freezes"), 2.5),
        (terminal("vit-lab compare --config configs/vit_base.yaml --offline --device cpu "
                  "--image data/cifar_train_00000.png", read("compare.txt"),
                  "152/152 tensors mapped · 100/100 stages · max |difference| 0.0 (CPU and T4)"), 3.5),
        (terminal("python scripts/demo.py --offline", read("demo.txt"),
                  "Real photos through the from-scratch model with verified weights"), 4.0),
        ([picture(Path("docs/demo/demo_predictions.png"),
                  "Top-3 classes and last-layer CLS attention (a diagnostic, not an explanation)")], 5.0),
    ]
    for path, caption in [
        ("results/figures/patch_size_ablation.png", "Patch size, width 96, 3 paired seeds: 71.6 / 63.4 / 55.9% (P=4/8/16)"),
        ("results/figures/position_ablation.png", "Learned positions beat none in every seed: 63.4 vs 56.0%"),
        ("results/figures/pooling_ablation.png", "Mean patch pooling beat CLS in every seed: 65.2 vs 63.4%"),
        ("results/figures/width_ablation.png", "Width 192 beat width 96 in every seed: 69.0 vs 63.4%"),
        ("results/colab_t4/width192_patch/width192_patch_accuracy.png", "T4: width-192 patch matrix 75.3 / 68.7 / 59.3% (P=4/8/16)"),
        ("results/colab_t4/upsampled128/upsampled128_accuracy.png", "T4: 128-px upsampled CIFAR-10 — more tokens, no new detail"),
        ("results/colab_t4/retina_full/retina_full_transfer.png", "T4: full RetinaMNIST fine-tune, test QWK 0.773 ± 0.038 (3 seeds)"),
        (shots / "colab_02_cuda_equivalence.png", "Colab T4: 100/100 equivalence stages, max difference 0.0"),
        (shots / "colab_03_cuda_benchmark.png", "Colab T4: 15.0 ms educational vs 14.6 ms timm fused (224 px, batch 1)"),
        (shots / "colab_05_retina_full_finetune.png", "Colab T4: RetinaMNIST methods side by side (test set, n = 400)"),
    ]:
        scenes.append(([picture(Path(path), caption)], 3.5))
    scenes.append(([card([("Every number comes from a saved run", SANS, FG),
                          ("configs · seeds · splits · environment · predictions", SANS, DIM),
                          ("CIFAR-10 test split never used for selection", SANS, DIM),
                          ("github.com/asifuddin01/ViT_reverse", SANS, ACCENT)])], 4.0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.output, fps=FPS, codec="libx264", quality=7,
                            macro_block_size=16) as writer:
        for frames, hold in scenes:
            for frame in frames:
                writer.append_data(np.asarray(frame))
            for _ in range(int(hold * FPS)):
                writer.append_data(np.asarray(frames[-1]))
    # Keyframes double as terminal screenshots for the README.
    scenes[2][0][-1].save(shots / "terminal_equivalence.png")
    scenes[3][0][-1].save(shots / "terminal_demo.png")
    total = sum(len(frames) + int(hold * FPS) for frames, hold in scenes) / FPS
    print(f"Wrote {args.output} ({total:.0f} s, {args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
