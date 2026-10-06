"""Selective ViT attention overlays and CLS representation summaries."""

from __future__ import annotations

import csv
import gc
import json
import math
import platform
from pathlib import Path
from typing import Any, Sequence

import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.datasets import CIFAR10

from vit_lab.models.vision_transformer import VisionTransformer
from vit_lab.reference.loader import load_reference_model
from vit_lab.reference.weight_map import load_exact_reference_weights


def capture_selected(
    model: VisionTransformer,
    image: torch.Tensor,
    *,
    attention_layers: Sequence[int],
    heads: Sequence[int],
    hidden_layers: Sequence[int],
) -> tuple[torch.Tensor, dict[tuple[int, int], torch.Tensor], dict[int, torch.Tensor]]:
    """Capture selected pre-dropout CLS rows and post-block CLS vectors only.

    Layer numbers are one-based. Hooks are removed even if inference raises.
    """
    depth = len(model.blocks)
    num_heads = model.blocks[0].attn.num_heads
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Expected one image with shape [1, C, H, W]")
    if any(layer < 1 or layer > depth for layer in (*attention_layers, *hidden_layers)):
        raise ValueError(f"Layer numbers must be in 1..{depth}")
    if any(head < 0 or head >= num_heads for head in heads):
        raise ValueError(f"Head indices must be in 0..{num_heads - 1}")
    maps: dict[tuple[int, int], torch.Tensor] = {}
    hidden: dict[int, torch.Tensor] = {}
    handles = []

    def attention_hook(layer: int):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...],
                 _output: torch.Tensor) -> None:
            probabilities = inputs[0]  # Input to dropout is the pre-dropout softmax.
            for head in heads:
                row = probabilities[0, head, 0, :].detach().cpu().clone()
                if not torch.isfinite(row).all() or abs(row.sum().item() - 1.0) > 1e-5:
                    raise ValueError("Attention row is not finite or normalized")
                maps[layer, head] = row
        return hook

    def hidden_hook(layer: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...],
                 output: torch.Tensor) -> None:
            hidden[layer] = output[0, 0, :].detach().cpu().clone()
        return hook

    try:
        for layer in set(attention_layers):
            handles.append(model.blocks[layer - 1].attn.attn_drop.register_forward_hook(
                attention_hook(layer)
            ))
        for layer in set(hidden_layers):
            handles.append(model.blocks[layer - 1].register_forward_hook(hidden_hook(layer)))
        with torch.inference_mode():
            logits = model(image).detach().cpu()
    finally:
        for handle in handles:
            handle.remove()
    if len(maps) != len(set(attention_layers)) * len(set(heads)):
        raise RuntimeError("A requested attention map was not captured")
    if len(hidden) != len(set(hidden_layers)):
        raise RuntimeError("A requested hidden state was not captured")
    return logits, maps, hidden


def representation_rows(image_id: str, hidden: dict[int, torch.Tensor]) -> list[dict[str, Any]]:
    """Cosine to final CLS state and drift from the previous selected state."""
    layers = sorted(hidden)
    if not layers:
        raise ValueError("At least one hidden layer is required")
    final = hidden[layers[-1]].reshape(1, -1)
    rows = []
    for position, layer in enumerate(layers):
        current = hidden[layer].reshape(1, -1)
        cosine_final = F.cosine_similarity(current, final).item()
        previous = hidden[layers[position - 1]].reshape(1, -1) if position else None
        cosine_previous = (
            F.cosine_similarity(current, previous).item() if previous is not None else None
        )
        rows.append({
            "image_id": image_id,
            "layer": layer,
            "cosine_to_final": cosine_final,
            "cosine_to_previous_selected": cosine_previous,
            "drift_from_previous_selected": 1 - cosine_previous if cosine_previous is not None else None,
        })
    return rows


def render_overlay(
    input_tensor: torch.Tensor,
    row: torch.Tensor,
    *,
    mean: Sequence[float],
    std: Sequence[float],
    maximum: float,
    image_id: str,
    layer: int,
    head: int,
    model_id: str,
    output: Path,
) -> None:
    """Overlay absolute CLS-to-patch probabilities with a shared 0..max scale."""
    if input_tensor.ndim != 3 or input_tensor.shape[0] != 3:
        raise ValueError("Expected a preprocessed RGB image [3, H, W]")
    patches = row.numel() - 1  # Exclude the CLS-to-CLS probability.
    grid = math.isqrt(patches)
    if grid * grid != patches or maximum <= 0:
        raise ValueError("Patch count must form a square grid and scale must be positive")
    height, width = input_tensor.shape[-2:]
    if height != width:
        raise ValueError("Overlay requires a square input")
    mean_tensor = torch.tensor(mean).reshape(3, 1, 1)
    std_tensor = torch.tensor(std).reshape(3, 1, 1)
    rgb = ((input_tensor.cpu() * std_tensor + mean_tensor).clamp(0, 1) * 255)
    display_size = max(448, height)
    base = Image.fromarray(rgb.permute(1, 2, 0).byte().numpy(), mode="RGB")
    base = base.resize((display_size, display_size), Image.Resampling.BICUBIC)
    patch_map = row[1:].reshape(1, 1, grid, grid)
    upsampled = F.interpolate(patch_map, size=(display_size, display_size), mode="bilinear",
                             align_corners=False)[0, 0].clamp(0, maximum)
    alpha = (upsampled / maximum * 175).byte().numpy()
    overlay = Image.composite(Image.new("RGB", base.size, (255, 62, 0)),
                              base, Image.fromarray(alpha, mode="L"))
    canvas_width = max(600, display_size)
    image_x = (canvas_width - display_size) // 2
    canvas = Image.new("RGB", (canvas_width, display_size + 102), "white")
    canvas.paste(overlay, (image_x, 63))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    draw.text((8, 4), f"{image_id} | layer {layer} head {head} | CLS to patches",
              fill="black", font=font)
    draw.text((8, 29), f"{model_id} | pre-dropout | scale 0..{maximum:.4f}",
              fill="black", font=font)
    draw.text((8, display_size + 70),
              "Attention diagnostic; not an explanation or localization.",
              fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def analyze_cifar10(
    config: dict[str, Any],
    *,
    data_root: Path,
    output_dir: Path,
    indices: Sequence[int] = (0, 1, 2, 3),
    attention_layers: Sequence[int] = (1, 6, 12),
    heads: Sequence[int] = (0, 1),
    hidden_layers: Sequence[int] = (1, 3, 6, 9, 12),
    device: str = "cpu",
    offline: bool = False,
) -> dict[str, Any]:
    """Analyze a fixed CIFAR-10 training image set using the verified ViT-Base weights."""
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Provide distinct CIFAR-10 training indices")
    if not attention_layers or not heads or not hidden_layers:
        raise ValueError("Provide at least one attention layer, head, and hidden layer")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    dataset = CIFAR10(root=data_root, train=True, download=False)
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError(f"CIFAR-10 training indices must be in 0..{len(dataset) - 1}")
    reference, hub_config, checksum = load_reference_model(
        config["reference"], local_files_only=offline
    )
    model = VisionTransformer(**config["model"])
    load_exact_reference_weights(model, reference)
    del reference
    gc.collect()
    model.to(device).eval()
    data_config = timm.data.resolve_data_config(hub_config["pretrained_cfg"])
    transform = timm.data.create_transform(**data_config, is_training=False)
    mean, std = data_config["mean"], data_config["std"]
    all_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    first_maps: dict[tuple[int, int], torch.Tensor] = {}
    first_tensor: torch.Tensor | None = None
    first_logits: torch.Tensor | None = None
    for position, index in enumerate(indices):
        image, label = dataset[index]
        tensor = transform(image.convert("RGB"))
        image_id = f"cifar10_train_{index:05d}"
        logits, maps, hidden = capture_selected(
            model, tensor.unsqueeze(0).to(device),
            attention_layers=attention_layers if position == 0 else (),
            heads=heads if position == 0 else (), hidden_layers=hidden_layers,
        )
        all_rows.extend(representation_rows(image_id, hidden))
        if position == 0:
            first_maps, first_tensor, first_logits = maps, tensor, logits
            for (layer, head), row in sorted(maps.items()):
                map_rows.append({
                    "image_id": image_id, "layer": layer, "head": head,
                    "cls_to_cls_probability": row[0].item(),
                    "patch_probability_mass": row[1:].sum().item(),
                    "row_sum": row.sum().item(),
                    "max_patch_probability": row[1:].max().item(),
                })
    assert first_tensor is not None and first_logits is not None
    repeated_logits, repeated_maps, _ = capture_selected(
        model, first_tensor.unsqueeze(0).to(device),
        attention_layers=attention_layers, heads=heads, hidden_layers=(),
    )
    deterministic = torch.equal(first_logits, repeated_logits) and all(
        torch.equal(first_maps[key], repeated_maps[key]) for key in first_maps
    )
    if not deterministic:
        raise ValueError("Repeated inference was not bitwise deterministic")
    scale_max = max(row[1:].max().item() for row in first_maps.values())
    output_dir.mkdir(parents=True, exist_ok=True)
    for (layer, head), row in sorted(first_maps.items()):
        render_overlay(
            first_tensor, row, mean=mean, std=std, maximum=scale_max,
            image_id=f"CIFAR-10 train {indices[0]:05d}", layer=layer, head=head,
            model_id=f"{config['reference']['model_name']} sha256:{checksum[:12]}",
            output=output_dir / f"attention_cifar_train_{indices[0]:05d}_l{layer:02d}_h{head:02d}.png",
        )
    for name, rows in (("representation_drift.csv", all_rows),
                       ("attention_summary.csv", map_rows)):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    metadata = {
        "dataset": "CIFAR-10 official training split",
        "indices": list(indices),
        "labels": [int(dataset.targets[index]) for index in indices],
        "attention_layers_one_based": list(attention_layers),
        "heads_zero_based": list(heads),
        "hidden_layers_one_based": list(hidden_layers),
        "attention_source": "educational model, input to attention dropout (pre-dropout)",
        "attention_query": "CLS token, patch keys only in overlays",
        "preprocessing": data_config,
        "reference": config["reference"],
        "checkpoint_sha256": checksum,
        "device": device,
        "torch": torch.__version__,
        "timm": timm.__version__,
        "platform": platform.platform(),
        "shared_attention_scale_min": 0.0,
        "shared_attention_scale_max": scale_max,
        "bitwise_deterministic_first_image_repeat": deterministic,
        "interpretation": "Attention diagnostic only; not an explanation or localization.",
    }
    (output_dir / "attention_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata
