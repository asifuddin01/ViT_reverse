"""Compare matching activations throughout the pretrained reference and our ViT."""

from __future__ import annotations

import csv
import json
import platform
from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn.functional as F
from PIL import Image

from vit_lab.models.vision_transformer import VisionTransformer
from vit_lab.reference.loader import load_reference_model
from vit_lab.reference.weight_map import load_exact_reference_weights


def _capture(model: torch.nn.Module, image: torch.Tensor) -> dict[str, torch.Tensor]:
    """Capture common module boundaries for one inference call."""
    values = {"input": image.detach().cpu()}
    handles = []
    modules = dict(model.named_modules())
    names = [
        "patch_embed", "pos_drop", "blocks.0.norm1", "blocks.0.attn.qkv",
        "blocks.0.attn.attn_drop", "blocks.0.attn.proj", "blocks.0.norm2",
        "blocks.0.mlp.fc1", "blocks.0.mlp.fc2",
    ]
    names.extend(f"blocks.{i}" for i in range(len(model.blocks)))
    names.extend(["norm", "head"])

    def save(name: str):
        def hook(_module: torch.nn.Module, _args: tuple[Any, ...], result: torch.Tensor) -> None:
            values[name] = result.detach().cpu()
        return hook

    def save_pooled(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        values["pooled_features"] = args[0].detach().cpu()

    try:
        for name in names:
            handles.append(modules[name].register_forward_hook(save(name)))
        handles.append(modules["head"].register_forward_pre_hook(save_pooled))
        with torch.inference_mode():
            model(image)
    finally:
        for handle in handles:
            handle.remove()
    return values


def _metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, float | bool]:
    if actual.shape != expected.shape:
        raise ValueError(f"Activation shapes differ: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    difference = (actual - expected).float()
    absolute = difference.abs()
    denominator = expected.float().abs().clamp_min(max(atol, 1e-8))
    if actual.numel() == 0:
        raise ValueError("Cannot compare empty activation")
    actual_flat = actual.float().reshape(1, -1)
    expected_flat = expected.float().reshape(1, -1)
    cosine = (
        1.0
        if actual_flat.norm() == 0 and expected_flat.norm() == 0
        else F.cosine_similarity(actual_flat, expected_flat).item()
    )
    cosine = max(-1.0, min(1.0, cosine))
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        passed = True
    except AssertionError:
        passed = False
    return {
        "max_absolute_error": absolute.max().item(),
        "mean_absolute_error": absolute.mean().item(),
        "mean_squared_error": difference.square().mean().item(),
        "mean_relative_error": (absolute / denominator).mean().item(),
        "cosine_similarity": cosine,
        "passed": passed,
    }


def compare_reference(
    config: dict[str, Any],
    *,
    output: Path,
    mapping_output: Path,
    image_path: Path | None = None,
    local_files_only: bool = False,
    device: str = "cpu",
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> list[dict[str, Any]]:
    """Write layer metrics and a complete, explicit weight mapping report."""
    if rtol < 0 or atol < 0:
        raise ValueError("Comparison tolerances must be nonnegative")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    reference, hub_config, checksum = load_reference_model(
        config["reference"],
        local_files_only=local_files_only,
        manual_attention=True,
    )
    target = VisionTransformer(**config["model"])
    mapping = load_exact_reference_weights(target, reference)
    reference.to(device).eval()
    target.to(device).eval()

    data_config = timm.data.resolve_data_config(hub_config["pretrained_cfg"])
    transform = timm.data.create_transform(**data_config, is_training=False)
    synthetic = Image.new("RGB", (256, 256), (127, 64, 192))
    torch.manual_seed(7)
    samples = {
        "random_seed7": torch.randn(1, 3, 224, 224),
        "zeros": torch.zeros(1, 3, 224, 224),
        "synthetic_rgb": transform(synthetic).unsqueeze(0),
    }
    if image_path is not None:
        with Image.open(image_path) as image:
            samples[f"image:{image_path.name}"] = transform(image.convert("RGB")).unsqueeze(0)

    rows = []
    for sample_name, image in samples.items():
        image = image.to(device)
        expected = _capture(reference, image)
        actual = _capture(target, image)
        if expected.keys() != actual.keys():
            raise ValueError("Captured stage names differ")
        for stage, expected_tensor in expected.items():
            actual_tensor = actual[stage]
            rows.append({
                "sample": sample_name,
                "stage": stage,
                "shape": str(tuple(expected_tensor.shape)),
                "dtype": str(expected_tensor.dtype),
                **_metrics(actual_tensor, expected_tensor, rtol=rtol, atol=atol),
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    mapping_output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with mapping_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)
    metadata = {
        "reference": config["reference"],
        "checkpoint_sha256": checksum,
        "model": config["model"],
        "device": device,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "rtol": rtol,
        "atol": atol,
        "sample_names": list(samples),
        "all_passed": all(row["passed"] for row in rows),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return rows
