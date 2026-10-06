"""Inspect the exact pretrained reference without depending on default model tags."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_reference(
    reference: dict[str, str],
    *,
    load_weights: bool = True,
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load a pinned timm ViT and return its architecture and tensor inventory."""
    repo_id = reference["repo_id"]
    revision = reference["revision"]
    download_args = {
        "repo_id": repo_id,
        "revision": revision,
        "local_files_only": local_files_only,
        "cache_dir": cache_dir,
    }
    config_path = hf_hub_download(filename="config.json", **download_args)
    with Path(config_path).open(encoding="utf-8") as handle:
        hub_config = json.load(handle)

    architecture = hub_config["architecture"]
    if architecture != reference["model_name"]:
        raise ValueError(
            f"Reference architecture mismatch: {architecture} != {reference['model_name']}"
        )
    model = timm.create_model(
        architecture,
        pretrained=False,
        pretrained_cfg=hub_config["pretrained_cfg"],
        num_classes=hub_config["num_classes"],
        global_pool=hub_config["global_pool"],
    )

    checkpoint_sha256 = None
    if load_weights:
        checkpoint_path = hf_hub_download(filename="model.safetensors", **download_args)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        expected = reference["checkpoint_sha256"]
        if checkpoint_sha256 != expected:
            raise ValueError(
                f"Checkpoint SHA-256 mismatch: {checkpoint_sha256} != {expected}"
            )
        model.load_state_dict(load_file(checkpoint_path), strict=True)

    model.eval()
    data_config = timm.data.resolve_data_config(model.pretrained_cfg)
    transform = timm.data.create_transform(**data_config, is_training=False)
    image = Image.new("RGB", (256, 256), (127, 64, 192))
    inputs = transform(image).unsqueeze(0)
    with torch.inference_mode():
        features = model.forward_features(inputs)
        logits = model.forward_head(features)

    inventory: dict[str, Any] = {
        "reference": {
            "repo_id": repo_id,
            "revision": revision,
            "architecture": architecture,
            "checkpoint_loaded": load_weights,
            "checkpoint_sha256": checkpoint_sha256,
            "hub_config": hub_config,
            "resolved_data_config": data_config,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": version("torchvision"),
            "timm": timm.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
        },
        "model": {
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "global_pool": model.global_pool,
            "has_class_token": model.has_class_token,
            "num_prefix_tokens": model.num_prefix_tokens,
            "norm_pre": type(model.norm_pre).__name__,
            "norm": type(model.norm).__name__,
            "fc_norm": type(model.fc_norm).__name__,
            "blocks": len(model.blocks),
            "first_block_norm_eps": model.blocks[0].norm1.eps,
            "first_block_qkv_bias": model.blocks[0].attn.qkv.bias is not None,
            "first_block_fused_attention": model.blocks[0].attn.fused_attn,
            "first_block_attention_dropout": model.blocks[0].attn.attn_drop.p,
            "first_block_drop_path": type(model.blocks[0].drop_path1).__name__,
        },
        "input_shape": list(inputs.shape),
        "feature_shape": list(features.shape),
        "logit_shape": list(logits.shape),
        "logit_prefix": logits[0, :5].tolist(),
        "state_dict": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in model.state_dict().items()
        },
        "modules": [
            {"name": name, "type": type(module).__name__}
            for name, module in model.named_modules()
        ],
    }
    return inventory
