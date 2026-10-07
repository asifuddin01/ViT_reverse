"""Compare trained CLS features before and after a whole-patch permutation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torchvision import transforms
from torchvision.datasets import CIFAR10

from vit_lab.models.vision_transformer import VisionTransformer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _measure(mode: str, images: torch.Tensor, swapped: torch.Tensor,
             *, batch_size: int) -> dict[str, object]:
    if mode == "learned":
        config_path = Path("configs/experiments/patch_size/patch_p08_seed7.yaml")
        checkpoint_path = Path("results/runs/patch_size/patch_p08_seed7/checkpoint.pt")
    else:
        config_path = Path("configs/experiments/position/position_none_seed7.yaml")
        checkpoint_path = Path("results/runs/position/position_none_seed7/checkpoint.pt")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = VisionTransformer(**config["model"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    original_features, swapped_features = [], []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            original_features.append(model.forward_features(images[start:start + batch_size])[:, 0])
            swapped_features.append(model.forward_features(swapped[start:start + batch_size])[:, 0])
    original_features = torch.cat(original_features)
    swapped_features = torch.cat(swapped_features)
    with torch.inference_mode():
        original_logits = model.head(original_features)
        swapped_logits = model.head(swapped_features)
    delta = (original_logits - swapped_logits).abs()
    result = {
        "position_embedding": mode,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "mean_absolute_logit_change": delta.mean().item(),
        "maximum_absolute_logit_change": delta.max().item(),
        "prediction_flips": int((original_logits.argmax(1) != swapped_logits.argmax(1)).sum().item()),
        "mean_cls_feature_cosine_similarity": F.cosine_similarity(
            original_features, swapped_features, dim=1).mean().item(),
    }
    if mode == "none" and result["maximum_absolute_logit_change"] > 1e-4:
        raise AssertionError("Position-free model changed under a whole-patch permutation")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=Path("results/ablations/position/patch_permutation.json"))
    args = parser.parse_args()
    if args.sample_count <= 0 or args.batch_size <= 0 or args.threads <= 0:
        parser.error("sample-count, batch-size, and threads must be positive")
    torch.set_num_threads(args.threads)
    learned_split_path = Path("results/runs/patch_size/patch_p08_seed7/split_indices.json")
    none_split_path = Path("results/runs/position/position_none_seed7/split_indices.json")
    learned_split = json.loads(learned_split_path.read_text(encoding="utf-8"))
    none_split = json.loads(none_split_path.read_text(encoding="utf-8"))
    if learned_split != none_split:
        raise ValueError("The seed-7 position variants used different train/validation splits")
    indices = learned_split["validation_indices"][:args.sample_count]
    if len(indices) != args.sample_count:
        raise ValueError("Requested more images than the saved validation split contains")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = CIFAR10(root="data", train=True, download=False, transform=transform)
    images = torch.stack([dataset[index][0] for index in indices])
    swapped = images.clone()
    swapped[:, :, :, :8] = images[:, :, :, 8:16]
    swapped[:, :, :, 8:16] = images[:, :, :, :8]
    cases = [_measure(mode, images, swapped, batch_size=args.batch_size)
             for mode in ("learned", "none")]
    result = {
        "dataset": "CIFAR-10 official training split, seed-7 held-out validation indices",
        "selection": "first saved validation indices in ascending order",
        "sample_count": args.sample_count,
        "split_sha256": hashlib.sha256(json.dumps(learned_split, sort_keys=True).encode()).hexdigest(),
        "transform": "ToTensor then (x-0.5)/0.5 per channel",
        "permutation": "swap the first two complete 8-pixel-wide patch columns in each 32x32 image",
        "device": "cpu",
        "torch_version": torch.__version__,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in cases:
        print(f"{row['position_embedding']}: mean |Δlogit| "
              f"{row['mean_absolute_logit_change']:.6f}, "
              f"prediction flips {row['prediction_flips']}/{args.sample_count}")


if __name__ == "__main__":
    main()
