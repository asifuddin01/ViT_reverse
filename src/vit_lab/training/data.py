"""Fixed CIFAR-10 training and validation data protocol."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10


def stratified_indices(
    labels: Sequence[int], *, validation_fraction: float = 0.1, seed: int = 7
) -> tuple[list[int], list[int]]:
    """Return disjoint train/validation indices with each class represented."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    if not labels:
        raise ValueError("labels must not be empty")
    generator = torch.Generator().manual_seed(seed)
    train, validation = [], []
    for label in sorted(set(labels)):
        class_indices = [i for i, value in enumerate(labels) if value == label]
        if len(class_indices) < 2:
            raise ValueError(f"Class {label} has fewer than two examples")
        order = torch.randperm(len(class_indices), generator=generator).tolist()
        selected = [class_indices[i] for i in order]
        count = max(1, min(len(selected) - 1, round(len(selected) * validation_fraction)))
        validation.extend(selected[:count])
        train.extend(selected[count:])
    return sorted(train), sorted(validation)


def cifar10_transforms(image_size: int = 32) -> tuple[transforms.Compose, transforms.Compose]:
    """Train/validation transforms; sizes above 32 bicubically upsample after augmentation.

    Upsampling adds tokens and compute, not image detail.
    """
    if image_size < 32:
        raise ValueError("image_size must be at least the native 32 pixels")
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    resize = ([] if image_size == 32 else
              [transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC,
                                 antialias=True)])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        *resize,
        normalize,
    ])
    validation_transform = transforms.Compose([transforms.ToTensor(), *resize, normalize])
    return train_transform, validation_transform


def cifar10_loaders(
    *,
    root: str | Path,
    batch_size: int,
    seed: int,
    generator: torch.Generator,
    download: bool = True,
    device: str = "cpu",
    image_size: int = 32,
) -> tuple[DataLoader, DataLoader, dict[str, object]]:
    """Create separate augmented-train and deterministic-validation views."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    train_transform, validation_transform = cifar10_transforms(image_size)
    root = Path(root)
    train_full = CIFAR10(root=root, train=True, transform=train_transform, download=download)
    validation_full = CIFAR10(
        root=root, train=True, transform=validation_transform, download=False
    )
    train_indices, validation_indices = stratified_indices(train_full.targets, seed=seed)
    pin_memory = device == "cuda"
    train_loader = DataLoader(
        Subset(train_full, train_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        Subset(validation_full, validation_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    split = {
        "dataset": "CIFAR-10 official training split",
        "seed": seed,
        "validation_fraction": 0.1,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "train_class_counts": dict(Counter(train_full.targets[i] for i in train_indices)),
        "validation_class_counts": dict(Counter(train_full.targets[i] for i in validation_indices)),
        "normalization_mean": [0.5, 0.5, 0.5],
        "normalization_std": [0.5, 0.5, 0.5],
    }
    return train_loader, validation_loader, split
