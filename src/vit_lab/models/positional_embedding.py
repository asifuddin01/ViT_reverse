"""Learned absolute position handling for patch-token sequences."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def resize_position_embedding(
    positions: torch.Tensor,
    *,
    old_grid: tuple[int, int],
    new_grid: tuple[int, int],
    prefix_tokens: int = 1,
) -> torch.Tensor:
    """Interpolate patch positions while preserving prefix-token positions."""
    if positions.ndim != 3 or positions.shape[0] != 1:
        raise ValueError("positions must have shape [1, tokens, embed_dim]")
    if positions.shape[1] != prefix_tokens + old_grid[0] * old_grid[1]:
        raise ValueError("positions length does not match old_grid")
    if min(*old_grid, *new_grid) <= 0:
        raise ValueError("grid dimensions must be positive")
    if old_grid == new_grid:
        return positions
    prefix = positions[:, :prefix_tokens]
    patches = positions[:, prefix_tokens:]
    patches = patches.reshape(1, old_grid[0], old_grid[1], -1).permute(0, 3, 1, 2)
    patches = F.interpolate(patches, size=new_grid, mode="bicubic", align_corners=False)
    patches = patches.permute(0, 2, 3, 1).reshape(1, -1, positions.shape[-1])
    return torch.cat((prefix, patches), dim=1)


def add_position(tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Add one learned position per token without implicit shape broadcasting."""
    if tokens.ndim != 3 or positions.shape != (1, tokens.shape[1], tokens.shape[2]):
        raise ValueError(
            f"Expected positions [1,{tokens.shape[1]},{tokens.shape[2]}], "
            f"got {tuple(positions.shape)}"
        )
    return tokens + positions
