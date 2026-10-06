"""Non-overlapping image patches projected into token embeddings."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PatchEmbedding(nn.Module):
    """Project each non-overlapping square patch with one shared linear map."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        *,
        strict_image_size: bool = True,
    ) -> None:
        super().__init__()
        if min(image_size, patch_size, in_channels, embed_dim) <= 0:
            raise ValueError("All patch embedding dimensions must be positive")
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.strict_image_size = strict_image_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def _check_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got shape {tuple(x.shape)}")
        _, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {channels}")
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Image height and width must be divisible by patch_size")
        if self.strict_image_size and (height, width) != (self.image_size, self.image_size):
            raise ValueError(
                f"Expected {self.image_size}x{self.image_size}, got {height}x{width}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return `[batch, patch_count, embed_dim]` in row-major grid order."""
        self._check_input(x)
        return self.proj(x).flatten(2).transpose(1, 2)

    def forward_explicit(self, x: torch.Tensor) -> torch.Tensor:
        """Educational equivalent using `unfold` and an explicit linear projection."""
        self._check_input(x)
        patches = F.unfold(
            x, kernel_size=self.patch_size, stride=self.patch_size
        ).transpose(1, 2)
        return F.linear(patches, self.proj.weight.flatten(1), self.proj.bias)
