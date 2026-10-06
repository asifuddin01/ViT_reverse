"""Token pooling before the classification head."""

from __future__ import annotations

import torch


def pool_tokens(tokens: torch.Tensor, method: str = "cls") -> torch.Tensor:
    """Select CLS or average patch tokens for classification."""
    if tokens.ndim != 3 or tokens.shape[1] < 1:
        raise ValueError("Expected nonempty token tensor [B, N, D]")
    if method == "cls":
        return tokens[:, 0]
    if method == "mean":
        return tokens.mean(dim=1)
    raise ValueError(f"Unknown pooling method: {method}")
