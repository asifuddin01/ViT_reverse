"""Per-token feed-forward network for a Transformer block."""

from __future__ import annotations

import torch
from torch import nn


class TokenMLP(nn.Module):
    """Apply the same two-layer MLP independently to every token."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if min(embed_dim, hidden_dim) <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid MLP dimension or dropout")
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))
