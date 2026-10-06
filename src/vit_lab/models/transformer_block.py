"""Pre-norm Transformer block matching the inspected reference operation order."""

from __future__ import annotations

import torch
from torch import nn

from vit_lab.models.attention import MultiHeadSelfAttention
from vit_lab.models.mlp import TokenMLP


class TransformerBlock(nn.Module):
    """Attention and MLP residual paths, each with its own preceding LayerNorm."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        *,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.attn = MultiHeadSelfAttention(
            embed_dim,
            num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.mlp = TokenMLP(embed_dim, int(embed_dim * mlp_ratio), dropout=dropout)

    def forward(
        self, x: torch.Tensor, *, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attention = self.attn(self.norm1(x), return_attention=return_attention)
        if return_attention:
            attention_output, probabilities = attention
        else:
            attention_output = attention
        x = x + attention_output
        x = x + self.mlp(self.norm2(x))
        return (x, probabilities) if return_attention else x
