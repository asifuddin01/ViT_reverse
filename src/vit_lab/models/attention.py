"""Explicit multi-head self-attention with inspectable probabilities."""

from __future__ import annotations

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Packed QKV projection followed by manual scaled dot-product attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        if not 0 <= attn_dropout < 1 or not 0 <= proj_dropout < 1:
            raise ValueError("dropout probabilities must be in [0, 1)")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(
        self, x: torch.Tensor, *, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.qkv.in_features:
            raise ValueError("Expected token tensor [B, N, embed_dim]")
        batch, tokens, _ = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        probabilities = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        attended = self.attn_drop(probabilities) @ v
        attended = attended.transpose(1, 2).reshape(batch, tokens, -1)
        output = self.proj_drop(self.proj(attended))
        return (output, probabilities) if return_attention else output
