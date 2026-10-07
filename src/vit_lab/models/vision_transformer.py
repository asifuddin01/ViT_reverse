"""From-scratch Vision Transformer with optional internal-state inspection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from vit_lab.models.classification_head import pool_tokens
from vit_lab.models.patch_embedding import PatchEmbedding
from vit_lab.models.positional_embedding import add_position, resize_position_embedding
from vit_lab.models.transformer_block import TransformerBlock


@dataclass
class VisionTransformerOutput:
    """Internal tensors returned only when analysis is requested."""

    logits: torch.Tensor
    patch_embeddings: torch.Tensor
    hidden_states: list[torch.Tensor] | None
    attention_maps: list[torch.Tensor] | None
    final_features: torch.Tensor


class VisionTransformer(nn.Module):
    """Class-token ViT with manual attention and the reference's pre-norm order."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_classes: int = 1000,
        *,
        pooling: str = "cls",
        position_embedding: str = "learned",
        dynamic_image_size: bool = False,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if (depth <= 0 or num_classes <= 0 or pooling not in ("cls", "mean")
                or position_embedding not in ("learned", "none")):
            raise ValueError("Invalid depth, class count, pooling, or position embedding")
        self.image_size = image_size
        self.patch_size = patch_size
        self.dynamic_image_size = dynamic_image_size
        self.pooling = pooling
        self.position_embedding = position_embedding
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim,
            strict_image_size=not dynamic_image_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if pooling == "cls" else None
        prefix_tokens = 1 if pooling == "cls" else 0
        position_shape = (1, self.patch_embed.num_patches + prefix_tokens, embed_dim)
        reference_position_shape = (1, self.patch_embed.num_patches + 1, embed_dim)
        self.pos_embed = (nn.Parameter(torch.zeros(position_shape))
                          if position_embedding == "learned" else None)
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio,
                dropout=dropout, attn_dropout=attn_dropout,
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)
        if self.pos_embed is not None and self.cls_token is None:
            # Match the CLS model's patch positions and random-number sequence.
            full_positions = torch.empty(reference_position_shape)
            nn.init.trunc_normal_(full_positions, std=0.02)
            with torch.no_grad():
                self.pos_embed.copy_(full_positions[:, 1:])
        elif self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            # Preserve the existing learned variant's RNG sequence for shared weights.
            nn.init.trunc_normal_(torch.empty(reference_position_shape), std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        else:
            nn.init.normal_(torch.empty(1, 1, embed_dim), std=1e-6)

    def _encode(
        self,
        images: torch.Tensor,
        *,
        return_attention: bool,
        return_hidden_states: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None, list[torch.Tensor] | None]:
        patches = self.patch_embed(images)
        tokens = patches
        if self.cls_token is not None:
            tokens = torch.cat((self.cls_token.expand(images.shape[0], -1, -1), tokens), dim=1)
        if self.pos_embed is not None:
            positions = self.pos_embed
            if tokens.shape[1] != positions.shape[1]:
                if not self.dynamic_image_size:
                    raise ValueError("Token count and positional embedding length differ")
                positions = resize_position_embedding(
                    positions,
                    old_grid=(self.patch_embed.grid_size, self.patch_embed.grid_size),
                    new_grid=(images.shape[2] // self.patch_size, images.shape[3] // self.patch_size),
                    prefix_tokens=int(self.cls_token is not None),
                )
            tokens = add_position(tokens, positions)
        tokens = self.pos_drop(tokens)
        hidden_states = [] if return_hidden_states else None
        attention_maps = [] if return_attention else None
        for block in self.blocks:
            if return_attention:
                tokens, probabilities = block(tokens, return_attention=True)
                attention_maps.append(probabilities)
            else:
                tokens = block(tokens)
            if return_hidden_states:
                hidden_states.append(tokens)
        return self.norm(tokens), patches, hidden_states, attention_maps

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return all normalized tokens before pooling and classification."""
        tokens, _, _, _ = self._encode(
            images, return_attention=False, return_hidden_states=False
        )
        return tokens

    def forward(
        self,
        images: torch.Tensor,
        *,
        return_attention: bool = False,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | VisionTransformerOutput:
        tokens, patches, hidden_states, attention_maps = self._encode(
            images,
            return_attention=return_attention,
            return_hidden_states=return_hidden_states,
        )
        features = pool_tokens(tokens, self.pooling)
        logits = self.head(features)
        if not (return_attention or return_hidden_states):
            return logits
        return VisionTransformerOutput(
            logits=logits,
            patch_embeddings=patches,
            hidden_states=hidden_states,
            attention_maps=attention_maps,
            final_features=features,
        )
