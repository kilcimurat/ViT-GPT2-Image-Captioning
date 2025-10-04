"""Learnable query transformer that bridges ViT features and GPT-2 cross-attention."""

from typing import Optional

import torch
import torch.nn as nn


class QFormerLayer(nn.Module):
    """Single transformer block with self-attention on queries and cross-attention to vision tokens."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        intermediate_size = int(hidden_size * mlp_ratio)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        query_states: torch.Tensor,
        visual_embeds: torch.Tensor,
        visual_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Propagate query embeddings through self-attn, cross-attn, and feed-forward blocks.

        Args:
            query_states: (batch, num_queries, hidden_size) query tokens.
            visual_embeds: (batch, num_patches, hidden_size) vision features from ViT.
            visual_attention_mask: optional (batch, num_patches) mask with 1 for valid tokens.

        Returns:
            Updated query states of shape (batch, num_queries, hidden_size).
        """
        residual = query_states
        self_attn_output, _ = self.self_attn(query_states, query_states, query_states, need_weights=False)
        query_states = self.norm1(residual + self.dropout1(self_attn_output))

        residual = query_states
        key_padding_mask = None
        if visual_attention_mask is not None:
            key_padding_mask = ~(visual_attention_mask.bool())
        cross_attn_output, _ = self.cross_attn(
            query_states,
            visual_embeds,
            visual_embeds,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query_states = self.norm2(residual + self.dropout2(cross_attn_output))

        residual = query_states
        hidden_states = self.linear2(self.dropout3(self.activation(self.linear1(query_states))))
        query_states = self.norm3(residual + hidden_states)
        return query_states


class QFormerBridge(nn.Module):
    """Stacks Q-Former layers over learnable query tokens to condition GPT-2 on visual context."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_query_tokens: int = 32,
        num_layers: int = 4,
        num_attention_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))
        self.layers = nn.ModuleList(
            [
                QFormerLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        visual_embeds: torch.Tensor,
        visual_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return query embeddings conditioned on visual tokens for GPT-2 cross-attention.

        Args:
            visual_embeds: (batch, num_patches, hidden_size) ViT features.
            visual_attention_mask: optional (batch, num_patches) mask (1 for valid tokens).

        Returns:
            Query embeddings shaped (batch, num_query_tokens, hidden_size).
        """
        if visual_embeds.dim() == 2:
            visual_embeds = visual_embeds.unsqueeze(0)
            if visual_attention_mask is not None:
                visual_attention_mask = visual_attention_mask.unsqueeze(0)
        batch_size = visual_embeds.size(0)
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        query_tokens = query_tokens.to(visual_embeds.dtype)

        for layer in self.layers:
            query_tokens = layer(query_tokens, visual_embeds, visual_attention_mask)

        return self.final_layer_norm(query_tokens)

    def get_query_attention_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Mask with ones indicating all query tokens are valid."""
        return torch.ones(batch_size, self.query_tokens.size(1), dtype=torch.long, device=device)
