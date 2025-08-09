import torch
import torch.nn as nn


class QFormer(nn.Module):
    """A lightweight Query Transformer to bridge vision and language models.

    This module maintains a set of learnable query tokens that attend to image
    features produced by the vision encoder (e.g. ViT). The resulting query
    embeddings can be consumed by a language model such as GPT‑2 via the
    ``encoder_hidden_states`` argument.
    """

    def __init__(
        self,
        image_dim: int = 768,
        num_query_tokens: int = 32,
        query_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(image_dim, query_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, query_dim))
        self.attn_layers = nn.ModuleList(
            [nn.MultiheadAttention(query_dim, num_heads, batch_first=True) for _ in range(num_layers)]
        )
        self.norm_layers = nn.ModuleList([nn.LayerNorm(query_dim) for _ in range(num_layers)])

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """Forward the Q-Former.

        Args:
            image_embeds: Tensor of shape ``(batch, seq_len, image_dim)`` from ViT.

        Returns:
            Tensor of shape ``(batch, num_query_tokens, query_dim)`` representing
            the attended query features.
        """
        x = self.proj(image_embeds)
        q = self.query_tokens.expand(image_embeds.size(0), -1, -1)
        for attn, norm in zip(self.attn_layers, self.norm_layers):
            attn_out, _ = attn(q, x, x)
            q = norm(q + attn_out)
        return q
