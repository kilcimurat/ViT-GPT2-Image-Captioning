import torch
import torch.nn as nn

class QFormerLayer(nn.Module):
    """A single Q-Former layer with self-attention, cross-attention and FFN."""
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)

        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm3 = nn.LayerNorm(hidden_size)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single layer.

        Args:
            query: Query tokens of shape (B, Q, C).
            key_value: Visual features to attend to, shape (B, N, C).
        Returns:
            Updated query tokens of shape (B, Q, C).
        """
        q = query + self.self_attn(query, query, query)[0]
        q = self.norm1(q)

        q = q + self.cross_attn(q, key_value, key_value)[0]
        q = self.norm2(q)

        q = q + self.ffn(q)
        q = self.norm3(q)
        return q


class QFormer(nn.Module):
    """Stack of Q-Former layers operating on learned query tokens."""
    def __init__(self, num_queries: int, hidden_size: int, num_layers: int, num_heads: int):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_size))
        self.layers = nn.ModuleList([
            QFormerLayer(hidden_size, num_heads) for _ in range(num_layers)
        ])

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """Generate query embeddings conditioned on visual features.

        Args:
            image_embeds: Visual embeddings of shape (B, N, C).
        Returns:
            Query embeddings of shape (B, num_queries, C).
        """
        batch_size = image_embeds.size(0)
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        for layer in self.layers:
            query_tokens = layer(query_tokens, image_embeds)
        return query_tokens
