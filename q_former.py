import torch
import torch.nn as nn


class QFormer(nn.Module):
    """Transformer-based query former used to bridge vision features and GPT-2."""

    def __init__(
        self,
        num_queries: int = 32,
        hidden_dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ln = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
            nn.Dropout(dropout),
        )

        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """Process vision features into a compact set of query tokens.

        Args:
            image_embeds: Tensor of shape (B, N, D) from the vision encoder.

        Returns:
            Tensor of shape (B, Q, D) representing query features.
        """

        b = image_embeds.size(0)
        queries = self.query_tokens.expand(b, -1, -1)

        queries = self.self_attn(queries)
        attn_output, _ = self.cross_attn(queries, image_embeds, image_embeds)
        queries = self.ln(attn_output + queries)

        ffn_output = self.ffn(queries)
        queries = self.ln2(ffn_output + queries)

        return queries

