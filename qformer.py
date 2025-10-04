import torch
import torch.nn as nn


class QFormerLayer(nn.Module):
    """A lightweight Query-Former layer with self-attn, cross-attn, and MLP."""

    def __init__(self, hidden_size: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        inner_size = int(hidden_size * mlp_ratio)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.self_dropout = nn.Dropout(dropout)
        self.self_norm = nn.LayerNorm(hidden_size)

        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.cross_dropout = nn.Dropout(dropout)
        self.cross_norm = nn.LayerNorm(hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Linear(inner_size, hidden_size),
        )
        self.mlp_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(hidden_size)

    def forward(self, query_states: torch.Tensor, visual_embeds: torch.Tensor, visual_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # query_states: (batch, num_queries, hidden)
        # visual_embeds: (batch, num_patches, hidden)
        query_states = self._self_attention(query_states)
        query_states = self._cross_attention(query_states, visual_embeds, visual_attention_mask)
        query_states = self._mlp(query_states)
        return query_states

    def _self_attention(self, query_states: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.self_attn(query_states, query_states, query_states, need_weights=False)
        query_states = query_states + self.self_dropout(attn_output)
        query_states = self.self_norm(query_states)
        return query_states

    def _cross_attention(self, query_states: torch.Tensor, visual_embeds: torch.Tensor, visual_attention_mask: torch.Tensor | None) -> torch.Tensor:
        key_padding_mask = None
        if visual_attention_mask is not None:
            key_padding_mask = ~visual_attention_mask.bool()
        attn_output, _ = self.cross_attn(query_states, visual_embeds, visual_embeds, need_weights=False, key_padding_mask=key_padding_mask)
        query_states = query_states + self.cross_dropout(attn_output)
        query_states = self.cross_norm(query_states)
        return query_states

    def _mlp(self, query_states: torch.Tensor) -> torch.Tensor:
        ff_output = self.mlp(query_states)
        query_states = query_states + self.mlp_dropout(ff_output)
        query_states = self.mlp_norm(query_states)
        return query_states


class QFormerBridge(nn.Module):
    """Bridges ViT patch embeddings and GPT-2 via learnable query tokens."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_queries: int = 32,
        num_layers: int = 4,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_size))
        self.layers = nn.ModuleList(
            [QFormerLayer(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, visual_embeds: torch.Tensor, visual_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = visual_embeds.size(0)
        query_tokens = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        query_states = query_tokens
        for layer in self.layers:
            query_states = layer(query_states, visual_embeds, visual_attention_mask)
        query_states = self.final_norm(query_states)
        return query_states

    def get_encoder_attention_mask(self, visual_embeds: torch.Tensor) -> torch.Tensor:
        batch_size = visual_embeds.size(0)
        num_queries = self.query_tokens.size(0)
        return torch.ones(batch_size, num_queries, dtype=torch.long, device=visual_embeds.device)
