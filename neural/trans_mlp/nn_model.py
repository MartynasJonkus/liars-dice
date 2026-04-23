from __future__ import annotations

import torch
from torch import nn


class PolicyNetwork(nn.Module):
    """
    Policy network with:
    - static feature branch
    - tiny transformer over current-round bid history
    - fusion MLP head
    """

    def __init__(
        self,
        static_dim: int,
        token_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
        max_bids: int = 40,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.static_dim = static_dim
        self.token_dim = token_dim
        self.num_actions = num_actions
        self.max_bids = max_bids
        self.d_model = d_model

        # Static branch
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, 64),
            nn.ReLU(),
        )

        # History branch
        self.bid_embed = nn.Linear(token_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_bids, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.bid_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Fusion head
        self.policy_head = nn.Sequential(
            nn.Linear(64 + d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(
        self,
        static_x: torch.Tensor,
        bid_history: torch.Tensor,
        bid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        static_repr = self.static_net(static_x)

        bid_enc = self.bid_embed(bid_history)
        bid_enc = bid_enc + self.pos_embedding[:, : bid_enc.size(1), :]

        if bid_mask is not None:
            all_masked = bid_mask.all(dim=1)
            if all_masked.any():
                bid_mask = bid_mask.clone()
                bid_mask[all_masked, 0] = False

        bid_enc = self.bid_transformer(
            bid_enc,
            src_key_padding_mask=bid_mask,
        )

        bid_pooled = self._pool_history(bid_enc, bid_mask)

        x = torch.cat([static_repr, bid_pooled], dim=-1)
        return self.policy_head(x)

    def _pool_history(
        self,
        bid_enc: torch.Tensor,
        bid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Pools sequence output into a single history embedding.

        Uses last valid token pooling.
        If a sequence has no valid tokens, falls back to the first token.
        """
        batch_size = bid_enc.size(0)
        device = bid_enc.device

        if bid_mask is None:
            return bid_enc[:, -1, :]

        valid_lengths = (~bid_mask).sum(dim=1)  # [B]
        last_indices = torch.clamp(valid_lengths - 1, min=0)

        batch_indices = torch.arange(batch_size, device=device)
        pooled = bid_enc[batch_indices, last_indices]

        return pooled
