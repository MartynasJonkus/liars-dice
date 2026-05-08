from __future__ import annotations

from typing import Any, Dict

import torch

from liars_dice.core.game import Observation
from neural.common.action_mapping import ActionMapper
from neural.trans_mlp.encoder_trans import ObservationEncoder
from neural.trans_mlp.model_trans import PolicyNetwork
from neural.common.neural_ismcts_base import NeuralISMCTSBase


class TransformerNeuralISMCTSAgent(NeuralISMCTSBase):
    def __init__(
        self,
        model: PolicyNetwork,
        encoder: ObservationEncoder,
        action_mapper: ActionMapper,
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            encoder=encoder,
            action_mapper=action_mapper,
            **kwargs,
        )

    def _encode_obs(self, obs: Observation) -> Dict[str, Any]:
        return self.encoder.encode(obs)

    def _make_prior_cache_key(self, encoded: Dict[str, Any]) -> Any:
        return (
            tuple(encoded["static_features"]),
            tuple(tuple(token) for token in encoded["bid_history"]),
            tuple(encoded["bid_mask"]),
        )

    def _call_model(self, encoded: Dict[str, Any]) -> torch.Tensor:
        static_x = torch.tensor(
            encoded["static_features"],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        bid_history = torch.tensor(
            encoded["bid_history"],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        bid_mask = torch.tensor(
            encoded["bid_mask"],
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)

        return self.model(static_x, bid_history, bid_mask).squeeze(0)
