from __future__ import annotations

from typing import Any, List

import torch

from liars_dice.core.game import Observation
from neural.common.action_mapping import ActionMapper
from neural.basic_mlp.encoder_mlp import ObservationEncoder
from neural.basic_mlp.model_mlp import PolicyNetwork
from neural.common.neural_ismcts_base import NeuralISMCTSBase


class MLPNeuralISMCTSAgent(NeuralISMCTSBase):
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

    def _encode_obs(self, obs: Observation) -> List[float]:
        return self.encoder.encode(obs)

    def _make_prior_cache_key(self, encoded: List[float]) -> Any:
        return tuple(encoded)

    def _call_model(self, encoded: List[float]) -> torch.Tensor:
        x = torch.tensor(
            encoded,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        return self.model(x).squeeze(0)
