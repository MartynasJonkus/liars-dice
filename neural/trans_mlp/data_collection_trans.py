from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from liars_dice.core.game import Observation
from neural.trans_mlp.encoder_trans import ObservationEncoder


@dataclass
class PolicySample:
    static_features: List[float]
    bid_history: List[List[float]]
    bid_mask: List[bool]
    target_policy: List[float]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "static_features": self.static_features,
            "bid_history": self.bid_history,
            "bid_mask": self.bid_mask,
            "target_policy": self.target_policy,
        }


def make_policy_sample(
    encoder: ObservationEncoder,
    obs: Observation,
    target_policy: List[float],
) -> PolicySample:
    encoded = encoder.encode(obs)

    return PolicySample(
        static_features=list(encoded["static_features"]),
        bid_history=[list(row) for row in encoded["bid_history"]],
        bid_mask=list(encoded["bid_mask"]),
        target_policy=list(target_policy),
    )


def save_samples_jsonl(samples: Iterable[PolicySample], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_json_dict(), separators=(",", ":")) + "\n")
            count += 1

    return count
