from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from liars_dice.core.game import Observation
from neural.basic_mlp.encoder_mlp import ObservationEncoder


@dataclass
class PolicySample:
    features: List[float]
    target_policy: List[float]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "features": self.features,
            "target_policy": self.target_policy,
        }


def make_policy_sample(
    encoder: ObservationEncoder,
    obs: Observation,
    target_policy: List[float],
) -> PolicySample:
    encoded = encoder.encode(obs)

    # If your MLP encoder returns a plain list, this is correct.
    # If it returns {"features": [...]}, change this to encoded["features"].
    if isinstance(encoded, dict):
        features = encoded["features"]
    else:
        features = encoded

    return PolicySample(
        features=list(features),
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
