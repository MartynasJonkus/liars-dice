from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from liars_dice.agents.neural.action_mapping import ActionMapper
from liars_dice.agents.neural.encoder import ObservationEncoder
from liars_dice.agents.neural.nn_model import PolicyNetwork

Action = Tuple[str, Any]


@dataclass
class PolicySample:
    features: List[float]
    target_policy: List[float]


class PolicyDataset(Dataset):
    def __init__(self, samples: List[PolicySample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        x = torch.tensor(sample.features, dtype=torch.float32)
        y = torch.tensor(sample.target_policy, dtype=torch.float32)
        return x, y


@dataclass
class TrainingConfig:
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    device: str = "cpu"


def train_policy_network(
    model: PolicyNetwork,
    samples: List[PolicySample],
    config: TrainingConfig,
) -> Dict[str, List[float]]:
    if not samples:
        raise ValueError("No training samples provided")

    dataset = PolicyDataset(samples)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = model.to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: Dict[str, List[float]] = {"loss": []}
    model.train()

    for epoch in range(config.epochs):
        running_loss = 0.0
        batches = 0
        for x, target_policy in loader:
            x = x.to(config.device)
            target_policy = target_policy.to(config.device)

            logits = model(x)
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = -(target_policy * log_probs).sum(dim=-1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batches += 1

        history["loss"].append(running_loss / max(1, batches))

    return history


def save_model_checkpoint(
    model: PolicyNetwork,
    encoder: ObservationEncoder,
    action_mapper: ActionMapper,
    path: str | Path,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "encoder_config": {
            "num_players": encoder.num_players,
            "max_dice_per_player": encoder.max_dice_per_player,
            "max_total_dice": encoder.max_total_dice,
            "history_len": encoder.history_len,
        },
        "action_mapper_config": {
            "max_total_dice": action_mapper.max_total_dice,
        },
    }
    torch.save(payload, Path(path))
