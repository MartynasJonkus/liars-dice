from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from neural.common.action_mapping import ActionMapper
from neural.basic_mlp.encoder_mlp import ObservationEncoder
from neural.basic_mlp.model_mlp import PolicyNetwork

Action = Tuple[str, Any]


@dataclass
class PolicySample:
    features: List[float]
    target_policy: List[float]


class PolicyDataset(Dataset):
    def __init__(self, samples: List[PolicySample]):
        if not samples:
            raise ValueError("PolicyDataset received no samples")
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
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    device: str = "cpu"


def soft_policy_loss(logits: torch.Tensor, target_policy: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy with a soft target policy distribution.
    target_policy is expected to sum to 1 for each sample.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(target_policy * log_probs).sum(dim=-1).mean()


def train_policy_network(
    model: PolicyNetwork,
    samples: List[PolicySample],
    config: TrainingConfig,
) -> Dict[str, List[float]]:
    """
    Simple training helper for cases where no validation split is needed.
    The main runner in train_mlp.py implements the train/validation workflow.
    """
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

    for _epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        batches = 0

        for x, target_policy in loader:
            x = x.to(config.device)
            target_policy = target_policy.to(config.device)

            logits = model(x)
            loss = soft_policy_loss(logits, target_policy)

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
    hidden_dim: int = 256,
    dropout: float = 0.1,
    extra_metadata: Dict[str, Any] | None = None,
) -> None:
    """
    Saves enough information to reconstruct the MLP policy model and its
    matching encoder/action mapper during evaluation or benchmarking.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "model_type": "mlp",
            "input_dim": encoder.input_dim,
            "num_actions": action_mapper.num_actions,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
        },
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

    if extra_metadata is not None:
        payload["metadata"] = extra_metadata

    torch.save(payload, path)
