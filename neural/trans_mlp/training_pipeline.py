from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from neural.action_mapping import ActionMapper
from neural.trans_mlp.data_collection import PolicySample
from neural.trans_mlp.encoder import ObservationEncoder


class PolicyDataset(Dataset):
    def __init__(self, samples: List[PolicySample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (
            torch.tensor(s.static_features, dtype=torch.float32),
            torch.tensor(s.bid_history, dtype=torch.float32),
            torch.tensor(s.bid_mask, dtype=torch.bool),
            torch.tensor(s.target_policy, dtype=torch.float32),
        )


@dataclass
class TrainingConfig:
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    device: str = "cpu"


def policy_cross_entropy_loss(
    logits: torch.Tensor,
    target_policy: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy with a soft target distribution (AlphaZero-style policy target).
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(target_policy * log_probs).sum(dim=-1).mean()


def build_dataloader(
    samples: List[PolicySample],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = PolicyDataset(samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def train_policy_network(
    model: torch.nn.Module,
    samples: List[PolicySample],
    config: TrainingConfig,
) -> Dict[str, List[float]]:
    if not samples:
        raise ValueError("Cannot train on an empty sample list")

    loader = build_dataloader(
        samples=samples,
        batch_size=config.batch_size,
        shuffle=True,
    )

    model.to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: Dict[str, List[float]] = {"loss": []}

    model.train()
    for epoch in range(config.epochs):
        running_loss = 0.0
        batch_count = 0

        for static_x, bid_history, bid_mask, target_policy in loader:
            static_x = static_x.to(config.device)
            bid_history = bid_history.to(config.device)
            bid_mask = bid_mask.to(config.device)
            target_policy = target_policy.to(config.device)

            logits = model(static_x, bid_history, bid_mask)
            loss = policy_cross_entropy_loss(logits, target_policy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batch_count += 1

        epoch_loss = running_loss / max(1, batch_count)
        history["loss"].append(epoch_loss)

    return history


@torch.no_grad()
def evaluate_policy_network(
    model: torch.nn.Module,
    samples: List[PolicySample],
    config: TrainingConfig,
) -> float:
    if not samples:
        raise ValueError("Cannot evaluate on an empty sample list")

    loader = build_dataloader(
        samples=samples,
        batch_size=config.batch_size,
        shuffle=False,
    )

    model.to(config.device)
    model.eval()

    total_loss = 0.0
    batch_count = 0

    for static_x, bid_history, bid_mask, target_policy in loader:
        static_x = static_x.to(config.device)
        bid_history = bid_history.to(config.device)
        bid_mask = bid_mask.to(config.device)
        target_policy = target_policy.to(config.device)

        logits = model(static_x, bid_history, bid_mask)
        loss = policy_cross_entropy_loss(logits, target_policy)

        total_loss += float(loss.item())
        batch_count += 1

    model.train()
    return total_loss / max(1, batch_count)


def save_model_checkpoint(
    model: torch.nn.Module,
    encoder: ObservationEncoder,
    action_mapper: ActionMapper,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "encoder_config": {
            "num_players": encoder.num_players,
            "max_dice_per_player": encoder.max_dice_per_player,
            "max_total_dice": encoder.max_total_dice,
            "max_bids": encoder.max_bids,
        },
        "action_mapper_config": {
            "max_total_dice": action_mapper.max_total_dice,
        },
        "model_config": {
            "static_dim": getattr(model, "static_dim", None),
            "token_dim": getattr(model, "token_dim", None),
            "num_actions": getattr(model, "num_actions", None),
            "max_bids": getattr(model, "max_bids", None),
            "d_model": getattr(model, "d_model", None),
        },
    }

    torch.save(payload, path)


def load_model_checkpoint(
    path: str | Path,
    model_cls,
    device: str = "cpu",
):
    """
    Reconstructs model + encoder + action mapper from a checkpoint.

    model_cls is expected to accept:
      static_dim, token_dim, num_actions, max_bids, d_model
    plus whatever defaults are present in the class.
    """
    from neural.action_mapping import ActionMapper
    from neural.trans_mlp.encoder import ObservationEncoder

    path = Path(path)
    payload = torch.load(path, map_location=device)

    encoder = ObservationEncoder(**payload["encoder_config"])
    action_mapper = ActionMapper(**payload["action_mapper_config"])

    model_cfg = payload["model_config"]

    model = model_cls(
        static_dim=model_cfg["static_dim"],
        token_dim=model_cfg["token_dim"],
        num_actions=model_cfg["num_actions"],
        max_bids=model_cfg["max_bids"],
        d_model=model_cfg["d_model"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    return model, encoder, action_mapper
