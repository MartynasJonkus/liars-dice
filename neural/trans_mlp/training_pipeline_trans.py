from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import torch
from torch.utils.data import DataLoader, Dataset

from neural.common.action_mapping import ActionMapper
from neural.trans_mlp.data_collection_trans import PolicySample
from neural.trans_mlp.encoder_trans import ObservationEncoder
from neural.trans_mlp.model_trans import PolicyNetwork


class PolicyDataset(Dataset):
    def __init__(self, samples: List[PolicySample]):
        if not samples:
            raise ValueError("PolicyDataset received no samples")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        return (
            torch.tensor(sample.static_features, dtype=torch.float32),
            torch.tensor(sample.bid_history, dtype=torch.float32),
            torch.tensor(sample.bid_mask, dtype=torch.bool),
            torch.tensor(sample.target_policy, dtype=torch.float32),
        )


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


def build_dataloader(
    samples: List[PolicySample],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        PolicyDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


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
    batches = 0

    for static_x, bid_history, bid_mask, target_policy in loader:
        static_x = static_x.to(config.device)
        bid_history = bid_history.to(config.device)
        bid_mask = bid_mask.to(config.device)
        target_policy = target_policy.to(config.device)

        logits = model(static_x, bid_history, bid_mask)
        loss = soft_policy_loss(logits, target_policy)

        total_loss += float(loss.item())
        batches += 1

    if batches == 0:
        raise ValueError("Validation loader produced no batches")

    return total_loss / batches


def train_policy_network(
    model: torch.nn.Module,
    train_samples: List[PolicySample],
    config: TrainingConfig,
    val_samples: List[PolicySample] | None = None,
    encoder: ObservationEncoder | None = None,
    action_mapper: ActionMapper | None = None,
    best_checkpoint_path: str | Path | None = None,
    last_checkpoint_path: str | Path | None = None,
    hidden_dim: int = 256,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
    extra_metadata: Dict[str, Any] | None = None,
) -> Dict[str, List[float]]:
    if not train_samples:
        raise ValueError("Cannot train on an empty training sample list")

    train_loader = build_dataloader(
        samples=train_samples,
        batch_size=config.batch_size,
        shuffle=True,
    )

    model.to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: Dict[str, List[float]] = {
        "train": [],
        "val": [],
    }

    best_score = float("inf")
    has_validation = val_samples is not None and len(val_samples) > 0

    for epoch in range(1, config.epochs + 1):
        model.train()

        running_loss = 0.0
        batches = 0

        for static_x, bid_history, bid_mask, target_policy in train_loader:
            static_x = static_x.to(config.device)
            bid_history = bid_history.to(config.device)
            bid_mask = bid_mask.to(config.device)
            target_policy = target_policy.to(config.device)

            logits = model(static_x, bid_history, bid_mask)
            loss = soft_policy_loss(logits, target_policy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batches += 1

        if batches == 0:
            raise ValueError("Training loader produced no batches")

        train_loss = running_loss / batches
        history["train"].append(train_loss)

        if has_validation:
            val_loss = evaluate_policy_network(
                model=model,
                samples=val_samples,
                config=config,
            )
            history["val"].append(val_loss)
            score = val_loss
            print(f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f}")
        else:
            score = train_loss
            print(f"Epoch {epoch:03d} | train={train_loss:.4f}")

        if score < best_score:
            best_score = score
            if best_checkpoint_path is not None:
                if encoder is None or action_mapper is None:
                    raise ValueError(
                        "encoder and action_mapper are required to save checkpoints"
                    )

                save_model_checkpoint(
                    model=model,
                    encoder=encoder,
                    action_mapper=action_mapper,
                    path=best_checkpoint_path,
                    hidden_dim=hidden_dim,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    extra_metadata={
                        **(extra_metadata or {}),
                        "best_epoch": epoch,
                        "best_score": best_score,
                        "selection_metric": (
                            "val_loss" if has_validation else "train_loss"
                        ),
                    },
                )

    if last_checkpoint_path is not None:
        if encoder is None or action_mapper is None:
            raise ValueError(
                "encoder and action_mapper are required to save checkpoints"
            )

        save_model_checkpoint(
            model=model,
            encoder=encoder,
            action_mapper=action_mapper,
            path=last_checkpoint_path,
            hidden_dim=hidden_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            extra_metadata={
                **(extra_metadata or {}),
                "final_train_loss": history["train"][-1],
                "final_val_loss": history["val"][-1] if history["val"] else None,
            },
        )

    return history


def save_model_checkpoint(
    model: torch.nn.Module,
    encoder: ObservationEncoder,
    action_mapper: ActionMapper,
    path: str | Path,
    hidden_dim: int = 256,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
    extra_metadata: Dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "model_type": "transformer",
            "static_dim": encoder.static_dim,
            "token_dim": encoder.bid_token_dim,
            "num_actions": action_mapper.num_actions,
            "hidden_dim": hidden_dim,
            "max_bids": encoder.max_bids,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
        },
        "encoder_config": {
            "num_players": encoder.num_players,
            "max_dice_per_player": encoder.max_dice_per_player,
            "max_total_dice": encoder.max_total_dice,
            "max_bids": encoder.max_bids,
        },
        "action_mapper_config": {
            "max_total_dice": action_mapper.max_total_dice,
        },
    }

    if extra_metadata is not None:
        payload["metadata"] = extra_metadata

    torch.save(payload, path)


def load_model_checkpoint(
    path: str | Path,
    model_cls: Type[torch.nn.Module] = PolicyNetwork,
    device: str = "cpu",
) -> Tuple[torch.nn.Module, ObservationEncoder, ActionMapper]:
    """
    Reconstructs model, encoder, and action mapper from a saved checkpoint.
    """
    path = Path(path)
    payload = torch.load(path, map_location=device)

    encoder = ObservationEncoder(**payload["encoder_config"])
    action_mapper = ActionMapper(**payload["action_mapper_config"])

    model_cfg = payload["model_config"]

    model = model_cls(
        static_dim=model_cfg["static_dim"],
        token_dim=model_cfg["token_dim"],
        num_actions=model_cfg["num_actions"],
        hidden_dim=model_cfg["hidden_dim"],
        max_bids=model_cfg["max_bids"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
    )

    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    return model, encoder, action_mapper
