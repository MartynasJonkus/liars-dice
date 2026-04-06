from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from liars_dice.agents.neural.action_mapping import ActionMapper
from liars_dice.agents.neural.encoder import ObservationEncoder
from liars_dice.agents.neural.nn_model import PolicyNetwork
from liars_dice.training.data_collection import PolicySample
from liars_dice.training.training_pipeline import save_model_checkpoint


class PolicyDataset(Dataset):
    def __init__(self, samples: List[PolicySample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        return (
            torch.tensor(sample.features, dtype=torch.float32),
            torch.tensor(sample.target_policy, dtype=torch.float32),
        )


def load_samples_jsonl(path: str | Path) -> List[PolicySample]:
    path = Path(path)
    samples: List[PolicySample] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} in {path}") from exc

            features = obj.get("features")
            target_policy = obj.get("target_policy")

            if not isinstance(features, list) or not isinstance(target_policy, list):
                raise ValueError(
                    f"Malformed sample on line {line_num}: expected 'features' and 'target_policy' lists"
                )

            samples.append(
                PolicySample(
                    features=features,
                    target_policy=target_policy,
                )
            )

    if not samples:
        raise ValueError(f"No samples loaded from {path}")

    return samples


def split_samples(
    samples: List[PolicySample],
    val_fraction: float,
    seed: int,
) -> Tuple[List[PolicySample], List[PolicySample]]:
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must be in [0, 1)")

    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    if val_fraction == 0.0:
        return shuffled, []

    val_size = int(len(shuffled) * val_fraction)
    val_size = max(1, val_size) if len(shuffled) > 1 else 0

    val_samples = shuffled[:val_size]
    train_samples = shuffled[val_size:]

    if not train_samples:
        raise ValueError("Validation split left no training samples")

    return train_samples, val_samples


def policy_cross_entropy_loss(
    logits: torch.Tensor,
    target_policy: torch.Tensor,
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(target_policy * log_probs).sum(dim=-1).mean()


@torch.no_grad()
def evaluate_policy_loss(
    model: PolicyNetwork,
    loader: DataLoader,
    device: str,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for features, target_policy in loader:
        features = features.to(device)
        target_policy = target_policy.to(device)

        logits = model(features)
        loss = policy_cross_entropy_loss(logits, target_policy)

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / max(1, total_batches)


def train_model(
    model: PolicyNetwork,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    checkpoint_dir: Path,
    encoder: ObservationEncoder,
    action_mapper: ActionMapper,
    save_every: int,
) -> Dict[str, List[float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for features, target_policy in train_loader:
            features = features.to(device)
            target_policy = target_policy.to(device)

            logits = model(features)
            loss = policy_cross_entropy_loss(logits, target_policy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batch_count += 1

        train_loss = running_loss / max(1, batch_count)
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss = evaluate_policy_loss(model, val_loader, device)
            history["val_loss"].append(val_loss)
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
            )
        else:
            val_loss = None
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f}")

        if save_every > 0 and epoch % save_every == 0:
            save_model_checkpoint(
                model=model,
                encoder=encoder,
                action_mapper=action_mapper,
                path=checkpoint_dir / f"policy_epoch_{epoch:03d}.pt",
            )

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model_checkpoint(
                model=model,
                encoder=encoder,
                action_mapper=action_mapper,
                path=checkpoint_dir / "policy_best.pt",
            )

    save_model_checkpoint(
        model=model,
        encoder=encoder,
        action_mapper=action_mapper,
        path=checkpoint_dir / "policy_last.pt",
    )

    return history


def save_training_history(path: Path, history: Dict[str, List[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a policy network from collected supervised JSONL data."
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to merged supervised_samples.jsonl",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="artifacts/policy_training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save numbered checkpoint every N epochs. Use 0 to disable.",
    )

    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--dice-per-player", type=int, default=5)
    parser.add_argument("--history-len", type=int, default=5)

    args = parser.parse_args()

    if args.num_players != 4:
        raise ValueError(
            "This trainer is intended for the fixed 4-player thesis setup."
        )

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    max_total_dice = args.num_players * args.dice_per_player

    action_mapper = ActionMapper(max_total_dice=max_total_dice)
    encoder = ObservationEncoder(
        num_players=args.num_players,
        max_dice_per_player=args.dice_per_player,
        max_total_dice=max_total_dice,
        history_len=args.history_len,
    )

    print(f"Loading samples from {args.data} ...")
    samples = load_samples_jsonl(args.data)
    print(f"Loaded {len(samples)} samples")

    expected_input_dim = encoder.input_dim
    expected_num_actions = action_mapper.num_actions

    for i, sample in enumerate(samples[:10]):
        if len(sample.features) != expected_input_dim:
            raise ValueError(
                f"Sample {i} has feature length {len(sample.features)} but expected {expected_input_dim}"
            )
        if len(sample.target_policy) != expected_num_actions:
            raise ValueError(
                f"Sample {i} has target length {len(sample.target_policy)} but expected {expected_num_actions}"
            )

    train_samples, val_samples = split_samples(
        samples=samples,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")

    train_dataset = PolicyDataset(train_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    if val_samples:
        val_dataset = PolicyDataset(val_samples)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
    else:
        val_loader = None

    model = PolicyNetwork(
        input_dim=encoder.input_dim,
        num_actions=action_mapper.num_actions,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(args.device)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training on {args.device} ...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=checkpoint_dir,
        encoder=encoder,
        action_mapper=action_mapper,
        save_every=args.save_every,
    )

    save_training_history(checkpoint_dir / "training_history.json", history)
    print(f"Training history written to: {checkpoint_dir / 'training_history.json'}")
    print(f"Final checkpoint written to: {checkpoint_dir / 'policy_last.pt'}")
    if val_loader is not None:
        print(
            f"Best validation checkpoint written to: {checkpoint_dir / 'policy_best.pt'}"
        )


if __name__ == "__main__":
    main()
