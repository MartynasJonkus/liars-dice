from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from neural.common.action_mapping import ActionMapper
from neural.basic_mlp.data_collection_mlp import PolicySample
from neural.basic_mlp.encoder_mlp import ObservationEncoder
from neural.basic_mlp.model_mlp import PolicyNetwork
from neural.basic_mlp.training_pipeline_mlp import (
    save_model_checkpoint,
    soft_policy_loss,
)

# =====================
# CONFIG
# =====================
DATA_PATH = "artifacts/data/mlp/supervised_samples.jsonl"
CHECKPOINT_DIR = "artifacts/training/mlp"

BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 256
DROPOUT = 0.1

VAL_FRACTION = 0.1
SEED = 12345
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_PLAYERS = 4
DICE_PER_PLAYER = 5
HISTORY_LEN = 10


# =====================
# DATASET
# =====================
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
            torch.tensor(sample.features, dtype=torch.float32),
            torch.tensor(sample.target_policy, dtype=torch.float32),
        )


def load_samples(path: str | Path) -> List[PolicySample]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    samples: List[PolicySample] = []
    skipped_zero_mass = 0
    skipped_bad_shape = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            features = obj.get("features")
            target_policy = obj.get("target_policy")

            if features is None or target_policy is None:
                raise KeyError(
                    f"Line {line_no} must contain 'features' and 'target_policy'"
                )

            target_policy = [float(x) for x in target_policy]
            mass = sum(target_policy)

            if mass <= 0.0:
                skipped_zero_mass += 1
                continue

            # Defensive renormalization in case JSON values have tiny numeric drift.
            target_policy = [x / mass for x in target_policy]

            if not features:
                skipped_bad_shape += 1
                continue

            samples.append(
                PolicySample(
                    features=[float(x) for x in features],
                    target_policy=target_policy,
                )
            )

    if not samples:
        raise ValueError(f"No valid samples loaded from {path}")

    if skipped_zero_mass > 0:
        print(f"Skipped {skipped_zero_mass} samples with zero target-policy mass")

    if skipped_bad_shape > 0:
        print(f"Skipped {skipped_bad_shape} samples with invalid feature shape")

    return samples


def split_samples(
    samples: List[PolicySample],
) -> Tuple[List[PolicySample], List[PolicySample]]:
    samples = list(samples)
    random.shuffle(samples)

    if VAL_FRACTION <= 0.0:
        return samples, []

    split = int(len(samples) * VAL_FRACTION)

    # Ensure small datasets still keep at least one training sample.
    if split >= len(samples):
        split = max(0, len(samples) - 1)

    val = samples[:split]
    train = samples[split:]

    return train, val


# =====================
# EVALUATION
# =====================
@torch.no_grad()
def evaluate(model: PolicyNetwork, loader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        loss = soft_policy_loss(logits, y)

        total_loss += float(loss.item())
        batches += 1

    if batches == 0:
        raise ValueError("Validation loader produced no batches")

    return total_loss / batches


# =====================
# TRAINING
# =====================
def train(
    model: PolicyNetwork,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    encoder: ObservationEncoder,
    action_mapper: ActionMapper,
) -> dict:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    history = {"train": [], "val": []}
    best_score = float("inf")

    checkpoint_dir = Path(CHECKPOINT_DIR)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"

    metadata = {
        "data_path": DATA_PATH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "val_fraction": VAL_FRACTION,
        "seed": SEED,
        "num_players": NUM_PLAYERS,
        "dice_per_player": DICE_PER_PLAYER,
        "history_len": HISTORY_LEN,
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        batches = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = soft_policy_loss(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batches += 1

        if batches == 0:
            raise ValueError("Training loader produced no batches")

        train_loss = running_loss / batches
        history["train"].append(train_loss)

        if val_loader is not None:
            val_loss = evaluate(model, val_loader)
            history["val"].append(val_loss)
            score = val_loss
            print(f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f}")
        else:
            score = train_loss
            print(f"Epoch {epoch:03d} | train={train_loss:.4f}")

        if score < best_score:
            best_score = score
            save_model_checkpoint(
                model=model,
                encoder=encoder,
                action_mapper=action_mapper,
                path=best_path,
                hidden_dim=HIDDEN_DIM,
                dropout=DROPOUT,
                extra_metadata={
                    **metadata,
                    "best_epoch": epoch,
                    "best_score": best_score,
                    "selection_metric": (
                        "val_loss" if val_loader is not None else "train_loss"
                    ),
                },
            )

    save_model_checkpoint(
        model=model,
        encoder=encoder,
        action_mapper=action_mapper,
        path=last_path,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        extra_metadata={
            **metadata,
            "final_train_loss": history["train"][-1],
            "final_val_loss": history["val"][-1] if history["val"] else None,
        },
    )

    return history


# =====================
# MAIN
# =====================
def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)

    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    max_total_dice = NUM_PLAYERS * DICE_PER_PLAYER

    action_mapper = ActionMapper(max_total_dice=max_total_dice)

    encoder = ObservationEncoder(
        num_players=NUM_PLAYERS,
        max_dice_per_player=DICE_PER_PLAYER,
        max_total_dice=max_total_dice,
        history_len=HISTORY_LEN,
    )

    print("Loading data...")
    samples = load_samples(DATA_PATH)
    print(f"Loaded {len(samples):,} valid samples")

    expected_input_dim = encoder.input_dim
    actual_input_dim = len(samples[0].features)

    if actual_input_dim != expected_input_dim:
        raise ValueError(
            f"Input dimension mismatch: data has {actual_input_dim}, "
            f"encoder expects {expected_input_dim}. "
            f"Check HISTORY_LEN and dataset path."
        )

    if len(samples[0].target_policy) != action_mapper.num_actions:
        raise ValueError(
            f"Target policy dimension mismatch: data has "
            f"{len(samples[0].target_policy)}, action mapper expects "
            f"{action_mapper.num_actions}."
        )

    train_samples, val_samples = split_samples(samples)

    print(f"Train: {len(train_samples):,}")
    print(f"Val:   {len(val_samples):,}")
    print(f"Input dim: {encoder.input_dim}")
    print(f"Actions:   {action_mapper.num_actions}")
    print(f"Device:    {DEVICE}")

    train_loader = DataLoader(
        PolicyDataset(train_samples),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = (
        DataLoader(
            PolicyDataset(val_samples),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        if val_samples
        else None
    )

    model = PolicyNetwork(
        input_dim=encoder.input_dim,
        num_actions=action_mapper.num_actions,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    print("Training...")
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        encoder=encoder,
        action_mapper=action_mapper,
    )

    history_path = Path(CHECKPOINT_DIR) / "history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("Training complete.")
    print(f"Saved best checkpoint to: {Path(CHECKPOINT_DIR) / 'best.pt'}")
    print(f"Saved last checkpoint to: {Path(CHECKPOINT_DIR) / 'last.pt'}")
    print(f"Saved history to: {history_path}")


if __name__ == "__main__":
    main()
