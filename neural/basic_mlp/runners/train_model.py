from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from neural.action_mapping import ActionMapper
from neural.basic_mlp.data_collection import PolicySample
from neural.basic_mlp.encoder import ObservationEncoder
from neural.basic_mlp.nn_model import PolicyNetwork
from neural.basic_mlp.training_pipeline import save_model_checkpoint

# =====================
# CONFIG (EDIT HERE)
# =====================
DATA_PATH = "artifacts/data_basic/supervised_samples.jsonl"
CHECKPOINT_DIR = "artifacts/training_basic"

BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 256
DROPOUT = 0.1

VAL_FRACTION = 0.1
SEED = 12345
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_EVERY = 5

NUM_PLAYERS = 4
DICE_PER_PLAYER = 5
HISTORY_LEN = 5


# =====================
# DATASET
# =====================
class PolicyDataset(Dataset):
    def __init__(self, samples: List[PolicySample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s.features, dtype=torch.float32),
            torch.tensor(s.target_policy, dtype=torch.float32),
        )


def load_samples(path: str | Path) -> List[PolicySample]:
    samples: List[PolicySample] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            samples.append(
                PolicySample(
                    features=obj["features"],
                    target_policy=obj["target_policy"],
                )
            )

    if not samples:
        raise ValueError("No samples loaded")

    return samples


def split_samples(samples: List[PolicySample]) -> Tuple[List, List]:
    random.shuffle(samples)
    split = int(len(samples) * VAL_FRACTION)

    val = samples[:split]
    train = samples[split:]

    return train, val


# =====================
# LOSS
# =====================
def policy_loss(logits, target):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


# =====================
# EVAL
# =====================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total = 0.0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        loss = policy_loss(logits, y)

        total += loss.item()

    return total / len(loader)


# =====================
# TRAIN
# =====================
def train(model, train_loader, val_loader):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    history = {"train": [], "val": []}
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = policy_loss(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()

        train_loss = running / len(train_loader)
        history["train"].append(train_loss)

        if val_loader:
            val_loss = evaluate(model, val_loader)
            history["val"].append(val_loss)

            print(f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f}")
        else:
            val_loss = None
            print(f"Epoch {epoch:03d} | train={train_loss:.4f}")

        # best model
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            save_model_checkpoint(
                model,
                encoder,
                action_mapper,
                Path(CHECKPOINT_DIR) / "best.pt",
            )

    # final model
    save_model_checkpoint(
        model,
        encoder,
        action_mapper,
        Path(CHECKPOINT_DIR) / "last.pt",
    )

    return history


# =====================
# MAIN
# =====================
if __name__ == "__main__":
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
    print(f"Loaded {len(samples)} samples")

    train_samples, val_samples = split_samples(samples)

    print(f"Train: {len(train_samples)}")
    print(f"Val:   {len(val_samples)}")

    train_loader = DataLoader(
        PolicyDataset(train_samples),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = (
        DataLoader(PolicyDataset(val_samples), batch_size=BATCH_SIZE)
        if val_samples
        else None
    )

    model = PolicyNetwork(
        input_dim=encoder.input_dim,
        num_actions=action_mapper.num_actions,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    print(f"Training on {DEVICE}...")
    history = train(model, train_loader, val_loader)

    with open(Path(CHECKPOINT_DIR) / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("Training complete.")
