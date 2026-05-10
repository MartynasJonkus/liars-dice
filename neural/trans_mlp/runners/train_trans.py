from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

import torch

from neural.common.action_mapping import ActionMapper
from neural.trans_mlp.data_collection_trans import PolicySample
from neural.trans_mlp.encoder_trans import ObservationEncoder
from neural.trans_mlp.model_trans import PolicyNetwork
from neural.trans_mlp.training_pipeline_trans import (
    TrainingConfig,
    train_policy_network,
)

# =====================
# CONFIG
# =====================
DATA_PATH = "artifacts/data/trans/supervised_samples.jsonl"
OUTPUT_DIR = "artifacts/training/trans"

NUM_PLAYERS = 4
DICE_PER_PLAYER = 5
MAX_BIDS = 10

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN_DIM = 256
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.1

VAL_FRACTION = 0.10
SEED = 12345


# =====================
# HELPERS
# =====================
def load_samples_jsonl(path: str | Path) -> List[PolicySample]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    samples: List[PolicySample] = []
    skipped_zero_mass = 0

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} in {path}") from exc

            target_policy = obj.get("target_policy")
            if target_policy is None:
                raise KeyError(f"Line {line_num} is missing 'target_policy'")

            target_policy = [float(x) for x in target_policy]
            mass = sum(target_policy)

            if mass <= 0.0:
                skipped_zero_mass += 1
                continue

            target_policy = [x / mass for x in target_policy]

            samples.append(
                PolicySample(
                    static_features=[float(x) for x in obj["static_features"]],
                    bid_history=[[float(v) for v in row] for row in obj["bid_history"]],
                    bid_mask=[bool(x) for x in obj["bid_mask"]],
                    target_policy=target_policy,
                )
            )

    if not samples:
        raise ValueError(f"No valid samples found in {path}")

    if skipped_zero_mass > 0:
        print(f"Skipped {skipped_zero_mass} samples with zero target-policy mass")

    return samples


def split_samples(
    samples: List[PolicySample],
    val_fraction: float,
    seed: int,
) -> Tuple[List[PolicySample], List[PolicySample]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")

    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    val_size = int(len(shuffled) * val_fraction)

    if val_fraction > 0.0 and val_size == 0 and len(shuffled) > 1:
        val_size = 1

    if val_size >= len(shuffled):
        val_size = max(0, len(shuffled) - 1)

    val_samples = shuffled[:val_size]
    train_samples = shuffled[val_size:]

    if not train_samples:
        raise ValueError("No training samples left after split")

    return train_samples, val_samples


def validate_sample_shapes(
    samples: List[PolicySample],
    encoder: ObservationEncoder,
    action_mapper: ActionMapper,
) -> None:
    expected_static_dim = encoder.static_dim
    expected_token_dim = encoder.bid_token_dim
    expected_max_bids = encoder.max_bids
    expected_num_actions = action_mapper.num_actions

    # Validate a subset to avoid scanning very large datasets twice.
    for i, sample in enumerate(samples[:100]):
        if len(sample.static_features) != expected_static_dim:
            raise ValueError(
                f"Sample {i}: static_features length "
                f"{len(sample.static_features)} != expected {expected_static_dim}"
            )

        if len(sample.bid_history) != expected_max_bids:
            raise ValueError(
                f"Sample {i}: bid_history length "
                f"{len(sample.bid_history)} != expected {expected_max_bids}"
            )

        for j, token in enumerate(sample.bid_history):
            if len(token) != expected_token_dim:
                raise ValueError(
                    f"Sample {i}, token {j}: token length "
                    f"{len(token)} != expected {expected_token_dim}"
                )

        if len(sample.bid_mask) != expected_max_bids:
            raise ValueError(
                f"Sample {i}: bid_mask length "
                f"{len(sample.bid_mask)} != expected {expected_max_bids}"
            )

        if len(sample.target_policy) != expected_num_actions:
            raise ValueError(
                f"Sample {i}: target_policy length "
                f"{len(sample.target_policy)} != expected {expected_num_actions}"
            )

        target_sum = sum(sample.target_policy)
        if abs(target_sum - 1.0) > 1e-4:
            raise ValueError(
                f"Sample {i}: target_policy sums to {target_sum:.6f}, expected 1.0"
            )


def save_history(path: str | Path, history: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


# =====================
# MAIN
# =====================
def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_total_dice = NUM_PLAYERS * DICE_PER_PLAYER

    encoder = ObservationEncoder(
        num_players=NUM_PLAYERS,
        max_dice_per_player=DICE_PER_PLAYER,
        max_total_dice=max_total_dice,
        max_bids=MAX_BIDS,
    )

    action_mapper = ActionMapper(max_total_dice=max_total_dice)

    print(f"Loading samples from: {DATA_PATH}")
    samples = load_samples_jsonl(DATA_PATH)
    print(f"Loaded {len(samples):,} valid samples")

    print("Validating sample shapes...")
    validate_sample_shapes(samples, encoder, action_mapper)
    print("Sample validation passed")

    train_samples, val_samples = split_samples(
        samples=samples,
        val_fraction=VAL_FRACTION,
        seed=SEED,
    )

    print(f"Train samples: {len(train_samples):,}")
    print(f"Val samples:   {len(val_samples):,}")
    print(f"Static dim:    {encoder.static_dim}")
    print(f"Token dim:     {encoder.bid_token_dim}")
    print(f"Max bids:      {encoder.max_bids}")
    print(f"Actions:       {action_mapper.num_actions}")
    print(f"Device:        {DEVICE}")

    model = PolicyNetwork(
        static_dim=encoder.static_dim,
        token_dim=encoder.bid_token_dim,
        num_actions=action_mapper.num_actions,
        hidden_dim=HIDDEN_DIM,
        max_bids=encoder.max_bids,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    )

    config = TrainingConfig(
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS,
        device=DEVICE,
    )

    metadata = {
        "data_path": DATA_PATH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "val_fraction": VAL_FRACTION,
        "seed": SEED,
        "num_players": NUM_PLAYERS,
        "dice_per_player": DICE_PER_PLAYER,
        "max_bids": MAX_BIDS,
        "hidden_dim": HIDDEN_DIM,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "dropout": DROPOUT,
    }

    print(f"Training on {DEVICE}...")

    history = train_policy_network(
        model=model,
        train_samples=train_samples,
        val_samples=val_samples,
        config=config,
        encoder=encoder,
        action_mapper=action_mapper,
        best_checkpoint_path=output_dir / "best.pt",
        last_checkpoint_path=output_dir / "last.pt",
        hidden_dim=HIDDEN_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        extra_metadata=metadata,
    )

    history_path = output_dir / "history.json"
    save_history(history_path, history)

    print("Training complete.")
    print(f"Saved training history to: {history_path}")
    print(f"Saved best checkpoint to: {output_dir / 'best.pt'}")
    print(f"Saved last checkpoint to: {output_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
