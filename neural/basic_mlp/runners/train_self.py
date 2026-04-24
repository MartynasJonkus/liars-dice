from __future__ import annotations

import copy
import json
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from liars_dice.core.game import LiarsDiceGame
from neural.action_mapping import ActionMapper
from neural.basic_mlp.encoder import ObservationEncoder
from neural.basic_mlp.neural_ismcts import NeuralISMCTSPUCTAgent
from neural.basic_mlp.nn_model import PolicyNetwork
from neural.basic_mlp.training_pipeline import save_model_checkpoint

# =====================
# CONFIG
# =====================
NUM_PLAYERS = 4
DICE_PER_PLAYER = 5
MAX_TOTAL_DICE = NUM_PLAYERS * DICE_PER_PLAYER

INITIAL_CHECKPOINT = "artifacts/training_basic/best.pt"
OUTPUT_DIR = "artifacts/self_play_mlp"

ITERATIONS = 10
SELF_PLAY_GAMES_PER_ITER = 100
EVAL_GAMES = 100

SIMS_PER_MOVE_SELF_PLAY = 300
SIMS_PER_MOVE_EVAL = 300

BUFFER_SIZE = 100_000
BATCH_SIZE = 512
EPOCHS_PER_ITER = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

TEMPERATURE = 1.0
ACCEPT_WIN_RATE = 0.55

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_SEED = 12345


# =====================
# DATA
# =====================
@dataclass
class SelfPlaySample:
    features: List[float]
    target_policy: torch.Tensor


# =====================
# LOADING
# =====================
def load_initial_model(
    checkpoint_path: str | Path,
) -> Tuple[PolicyNetwork, ObservationEncoder, ActionMapper]:
    payload = torch.load(checkpoint_path, map_location=DEVICE)

    encoder = ObservationEncoder(**payload["encoder_config"])
    action_mapper = ActionMapper(**payload["action_mapper_config"])

    model = PolicyNetwork(
        input_dim=encoder.input_dim,
        num_actions=action_mapper.num_actions,
        hidden_dim=256,
        dropout=0.1,
    )

    model.load_state_dict(payload["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, encoder, action_mapper


# =====================
# SELF-PLAY
# =====================
def sample_action_from_policy(
    root_policy: Dict,
    rng: random.Random,
    temperature: float,
):
    actions = list(root_policy.keys())
    probs = torch.tensor(
        [root_policy[a] for a in actions],
        dtype=torch.float32,
    )

    if temperature <= 0.0:
        best_prob = probs.max().item()
        candidates = [
            a for a, p in zip(actions, probs.tolist()) if abs(p - best_prob) < 1e-12
        ]
        return rng.choice(candidates)

    probs = probs.pow(1.0 / temperature)
    probs_sum = probs.sum()

    if probs_sum.item() <= 0.0:
        return rng.choice(actions)

    probs = probs / probs_sum
    idx = torch.multinomial(probs, 1).item()
    return actions[idx]


def run_self_play_game(
    model: PolicyNetwork,
    encoder: ObservationEncoder,
    action_mapper: ActionMapper,
    seed: int,
) -> List[SelfPlaySample]:
    rng = random.Random(seed)

    agents = [
        NeuralISMCTSPUCTAgent(
            model=model,
            encoder=encoder,
            action_mapper=action_mapper,
            seed=rng.randint(0, 10**9),
            sims_per_move=SIMS_PER_MOVE_SELF_PLAY,
            time_limit_s=None,
        )
        for _ in range(NUM_PLAYERS)
    ]

    game = LiarsDiceGame(
        num_players=NUM_PLAYERS,
        dice_per_player=DICE_PER_PLAYER,
        seed=rng.randint(0, 10**9),
    )

    samples: List[SelfPlaySample] = []

    while True:
        pid = game._current
        obs = game.observe(pid)

        action, root_policy, _sim_count = agents[pid].search_policy(game, obs)

        if root_policy:
            features = encoder.encode(obs)

            target = torch.zeros(action_mapper.num_actions, dtype=torch.float32)
            for a, p in root_policy.items():
                idx = action_mapper.action_to_index(a)
                target[idx] = float(p)

            target_sum = target.sum()
            if target_sum.item() > 0.0:
                target = target / target_sum

                samples.append(
                    SelfPlaySample(
                        features=features,
                        target_policy=target,
                    )
                )

                action = sample_action_from_policy(
                    root_policy=root_policy,
                    rng=rng,
                    temperature=TEMPERATURE,
                )

        info = game.step(action)
        if info.get("terminal"):
            break

    return samples


# =====================
# TRAINING
# =====================
def policy_cross_entropy_loss(
    logits: torch.Tensor,
    target_policy: torch.Tensor,
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(target_policy * log_probs).sum(dim=-1).mean()


def train_one_iteration(
    model: PolicyNetwork,
    replay_buffer: deque[SelfPlaySample],
    optimizer: torch.optim.Optimizer,
) -> List[float]:
    model.train()
    losses: List[float] = []

    if len(replay_buffer) < BATCH_SIZE:
        return losses

    steps_per_epoch = max(1, len(replay_buffer) // BATCH_SIZE)

    for epoch in range(EPOCHS_PER_ITER):
        running_loss = 0.0

        for _ in range(steps_per_epoch):
            batch = random.sample(list(replay_buffer), BATCH_SIZE)

            features = torch.tensor(
                [s.features for s in batch],
                dtype=torch.float32,
                device=DEVICE,
            )

            target_policy = torch.stack([s.target_policy for s in batch]).to(DEVICE)

            logits = model(features)
            loss = policy_cross_entropy_loss(logits, target_policy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        epoch_loss = running_loss / steps_per_epoch
        losses.append(epoch_loss)
        print(f"  Train epoch {epoch + 1:02d} | loss={epoch_loss:.4f}")

    model.eval()
    return losses


# =====================
# EVALUATION
# =====================
def run_eval_game(
    new_model: PolicyNetwork,
    best_model: PolicyNetwork,
    encoder: ObservationEncoder,
    action_mapper: ActionMapper,
    seed: int,
) -> Tuple[int, Dict[str, int]]:
    """
    Runs one 4-player game with 2 new-model agents and 2 best-model agents.

    Returns:
      winner_group:
        1 if new model wins
        0 if best model wins
      placements:
        counts by group labels
    """
    rng = random.Random(seed)

    seat_groups = ["new", "new", "best", "best"]
    rng.shuffle(seat_groups)

    agents = []
    for group in seat_groups:
        model = new_model if group == "new" else best_model
        agents.append(
            NeuralISMCTSPUCTAgent(
                model=model,
                encoder=encoder,
                action_mapper=action_mapper,
                seed=rng.randint(0, 10**9),
                sims_per_move=SIMS_PER_MOVE_EVAL,
                time_limit_s=None,
            )
        )

    game = LiarsDiceGame(
        num_players=NUM_PLAYERS,
        dice_per_player=DICE_PER_PLAYER,
        seed=rng.randint(0, 10**9),
    )

    eliminated: List[int] = []

    while True:
        pid = game._current
        obs = game.observe(pid)

        action = agents[pid].select_action(game, obs)
        info = game.step(action)

        for p in range(NUM_PLAYERS):
            if game._dice_left[p] == 0 and p not in eliminated:
                eliminated.append(p)

        if info.get("terminal"):
            winner = info["winner"]
            if winner not in eliminated:
                eliminated.append(winner)
            break

    winner_group = 1 if seat_groups[winner] == "new" else 0

    placements_best_to_worst = eliminated[::-1]
    placement_counts = {
        "new_first": 0,
        "new_second": 0,
        "new_third": 0,
        "new_fourth": 0,
        "best_first": 0,
        "best_second": 0,
        "best_third": 0,
        "best_fourth": 0,
    }

    place_names = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
    }

    for place, seat in enumerate(placements_best_to_worst, start=1):
        group = seat_groups[seat]
        key = f"{group}_{place_names[place]}"
        placement_counts[key] += 1

    return winner_group, placement_counts


def evaluate_new_vs_best(
    new_model: PolicyNetwork,
    best_model: PolicyNetwork,
    encoder: ObservationEncoder,
    action_mapper: ActionMapper,
    iteration: int,
) -> Dict[str, float]:
    new_wins = 0
    placement_totals = defaultdict_int()

    for i in tqdm(range(EVAL_GAMES), desc="Evaluation"):
        winner_group, placement_counts = run_eval_game(
            new_model=new_model,
            best_model=best_model,
            encoder=encoder,
            action_mapper=action_mapper,
            seed=BASE_SEED + iteration * 10_000_000 + i,
        )

        new_wins += winner_group

        for k, v in placement_counts.items():
            placement_totals[k] += v

    win_rate = new_wins / EVAL_GAMES

    result = {
        "new_win_rate": win_rate,
    }

    for k, v in placement_totals.items():
        result[k] = float(v)

    return result


def defaultdict_int():
    return defaultdict(int)


# =====================
# SAVE HELPERS
# =====================
def save_history(path: str | Path, history: List[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


# =====================
# MAIN
# =====================
def main() -> None:
    random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading initial checkpoint: {INITIAL_CHECKPOINT}")
    model, encoder, action_mapper = load_initial_model(INITIAL_CHECKPOINT)

    best_model = copy.deepcopy(model)
    best_model.to(DEVICE)
    best_model.eval()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    replay_buffer: deque[SelfPlaySample] = deque(maxlen=BUFFER_SIZE)
    history: List[Dict] = []

    save_model_checkpoint(
        model=best_model,
        encoder=encoder,
        action_mapper=action_mapper,
        path=output_dir / "initial.pt",
    )

    for iteration in range(ITERATIONS):
        print(f"\n=== ITERATION {iteration} ===")

        # ---------------------
        # Self-play collection
        # ---------------------
        collected_samples: List[SelfPlaySample] = []

        for game_idx in tqdm(
            range(SELF_PLAY_GAMES_PER_ITER),
            desc="Self-play",
        ):
            samples = run_self_play_game(
                model=model,
                encoder=encoder,
                action_mapper=action_mapper,
                seed=BASE_SEED + iteration * 1_000_000 + game_idx,
            )
            collected_samples.extend(samples)

        replay_buffer.extend(collected_samples)

        print(f"Collected samples: {len(collected_samples)}")
        print(f"Replay buffer size: {len(replay_buffer)}")

        # ---------------------
        # Train
        # ---------------------
        train_losses = train_one_iteration(
            model=model,
            replay_buffer=replay_buffer,
            optimizer=optimizer,
        )

        # ---------------------
        # Evaluate
        # ---------------------
        eval_result = evaluate_new_vs_best(
            new_model=model,
            best_model=best_model,
            encoder=encoder,
            action_mapper=action_mapper,
            iteration=iteration,
        )

        new_win_rate = eval_result["new_win_rate"]
        accepted = new_win_rate >= ACCEPT_WIN_RATE

        print(f"New model win rate vs best: {new_win_rate:.3f}")
        print("Accepted" if accepted else "Rejected")

        if accepted:
            best_model = copy.deepcopy(model)
            best_model.to(DEVICE)
            best_model.eval()

            save_model_checkpoint(
                model=best_model,
                encoder=encoder,
                action_mapper=action_mapper,
                path=output_dir / "best.pt",
            )
        else:
            model.load_state_dict(best_model.state_dict())
            model.to(DEVICE)
            model.eval()

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            )

        save_model_checkpoint(
            model=model,
            encoder=encoder,
            action_mapper=action_mapper,
            path=output_dir / "last.pt",
        )

        iteration_record = {
            "iteration": iteration,
            "collected_samples": len(collected_samples),
            "buffer_size": len(replay_buffer),
            "train_losses": train_losses,
            "new_win_rate_vs_best": new_win_rate,
            "accepted": accepted,
            **eval_result,
        }

        history.append(iteration_record)
        save_history(output_dir / "history.json", history)

    print("\nDone.")
    print(f"Best checkpoint: {output_dir / 'best.pt'}")
    print(f"Last checkpoint: {output_dir / 'last.pt'}")
    print(f"History: {output_dir / 'history.json'}")


if __name__ == "__main__":
    main()
