from __future__ import annotations

import copy
import json
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from liars_dice.core.game import LiarsDiceGame
from neural.common.action_mapping import ActionMapper
from neural.basic_mlp.encoder_mlp import ObservationEncoder
from neural.basic_mlp.model_mlp import PolicyNetwork
from neural.basic_mlp.neural_ismcts_mlp import MLPNeuralISMCTSAgent
from neural.basic_mlp.training_pipeline_mlp import save_model_checkpoint

# =====================
# CONFIG
# =====================
NUM_PLAYERS = 4
DICE_PER_PLAYER = 5
MAX_TOTAL_DICE = NUM_PLAYERS * DICE_PER_PLAYER

INITIAL_CHECKPOINT = "artifacts/training/mlp/best.pt"
OUTPUT_DIR = "artifacts/self_play/mlp"

ITERATIONS = 10
SELF_PLAY_GAMES_PER_ITER = 100
EVAL_GAMES = 100

SIMS_PER_MOVE_SELF_PLAY = 500
SIMS_PER_MOVE_EVAL = 500

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
    target_policy: List[float]


# =====================
# IO HELPERS
# =====================
def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def script_config() -> Dict[str, Any]:
    return {
        "num_players": NUM_PLAYERS,
        "dice_per_player": DICE_PER_PLAYER,
        "max_total_dice": MAX_TOTAL_DICE,
        "initial_checkpoint": INITIAL_CHECKPOINT,
        "output_dir": OUTPUT_DIR,
        "iterations": ITERATIONS,
        "self_play_games_per_iter": SELF_PLAY_GAMES_PER_ITER,
        "eval_games": EVAL_GAMES,
        "sims_per_move_self_play": SIMS_PER_MOVE_SELF_PLAY,
        "sims_per_move_eval": SIMS_PER_MOVE_EVAL,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_per_iter": EPOCHS_PER_ITER,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "temperature": TEMPERATURE,
        "accept_win_rate": ACCEPT_WIN_RATE,
        "device": DEVICE,
        "base_seed": BASE_SEED,
    }


# =====================
# LOADING
# =====================
def load_initial_model(
    checkpoint_path: str | Path,
) -> Tuple[PolicyNetwork, ObservationEncoder, ActionMapper, Dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=DEVICE)

    encoder = ObservationEncoder(**payload["encoder_config"])
    action_mapper = ActionMapper(**payload["action_mapper_config"])

    model_cfg = payload.get("model_config", {})

    input_dim = int(model_cfg.get("input_dim", encoder.input_dim))
    num_actions = int(model_cfg.get("num_actions", action_mapper.num_actions))
    hidden_dim = int(model_cfg.get("hidden_dim", 256))
    dropout = float(model_cfg.get("dropout", 0.1))

    if input_dim != encoder.input_dim:
        raise ValueError(
            f"Checkpoint input_dim={input_dim}, but encoder.input_dim={encoder.input_dim}"
        )

    if num_actions != action_mapper.num_actions:
        raise ValueError(
            f"Checkpoint num_actions={num_actions}, "
            f"but action_mapper.num_actions={action_mapper.num_actions}"
        )

    model = PolicyNetwork(
        input_dim=encoder.input_dim,
        num_actions=action_mapper.num_actions,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    model.load_state_dict(payload["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    checkpoint_info = {
        "model_config": {
            "input_dim": encoder.input_dim,
            "num_actions": action_mapper.num_actions,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
        },
        "encoder_config": payload["encoder_config"],
        "action_mapper_config": payload["action_mapper_config"],
        "checkpoint_metadata": payload.get("metadata", {}),
    }

    return model, encoder, action_mapper, checkpoint_info


# =====================
# SELF-PLAY
# =====================
def sample_action_from_policy(
    root_policy: Dict[Tuple[str, Any], float],
    rng: random.Random,
    temperature: float,
) -> Tuple[str, Any]:
    actions = list(root_policy.keys())

    if not actions:
        raise ValueError("Cannot sample from an empty root policy")

    probs = torch.tensor(
        [float(root_policy[a]) for a in actions],
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


def root_policy_to_target_vector(
    root_policy: Dict[Tuple[str, Any], float],
    action_mapper: ActionMapper,
) -> List[float]:
    target = torch.zeros(action_mapper.num_actions, dtype=torch.float32)

    for action, prob in root_policy.items():
        idx = action_mapper.action_to_index(action)
        target[idx] = float(prob)

    total = target.sum()
    if total.item() <= 0.0:
        return []

    target = target / total
    return [float(x) for x in target.tolist()]


def run_self_play_game(
    model: PolicyNetwork,
    encoder: ObservationEncoder,
    action_mapper: ActionMapper,
    seed: int,
) -> Tuple[List[SelfPlaySample], Dict[str, Any]]:
    rng = random.Random(seed)

    agents = [
        MLPNeuralISMCTSAgent(
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
    total_sims = 0
    moves = 0

    while True:
        pid = game._current
        obs = game.observe(pid)

        action, root_policy, sim_count = agents[pid].search_policy(game, obs)
        total_sims += int(sim_count)
        moves += 1

        if root_policy:
            target_policy = root_policy_to_target_vector(root_policy, action_mapper)

            if target_policy:
                features = encoder.encode(obs)

                samples.append(
                    SelfPlaySample(
                        features=[float(x) for x in features],
                        target_policy=target_policy,
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

    stats = {
        "samples": len(samples),
        "moves": moves,
        "total_sims": total_sims,
        "avg_sims_per_move": total_sims / moves if moves > 0 else 0.0,
    }

    return samples, stats


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

    buffer_list = list(replay_buffer)
    steps_per_epoch = max(1, len(buffer_list) // BATCH_SIZE)

    for epoch in range(EPOCHS_PER_ITER):
        running_loss = 0.0

        for _ in range(steps_per_epoch):
            batch = random.sample(buffer_list, BATCH_SIZE)

            features = torch.tensor(
                [sample.features for sample in batch],
                dtype=torch.float32,
                device=DEVICE,
            )

            target_policy = torch.tensor(
                [sample.target_policy for sample in batch],
                dtype=torch.float32,
                device=DEVICE,
            )

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
) -> Tuple[int, Dict[str, int], Dict[str, Any]]:
    """
    Runs one 4-player evaluation game with 2 new-model agents and 2 best-model agents.

    Returns:
      winner_group:
        1 if a new-model seat wins,
        0 if a best-model seat wins.
      placement_counts:
        placement counts by group.
      stats:
        move/simulation counts.
    """
    rng = random.Random(seed)

    seat_groups = ["new", "new", "best", "best"]
    rng.shuffle(seat_groups)

    agents = []
    for group in seat_groups:
        model = new_model if group == "new" else best_model

        agents.append(
            MLPNeuralISMCTSAgent(
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
    total_sims = 0
    moves = 0

    while True:
        pid = game._current
        obs = game.observe(pid)

        action = agents[pid].select_action(game, obs)
        total_sims += int(getattr(agents[pid], "_last_sim_count", 0))
        moves += 1

        info = game.step(action)

        for player_id in range(NUM_PLAYERS):
            if game._dice_left[player_id] == 0 and player_id not in eliminated:
                eliminated.append(player_id)

        if info.get("terminal"):
            winner = int(info["winner"])
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

    stats = {
        "moves": moves,
        "total_sims": total_sims,
        "avg_sims_per_move": total_sims / moves if moves > 0 else 0.0,
    }

    return winner_group, placement_counts, stats


def evaluate_new_vs_best(
    new_model: PolicyNetwork,
    best_model: PolicyNetwork,
    encoder: ObservationEncoder,
    action_mapper: ActionMapper,
    iteration: int,
) -> Dict[str, float]:
    new_wins = 0
    placement_totals = defaultdict(int)

    total_moves = 0
    total_sims = 0

    for game_idx in tqdm(range(EVAL_GAMES), desc="Evaluation"):
        winner_group, placement_counts, stats = run_eval_game(
            new_model=new_model,
            best_model=best_model,
            encoder=encoder,
            action_mapper=action_mapper,
            seed=BASE_SEED + iteration * 10_000_000 + game_idx,
        )

        new_wins += winner_group

        for key, value in placement_counts.items():
            placement_totals[key] += value

        total_moves += int(stats["moves"])
        total_sims += int(stats["total_sims"])

    win_rate = new_wins / EVAL_GAMES

    result: Dict[str, float] = {
        "new_win_rate": win_rate,
        "eval_games": float(EVAL_GAMES),
        "eval_total_moves": float(total_moves),
        "eval_total_sims": float(total_sims),
        "eval_avg_sims_per_move": (
            total_sims / total_moves if total_moves > 0 else 0.0
        ),
    }

    for key, value in placement_totals.items():
        result[key] = float(value)

    return result


# =====================
# CHECKPOINT HELPERS
# =====================
def checkpoint_metadata(
    iteration: int,
    accepted: bool,
    train_losses: List[float],
    eval_result: Dict[str, float],
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        **script_config(),
        "iteration": iteration,
        "accepted": accepted,
        "train_losses": train_losses,
        "eval_result": eval_result,
    }

    if extra is not None:
        metadata.update(extra)

    return metadata


# =====================
# MAIN
# =====================
def main() -> None:
    random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates_dir = output_dir / "candidates"
    accepted_dir = output_dir / "accepted"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir.mkdir(parents=True, exist_ok=True)

    save_json(output_dir / "config.json", script_config())

    print(f"Loading initial checkpoint: {INITIAL_CHECKPOINT}")
    model, encoder, action_mapper, checkpoint_info = load_initial_model(
        INITIAL_CHECKPOINT
    )

    save_json(output_dir / "initial_checkpoint_info.json", checkpoint_info)

    model_cfg = checkpoint_info["model_config"]
    hidden_dim = int(model_cfg["hidden_dim"])
    dropout = float(model_cfg["dropout"])

    best_model = copy.deepcopy(model)
    best_model.to(DEVICE)
    best_model.eval()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    replay_buffer: deque[SelfPlaySample] = deque(maxlen=BUFFER_SIZE)
    history: List[Dict[str, Any]] = []

    save_model_checkpoint(
        model=best_model,
        encoder=encoder,
        action_mapper=action_mapper,
        path=output_dir / "initial.pt",
        hidden_dim=hidden_dim,
        dropout=dropout,
        extra_metadata={
            **script_config(),
            "source_checkpoint": INITIAL_CHECKPOINT,
            "description": "Initial supervised checkpoint before self-play.",
        },
    )

    accepted_iterations: List[int] = []

    overall_start = time.perf_counter()

    for iteration in range(ITERATIONS):
        iteration_start = time.perf_counter()

        print(f"\n=== ITERATION {iteration} ===")

        # ---------------------
        # Self-play collection
        # ---------------------
        collect_start = time.perf_counter()

        collected_samples: List[SelfPlaySample] = []
        self_play_total_moves = 0
        self_play_total_sims = 0

        for game_idx in tqdm(
            range(SELF_PLAY_GAMES_PER_ITER),
            desc="Self-play",
        ):
            samples, stats = run_self_play_game(
                model=model,
                encoder=encoder,
                action_mapper=action_mapper,
                seed=BASE_SEED + iteration * 1_000_000 + game_idx,
            )

            collected_samples.extend(samples)
            self_play_total_moves += int(stats["moves"])
            self_play_total_sims += int(stats["total_sims"])

        replay_buffer.extend(collected_samples)

        collect_time_s = time.perf_counter() - collect_start

        print(f"Collected samples: {len(collected_samples)}")
        print(f"Replay buffer size: {len(replay_buffer)}")

        # ---------------------
        # Train candidate
        # ---------------------
        train_start = time.perf_counter()

        train_losses = train_one_iteration(
            model=model,
            replay_buffer=replay_buffer,
            optimizer=optimizer,
        )

        train_time_s = time.perf_counter() - train_start

        # ---------------------
        # Evaluate candidate
        # ---------------------
        eval_start = time.perf_counter()

        eval_result = evaluate_new_vs_best(
            new_model=model,
            best_model=best_model,
            encoder=encoder,
            action_mapper=action_mapper,
            iteration=iteration,
        )

        eval_time_s = time.perf_counter() - eval_start

        new_win_rate = float(eval_result["new_win_rate"])
        accepted = new_win_rate >= ACCEPT_WIN_RATE

        print(f"New model win rate vs best: {new_win_rate:.3f}")
        print("Accepted" if accepted else "Rejected")

        candidate_path = candidates_dir / f"iteration_{iteration:03d}.pt"

        save_model_checkpoint(
            model=model,
            encoder=encoder,
            action_mapper=action_mapper,
            path=candidate_path,
            hidden_dim=hidden_dim,
            dropout=dropout,
            extra_metadata=checkpoint_metadata(
                iteration=iteration,
                accepted=accepted,
                train_losses=train_losses,
                eval_result=eval_result,
                extra={
                    "checkpoint_type": "candidate",
                    "replay_buffer_size": len(replay_buffer),
                    "collected_samples": len(collected_samples),
                },
            ),
        )

        if accepted:
            best_model = copy.deepcopy(model)
            best_model.to(DEVICE)
            best_model.eval()

            accepted_iterations.append(iteration)

            best_path = output_dir / "best.pt"
            accepted_path = accepted_dir / f"iteration_{iteration:03d}.pt"

            metadata = checkpoint_metadata(
                iteration=iteration,
                accepted=True,
                train_losses=train_losses,
                eval_result=eval_result,
                extra={
                    "checkpoint_type": "accepted_best",
                    "replay_buffer_size": len(replay_buffer),
                    "collected_samples": len(collected_samples),
                },
            )

            save_model_checkpoint(
                model=best_model,
                encoder=encoder,
                action_mapper=action_mapper,
                path=best_path,
                hidden_dim=hidden_dim,
                dropout=dropout,
                extra_metadata=metadata,
            )

            save_model_checkpoint(
                model=best_model,
                encoder=encoder,
                action_mapper=action_mapper,
                path=accepted_path,
                hidden_dim=hidden_dim,
                dropout=dropout,
                extra_metadata=metadata,
            )

        else:
            # Revert trainable model back to current best.
            model.load_state_dict(best_model.state_dict())
            model.to(DEVICE)
            model.eval()

            # Reset optimizer after reverting weights.
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
            hidden_dim=hidden_dim,
            dropout=dropout,
            extra_metadata=checkpoint_metadata(
                iteration=iteration,
                accepted=accepted,
                train_losses=train_losses,
                eval_result=eval_result,
                extra={
                    "checkpoint_type": "last_current_model",
                    "replay_buffer_size": len(replay_buffer),
                    "collected_samples": len(collected_samples),
                },
            ),
        )

        iteration_time_s = time.perf_counter() - iteration_start

        iteration_record: Dict[str, Any] = {
            "iteration": iteration,
            "accepted": accepted,
            "accepted_iterations_so_far": list(accepted_iterations),
            "new_win_rate_vs_best": new_win_rate,
            "collected_samples": len(collected_samples),
            "buffer_size": len(replay_buffer),
            "train_losses": train_losses,
            "self_play_games": SELF_PLAY_GAMES_PER_ITER,
            "self_play_total_moves": self_play_total_moves,
            "self_play_total_sims": self_play_total_sims,
            "self_play_avg_sims_per_move": (
                self_play_total_sims / self_play_total_moves
                if self_play_total_moves > 0
                else 0.0
            ),
            "collect_time_s": collect_time_s,
            "train_time_s": train_time_s,
            "eval_time_s": eval_time_s,
            "iteration_time_s": iteration_time_s,
            "candidate_checkpoint": str(candidate_path),
            **eval_result,
        }

        history.append(iteration_record)
        save_json(output_dir / "history.json", history)

    total_time_s = time.perf_counter() - overall_start

    summary = {
        **script_config(),
        "total_time_s": total_time_s,
        "iterations_completed": ITERATIONS,
        "accepted_iterations": accepted_iterations,
        "num_accepted": len(accepted_iterations),
        "final_buffer_size": len(replay_buffer),
        "best_checkpoint": str(output_dir / "best.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "history_file": str(output_dir / "history.json"),
    }

    save_json(output_dir / "summary.json", summary)

    print("\nDone.")
    print(f"Accepted iterations: {accepted_iterations}")
    print(f"Best checkpoint: {output_dir / 'best.pt'}")
    print(f"Last checkpoint: {output_dir / 'last.pt'}")
    print(f"History: {output_dir / 'history.json'}")
    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
