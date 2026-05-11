from __future__ import annotations

import copy
import json
import os
import random
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from liars_dice.core.game import LiarsDiceGame

from neural.common.action_mapping import ActionMapper
from neural.trans_mlp.encoder_trans import ObservationEncoder
from neural.trans_mlp.model_trans import PolicyNetwork
from neural.trans_mlp.neural_ismcts_trans import TransformerNeuralISMCTSAgent
from neural.trans_mlp.training_pipeline_trans import save_model_checkpoint


# =====================
# CONFIG
# =====================
NUM_PLAYERS = 4
DICE_PER_PLAYER = 5
MAX_TOTAL_DICE = NUM_PLAYERS * DICE_PER_PLAYER

INITIAL_CHECKPOINT = "artifacts/training/trans/best.pt"
OUTPUT_DIR = "artifacts/self_play/trans"

ITERATIONS = 10
SELF_PLAY_GAMES_PER_ITER = 100
EVAL_GAMES = 100

SIMS_PER_MOVE_SELF_PLAY = 500
SIMS_PER_MOVE_EVAL = 500

WORKERS = max(1, min((os.cpu_count() or 2) - 1, SELF_PLAY_GAMES_PER_ITER))

BUFFER_SIZE = 100_000
BATCH_SIZE = 512
EPOCHS_PER_ITER = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

TEMPERATURE = 1.0
ACCEPT_WIN_RATE = 0.55

TRAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORKER_DEVICE = "cpu"

BASE_SEED = 12345


# =====================
# WORKER GLOBALS
# =====================
_worker_model: PolicyNetwork | None = None
_worker_best_model: PolicyNetwork | None = None
_worker_encoder: ObservationEncoder | None = None
_worker_action_mapper: ActionMapper | None = None
_worker_model_cfg: Dict[str, Any] | None = None


# =====================
# DATA
# =====================
@dataclass
class SelfPlaySample:
    static_features: List[float]
    bid_history: List[List[float]]
    bid_mask: List[bool]
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
        "workers": WORKERS,
        "sims_per_move_self_play": SIMS_PER_MOVE_SELF_PLAY,
        "sims_per_move_eval": SIMS_PER_MOVE_EVAL,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_per_iter": EPOCHS_PER_ITER,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "temperature": TEMPERATURE,
        "accept_win_rate": ACCEPT_WIN_RATE,
        "train_device": TRAIN_DEVICE,
        "worker_device": WORKER_DEVICE,
        "base_seed": BASE_SEED,
    }


def state_dict_to_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


# =====================
# MODEL HELPERS
# =====================
def build_transformer_model(
    model_cfg: Dict[str, Any],
    encoder: ObservationEncoder,
    action_mapper: ActionMapper,
    device: str,
) -> PolicyNetwork:
    model = PolicyNetwork(
        static_dim=encoder.static_dim,
        token_dim=encoder.bid_token_dim,
        num_actions=action_mapper.num_actions,
        hidden_dim=int(model_cfg["hidden_dim"]),
        max_bids=encoder.max_bids,
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        num_layers=int(model_cfg["num_layers"]),
        dim_feedforward=int(model_cfg["dim_feedforward"]),
        dropout=float(model_cfg["dropout"]),
    )

    model.to(device)
    model.eval()
    return model


def load_initial_model(
    checkpoint_path: str | Path,
) -> Tuple[PolicyNetwork, ObservationEncoder, ActionMapper, Dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=TRAIN_DEVICE)

    encoder = ObservationEncoder(**payload["encoder_config"])
    action_mapper = ActionMapper(**payload["action_mapper_config"])

    model_cfg = payload["model_config"]

    if int(model_cfg["static_dim"]) != encoder.static_dim:
        raise ValueError("Checkpoint static_dim does not match encoder.static_dim")

    if int(model_cfg["token_dim"]) != encoder.bid_token_dim:
        raise ValueError("Checkpoint token_dim does not match encoder.bid_token_dim")

    if int(model_cfg["max_bids"]) != encoder.max_bids:
        raise ValueError("Checkpoint max_bids does not match encoder.max_bids")

    if int(model_cfg["num_actions"]) != action_mapper.num_actions:
        raise ValueError("Checkpoint num_actions does not match action mapper")

    model = build_transformer_model(
        model_cfg=model_cfg,
        encoder=encoder,
        action_mapper=action_mapper,
        device=TRAIN_DEVICE,
    )

    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    checkpoint_info = {
        "model_config": {
            "static_dim": encoder.static_dim,
            "token_dim": encoder.bid_token_dim,
            "num_actions": action_mapper.num_actions,
            "hidden_dim": int(model_cfg["hidden_dim"]),
            "max_bids": encoder.max_bids,
            "d_model": int(model_cfg["d_model"]),
            "nhead": int(model_cfg["nhead"]),
            "num_layers": int(model_cfg["num_layers"]),
            "dim_feedforward": int(model_cfg["dim_feedforward"]),
            "dropout": float(model_cfg["dropout"]),
        },
        "encoder_config": payload["encoder_config"],
        "action_mapper_config": payload["action_mapper_config"],
        "checkpoint_metadata": payload.get("metadata", {}),
    }

    return model, encoder, action_mapper, checkpoint_info


# =====================
# WORKER INITIALIZERS
# =====================
def init_self_play_worker(
    model_state_dict: Dict[str, torch.Tensor],
    model_cfg: Dict[str, Any],
    encoder_config: Dict[str, Any],
    action_mapper_config: Dict[str, Any],
) -> None:
    global _worker_model
    global _worker_encoder
    global _worker_action_mapper
    global _worker_model_cfg

    torch.set_num_threads(1)

    _worker_encoder = ObservationEncoder(**encoder_config)
    _worker_action_mapper = ActionMapper(**action_mapper_config)
    _worker_model_cfg = model_cfg

    _worker_model = build_transformer_model(
        model_cfg=model_cfg,
        encoder=_worker_encoder,
        action_mapper=_worker_action_mapper,
        device=WORKER_DEVICE,
    )

    _worker_model.load_state_dict(model_state_dict)
    _worker_model.eval()


def init_eval_worker(
    new_state_dict: Dict[str, torch.Tensor],
    best_state_dict: Dict[str, torch.Tensor],
    model_cfg: Dict[str, Any],
    encoder_config: Dict[str, Any],
    action_mapper_config: Dict[str, Any],
) -> None:
    global _worker_model
    global _worker_best_model
    global _worker_encoder
    global _worker_action_mapper
    global _worker_model_cfg

    torch.set_num_threads(1)

    _worker_encoder = ObservationEncoder(**encoder_config)
    _worker_action_mapper = ActionMapper(**action_mapper_config)
    _worker_model_cfg = model_cfg

    _worker_model = build_transformer_model(
        model_cfg=model_cfg,
        encoder=_worker_encoder,
        action_mapper=_worker_action_mapper,
        device=WORKER_DEVICE,
    )
    _worker_model.load_state_dict(new_state_dict)
    _worker_model.eval()

    _worker_best_model = build_transformer_model(
        model_cfg=model_cfg,
        encoder=_worker_encoder,
        action_mapper=_worker_action_mapper,
        device=WORKER_DEVICE,
    )
    _worker_best_model.load_state_dict(best_state_dict)
    _worker_best_model.eval()


# =====================
# SELF-PLAY HELPERS
# =====================
def sample_action_from_policy(
    root_policy: Dict[Tuple[str, Any], float],
    rng: random.Random,
    temperature: float,
) -> Tuple[str, Any]:
    actions = list(root_policy.keys())

    if not actions:
        raise ValueError("Cannot sample from an empty root policy")

    probs = torch.tensor([float(root_policy[a]) for a in actions], dtype=torch.float32)

    if temperature <= 0.0:
        best_prob = probs.max().item()
        candidates = [
            a for a, p in zip(actions, probs.tolist()) if abs(p - best_prob) < 1e-12
        ]
        return rng.choice(candidates)

    probs = probs.pow(1.0 / temperature)
    total = probs.sum()

    if total.item() <= 0.0:
        return rng.choice(actions)

    probs = probs / total
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


def encode_self_play_sample(
    encoder: ObservationEncoder,
    obs,
    target_policy: List[float],
) -> SelfPlaySample:
    encoded = encoder.encode(obs)

    return SelfPlaySample(
        static_features=[float(x) for x in encoded["static_features"]],
        bid_history=[[float(v) for v in row] for row in encoded["bid_history"]],
        bid_mask=[bool(x) for x in encoded["bid_mask"]],
        target_policy=[float(x) for x in target_policy],
    )


def run_self_play_game_worker(seed: int) -> Tuple[List[SelfPlaySample], Dict[str, Any]]:
    if _worker_model is None or _worker_encoder is None or _worker_action_mapper is None:
        raise RuntimeError("Self-play worker was not initialized")

    rng = random.Random(seed)

    agents = [
        TransformerNeuralISMCTSAgent(
            model=_worker_model,
            encoder=_worker_encoder,
            action_mapper=_worker_action_mapper,
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
            target_policy = root_policy_to_target_vector(
                root_policy=root_policy,
                action_mapper=_worker_action_mapper,
            )

            if target_policy:
                samples.append(
                    encode_self_play_sample(
                        encoder=_worker_encoder,
                        obs=obs,
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


def collect_self_play_parallel(
    model: PolicyNetwork,
    checkpoint_info: Dict[str, Any],
    iteration: int,
) -> Tuple[List[SelfPlaySample], Dict[str, Any]]:
    seeds = [
        BASE_SEED + iteration * 1_000_000 + game_idx
        for game_idx in range(SELF_PLAY_GAMES_PER_ITER)
    ]

    model_state = state_dict_to_cpu(model)
    model_cfg = checkpoint_info["model_config"]
    encoder_config = checkpoint_info["encoder_config"]
    action_mapper_config = checkpoint_info["action_mapper_config"]

    all_samples: List[SelfPlaySample] = []
    total_moves = 0
    total_sims = 0

    with ProcessPoolExecutor(
        max_workers=WORKERS,
        initializer=init_self_play_worker,
        initargs=(model_state, model_cfg, encoder_config, action_mapper_config),
    ) as executor:
        futures = [executor.submit(run_self_play_game_worker, seed) for seed in seeds]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Self-play games",
        ):
            samples, stats = future.result()

            all_samples.extend(samples)
            total_moves += int(stats["moves"])
            total_sims += int(stats["total_sims"])

    summary = {
        "games": SELF_PLAY_GAMES_PER_ITER,
        "samples": len(all_samples),
        "total_moves": total_moves,
        "total_sims": total_sims,
        "avg_sims_per_move": total_sims / total_moves if total_moves > 0 else 0.0,
    }

    return all_samples, summary


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

            static_x = torch.tensor(
                [sample.static_features for sample in batch],
                dtype=torch.float32,
                device=TRAIN_DEVICE,
            )

            bid_history = torch.tensor(
                [sample.bid_history for sample in batch],
                dtype=torch.float32,
                device=TRAIN_DEVICE,
            )

            bid_mask = torch.tensor(
                [sample.bid_mask for sample in batch],
                dtype=torch.bool,
                device=TRAIN_DEVICE,
            )

            target_policy = torch.tensor(
                [sample.target_policy for sample in batch],
                dtype=torch.float32,
                device=TRAIN_DEVICE,
            )

            logits = model(static_x, bid_history, bid_mask)
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
def run_eval_game_worker(seed: int) -> Tuple[int, Dict[str, int], Dict[str, Any]]:
    if (
        _worker_model is None
        or _worker_best_model is None
        or _worker_encoder is None
        or _worker_action_mapper is None
    ):
        raise RuntimeError("Eval worker was not initialized")

    rng = random.Random(seed)

    seat_groups = ["new", "new", "best", "best"]
    rng.shuffle(seat_groups)

    agents = []
    for group in seat_groups:
        model = _worker_model if group == "new" else _worker_best_model

        agents.append(
            TransformerNeuralISMCTSAgent(
                model=model,
                encoder=_worker_encoder,
                action_mapper=_worker_action_mapper,
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
    winner = None

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

    if winner is None:
        raise RuntimeError("Evaluation game ended without winner")

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


def evaluate_new_vs_best_parallel(
    new_model: PolicyNetwork,
    best_model: PolicyNetwork,
    checkpoint_info: Dict[str, Any],
    iteration: int,
) -> Dict[str, float]:
    seeds = [
        BASE_SEED + iteration * 10_000_000 + game_idx
        for game_idx in range(EVAL_GAMES)
    ]

    new_state = state_dict_to_cpu(new_model)
    best_state = state_dict_to_cpu(best_model)

    model_cfg = checkpoint_info["model_config"]
    encoder_config = checkpoint_info["encoder_config"]
    action_mapper_config = checkpoint_info["action_mapper_config"]

    new_wins = 0
    placement_totals = defaultdict(int)

    total_moves = 0
    total_sims = 0

    with ProcessPoolExecutor(
        max_workers=WORKERS,
        initializer=init_eval_worker,
        initargs=(new_state, best_state, model_cfg, encoder_config, action_mapper_config),
    ) as executor:
        futures = [executor.submit(run_eval_game_worker, seed) for seed in seeds]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Evaluation games",
        ):
            winner_group, placement_counts, stats = future.result()

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


def save_transformer_checkpoint(
    model: PolicyNetwork,
    encoder: ObservationEncoder,
    action_mapper: ActionMapper,
    path: str | Path,
    model_cfg: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    save_model_checkpoint(
        model=model,
        encoder=encoder,
        action_mapper=action_mapper,
        path=path,
        hidden_dim=int(model_cfg["hidden_dim"]),
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        num_layers=int(model_cfg["num_layers"]),
        dim_feedforward=int(model_cfg["dim_feedforward"]),
        dropout=float(model_cfg["dropout"]),
        extra_metadata=metadata,
    )


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

    best_model = copy.deepcopy(model)
    best_model.to(TRAIN_DEVICE)
    best_model.eval()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    replay_buffer: deque[SelfPlaySample] = deque(maxlen=BUFFER_SIZE)
    history: List[Dict[str, Any]] = []

    save_transformer_checkpoint(
        model=best_model,
        encoder=encoder,
        action_mapper=action_mapper,
        path=output_dir / "initial.pt",
        model_cfg=model_cfg,
        metadata={
            **script_config(),
            "source_checkpoint": INITIAL_CHECKPOINT,
            "description": "Initial supervised Transformer checkpoint before self-play.",
        },
    )

    accepted_iterations: List[int] = []
    overall_start = time.perf_counter()

    for iteration in range(ITERATIONS):
        iteration_start = time.perf_counter()

        print(f"\n=== ITERATION {iteration} ===")

        collect_start = time.perf_counter()

        collected_samples, self_play_stats = collect_self_play_parallel(
            model=model,
            checkpoint_info=checkpoint_info,
            iteration=iteration,
        )

        replay_buffer.extend(collected_samples)

        collect_time_s = time.perf_counter() - collect_start

        print(f"Collected samples: {len(collected_samples)}")
        print(f"Replay buffer size: {len(replay_buffer)}")

        train_start = time.perf_counter()

        train_losses = train_one_iteration(
            model=model,
            replay_buffer=replay_buffer,
            optimizer=optimizer,
        )

        train_time_s = time.perf_counter() - train_start

        eval_start = time.perf_counter()

        eval_result = evaluate_new_vs_best_parallel(
            new_model=model,
            best_model=best_model,
            checkpoint_info=checkpoint_info,
            iteration=iteration,
        )

        eval_time_s = time.perf_counter() - eval_start

        new_win_rate = float(eval_result["new_win_rate"])
        accepted = new_win_rate >= ACCEPT_WIN_RATE

        print(f"New model win rate vs best: {new_win_rate:.3f}")
        print("Accepted" if accepted else "Rejected")

        candidate_path = candidates_dir / f"iteration_{iteration:03d}.pt"

        save_transformer_checkpoint(
            model=model,
            encoder=encoder,
            action_mapper=action_mapper,
            path=candidate_path,
            model_cfg=model_cfg,
            metadata=checkpoint_metadata(
                iteration=iteration,
                accepted=accepted,
                train_losses=train_losses,
                eval_result=eval_result,
                extra={
                    "checkpoint_type": "candidate",
                    "replay_buffer_size": len(replay_buffer),
                    "collected_samples": len(collected_samples),
                    "self_play_stats": self_play_stats,
                },
            ),
        )

        if accepted:
            best_model = copy.deepcopy(model)
            best_model.to(TRAIN_DEVICE)
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
                    "self_play_stats": self_play_stats,
                },
            )

            save_transformer_checkpoint(
                model=best_model,
                encoder=encoder,
                action_mapper=action_mapper,
                path=best_path,
                model_cfg=model_cfg,
                metadata=metadata,
            )

            save_transformer_checkpoint(
                model=best_model,
                encoder=encoder,
                action_mapper=action_mapper,
                path=accepted_path,
                model_cfg=model_cfg,
                metadata=metadata,
            )

        else:
            model.load_state_dict(best_model.state_dict())
            model.to(TRAIN_DEVICE)
            model.eval()

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            )

        save_transformer_checkpoint(
            model=model,
            encoder=encoder,
            action_mapper=action_mapper,
            path=output_dir / "last.pt",
            model_cfg=model_cfg,
            metadata=checkpoint_metadata(
                iteration=iteration,
                accepted=accepted,
                train_losses=train_losses,
                eval_result=eval_result,
                extra={
                    "checkpoint_type": "last_current_model",
                    "replay_buffer_size": len(replay_buffer),
                    "collected_samples": len(collected_samples),
                    "self_play_stats": self_play_stats,
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
            "self_play_stats": self_play_stats,
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