from __future__ import annotations

import importlib
import random
import time
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from liars_dice.core.game import LiarsDiceGame

# =====================
# CONFIG
# =====================
NUM_GAMES = 10
NUM_PLAYERS = 4
DICE_PER_PLAYER = 5
BASE_SEED = 12345

TIME_LIMIT_S = 0.200  # 200 ms per move for throughput benchmark
SIMS_PER_MOVE = 500  # 500 sims per move for cost-per-sim benchmark

CHECKPOINT_PATH = "artifacts/policy_training/best.pt"

AGENTS = {
    "PUCT": "liars_dice.agents.ismcts_2_puct:ISMCTSPUCTAgent",
    "History": "liars_dice.agents.ismcts_3_history:ISMCTSHistoryAgent",
    "Neural": "liars_dice.agents.neural.neural_ismcts:NeuralISMCTSPUCTAgent",
}


# =====================
# WORKER-LOCAL CACHE
# =====================
_worker_neural_bundle = None


# =====================
# LOADERS
# =====================
def _load_class(path: str):
    module_name, class_name = path.split(":")
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


def init_worker_neural(checkpoint_path: str):
    global _worker_neural_bundle

    from neural.action_mapping import ActionMapper
    from neural.basic_mlp.encoder import ObservationEncoder
    from neural.basic_mlp.nn_model import PolicyNetwork

    payload = torch.load(checkpoint_path, map_location="cpu")

    encoder = ObservationEncoder(**payload["encoder_config"])
    mapper = ActionMapper(**payload["action_mapper_config"])

    model_cfg = payload.get("model_config", {})
    if model_cfg and model_cfg.get("static_dim") is not None:
        # transformer-style checkpoint
        model = PolicyNetwork(
            static_dim=model_cfg["static_dim"],
            token_dim=model_cfg["token_dim"],
            num_actions=model_cfg["num_actions"],
            max_bids=model_cfg["max_bids"],
            d_model=model_cfg["d_model"],
        )
    else:
        # original flat MLP checkpoint
        model = PolicyNetwork(
            input_dim=encoder.input_dim,
            num_actions=mapper.num_actions,
            hidden_dim=256,
            dropout=0.1,
        )

    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    _worker_neural_bundle = (model, encoder, mapper)


def make_neural_agent(seed: int, sims_per_move: int | None, time_limit_s: float | None):
    from neural.basic_mlp.neural_ismcts import NeuralISMCTSPUCTAgent

    model, encoder, mapper = _worker_neural_bundle
    return NeuralISMCTSPUCTAgent(
        model=model,
        encoder=encoder,
        action_mapper=mapper,
        seed=seed,
        sims_per_move=sims_per_move,
        time_limit_s=time_limit_s,
    )


def load_baseline_agent(
    path: str, seed: int, sims_per_move: int | None, time_limit_s: float | None
):
    Cls = _load_class(path)
    return Cls(
        seed=seed,
        sims_per_move=sims_per_move,
        time_limit_s=time_limit_s,
    )


# =====================
# SINGLE GAME RUNNERS
# =====================
def _run_game(
    agent_name: str,
    agent_path: str,
    seed: int,
    sims_per_move: int | None,
    time_limit_s: float | None,
):
    rng = random.Random(seed)

    agents = []
    for _ in range(NUM_PLAYERS):
        agent_seed = rng.randint(0, 10**9)
        if agent_name == "Neural":
            agent = make_neural_agent(
                seed=agent_seed,
                sims_per_move=sims_per_move,
                time_limit_s=time_limit_s,
            )
        else:
            agent = load_baseline_agent(
                path=agent_path,
                seed=agent_seed,
                sims_per_move=sims_per_move,
                time_limit_s=time_limit_s,
            )
        agents.append(agent)

    game = LiarsDiceGame(
        num_players=NUM_PLAYERS,
        dice_per_player=DICE_PER_PLAYER,
        seed=rng.randint(0, 10**9),
    )

    total_time = 0.0
    total_sims = 0
    total_moves = 0

    while True:
        pid = game._current
        obs = game.observe(pid)

        t0 = time.perf_counter()
        action = agents[pid].select_action(game, obs)
        t1 = time.perf_counter()

        move_time = t1 - t0
        move_sims = getattr(agents[pid], "_last_sim_count", None)

        if move_sims is None:
            raise RuntimeError(
                f"{type(agents[pid]).__name__} did not expose _last_sim_count"
            )

        total_time += move_time
        total_sims += move_sims
        total_moves += 1

        info = game.step(action)
        if info.get("terminal"):
            break

    return total_time, total_sims, total_moves


def run_game_time_limited(args):
    agent_name, agent_path, seed = args
    return _run_game(
        agent_name=agent_name,
        agent_path=agent_path,
        seed=seed,
        sims_per_move=None,
        time_limit_s=TIME_LIMIT_S,
    )


def run_game_sim_limited(args):
    agent_name, agent_path, seed = args
    return _run_game(
        agent_name=agent_name,
        agent_path=agent_path,
        seed=seed,
        sims_per_move=SIMS_PER_MOVE,
        time_limit_s=None,
    )


# =====================
# BENCHMARKS
# =====================
def benchmark_time_limited(agent_name: str, agent_path: str) -> Dict[str, Any]:
    args = [(agent_name, agent_path, BASE_SEED + i) for i in range(NUM_GAMES)]

    with Pool(
        initializer=init_worker_neural if agent_name == "Neural" else None,
        initargs=(CHECKPOINT_PATH,) if agent_name == "Neural" else (),
    ) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(run_game_time_limited, args),
                total=len(args),
                desc=f"{agent_name} time-limited",
            )
        )

    total_time = sum(t for t, _, _ in results)
    total_sims = sum(s for _, s, _ in results)
    total_moves = sum(m for _, _, m in results)

    sims_per_sec = total_sims / total_time if total_time > 0 else 0.0
    ms_per_move = (1000.0 * total_time / total_moves) if total_moves > 0 else 0.0
    sims_per_move_avg = total_sims / total_moves if total_moves > 0 else 0.0

    return {
        "agent": agent_name,
        "mode": "time_limited",
        "games": NUM_GAMES,
        "time_limit_s": TIME_LIMIT_S,
        "total_decision_time_s": total_time,
        "total_simulations": total_sims,
        "total_moves": total_moves,
        "simulations_per_second": sims_per_sec,
        "avg_ms_per_move": ms_per_move,
        "avg_sims_per_move": sims_per_move_avg,
    }


def benchmark_sim_limited(agent_name: str, agent_path: str) -> Dict[str, Any]:
    args = [(agent_name, agent_path, BASE_SEED + 100_000 + i) for i in range(NUM_GAMES)]

    with Pool(
        initializer=init_worker_neural if agent_name == "Neural" else None,
        initargs=(CHECKPOINT_PATH,) if agent_name == "Neural" else (),
    ) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(run_game_sim_limited, args),
                total=len(args),
                desc=f"{agent_name} sim-limited",
            )
        )

    total_time = sum(t for t, _, _ in results)
    total_sims = sum(s for _, s, _ in results)
    total_moves = sum(m for _, _, m in results)

    time_per_sim_s = total_time / total_sims if total_sims > 0 else 0.0
    ms_per_sim = time_per_sim_s * 1000.0
    ms_per_move = (1000.0 * total_time / total_moves) if total_moves > 0 else 0.0
    sims_per_move_avg = total_sims / total_moves if total_moves > 0 else 0.0

    return {
        "agent": agent_name,
        "mode": "sim_limited",
        "games": NUM_GAMES,
        "sims_per_move_target": SIMS_PER_MOVE,
        "total_decision_time_s": total_time,
        "total_simulations": total_sims,
        "total_moves": total_moves,
        "time_per_sim_s": time_per_sim_s,
        "time_per_sim_ms": ms_per_sim,
        "avg_ms_per_move": ms_per_move,
        "avg_sims_per_move": sims_per_move_avg,
    }


# =====================
# MAIN
# =====================
if __name__ == "__main__":
    time_results: List[Dict[str, Any]] = []
    sim_results: List[Dict[str, Any]] = []

    print("\n=== TIME-LIMITED: simulations per second ===")
    for agent_name, agent_path in AGENTS.items():
        result = benchmark_time_limited(agent_name, agent_path)
        time_results.append(result)
        print(
            f"{result['agent']:8s} | "
            f"{result['simulations_per_second']:.2f} sims/sec | "
            f"{result['avg_ms_per_move']:.2f} ms/move | "
            f"{result['avg_sims_per_move']:.2f} sims/move"
        )

    print("\n=== SIM-LIMITED: time per simulation ===")
    for agent_name, agent_path in AGENTS.items():
        result = benchmark_sim_limited(agent_name, agent_path)
        sim_results.append(result)
        print(
            f"{result['agent']:8s} | "
            f"{result['time_per_sim_ms']:.4f} ms/sim | "
            f"{result['avg_ms_per_move']:.2f} ms/move | "
            f"{result['avg_sims_per_move']:.2f} sims/move"
        )
