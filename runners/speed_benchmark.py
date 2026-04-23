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

TIME_LIMIT_S = 0.200  # throughput benchmark
SIMS_PER_MOVE = 500  # cost-per-simulation benchmark

AGENTS: Dict[str, Dict[str, Any]] = {
    "Neural_Transformer": {
        "class_path": "neural.trans_mlp.neural_ismcts:NeuralISMCTSPUCTAgent",
        "kind": "neural",
        "checkpoint_path": "artifacts/training_trans/best.pt",
    },
    "PUCT": {
        "class_path": "liars_dice.agents.ismcts_2_puct:ISMCTSPUCTAgent",
        "kind": "standard",
    },
    "History": {
        "class_path": "liars_dice.agents.ismcts_3_history:ISMCTSHistoryAgent",
        "kind": "standard",
    },
}


# =====================
# WORKER-LOCAL CACHE
# =====================
_WORKER_CACHE: Dict[str, Any] = {}


# =====================
# HELPERS
# =====================
def _load_class(path: str):
    module_name, class_name = path.split(":")
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


def _looks_like_sequence_checkpoint(payload: Dict[str, Any]) -> bool:
    encoder_cfg = payload.get("encoder_config", {})
    model_cfg = payload.get("model_config", {})
    return ("max_bids" in encoder_cfg) or (model_cfg.get("static_dim") is not None)


def _load_neural_bundle(checkpoint_path: str) -> Tuple[Any, Any, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")

    from neural.action_mapping import ActionMapper

    mapper = ActionMapper(**payload["action_mapper_config"])

    if _looks_like_sequence_checkpoint(payload):
        from neural.trans_mlp.encoder import ObservationEncoder
        from neural.trans_mlp.nn_model import PolicyNetwork

        encoder = ObservationEncoder(**payload["encoder_config"])
        model_cfg = payload["model_config"]

        model = PolicyNetwork(
            static_dim=model_cfg["static_dim"],
            token_dim=model_cfg["token_dim"],
            num_actions=model_cfg["num_actions"],
            max_bids=model_cfg["max_bids"],
            d_model=model_cfg["d_model"],
        )
    else:
        from neural.basic_mlp.encoder import ObservationEncoder
        from neural.basic_mlp.nn_model import PolicyNetwork

        encoder = ObservationEncoder(**payload["encoder_config"])

        model = PolicyNetwork(
            input_dim=encoder.input_dim,
            num_actions=mapper.num_actions,
            hidden_dim=256,
            dropout=0.1,
        )

    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    return model, encoder, mapper


def _init_worker(agent_spec: Dict[str, Any]) -> None:
    """
    Called once per worker process.
    For neural agents, preload checkpoint/model/encoder/mapper.
    For standard agents, no extra state is required.
    """
    global _WORKER_CACHE
    _WORKER_CACHE = {"agent_spec": agent_spec}

    if agent_spec["kind"] == "neural":
        checkpoint_path = agent_spec["checkpoint_path"]
        model, encoder, mapper = _load_neural_bundle(checkpoint_path)
        _WORKER_CACHE["neural_bundle"] = (model, encoder, mapper)


def _make_agent(
    agent_spec: Dict[str, Any],
    seed: int,
    sims_per_move: int | None,
    time_limit_s: float | None,
):
    cls = _load_class(agent_spec["class_path"])

    if agent_spec["kind"] == "neural":
        model, encoder, mapper = _WORKER_CACHE["neural_bundle"]
        return cls(
            model=model,
            encoder=encoder,
            action_mapper=mapper,
            seed=seed,
            sims_per_move=sims_per_move,
            time_limit_s=time_limit_s,
        )

    # Standard agents: pass search-budget args if supported.
    # This assumes your old ISMCTS agents / heuristic agent accept these.
    return cls(
        seed=seed,
        sims_per_move=sims_per_move,
        time_limit_s=time_limit_s,
    )


# =====================
# SINGLE GAME RUNNER
# =====================
def _run_game(
    agent_name: str,
    seed: int,
    sims_per_move: int | None,
    time_limit_s: float | None,
):
    agent_spec = _WORKER_CACHE["agent_spec"]
    rng = random.Random(seed)

    agents = []
    for _ in range(NUM_PLAYERS):
        agent_seed = rng.randint(0, 10**9)
        agent = _make_agent(
            agent_spec=agent_spec,
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
    agent_name, seed = args
    return _run_game(
        agent_name=agent_name,
        seed=seed,
        sims_per_move=None,
        time_limit_s=TIME_LIMIT_S,
    )


def run_game_sim_limited(args):
    agent_name, seed = args
    return _run_game(
        agent_name=agent_name,
        seed=seed,
        sims_per_move=SIMS_PER_MOVE,
        time_limit_s=None,
    )


# =====================
# BENCHMARKS
# =====================
def benchmark_time_limited(
    agent_name: str, agent_spec: Dict[str, Any]
) -> Dict[str, Any]:
    args = [(agent_name, BASE_SEED + i) for i in range(NUM_GAMES)]

    with Pool(
        initializer=_init_worker,
        initargs=(agent_spec,),
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


def benchmark_sim_limited(
    agent_name: str, agent_spec: Dict[str, Any]
) -> Dict[str, Any]:
    args = [(agent_name, BASE_SEED + 100_000 + i) for i in range(NUM_GAMES)]

    with Pool(
        initializer=_init_worker,
        initargs=(agent_spec,),
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
    for agent_name, agent_spec in AGENTS.items():
        result = benchmark_time_limited(agent_name, agent_spec)
        time_results.append(result)
        print(
            f"{result['agent']:18s} | "
            f"{result['simulations_per_second']:.2f} sims/sec | "
            f"{result['avg_ms_per_move']:.2f} ms/move | "
            f"{result['avg_sims_per_move']:.2f} sims/move"
        )

    print("\n=== SIM-LIMITED: time per simulation ===")
    for agent_name, agent_spec in AGENTS.items():
        result = benchmark_sim_limited(agent_name, agent_spec)
        sim_results.append(result)
        print(
            f"{result['agent']:18s} | "
            f"{result['time_per_sim_ms']:.4f} ms/sim | "
            f"{result['avg_ms_per_move']:.2f} ms/move | "
            f"{result['avg_sims_per_move']:.2f} sims/move"
        )
