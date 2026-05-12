from __future__ import annotations

import csv
import importlib
import inspect
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from liars_dice.core.game import LiarsDiceGame

# =====================
# CONFIG
# =====================
NUM_GAMES = 100

NUM_PLAYERS = 4
DICE_PER_PLAYER = 5
BASE_SEED = 12345

TIME_LIMIT_S = 0.200
SIMS_PER_MOVE = 500

OUT_DIR = Path("artifacts/benchmarks")
OUT_TIME_JSON = OUT_DIR / "speed_time_limited.json"
OUT_SIM_JSON = OUT_DIR / "speed_sim_limited.json"
OUT_TIME_CSV = OUT_DIR / "speed_time_limited.csv"
OUT_SIM_CSV = OUT_DIR / "speed_sim_limited.csv"


AGENTS: Dict[str, Dict[str, Any]] = {
    "ISMCTS_Basic": {
        "kind": "standard",
        "class_path": "liars_dice.agents.ISMCTS_Basic:ISMCTSHeuristicAgent",
    },
    "PUCT": {
        "kind": "standard",
        "class_path": "liars_dice.agents.ISMCTS_PUCT:ISMCTSPUCTAgent",
    },
    "History": {
        "kind": "standard",
        "class_path": "liars_dice.agents.ISMCTS_History:ISMCTSHistoryAgent",
    },
    "Neural_MLP": {
        "kind": "neural",
        "class_path": "neural.basic_mlp.neural_ismcts_mlp:MLPNeuralISMCTSAgent",
        "checkpoint_path": "artifacts/training/mlp/best.pt",
    },
    "Neural_Transformer": {
        "kind": "neural",
        "class_path": "neural.trans_mlp.neural_ismcts_trans:TransformerNeuralISMCTSAgent",
        "checkpoint_path": "artifacts/training/trans/best.pt",
    },
    "Neural_MLP_SELF": {
        "kind": "neural",
        "class_path": "neural.basic_mlp.neural_ismcts_mlp:MLPNeuralISMCTSAgent",
        "checkpoint_path": "artifacts/self_play/mlp/best.pt",
    },
    "Neural_Transformer_SELF": {
        "kind": "neural",
        "class_path": "neural.trans_mlp.neural_ismcts_trans:TransformerNeuralISMCTSAgent",
        "checkpoint_path": "artifacts/self_play/trans/best.pt",
    },
}


# =====================
# WORKER-LOCAL CACHE
# =====================
_WORKER_NEURAL_BUNDLE: Tuple[Any, Any, Any] | None = None


# =====================
# LOADING HELPERS
# =====================
def _load_class(path: str):
    module_name, class_name = path.split(":")
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


def _construct_with_supported_kwargs(cls, kwargs: Dict[str, Any]):
    """
    Constructs an agent while only passing constructor arguments it supports.
    This lets the benchmark handle agents whose constructors do not all accept
    exactly the same keyword arguments.
    """
    sig = inspect.signature(cls.__init__)
    supported = {name for name in sig.parameters if name != "self"}

    filtered = {key: value for key, value in kwargs.items() if key in supported}

    return cls(**filtered)


def _is_transformer_checkpoint(payload: Dict[str, Any]) -> bool:
    model_cfg = payload.get("model_config", {})
    model_type = model_cfg.get("model_type")

    if model_type == "transformer":
        return True

    return (
        "static_dim" in model_cfg
        and "token_dim" in model_cfg
        and "max_bids" in model_cfg
    )


def _load_neural_bundle(checkpoint_path: str | Path) -> Tuple[Any, Any, Any]:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")

    from neural.common.action_mapping import ActionMapper

    action_mapper = ActionMapper(**payload["action_mapper_config"])
    model_cfg = payload["model_config"]

    if _is_transformer_checkpoint(payload):
        from neural.trans_mlp.encoder_trans import ObservationEncoder
        from neural.trans_mlp.model_trans import PolicyNetwork

        encoder = ObservationEncoder(**payload["encoder_config"])

        model = PolicyNetwork(
            static_dim=int(model_cfg["static_dim"]),
            token_dim=int(model_cfg["token_dim"]),
            num_actions=int(model_cfg["num_actions"]),
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            max_bids=int(model_cfg["max_bids"]),
            d_model=int(model_cfg.get("d_model", 64)),
            nhead=int(model_cfg.get("nhead", 4)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", 128)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )

    else:
        from neural.basic_mlp.encoder_mlp import ObservationEncoder
        from neural.basic_mlp.model_mlp import PolicyNetwork

        encoder = ObservationEncoder(**payload["encoder_config"])

        model = PolicyNetwork(
            input_dim=int(model_cfg.get("input_dim", encoder.input_dim)),
            num_actions=int(model_cfg.get("num_actions", action_mapper.num_actions)),
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )

    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    return model, encoder, action_mapper


def init_worker_neural(checkpoint_path: str) -> None:
    global _WORKER_NEURAL_BUNDLE
    torch.set_num_threads(1)
    _WORKER_NEURAL_BUNDLE = _load_neural_bundle(checkpoint_path)


def make_agent(
    agent_name: str,
    agent_spec: Dict[str, Any],
    seed: int,
    sims_per_move: int | None,
    time_limit_s: float | None,
):
    cls = _load_class(agent_spec["class_path"])

    if agent_spec["kind"] == "neural":
        if _WORKER_NEURAL_BUNDLE is None:
            raise RuntimeError(
                f"Worker neural bundle was not initialized for {agent_name}"
            )

        model, encoder, action_mapper = _WORKER_NEURAL_BUNDLE

        return cls(
            model=model,
            encoder=encoder,
            action_mapper=action_mapper,
            seed=seed,
            sims_per_move=sims_per_move,
            time_limit_s=time_limit_s,
        )

    return _construct_with_supported_kwargs(
        cls,
        {
            "seed": seed,
            "sims_per_move": sims_per_move,
            "time_limit_s": time_limit_s,
        },
    )


# =====================
# GAME RUNNER
# =====================
def _run_game(
    agent_name: str,
    agent_spec: Dict[str, Any],
    seed: int,
    sims_per_move: int | None,
    time_limit_s: float | None,
) -> Tuple[float, int, int]:
    rng = random.Random(seed)

    agents = []
    for _ in range(NUM_PLAYERS):
        agent_seed = rng.randint(0, 10**9)

        agent = make_agent(
            agent_name=agent_name,
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
                f"{agent_name} / {type(agents[pid]).__name__} did not expose "
                f"_last_sim_count. This benchmark expects search agents only."
            )

        total_time += move_time
        total_sims += int(move_sims)
        total_moves += 1

        info = game.step(action)
        if info.get("terminal"):
            break

    return total_time, total_sims, total_moves


def run_game_time_limited(args):
    agent_name, agent_spec, seed = args

    return _run_game(
        agent_name=agent_name,
        agent_spec=agent_spec,
        seed=seed,
        sims_per_move=None,
        time_limit_s=TIME_LIMIT_S,
    )


def run_game_sim_limited(args):
    agent_name, agent_spec, seed = args

    return _run_game(
        agent_name=agent_name,
        agent_spec=agent_spec,
        seed=seed,
        sims_per_move=SIMS_PER_MOVE,
        time_limit_s=None,
    )


# =====================
# BENCHMARKS
# =====================
def _pool_for_agent(agent_spec: Dict[str, Any]):
    if agent_spec["kind"] == "neural":
        checkpoint_path = agent_spec["checkpoint_path"]

        return Pool(
            initializer=init_worker_neural,
            initargs=(checkpoint_path,),
        )

    return Pool()


def benchmark_time_limited(
    agent_name: str, agent_spec: Dict[str, Any]
) -> Dict[str, Any]:
    args = [(agent_name, agent_spec, BASE_SEED + i) for i in range(NUM_GAMES)]

    with _pool_for_agent(agent_spec) as pool:
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
    avg_ms_per_move = 1000.0 * total_time / total_moves if total_moves > 0 else 0.0
    avg_sims_per_move = total_sims / total_moves if total_moves > 0 else 0.0

    return {
        "agent": agent_name,
        "mode": "time_limited",
        "games": NUM_GAMES,
        "time_limit_s": TIME_LIMIT_S,
        "total_decision_time_s": total_time,
        "total_simulations": total_sims,
        "total_moves": total_moves,
        "simulations_per_second": sims_per_sec,
        "avg_ms_per_move": avg_ms_per_move,
        "avg_sims_per_move": avg_sims_per_move,
    }


def benchmark_sim_limited(
    agent_name: str, agent_spec: Dict[str, Any]
) -> Dict[str, Any]:
    args = [(agent_name, agent_spec, BASE_SEED + 100_000 + i) for i in range(NUM_GAMES)]

    with _pool_for_agent(agent_spec) as pool:
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
    time_per_sim_ms = time_per_sim_s * 1000.0
    avg_ms_per_move = 1000.0 * total_time / total_moves if total_moves > 0 else 0.0
    avg_sims_per_move = total_sims / total_moves if total_moves > 0 else 0.0
    sims_per_sec = total_sims / total_time if total_time > 0 else 0.0

    return {
        "agent": agent_name,
        "mode": "sim_limited",
        "games": NUM_GAMES,
        "sims_per_move_target": SIMS_PER_MOVE,
        "total_decision_time_s": total_time,
        "total_simulations": total_sims,
        "total_moves": total_moves,
        "time_per_sim_s": time_per_sim_s,
        "time_per_sim_ms": time_per_sim_ms,
        "simulations_per_second": sims_per_sec,
        "avg_ms_per_move": avg_ms_per_move,
        "avg_sims_per_move": avg_sims_per_move,
    }


# =====================
# OUTPUT HELPERS
# =====================
def save_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    all_keys = sorted({key for row in rows for key in row.keys()})

    preferred_order = [
        "agent",
        "mode",
        "games",
        "time_limit_s",
        "sims_per_move_target",
        "avg_ms_per_move",
        "avg_sims_per_move",
        "time_per_sim_ms",
        "simulations_per_second",
        "total_decision_time_s",
        "total_simulations",
        "total_moves",
    ]

    fieldnames = [key for key in preferred_order if key in all_keys]
    fieldnames.extend(key for key in all_keys if key not in fieldnames)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_time_result(result: Dict[str, Any]) -> None:
    print(
        f"{result['agent']:24s} | "
        f"{result['simulations_per_second']:9.2f} sims/sec | "
        f"{result['avg_ms_per_move']:8.2f} ms/move | "
        f"{result['avg_sims_per_move']:8.2f} sims/move"
    )


def print_sim_result(result: Dict[str, Any]) -> None:
    print(
        f"{result['agent']:24s} | "
        f"{result['time_per_sim_ms']:8.4f} ms/sim | "
        f"{result['avg_ms_per_move']:8.2f} ms/move | "
        f"{result['avg_sims_per_move']:8.2f} sims/move | "
        f"{result['simulations_per_second']:9.2f} sims/sec"
    )


def validate_agent_specs() -> None:
    for agent_name, spec in AGENTS.items():
        if spec["kind"] not in {"standard", "neural"}:
            raise ValueError(f"{agent_name}: invalid kind {spec['kind']}")

        _load_class(spec["class_path"])

        if spec["kind"] == "neural":
            checkpoint_path = Path(spec["checkpoint_path"])
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"{agent_name}: checkpoint not found: {checkpoint_path}"
                )


# =====================
# MAIN
# =====================
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    validate_agent_specs()

    time_results: List[Dict[str, Any]] = []
    sim_results: List[Dict[str, Any]] = []

    print("\n=== TIME-LIMITED: simulations per second ===")
    print(f"Games per agent: {NUM_GAMES}")
    print(f"Time limit per move: {TIME_LIMIT_S:.3f}s\n")

    for agent_name, agent_spec in AGENTS.items():
        result = benchmark_time_limited(agent_name, agent_spec)
        time_results.append(result)
        print_time_result(result)

    print("\n=== SIM-LIMITED: time per simulation ===")
    print(f"Games per agent: {NUM_GAMES}")
    print(f"Simulations per move: {SIMS_PER_MOVE}\n")

    for agent_name, agent_spec in AGENTS.items():
        result = benchmark_sim_limited(agent_name, agent_spec)
        sim_results.append(result)
        print_sim_result(result)

    save_json(OUT_TIME_JSON, time_results)
    save_json(OUT_SIM_JSON, sim_results)
    save_csv(OUT_TIME_CSV, time_results)
    save_csv(OUT_SIM_CSV, sim_results)

    print("\nSaved benchmark results:")
    print(f"  {OUT_TIME_JSON}")
    print(f"  {OUT_SIM_JSON}")
    print(f"  {OUT_TIME_CSV}")
    print(f"  {OUT_SIM_CSV}")


if __name__ == "__main__":
    main()
