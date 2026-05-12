from __future__ import annotations

import csv
import importlib
import inspect
import itertools
import json
import os
import random
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from liars_dice.core.game import LiarsDiceGame

# =====================
# CONFIG
# =====================
NUM_GAMES_PER_CONFIG = 10
NUM_PLAYERS = 4
DICE_PER_PLAYER = 5
BASE_SEED = 12345

SIMS_PER_MOVE = None
TIME_LIMIT_S = 0.2

WORKERS = max(1, (os.cpu_count() or 2) - 1)

OUT_DIR = Path("evaluation/results/combinations")
OUTPUT_GAMES_CSV = OUT_DIR / "mixed_games.csv"
OUTPUT_SEATS_CSV = OUT_DIR / "mixed_seats.csv"
OUTPUT_CONFIG_SUMMARY_CSV = OUT_DIR / "mixed_config_summary.csv"
OUTPUT_OVERALL_SUMMARY_CSV = OUT_DIR / "mixed_overall_summary.csv"
OUTPUT_METADATA_JSON = OUT_DIR / "mixed_metadata.json"


# Specify the exact agents to include in the mixed-combination tournament.
# The script will evaluate all 4-agent combinations from this list.
EVALUATED_AGENTS: List[str] = [
    "PUCT",
    "History",
    "Neural_MLP",
    "Neural_Transformer",
    "Neural_MLP_SELF",
    "Neural_Transformer_SELF",
]


# Adjust class_path values if your local module names differ.
AGENTS: Dict[str, Dict[str, Any]] = {
    "Heuristic": {
        "kind": "standard",
        "class_path": "liars_dice.agents.Heuristic:HeuristicAgent",
    },
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
_WORKER_CACHE: Dict[str, Any] = {}


# =====================
# HELPERS
# =====================
def load_class(path: str):
    module_name, class_name = path.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def construct_with_supported_kwargs(cls, kwargs: Dict[str, Any]):
    sig = inspect.signature(cls.__init__)
    supported = {name for name in sig.parameters if name != "self"}

    filtered = {key: value for key, value in kwargs.items() if key in supported}

    return cls(**filtered)


def is_transformer_checkpoint(payload: Dict[str, Any]) -> bool:
    model_cfg = payload.get("model_config", {})
    model_type = model_cfg.get("model_type")

    if model_type == "transformer":
        return True

    return (
        "static_dim" in model_cfg
        and "token_dim" in model_cfg
        and "max_bids" in model_cfg
    )


def load_neural_bundle(checkpoint_path: str | Path) -> Tuple[Any, Any, Any]:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")

    from neural.common.action_mapping import ActionMapper

    action_mapper = ActionMapper(**payload["action_mapper_config"])
    model_cfg = payload["model_config"]

    if is_transformer_checkpoint(payload):
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


def init_worker() -> None:
    global _WORKER_CACHE

    torch.set_num_threads(1)

    _WORKER_CACHE = {
        "classes": {},
        "bundles": {},
    }

    used_agents = set(EVALUATED_AGENTS)

    for agent_label in used_agents:
        spec = AGENTS[agent_label]
        _WORKER_CACHE["classes"][agent_label] = load_class(spec["class_path"])

        if spec["kind"] == "neural":
            _WORKER_CACHE["bundles"][agent_label] = load_neural_bundle(
                spec["checkpoint_path"]
            )


def make_agent(agent_label: str, seed: int):
    spec = AGENTS[agent_label]
    cls = _WORKER_CACHE["classes"][agent_label]

    if spec["kind"] == "neural":
        model, encoder, action_mapper = _WORKER_CACHE["bundles"][agent_label]

        return cls(
            model=model,
            encoder=encoder,
            action_mapper=action_mapper,
            seed=seed,
            sims_per_move=SIMS_PER_MOVE,
            time_limit_s=TIME_LIMIT_S,
        )

    return construct_with_supported_kwargs(
        cls,
        {
            "seed": seed,
            "sims_per_move": SIMS_PER_MOVE,
            "time_limit_s": TIME_LIMIT_S,
        },
    )


def generate_combination_configs(agent_labels: List[str]) -> List[Tuple[str, ...]]:
    if len(agent_labels) < NUM_PLAYERS:
        raise ValueError(
            f"Need at least {NUM_PLAYERS} agents to generate 4-player combinations"
        )

    return list(itertools.combinations(agent_labels, NUM_PLAYERS))


def config_name(config: Tuple[str, ...]) -> str:
    return "__".join(config)


def compute_placements_from_elimination(eliminated: List[int]) -> Dict[int, int]:
    if len(eliminated) != NUM_PLAYERS:
        raise RuntimeError(
            f"Expected {NUM_PLAYERS} eliminated/winner entries, got {len(eliminated)}: "
            f"{eliminated}"
        )

    placements_best_to_worst = eliminated[::-1]

    return {seat: place for place, seat in enumerate(placements_best_to_worst, start=1)}


# =====================
# SINGLE GAME
# =====================
def run_single_game(
    args: Tuple[int, int, Tuple[str, ...]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    global_game_id, local_game_id, config = args

    rng = random.Random(BASE_SEED + global_game_id)

    seat_labels = list(config)
    rng.shuffle(seat_labels)

    agents = []
    time_spent = [0.0 for _ in range(NUM_PLAYERS)]
    sim_counts = [0 for _ in range(NUM_PLAYERS)]
    move_counts = [0 for _ in range(NUM_PLAYERS)]

    for seat in range(NUM_PLAYERS):
        agent_seed = rng.randint(0, 10**9)
        agent = make_agent(seat_labels[seat], seed=agent_seed)
        agents.append(agent)

    game = LiarsDiceGame(
        num_players=NUM_PLAYERS,
        dice_per_player=DICE_PER_PLAYER,
        seed=rng.randint(0, 10**9),
    )

    eliminated: List[int] = []
    winner: int | None = None
    total_turns = 0

    while True:
        pid = game._current
        obs = game.observe(pid)

        t0 = time.perf_counter()
        action = agents[pid].select_action(game, obs)
        t1 = time.perf_counter()

        move_time = t1 - t0
        move_sims = getattr(agents[pid], "_last_sim_count", 0)

        if move_sims is None:
            move_sims = 0

        time_spent[pid] += move_time
        sim_counts[pid] += int(move_sims)
        move_counts[pid] += 1
        total_turns += 1

        info = game.step(action)

        for seat in range(NUM_PLAYERS):
            if game._dice_left[seat] == 0 and seat not in eliminated:
                eliminated.append(seat)

        if info.get("terminal"):
            winner = int(info["winner"])
            if winner not in eliminated:
                eliminated.append(winner)
            break

    if winner is None:
        raise RuntimeError("Game ended without winner")

    placements_by_seat = compute_placements_from_elimination(eliminated)

    placement_agents = {}
    for place in range(1, NUM_PLAYERS + 1):
        seat = next(
            seat for seat, placement in placements_by_seat.items() if placement == place
        )
        placement_agents[place] = seat_labels[seat]

    cfg_name = config_name(config)

    game_row = {
        "global_game_id": global_game_id,
        "local_game_id": local_game_id,
        "config": cfg_name,
        "winner_agent": seat_labels[winner],
        "winner_seat": winner,
        "turns": total_turns,
        "seat_0_agent": seat_labels[0],
        "seat_1_agent": seat_labels[1],
        "seat_2_agent": seat_labels[2],
        "seat_3_agent": seat_labels[3],
        "placement_1": placement_agents[1],
        "placement_2": placement_agents[2],
        "placement_3": placement_agents[3],
        "placement_4": placement_agents[4],
    }

    seat_rows: List[Dict[str, Any]] = []

    for seat, agent_label in enumerate(seat_labels):
        placement = placements_by_seat[seat]
        decision_time = time_spent[seat]
        moves = move_counts[seat]
        sims = sim_counts[seat]

        seat_rows.append(
            {
                "global_game_id": global_game_id,
                "local_game_id": local_game_id,
                "config": cfg_name,
                "seat": seat,
                "agent": agent_label,
                "placement": placement,
                "is_winner": int(placement == 1),
                "decision_time_sec": decision_time,
                "moves": moves,
                "sims": sims,
                "avg_move_time_sec": decision_time / moves if moves > 0 else 0.0,
                "avg_sims_per_move": sims / moves if moves > 0 else 0.0,
                "sims_per_sec": sims / decision_time if decision_time > 0 else 0.0,
            }
        )

    return game_row, seat_rows


# =====================
# SUMMARY
# =====================
def summarize_by_agent(df_seats: pd.DataFrame) -> pd.DataFrame:
    placement_counts = (
        df_seats.groupby("agent")["placement"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={1: "first", 2: "second", 3: "third", 4: "fourth"})
        .reset_index()
    )

    for col in ["first", "second", "third", "fourth"]:
        if col not in placement_counts.columns:
            placement_counts[col] = 0

    mean_place = (
        df_seats.groupby("agent")["placement"]
        .mean()
        .reset_index()
        .rename(columns={"placement": "mean_placement"})
    )

    games_played = df_seats.groupby("agent").size().reset_index(name="games")

    timing = (
        df_seats.groupby("agent")
        .agg(
            total_decision_time_sec=("decision_time_sec", "sum"),
            total_sims=("sims", "sum"),
            total_moves=("moves", "sum"),
        )
        .reset_index()
    )

    summary = placement_counts.merge(mean_place, on="agent", how="left")
    summary = summary.merge(games_played, on="agent", how="left")
    summary = summary.merge(timing, on="agent", how="left")

    summary["win_rate"] = summary["first"] / summary["games"]
    summary["avg_move_time_sec"] = (
        summary["total_decision_time_sec"] / summary["total_moves"]
    )
    summary["avg_sims_per_move"] = summary["total_sims"] / summary["total_moves"]
    summary["sims_per_sec"] = summary.apply(
        lambda r: (
            r["total_sims"] / r["total_decision_time_sec"]
            if r["total_decision_time_sec"] > 0
            else 0.0
        ),
        axis=1,
    )

    summary = summary[
        [
            "agent",
            "first",
            "second",
            "third",
            "fourth",
            "mean_placement",
            "games",
            "win_rate",
            "avg_move_time_sec",
            "avg_sims_per_move",
            "total_decision_time_sec",
            "total_sims",
            "total_moves",
            "sims_per_sec",
        ]
    ]

    return summary.sort_values(
        by=["mean_placement", "win_rate", "first"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def summarize_by_config_agent(df_seats: pd.DataFrame) -> pd.DataFrame:
    placement_counts = (
        df_seats.groupby(["config", "agent"])["placement"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={1: "first", 2: "second", 3: "third", 4: "fourth"})
        .reset_index()
    )

    for col in ["first", "second", "third", "fourth"]:
        if col not in placement_counts.columns:
            placement_counts[col] = 0

    mean_place = (
        df_seats.groupby(["config", "agent"])["placement"]
        .mean()
        .reset_index()
        .rename(columns={"placement": "mean_placement"})
    )

    games_played = (
        df_seats.groupby(["config", "agent"]).size().reset_index(name="games")
    )

    timing = (
        df_seats.groupby(["config", "agent"])
        .agg(
            total_decision_time_sec=("decision_time_sec", "sum"),
            total_sims=("sims", "sum"),
            total_moves=("moves", "sum"),
        )
        .reset_index()
    )

    summary = placement_counts.merge(mean_place, on=["config", "agent"], how="left")
    summary = summary.merge(games_played, on=["config", "agent"], how="left")
    summary = summary.merge(timing, on=["config", "agent"], how="left")

    summary["win_rate"] = summary["first"] / summary["games"]
    summary["avg_move_time_sec"] = (
        summary["total_decision_time_sec"] / summary["total_moves"]
    )
    summary["avg_sims_per_move"] = summary["total_sims"] / summary["total_moves"]
    summary["sims_per_sec"] = summary.apply(
        lambda r: (
            r["total_sims"] / r["total_decision_time_sec"]
            if r["total_decision_time_sec"] > 0
            else 0.0
        ),
        axis=1,
    )

    return summary.sort_values(
        by=["config", "mean_placement", "win_rate", "first"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


# =====================
# VALIDATION / METADATA
# =====================
def validate_config() -> None:
    for agent_label in EVALUATED_AGENTS:
        if agent_label not in AGENTS:
            raise KeyError(f"EVALUATED_AGENTS contains unknown agent: {agent_label}")

        spec = AGENTS[agent_label]

        if spec["kind"] not in {"standard", "neural"}:
            raise ValueError(f"{agent_label}: invalid kind {spec['kind']}")

        load_class(spec["class_path"])

        if spec["kind"] == "neural":
            checkpoint_path = Path(spec["checkpoint_path"])
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"{agent_label}: checkpoint not found: {checkpoint_path}"
                )


def save_metadata(configs: List[Tuple[str, ...]]) -> None:
    metadata = {
        "num_games_per_config": NUM_GAMES_PER_CONFIG,
        "num_players": NUM_PLAYERS,
        "dice_per_player": DICE_PER_PLAYER,
        "base_seed": BASE_SEED,
        "sims_per_move": SIMS_PER_MOVE,
        "time_limit_s": TIME_LIMIT_S,
        "workers": WORKERS,
        "evaluated_agents": EVALUATED_AGENTS,
        "num_configs": len(configs),
        "total_games": len(configs) * NUM_GAMES_PER_CONFIG,
        "configs": [list(cfg) for cfg in configs],
        "agent_specs": AGENTS,
    }

    with OUTPUT_METADATA_JSON.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# =====================
# MAIN
# =====================
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    validate_config()

    configs = generate_combination_configs(EVALUATED_AGENTS)
    save_metadata(configs)

    args: List[Tuple[int, int, Tuple[str, ...]]] = []

    global_game_id = 0
    for config in configs:
        for local_game_id in range(NUM_GAMES_PER_CONFIG):
            args.append((global_game_id, local_game_id, config))
            global_game_id += 1

    print("Mixed-agent combination evaluation")
    print(f"Agents: {EVALUATED_AGENTS}")
    print(f"Configurations: {len(configs)}")
    print(f"Games per configuration: {NUM_GAMES_PER_CONFIG}")
    print(f"Total games: {len(args)}")
    print(f"Workers: {WORKERS}")
    print(f"Time limit per move: {TIME_LIMIT_S}")
    print(f"Simulations per move: {SIMS_PER_MOVE}")
    print()

    print("Configurations:")
    for cfg in configs:
        print("  ", cfg)

    all_game_rows: List[Dict[str, Any]] = []
    all_seat_rows: List[Dict[str, Any]] = []

    with Pool(processes=WORKERS, initializer=init_worker) as pool:
        for game_row, seat_rows in tqdm(
            pool.imap_unordered(run_single_game, args),
            total=len(args),
            desc="Running mixed games",
        ):
            all_game_rows.append(game_row)
            all_seat_rows.extend(seat_rows)

    df_games = pd.DataFrame(all_game_rows)
    df_seats = pd.DataFrame(all_seat_rows)

    df_games = df_games.sort_values("global_game_id").reset_index(drop=True)
    df_seats = df_seats.sort_values(["global_game_id", "seat"]).reset_index(drop=True)

    df_games.to_csv(OUTPUT_GAMES_CSV, index=False)
    df_seats.to_csv(OUTPUT_SEATS_CSV, index=False)

    config_summary = summarize_by_config_agent(df_seats)
    overall_summary = summarize_by_agent(df_seats)

    config_summary.to_csv(OUTPUT_CONFIG_SUMMARY_CSV, index=False)
    overall_summary.to_csv(OUTPUT_OVERALL_SUMMARY_CSV, index=False)

    print("\nOverall summary:")
    print(overall_summary.to_string(index=False))

    print("\nSaved:")
    print(f"  Raw game results      -> {OUTPUT_GAMES_CSV}")
    print(f"  Per-seat results      -> {OUTPUT_SEATS_CSV}")
    print(f"  Per-config summary    -> {OUTPUT_CONFIG_SUMMARY_CSV}")
    print(f"  Overall summary       -> {OUTPUT_OVERALL_SUMMARY_CSV}")
    print(f"  Metadata              -> {OUTPUT_METADATA_JSON}")


if __name__ == "__main__":
    main()
