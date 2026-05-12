from __future__ import annotations

import csv
import importlib
import inspect
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from liars_dice.core.game import LiarsDiceGame

# =====================
# CONFIG
# =====================
NUM_GAMES_PER_MATCHUP = 1000
WORKERS = max(1, (os.cpu_count() or 2) - 1)

NUM_PLAYERS = 4
DICE_PER_PLAYER = 5
BASE_SEED = 12345

# Main evaluation budget.
TIME_LIMIT_S = 0.200
SIMS_PER_MOVE = None

# Alternative diagnostic mode:
# TIME_LIMIT_S = None
# SIMS_PER_MOVE = 500

OUT_DIR = Path("artifacts/evaluation/head_to_head")
OUT_JSON = OUT_DIR / "head_to_head_results.json"
OUT_CSV = OUT_DIR / "head_to_head_results.csv"
OUT_GAMES_JSONL = OUT_DIR / "head_to_head_games.jsonl"


# =====================
# AGENT SPECS
# =====================
# Adjust class_path values if your module names differ.
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
# MATCHUPS
# =====================
# Format:
#   (agent_a, agent_b, comparison_label)
#
# In every game:
#   2 seats use agent_a
#   2 seats use agent_b
# seating order is shuffled.
MATCHUPS: List[Tuple[str, str, str]] = [
    ("ISMCTS_Basic", "Heuristic", "basic_vs_heuristic"),
    ("PUCT", "ISMCTS_Basic", "puct_vs_basic"),
    ("PUCT", "Heuristic", "puct_vs_heuristic"),
    ("History", "PUCT", "history_vs_puct"),
    ("Neural_MLP", "History", "mlp_vs_history"),
    ("Neural_Transformer", "Neural_MLP", "transformer_vs_mlp"),
    ("Neural_MLP_SELF", "Neural_MLP", "mlp_self_vs_mlp"),
    (
        "Neural_Transformer_SELF",
        "Neural_Transformer",
        "transformer_self_vs_transformer",
    ),
    ("Neural_MLP_SELF", "Neural_Transformer_SELF", "mlp_self_vs_transformer_self"),
]


# =====================
# WORKER CACHE
# =====================
_WORKER_NEURAL_BUNDLES: Dict[str, Tuple[Any, Any, Any]] = {}


# =====================
# GENERAL HELPERS
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


def init_worker(neural_specs: Dict[str, Dict[str, Any]]) -> None:
    """
    Load all neural checkpoints once per worker process.
    """
    global _WORKER_NEURAL_BUNDLES

    torch.set_num_threads(1)

    _WORKER_NEURAL_BUNDLES = {}

    for agent_name, spec in neural_specs.items():
        _WORKER_NEURAL_BUNDLES[agent_name] = load_neural_bundle(spec["checkpoint_path"])


def make_agent(
    agent_name: str,
    seed: int,
):
    spec = AGENTS[agent_name]
    cls = load_class(spec["class_path"])

    if spec["kind"] == "neural":
        if agent_name not in _WORKER_NEURAL_BUNDLES:
            raise RuntimeError(f"Neural bundle not loaded for {agent_name}")

        model, encoder, action_mapper = _WORKER_NEURAL_BUNDLES[agent_name]

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


# =====================
# GAME LOGIC
# =====================
def run_one_head_to_head_game(args: Tuple[str, str, int, int]) -> Dict[str, Any]:
    agent_a, agent_b, matchup_index, game_index = args

    seed = BASE_SEED + matchup_index * 10_000_000 + game_index
    rng = random.Random(seed)

    seat_agents = [agent_a, agent_a, agent_b, agent_b]
    rng.shuffle(seat_agents)

    agents = [
        make_agent(
            agent_name=agent_name,
            seed=rng.randint(0, 10**9),
        )
        for agent_name in seat_agents
    ]

    game = LiarsDiceGame(
        num_players=NUM_PLAYERS,
        dice_per_player=DICE_PER_PLAYER,
        seed=rng.randint(0, 10**9),
    )

    eliminated: List[int] = []
    total_decision_time_by_side = defaultdict(float)
    total_sims_by_side = defaultdict(int)
    total_moves_by_side = defaultdict(int)

    winner: int | None = None

    while True:
        pid = game._current
        side = "A" if seat_agents[pid] == agent_a else "B"

        obs = game.observe(pid)

        t0 = time.perf_counter()
        action = agents[pid].select_action(game, obs)
        t1 = time.perf_counter()

        move_time = t1 - t0
        move_sims = getattr(agents[pid], "_last_sim_count", 0)

        if move_sims is None:
            move_sims = 0

        total_decision_time_by_side[side] += move_time
        total_sims_by_side[side] += int(move_sims)
        total_moves_by_side[side] += 1

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
        raise RuntimeError("Game ended without winner")

    winner_side = "A" if seat_agents[winner] == agent_a else "B"

    placements_best_to_worst = eliminated[::-1]

    placement_by_side = {
        "A_first": 0,
        "A_second": 0,
        "A_third": 0,
        "A_fourth": 0,
        "B_first": 0,
        "B_second": 0,
        "B_third": 0,
        "B_fourth": 0,
    }

    place_names = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
    }

    for place, seat in enumerate(placements_best_to_worst, start=1):
        side = "A" if seat_agents[seat] == agent_a else "B"
        placement_by_side[f"{side}_{place_names[place]}"] += 1

    return {
        "matchup_index": matchup_index,
        "game_index": game_index,
        "agent_a": agent_a,
        "agent_b": agent_b,
        "seat_agents": seat_agents,
        "winner": winner,
        "winner_side": winner_side,
        "placements_best_to_worst": placements_best_to_worst,
        **placement_by_side,
        "A_total_decision_time_s": total_decision_time_by_side["A"],
        "B_total_decision_time_s": total_decision_time_by_side["B"],
        "A_total_sims": total_sims_by_side["A"],
        "B_total_sims": total_sims_by_side["B"],
        "A_total_moves": total_moves_by_side["A"],
        "B_total_moves": total_moves_by_side["B"],
    }


# =====================
# AGGREGATION
# =====================
def side_summary(
    side: str,
    agent_name: str,
    game_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    first = sum(int(r[f"{side}_first"]) for r in game_rows)
    second = sum(int(r[f"{side}_second"]) for r in game_rows)
    third = sum(int(r[f"{side}_third"]) for r in game_rows)
    fourth = sum(int(r[f"{side}_fourth"]) for r in game_rows)

    games = len(game_rows)
    seats = games * 2

    wins = sum(1 for r in game_rows if r["winner_side"] == side)

    total_decision_time_s = sum(
        float(r[f"{side}_total_decision_time_s"]) for r in game_rows
    )
    total_sims = sum(int(r[f"{side}_total_sims"]) for r in game_rows)
    total_moves = sum(int(r[f"{side}_total_moves"]) for r in game_rows)

    mean_placement = (first * 1 + second * 2 + third * 3 + fourth * 4) / seats

    win_rate = wins / games if games > 0 else 0.0
    avg_move_time_s = total_decision_time_s / total_moves if total_moves > 0 else 0.0
    sims_per_sec = (
        total_sims / total_decision_time_s if total_decision_time_s > 0 else 0.0
    )
    avg_sims_per_move = total_sims / total_moves if total_moves > 0 else 0.0

    return {
        "agent": agent_name,
        "wins": wins,
        "win_rate": win_rate,
        "first": first,
        "second": second,
        "third": third,
        "fourth": fourth,
        "mean_placement": mean_placement,
        "seats": seats,
        "games": games,
        "total_decision_time_s": total_decision_time_s,
        "total_sims": total_sims,
        "total_moves": total_moves,
        "avg_move_time_s": avg_move_time_s,
        "avg_sims_per_move": avg_sims_per_move,
        "sims_per_sec": sims_per_sec,
    }


def aggregate_matchup(
    agent_a: str,
    agent_b: str,
    label: str,
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    summary_a = side_summary("A", agent_a, rows)
    summary_b = side_summary("B", agent_b, rows)

    delta_mean_placement = summary_a["mean_placement"] - summary_b["mean_placement"]
    delta_win_rate = summary_a["win_rate"] - summary_b["win_rate"]

    return {
        "label": label,
        "agent_a": agent_a,
        "agent_b": agent_b,
        "games": len(rows),
        "time_limit_s": TIME_LIMIT_S,
        "sims_per_move": SIMS_PER_MOVE,
        "agent_a_summary": summary_a,
        "agent_b_summary": summary_b,
        "delta_mean_placement_a_minus_b": delta_mean_placement,
        "delta_win_rate_a_minus_b": delta_win_rate,
    }


def flatten_result_for_csv(result: Dict[str, Any]) -> Dict[str, Any]:
    a = result["agent_a_summary"]
    b = result["agent_b_summary"]

    return {
        "label": result["label"],
        "agent_a": result["agent_a"],
        "agent_b": result["agent_b"],
        "games": result["games"],
        "time_limit_s": result["time_limit_s"],
        "sims_per_move": result["sims_per_move"],
        "agent_a_win_rate": a["win_rate"],
        "agent_b_win_rate": b["win_rate"],
        "agent_a_wins": a["wins"],
        "agent_b_wins": b["wins"],
        "agent_a_mean_placement": a["mean_placement"],
        "agent_b_mean_placement": b["mean_placement"],
        "delta_mean_placement_a_minus_b": result["delta_mean_placement_a_minus_b"],
        "delta_win_rate_a_minus_b": result["delta_win_rate_a_minus_b"],
        "agent_a_first": a["first"],
        "agent_a_second": a["second"],
        "agent_a_third": a["third"],
        "agent_a_fourth": a["fourth"],
        "agent_b_first": b["first"],
        "agent_b_second": b["second"],
        "agent_b_third": b["third"],
        "agent_b_fourth": b["fourth"],
        "agent_a_avg_move_time_s": a["avg_move_time_s"],
        "agent_b_avg_move_time_s": b["avg_move_time_s"],
        "agent_a_avg_sims_per_move": a["avg_sims_per_move"],
        "agent_b_avg_sims_per_move": b["avg_sims_per_move"],
        "agent_a_sims_per_sec": a["sims_per_sec"],
        "agent_b_sims_per_sec": b["sims_per_sec"],
    }


# =====================
# OUTPUT
# =====================
def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =====================
# VALIDATION
# =====================
def validate_specs() -> None:
    used_agents = set()

    for agent_a, agent_b, _label in MATCHUPS:
        used_agents.add(agent_a)
        used_agents.add(agent_b)

    for agent_name in used_agents:
        if agent_name not in AGENTS:
            raise KeyError(f"Agent used in MATCHUPS but not defined: {agent_name}")

        spec = AGENTS[agent_name]

        if spec["kind"] not in {"standard", "neural"}:
            raise ValueError(f"{agent_name}: invalid kind {spec['kind']}")

        load_class(spec["class_path"])

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
    validate_specs()

    used_neural_specs = {
        agent_name: AGENTS[agent_name]
        for matchup in MATCHUPS
        for agent_name in matchup[:2]
        if AGENTS[agent_name]["kind"] == "neural"
    }

    all_game_rows: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []

    print("Head-to-head evaluation")
    print(f"Games per matchup: {NUM_GAMES_PER_MATCHUP}")
    print(f"Workers: {WORKERS}")
    print(f"Time limit per move: {TIME_LIMIT_S}")
    print(f"Simulations per move: {SIMS_PER_MOVE}")
    print()

    for matchup_index, (agent_a, agent_b, label) in enumerate(MATCHUPS):
        print(f"\n=== {label}: {agent_a} vs {agent_b} ===")

        args = [
            (agent_a, agent_b, matchup_index, game_index)
            for game_index in range(NUM_GAMES_PER_MATCHUP)
        ]

        matchup_rows: List[Dict[str, Any]] = []

        with ProcessPoolExecutor(
            max_workers=WORKERS,
            initializer=init_worker,
            initargs=(used_neural_specs,),
        ) as executor:
            futures = [executor.submit(run_one_head_to_head_game, arg) for arg in args]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=label,
            ):
                row = future.result()
                matchup_rows.append(row)

        matchup_rows.sort(key=lambda r: r["game_index"])

        result = aggregate_matchup(
            agent_a=agent_a,
            agent_b=agent_b,
            label=label,
            rows=matchup_rows,
        )

        all_game_rows.extend(matchup_rows)
        all_results.append(result)

        a_summary = result["agent_a_summary"]
        b_summary = result["agent_b_summary"]

        print(
            f"{agent_a}: win_rate={a_summary['win_rate']:.3f}, "
            f"mean_place={a_summary['mean_placement']:.3f}, "
            f"placements=({a_summary['first']}, {a_summary['second']}, "
            f"{a_summary['third']}, {a_summary['fourth']})"
        )

        print(
            f"{agent_b}: win_rate={b_summary['win_rate']:.3f}, "
            f"mean_place={b_summary['mean_placement']:.3f}, "
            f"placements=({b_summary['first']}, {b_summary['second']}, "
            f"{b_summary['third']}, {b_summary['fourth']})"
        )

    flat_rows = [flatten_result_for_csv(result) for result in all_results]

    save_json(OUT_JSON, all_results)
    save_csv(OUT_CSV, flat_rows)
    save_jsonl(OUT_GAMES_JSONL, all_game_rows)

    print("\nSaved results:")
    print(f"  {OUT_JSON}")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_GAMES_JSONL}")


if __name__ == "__main__":
    main()
