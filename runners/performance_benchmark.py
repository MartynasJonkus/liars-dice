from __future__ import annotations

import importlib
import random
import time
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from liars_dice.core.game import LiarsDiceGame

# =====================
# CONFIG
# =====================
NUM_GAMES_PER_CONFIG = 100
NUM_PLAYERS = 4
DICE_PER_PLAYER = 5
BASE_SEED = 12345

SIMS_PER_MOVE = None
TIME_LIMIT_S = 0.2  # e.g. 0.2 for time-limited, else None

OUTPUT_GAMES_CSV = "pre_self_play_games.csv"
OUTPUT_SUMMARY_CSV = "pre_self_play_summary.csv"

AGENTS: Dict[str, Dict[str, Any]] = {
    "Neural_MLP": {
        "class_path": "neural.basic_mlp.neural_ismcts:NeuralISMCTSPUCTAgent",
        "kind": "neural",
        "checkpoint_path": "artifacts/training_basic/best.pt",
    },
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
    "Heuristic": {
        "class_path": "liars_dice.agents.baseline_heuristic:HeuristicAgent",
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


def _init_worker() -> None:
    global _WORKER_CACHE
    _WORKER_CACHE = {
        "bundles": {},
        "classes": {},
    }

    for agent_label, spec in AGENTS.items():
        _WORKER_CACHE["classes"][agent_label] = _load_class(spec["class_path"])
        if spec["kind"] == "neural":
            _WORKER_CACHE["bundles"][agent_label] = _load_neural_bundle(
                spec["checkpoint_path"]
            )


def _make_agent(agent_label: str, seed: int):
    spec = AGENTS[agent_label]
    cls = _WORKER_CACHE["classes"][agent_label]

    if spec["kind"] == "neural":
        model, encoder, mapper = _WORKER_CACHE["bundles"][agent_label]
        return cls(
            model=model,
            encoder=encoder,
            action_mapper=mapper,
            seed=seed,
            sims_per_move=SIMS_PER_MOVE,
            time_limit_s=TIME_LIMIT_S,
        )

    try:
        return cls(
            seed=seed,
            sims_per_move=SIMS_PER_MOVE,
            time_limit_s=TIME_LIMIT_S,
        )
    except TypeError:
        return cls(seed=seed)


def _generate_leave_one_out_configs(agent_labels: List[str]) -> List[Tuple[str, ...]]:
    """
    5 configurations:
      all agents except one
    """
    configs = []
    for omitted in agent_labels:
        lineup = tuple(a for a in agent_labels if a != omitted)
        configs.append(lineup)
    return configs


def _compute_placements_from_elimination(
    eliminated: List[int],
    winner: int,
    num_players: int,
) -> Dict[int, int]:
    placements_ordered_best_to_worst = eliminated[::-1]
    return {
        seat: place
        for place, seat in enumerate(placements_ordered_best_to_worst, start=1)
    }


# =====================
# SINGLE GAME
# =====================
def run_single_game(args):
    game_id, config = args
    rng = random.Random(BASE_SEED + game_id)

    # Shuffle seat order every game
    seat_labels = list(config)
    rng.shuffle(seat_labels)

    agents = []
    time_spent = []
    sim_counts = []

    for _seat in range(NUM_PLAYERS):
        agent_seed = rng.randint(0, 10**9)
        agent = _make_agent(seat_labels[_seat], seed=agent_seed)
        agents.append(agent)
        time_spent.append(0.0)
        sim_counts.append(0)

    game = LiarsDiceGame(
        num_players=NUM_PLAYERS,
        dice_per_player=DICE_PER_PLAYER,
        seed=rng.randint(0, 10**9),
    )

    eliminated: List[int] = []

    while True:
        pid = game._current
        obs = game.observe(pid)

        t0 = time.perf_counter()
        action = agents[pid].select_action(game, obs)
        t1 = time.perf_counter()

        time_spent[pid] += t1 - t0
        sim_counts[pid] += getattr(agents[pid], "_last_sim_count", 0)

        info = game.step(action)

        for p in range(NUM_PLAYERS):
            if game._dice_left[p] == 0 and p not in eliminated:
                eliminated.append(p)

        if info.get("terminal"):
            winner = info["winner"]
            if winner not in eliminated:
                eliminated.append(winner)
            break

    placements_by_seat = _compute_placements_from_elimination(
        eliminated=eliminated,
        winner=winner,
        num_players=NUM_PLAYERS,
    )

    placement_agents = {}
    for place in range(1, NUM_PLAYERS + 1):
        seat = next(seat for seat, p in placements_by_seat.items() if p == place)
        placement_agents[place] = seat_labels[seat]

    config_name = "_".join(sorted(config))

    game_row = {
        "game_id": game_id,
        "config": config_name,
        "seat_0_agent": seat_labels[0],
        "seat_1_agent": seat_labels[1],
        "seat_2_agent": seat_labels[2],
        "seat_3_agent": seat_labels[3],
        "placement_1": placement_agents[1],
        "placement_2": placement_agents[2],
        "placement_3": placement_agents[3],
        "placement_4": placement_agents[4],
    }

    time_rows = []
    for seat, agent_label in enumerate(seat_labels):
        time_rows.append(
            {
                "game_id": game_id,
                "config": config_name,
                "agent": agent_label,
                "time_sec": time_spent[seat],
                "sims": sim_counts[seat],
            }
        )

    placement_rows = []
    for seat, agent_label in enumerate(seat_labels):
        placement_rows.append(
            {
                "game_id": game_id,
                "config": config_name,
                "agent": agent_label,
                "placement": placements_by_seat[seat],
            }
        )

    return game_row, time_rows, placement_rows


# =====================
# MAIN
# =====================
def main() -> None:
    agent_labels = list(AGENTS.keys())
    configs = _generate_leave_one_out_configs(agent_labels)

    args = []
    game_id = 0
    for config in configs:
        for _ in range(NUM_GAMES_PER_CONFIG):
            args.append((game_id, config))
            game_id += 1

    print(
        f"Running {len(configs)} configurations x {NUM_GAMES_PER_CONFIG} games "
        f"= {len(args)} total games"
    )

    print("Configurations:")
    for cfg in configs:
        print("  ", cfg)

    all_game_rows: List[Dict[str, Any]] = []
    all_time_rows: List[Dict[str, Any]] = []
    all_placement_rows: List[Dict[str, Any]] = []

    with Pool(initializer=_init_worker) as pool:
        for game_row, time_rows, placement_rows in tqdm(
            pool.imap_unordered(run_single_game, args),
            total=len(args),
            desc="Running configurations",
        ):
            all_game_rows.append(game_row)
            all_time_rows.extend(time_rows)
            all_placement_rows.extend(placement_rows)

    # Raw per-game results
    df_games = pd.DataFrame(all_game_rows)
    df_games.to_csv(OUTPUT_GAMES_CSV, index=False)

    # Timing / simulation summary
    df_time = pd.DataFrame(all_time_rows)
    timing_summary = (
        df_time.groupby("agent")
        .agg(
            avg_move_time_sec=("time_sec", "mean"),
            total_time_sec=("time_sec", "sum"),
            total_sims=("sims", "sum"),
        )
        .reset_index()
    )
    timing_summary["sims_per_sec"] = timing_summary.apply(
        lambda r: (
            (r["total_sims"] / r["total_time_sec"]) if r["total_time_sec"] > 0 else 0.0
        ),
        axis=1,
    )

    # Placement summary
    df_place = pd.DataFrame(all_placement_rows)
    placement_summary = (
        df_place.groupby("agent")["placement"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(
            columns={
                1: "first",
                2: "second",
                3: "third",
                4: "fourth",
            }
        )
        .reset_index()
    )

    mean_place = (
        df_place.groupby("agent")["placement"]
        .mean()
        .reset_index()
        .rename(columns={"placement": "mean_placement"})
    )

    games_played = df_place.groupby("agent").size().reset_index(name="games")

    summary = placement_summary.merge(mean_place, on="agent", how="left")
    summary = summary.merge(games_played, on="agent", how="left")
    summary["win_rate"] = summary["first"] / summary["games"]
    summary = summary.merge(timing_summary, on="agent", how="left")

    summary = summary.sort_values(
        by=["mean_placement", "win_rate", "first"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print("\nSaved:")
    print(f"  Raw per-game results -> {OUTPUT_GAMES_CSV}")
    print(f"  Aggregated summary   -> {OUTPUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
