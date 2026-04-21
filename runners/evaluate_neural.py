from __future__ import annotations

import importlib
import json
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from liars_dice.core.game import LiarsDiceGame
from neural.action_mapping import ActionMapper
from neural.basic_mlp.encoder import ObservationEncoder
from neural.basic_mlp.neural_ismcts import NeuralISMCTSPUCTAgent
from neural.basic_mlp.nn_model import PolicyNetwork

# =====================
# CONFIG
# =====================
CHECKPOINT_PATH = "artifacts/policy_training/best.pt"
RESULTS_JSON = "artifacts/eval/neural_vs_baselines.json"

NUM_GAMES = 100
DICE_PER_PLAYER = 5
NUM_PLAYERS = 4

BASE_SEED = 12345
DEVICE = "cpu"  # use "cuda" only if each worker should use GPU; usually keep CPU for multiprocessing

# Matchups:
# "N2_B2" = 2 Neural vs 2 Baseline
# "N1_B3" = 1 Neural vs 3 Baseline
MATCHUP_CONFIGS = [
    {
        "name": "Neural_vs_PUCT",
        "baseline_path": "liars_dice.agents.ismcts_2_puct:ISMCTSPUCTAgent",
        "lineups": ["N2_B2", "N1_B3"],
    },
    {
        "name": "Neural_vs_History",
        "baseline_path": "liars_dice.agents.ismcts_3_history:ISMCTSHistoryAgent",
        "lineups": ["N2_B2", "N1_B3"],
    },
]

# Neural agent parameters
NEURAL_KWARGS = {
    "label": "Neural-ISMCTS",
    "sims_per_move": 400,
    "puct_c": 1.5,
    "prior_floor": 1e-6,
    "rollout_theta": 0.40,
    "rollout_alpha": 0.70,
    "rollout_eps": 0.15,
    "rollout_max_steps": 40,
}

# Baseline kwargs keyed by full import path
BASELINE_KWARGS = {
    "liars_dice.agents.ismcts_2_puct:ISMCTSPUCTAgent": {
        "label": "ISMCTS-PUCT",
        "sims_per_move": 400,
        "puct_c": 1.5,
        "prior_tau": 1.0,
        "liar_exp": 0.5,
        "prior_floor": 1e-6,
        "rollout_theta": 0.40,
        "rollout_alpha": 0.70,
        "rollout_eps": 0.15,
        "rollout_max_steps": 40,
    },
    "liars_dice.agents.ismcts_3_history:ISMCTSHistoryAgent": {
        "label": "ISMCTS-History",
        "sims_per_move": 400,
        "puct_c": 1.5,
        "prior_tau": 1.0,
        "liar_exp": 0.5,
        "prior_floor": 1e-6,
        "hist_beta": 1.0,
        "hist_gamma": 0.0,
        "rollout_theta": 0.40,
        "rollout_alpha": 0.70,
        "rollout_eps": 0.15,
        "rollout_max_steps": 40,
    },
}


# =====================
# WORKER-LOCAL CACHE
# =====================
_WORKER_STATE: Dict[str, Any] = {}


# =====================
# HELPERS
# =====================
def _load_class(path: str):
    module_name, class_name = path.split(":")
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


def _load_neural_bundle(checkpoint_path: str, device: str) -> Dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=device)

    encoder_cfg = payload["encoder_config"]
    mapper_cfg = payload["action_mapper_config"]
    model_state = payload["model_state_dict"]

    action_mapper = ActionMapper(**mapper_cfg)
    encoder = ObservationEncoder(**encoder_cfg)

    model = PolicyNetwork(
        input_dim=encoder.input_dim,
        num_actions=action_mapper.num_actions,
        hidden_dim=256,
        dropout=0.1,
    )
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return {
        "model": model,
        "encoder": encoder,
        "action_mapper": action_mapper,
    }


def _init_worker(checkpoint_path: str, device: str, baseline_paths: List[str]) -> None:
    global _WORKER_STATE

    torch.set_num_threads(1)

    _WORKER_STATE["device"] = device
    _WORKER_STATE["neural_bundle"] = _load_neural_bundle(checkpoint_path, device)
    _WORKER_STATE["baseline_classes"] = {
        path: _load_class(path) for path in baseline_paths
    }


def _make_neural_agent(seed: int):
    bundle = _WORKER_STATE["neural_bundle"]
    return NeuralISMCTSPUCTAgent(
        model=bundle["model"],
        encoder=bundle["encoder"],
        action_mapper=bundle["action_mapper"],
        seed=seed,
        device=_WORKER_STATE["device"],
        **NEURAL_KWARGS,
    )


def _make_baseline_agent(path: str, seed: int):
    cls = _WORKER_STATE["baseline_classes"][path]
    kwargs = dict(BASELINE_KWARGS[path])
    kwargs["seed"] = seed
    return cls(**kwargs)


def _compute_team_best_places(
    placements_by_seat: Dict[int, int],
    seat_team_labels: List[str],
) -> Dict[str, int]:
    per_team_best = {}
    for seat, place in placements_by_seat.items():
        team = seat_team_labels[seat]
        if team not in per_team_best:
            per_team_best[team] = place
        else:
            per_team_best[team] = min(per_team_best[team], place)
    return per_team_best


def _placements_from_elimination_order(
    elimination_order: List[int],
    winner: int,
    num_players: int,
) -> Dict[int, int]:
    placements: Dict[int, int] = {}

    for idx, pid in enumerate(elimination_order):
        placements[pid] = num_players - idx

    placements[winner] = 1
    return placements


# =====================
# SINGLE GAME
# =====================
def run_single_game(args):
    (
        game_id,
        matchup_name,
        config_label,
        baseline_path,
        dice_per_player,
        base_seed,
    ) = args

    rng = random.Random(base_seed + game_id)

    if config_label == "N2_B2":
        seat_roles = ["Neural", "Neural", "Baseline", "Baseline"]
    elif config_label == "N1_B3":
        seat_roles = ["Neural", "Baseline", "Baseline", "Baseline"]
    else:
        raise ValueError(f"Unknown config_label: {config_label}")

    rng.shuffle(seat_roles)

    agents = []
    seat_team_labels = []

    for seat, role in enumerate(seat_roles):
        seed = rng.randint(0, 10**9)
        if role == "Neural":
            agent = _make_neural_agent(seed=seed)
            team = "Neural"
        else:
            agent = _make_baseline_agent(path=baseline_path, seed=seed)
            team = "Baseline"
        agents.append(agent)
        seat_team_labels.append(team)

    game = LiarsDiceGame(
        num_players=NUM_PLAYERS,
        dice_per_player=dice_per_player,
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

    placements_ordered_best_to_worst = eliminated[::-1]
    placements_by_seat = {
        seat: place
        for place, seat in enumerate(placements_ordered_best_to_worst, start=1)
    }

    team_best_places = _compute_team_best_places(
        placements_by_seat=placements_by_seat,
        seat_team_labels=seat_team_labels,
    )

    return {
        "game_id": game_id,
        "matchup": matchup_name,
        "config": config_label,
        "team_best_places": team_best_places,
    }


# =====================
# AGGREGATION
# =====================
def aggregate_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[row["matchup"]][row["config"]].append(row)

    out: Dict[str, Any] = {}

    for matchup_name, configs in grouped.items():
        out[matchup_name] = {}

        for config_label, config_rows in configs.items():
            placement_counts = {
                "Neural": {1: 0, 2: 0, 3: 0, 4: 0},
                "Baseline": {1: 0, 2: 0, 3: 0, 4: 0},
            }
            placement_sum = {"Neural": 0, "Baseline": 0}
            games = len(config_rows)

            for row in config_rows:
                for team in ["Neural", "Baseline"]:
                    place = row["team_best_places"][team]
                    placement_counts[team][place] += 1
                    placement_sum[team] += place

            summary = {}
            for team in ["Neural", "Baseline"]:
                summary[team] = {
                    "first": placement_counts[team][1],
                    "second": placement_counts[team][2],
                    "third": placement_counts[team][3],
                    "fourth": placement_counts[team][4],
                    "games": games,
                    "win_rate": placement_counts[team][1] / games,
                    "mean_placement": placement_sum[team] / games,
                }

            out[matchup_name][config_label] = summary

    return out


def print_results(results: Dict[str, Any]) -> None:
    for matchup_name, configs in results.items():
        print(f"\n=== {matchup_name} ===")
        for config_label, summary in configs.items():
            print(f"\n  [{config_label}]")
            for team, stats in summary.items():
                print(
                    f"  {team:8s} | "
                    f"1st={stats['first']:4d} "
                    f"2nd={stats['second']:4d} "
                    f"3rd={stats['third']:4d} "
                    f"4th={stats['fourth']:4d} "
                    f"| win_rate={stats['win_rate']:.4f} "
                    f"| mean_place={stats['mean_placement']:.4f}"
                )


# =====================
# MAIN
# =====================
def main():
    Path(RESULTS_JSON).parent.mkdir(parents=True, exist_ok=True)

    args = []
    game_id = 0
    baseline_paths = sorted({m["baseline_path"] for m in MATCHUP_CONFIGS})

    for matchup in MATCHUP_CONFIGS:
        for config_label in matchup["lineups"]:
            for _ in range(NUM_GAMES):
                args.append(
                    (
                        game_id,
                        matchup["name"],
                        config_label,
                        matchup["baseline_path"],
                        DICE_PER_PLAYER,
                        BASE_SEED,
                    )
                )
                game_id += 1

    with Pool(
        initializer=_init_worker,
        initargs=(CHECKPOINT_PATH, DEVICE, baseline_paths),
    ) as pool:
        rows = list(
            tqdm(
                pool.imap_unordered(run_single_game, args),
                total=len(args),
                desc="Running evaluation",
            )
        )

    results = aggregate_results(rows)
    print_results(results)

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
