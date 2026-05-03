from __future__ import annotations

import csv
import importlib
import importlib.util
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from liars_dice.core.game import LiarsDiceGame

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("sweep_results/ismcts_history_hist_beta_vs_ismcts_puct")

# Sweep agent: history-aware ISMCTS-PUCT variant
SWEEP_AGENT_SPEC = "liars_dice.agents.ISMCTS_History:ISMCTSHistoryAgent"

# Opponent agent: non-history PUCT variant
OPPONENT_AGENT_SPEC = "liars_dice.agents.ISMCTS_PUCT:ISMCTSPUCTAgent"

# Sweep the history-determinization face-frequency strength.
SWEEP_PARAMETER_NAME = "hist_beta"
SWEEP_PARAMETER_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

GAMES_PER_VALUE = 200

NUM_PLAYERS = 4
DICE_PER_PLAYER = 5

# One ISMCTS-History agent and three ISMCTS-PUCT opponents.
RANDOMIZE_SWEEP_AGENT_SEAT = True
FIXED_SWEEP_AGENT_SEAT = 0

EXECUTOR_KIND = "process"
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)

BASE_SEED = 12345

# ISMCTS-History configuration.
# hist_beta is injected automatically by the sweep script.
SWEEP_AGENT_KWARGS: Dict[str, Any] = {
    "label": "ISMCTS-History-Sweep",
    "sims_per_move": None,
    "time_limit_s": 0.2,
    "puct_c": 1.5,
    "prior_tau": 1.0,
    "liar_exp": 0.5,
    "prior_floor": 1e-6,
    # hist_beta is injected automatically.
    "rollout_theta": 0.30,
    "rollout_alpha": 0.60,
    "rollout_eps": 0.2,
    "rollout_max_steps": 40,
}

# ISMCTS-PUCT opponent configuration.
OPPONENT_AGENT_KWARGS: Dict[str, Any] = {
    "label": "ISMCTS-PUCT-Opponent",
    "sims_per_move": None,
    "time_limit_s": 0.2,
    "puct_c": 1.5,
    "prior_tau": 1.0,
    "liar_exp": 0.5,
    "prior_floor": 1e-6,
    "rollout_theta": 0.30,
    "rollout_alpha": 0.60,
    "rollout_eps": 0.2,
    "rollout_max_steps": 40,
}

SUMMARY_CSV_NAME = "summary.csv"
GAMES_CSV_NAME = "games.csv"
SUMMARY_JSON_NAME = "summary.json"
PLOT_NAME = "winrate_plot.png"


# ============================================================
# IMPLEMENTATION
# Usually no need to edit below this line.
# ============================================================


@dataclass(frozen=True)
class GameJob:
    parameter_value: Any
    game_index: int
    seed: int


@dataclass
class GameResult:
    parameter_value: Any
    game_index: int
    seed: int
    sweep_seat: int
    winner: int
    sweep_won: bool
    sweep_place: int
    placements: Dict[int, int]
    final_dice_left: List[int]
    turns: int
    duration_s: float


def load_class(spec: str):
    """
    Loads a class from either:
      - 'package.module:ClassName'
      - 'path/to/file.py:ClassName'
    """
    if ":" not in spec:
        raise ValueError(
            f"Invalid class spec '{spec}'. Expected 'module:ClassName' or 'path.py:ClassName'."
        )

    module_part, class_name = spec.split(":", 1)

    if module_part.endswith(".py") or "/" in module_part or "\\" in module_part:
        file_path = Path(module_part).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Could not find Python file: {file_path}")

        module_name = f"_dynamic_agent_{file_path.stem}_{abs(hash(str(file_path)))}"
        spec_obj = importlib.util.spec_from_file_location(module_name, file_path)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"Could not load module from file: {file_path}")

        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(module_part)

    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Module '{module_part}' has no class '{class_name}'."
        ) from exc


def make_agent(agent_spec: str, kwargs: Dict[str, Any], seed: int):
    cls = load_class(agent_spec)

    agent_kwargs = dict(kwargs)

    # Your existing agents generally accept seed, but this keeps compatibility
    # with simpler agents that do not.
    try:
        return cls(**agent_kwargs, seed=seed)
    except TypeError:
        return cls(**agent_kwargs)


def choose_sweep_seat(rng: random.Random) -> int:
    if RANDOMIZE_SWEEP_AGENT_SEAT:
        return rng.randrange(NUM_PLAYERS)
    return FIXED_SWEEP_AGENT_SEAT


def compute_placements(elimination_order: List[int], winner: int) -> Dict[int, int]:
    """
    For 4 players:
      first eliminated  -> 4th
      second eliminated -> 3rd
      third eliminated  -> 2nd
      winner            -> 1st
    """
    placements: Dict[int, int] = {}

    place = NUM_PLAYERS
    for pid in elimination_order:
        placements[pid] = place
        place -= 1

    placements[winner] = 1

    # Defensive fallback in case something unusual happens.
    for pid in range(NUM_PLAYERS):
        placements.setdefault(pid, place)
        place -= 1

    return placements


def play_one_game(job: GameJob) -> GameResult:
    start_time = time.perf_counter()
    rng = random.Random(job.seed)

    sweep_seat = choose_sweep_seat(rng)

    sweep_kwargs = dict(SWEEP_AGENT_KWARGS)
    sweep_kwargs[SWEEP_PARAMETER_NAME] = job.parameter_value

    agents = []
    for seat in range(NUM_PLAYERS):
        if seat == sweep_seat:
            agent = make_agent(
                SWEEP_AGENT_SPEC,
                sweep_kwargs,
                seed=job.seed * 1000 + seat,
            )
        else:
            agent = make_agent(
                OPPONENT_AGENT_SPEC,
                OPPONENT_AGENT_KWARGS,
                seed=job.seed * 1000 + seat,
            )
        agents.append(agent)

    game = LiarsDiceGame(
        num_players=NUM_PLAYERS,
        dice_per_player=DICE_PER_PLAYER,
        seed=job.seed,
    )

    elimination_order: List[int] = []
    previously_alive = {pid for pid in range(NUM_PLAYERS)}
    turns = 0

    while game.num_alive() > 1:
        actor = game._current
        obs = game.observe(actor)

        action = agents[actor].select_action(game, obs)
        info = game.step(action)
        turns += 1

        currently_alive = {pid for pid, dice in enumerate(game._dice_left) if dice > 0}
        eliminated_now = sorted(previously_alive - currently_alive)

        for pid in eliminated_now:
            if pid not in elimination_order:
                elimination_order.append(pid)

        previously_alive = currently_alive

        if info.get("terminal", False):
            break

    winner = game._winner()
    if winner is None:
        raise RuntimeError("Game ended without a winner.")

    placements = compute_placements(elimination_order, winner)
    sweep_place = placements[sweep_seat]

    # Notify agents. Existing agents may ignore this.
    final_info = {
        "terminal": True,
        "winner": winner,
        "placements": placements,
        "final_dice_left": list(game._dice_left),
    }

    for pid, agent in enumerate(agents):
        try:
            agent.notify_result(game.observe(pid), final_info)
        except Exception:
            # Do not let logging hooks break the sweep.
            pass

    duration_s = time.perf_counter() - start_time

    return GameResult(
        parameter_value=job.parameter_value,
        game_index=job.game_index,
        seed=job.seed,
        sweep_seat=sweep_seat,
        winner=winner,
        sweep_won=(winner == sweep_seat),
        sweep_place=sweep_place,
        placements=placements,
        final_dice_left=list(game._dice_left),
        turns=turns,
        duration_s=duration_s,
    )


def aggregate_results(results: List[GameResult]) -> List[Dict[str, Any]]:
    grouped: Dict[Any, List[GameResult]] = defaultdict(list)

    for result in results:
        grouped[result.parameter_value].append(result)

    summary_rows: List[Dict[str, Any]] = []

    for parameter_value in SWEEP_PARAMETER_VALUES:
        group = grouped[parameter_value]
        n = len(group)

        if n == 0:
            continue

        place_counts = Counter(result.sweep_place for result in group)
        wins = place_counts[1]

        avg_place = sum(result.sweep_place for result in group) / n
        avg_turns = sum(result.turns for result in group) / n
        avg_duration_s = sum(result.duration_s for result in group) / n

        seat_counts = Counter(result.sweep_seat for result in group)

        row = {
            "parameter": SWEEP_PARAMETER_NAME,
            "value": parameter_value,
            "games": n,
            "wins": wins,
            "winrate": wins / n,
            "place_1_count": place_counts[1],
            "place_2_count": place_counts[2],
            "place_3_count": place_counts[3],
            "place_4_count": place_counts[4],
            "place_1_rate": place_counts[1] / n,
            "place_2_rate": place_counts[2] / n,
            "place_3_rate": place_counts[3] / n,
            "place_4_rate": place_counts[4] / n,
            "avg_place": avg_place,
            "avg_turns": avg_turns,
            "avg_game_duration_s": avg_duration_s,
            "sweep_seat_0_count": seat_counts[0],
            "sweep_seat_1_count": seat_counts[1],
            "sweep_seat_2_count": seat_counts[2],
            "sweep_seat_3_count": seat_counts[3],
        }

        summary_rows.append(row)

    return summary_rows


def write_games_csv(results: List[GameResult], path: Path) -> None:
    fieldnames = [
        "parameter",
        "value",
        "game_index",
        "seed",
        "sweep_seat",
        "winner",
        "sweep_won",
        "sweep_place",
        "player_0_place",
        "player_1_place",
        "player_2_place",
        "player_3_place",
        "final_dice_left",
        "turns",
        "duration_s",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "parameter": SWEEP_PARAMETER_NAME,
                    "value": result.parameter_value,
                    "game_index": result.game_index,
                    "seed": result.seed,
                    "sweep_seat": result.sweep_seat,
                    "winner": result.winner,
                    "sweep_won": int(result.sweep_won),
                    "sweep_place": result.sweep_place,
                    "player_0_place": result.placements[0],
                    "player_1_place": result.placements[1],
                    "player_2_place": result.placements[2],
                    "player_3_place": result.placements[3],
                    "final_dice_left": json.dumps(result.final_dice_left),
                    "turns": result.turns,
                    "duration_s": result.duration_s,
                }
            )


def write_summary_csv(summary_rows: List[Dict[str, Any]], path: Path) -> None:
    if not summary_rows:
        raise RuntimeError("No summary rows to write.")

    fieldnames = list(summary_rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def write_summary_json(
    summary_rows: List[Dict[str, Any]],
    results: List[GameResult],
    path: Path,
) -> None:
    payload = {
        "config": {
            "sweep_agent_spec": SWEEP_AGENT_SPEC,
            "opponent_agent_spec": OPPONENT_AGENT_SPEC,
            "sweep_parameter_name": SWEEP_PARAMETER_NAME,
            "sweep_parameter_values": SWEEP_PARAMETER_VALUES,
            "games_per_value": GAMES_PER_VALUE,
            "num_players": NUM_PLAYERS,
            "dice_per_player": DICE_PER_PLAYER,
            "randomize_sweep_agent_seat": RANDOMIZE_SWEEP_AGENT_SEAT,
            "fixed_sweep_agent_seat": FIXED_SWEEP_AGENT_SEAT,
            "executor_kind": EXECUTOR_KIND,
            "max_workers": MAX_WORKERS,
            "base_seed": BASE_SEED,
            "sweep_agent_kwargs": SWEEP_AGENT_KWARGS,
            "opponent_agent_kwargs": OPPONENT_AGENT_KWARGS,
        },
        "summary": summary_rows,
        "num_raw_game_results": len(results),
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def plot_winrates(summary_rows: List[Dict[str, Any]], path: Path) -> None:
    xs = [str(row["value"]) for row in summary_rows]
    ys = [100.0 * row["winrate"] for row in summary_rows]

    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel(SWEEP_PARAMETER_NAME)
    plt.ylabel("Sweep agent winrate (%)")
    plt.title(f"Parameter sweep: {SWEEP_PARAMETER_NAME}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.show()


def build_jobs() -> List[GameJob]:
    jobs: List[GameJob] = []

    for value_index, parameter_value in enumerate(SWEEP_PARAMETER_VALUES):
        for game_index in range(GAMES_PER_VALUE):
            seed = BASE_SEED + value_index * 1_000_000 + game_index
            jobs.append(
                GameJob(
                    parameter_value=parameter_value,
                    game_index=game_index,
                    seed=seed,
                )
            )

    return jobs


def run_parallel(jobs: List[GameJob]) -> List[GameResult]:
    if EXECUTOR_KIND == "process":
        executor_cls = ProcessPoolExecutor
    elif EXECUTOR_KIND == "thread":
        executor_cls = ThreadPoolExecutor
    else:
        raise ValueError("EXECUTOR_KIND must be either 'process' or 'thread'.")

    results: List[GameResult] = []
    total = len(jobs)
    completed = 0

    with executor_cls(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(play_one_game, job) for job in jobs]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            completed += 1
            if completed % max(1, total // 20) == 0 or completed == total:
                print(f"Completed {completed}/{total} games")

    results.sort(
        key=lambda r: (SWEEP_PARAMETER_VALUES.index(r.parameter_value), r.game_index)
    )
    return results


def print_summary(summary_rows: List[Dict[str, Any]]) -> None:
    print()
    print("Sweep summary")
    print("=" * 80)

    for row in summary_rows:
        print(
            f"{SWEEP_PARAMETER_NAME}={row['value']!r} | "
            f"games={row['games']} | "
            f"winrate={row['winrate']:.3f} | "
            f"placements=[1st:{row['place_1_count']}, "
            f"2nd:{row['place_2_count']}, "
            f"3rd:{row['place_3_count']}, "
            f"4th:{row['place_4_count']}] | "
            f"avg_place={row['avg_place']:.3f}"
        )

    best = max(summary_rows, key=lambda r: r["winrate"])
    print("=" * 80)
    print(
        f"Best by winrate: {SWEEP_PARAMETER_NAME}={best['value']!r} "
        f"with winrate={best['winrate']:.3f}"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs()

    print(f"Running parameter sweep for {SWEEP_AGENT_SPEC}")
    print(f"Opponent: {OPPONENT_AGENT_SPEC}")
    print(f"Parameter: {SWEEP_PARAMETER_NAME}")
    print(f"Values: {SWEEP_PARAMETER_VALUES}")
    print(f"Games per value: {GAMES_PER_VALUE}")
    print(f"Total games: {len(jobs)}")
    print(f"Executor: {EXECUTOR_KIND}, workers={MAX_WORKERS}")
    print()

    start = time.perf_counter()
    results = run_parallel(jobs)
    elapsed = time.perf_counter() - start

    summary_rows = aggregate_results(results)

    games_csv_path = OUTPUT_DIR / GAMES_CSV_NAME
    summary_csv_path = OUTPUT_DIR / SUMMARY_CSV_NAME
    summary_json_path = OUTPUT_DIR / SUMMARY_JSON_NAME
    plot_path = OUTPUT_DIR / PLOT_NAME

    write_games_csv(results, games_csv_path)
    write_summary_csv(summary_rows, summary_csv_path)
    write_summary_json(summary_rows, results, summary_json_path)
    plot_winrates(summary_rows, plot_path)

    print_summary(summary_rows)

    print()
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Wrote raw game results to: {games_csv_path}")
    print(f"Wrote summary CSV to: {summary_csv_path}")
    print(f"Wrote summary JSON to: {summary_json_path}")
    print(f"Wrote plot to: {plot_path}")


if __name__ == "__main__":
    main()
