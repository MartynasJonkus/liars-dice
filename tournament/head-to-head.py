import importlib
import random
import time
import pandas as pd
from multiprocessing import Pool
from typing import Tuple, Dict, Any, List
from tqdm import tqdm

from liars_dice.core.game import LiarsDiceGame

# ----------------------------
# Agent loading
# ----------------------------

def load_agent(path: str, seed: int):
    module, cls = path.split(":")
    mod = importlib.import_module(module)
    Cls = getattr(mod, cls)
    agent = Cls(seed=seed)
    return agent, agent.name


# ----------------------------
# Single game runner
# ----------------------------

def run_single_game(args):
    (
        game_id,
        config_label,
        agent_A_path,
        agent_B_path,
        dice_per_player,
    ) = args

    rng = random.Random()

    if config_label == "A2_B1":
        agent_paths = [agent_A_path, agent_A_path, agent_B_path]
    else:  # A1_B2
        agent_paths = [agent_A_path, agent_B_path, agent_B_path]

    rng.shuffle(agent_paths)

    agents = []
    names = []
    time_spent = []

    for path in agent_paths:
        agent, name = load_agent(path, seed=rng.randint(0, 10**9))
        agents.append(agent)
        names.append(name)
        time_spent.append(0.0)

    game = LiarsDiceGame(
        num_players=3,
        dice_per_player=dice_per_player,
        seed=rng.randint(0, 10**9),
    )

    eliminated = []

    while True:
        pid = game._current
        obs = game.observe(pid)

        t0 = time.perf_counter()
        action = agents[pid].select_action(game, obs)
        t1 = time.perf_counter()

        time_spent[pid] += (t1 - t0)

        info = game.step(action)

        for p in range(3):
            if game._dice_left[p] == 0 and p not in eliminated:
                eliminated.append(p)

        if info.get("terminal"):
            winner = info["winner"]
            if winner not in eliminated:
                eliminated.append(winner)
            break

    placements = eliminated[::-1]

    placement_names = {
        "placement_1": names[placements[0]],
        "placement_2": names[placements[1]],
        "placement_3": names[placements[2]],
    }

    time_rows = []
    for pid, name in enumerate(names):
        time_rows.append({
            "game_id": game_id,
            "agent": name,
            "time": time_spent[pid],
        })

    return {
        "game_id": game_id,
        "config": config_label,
        **placement_names,
    }, time_rows


# ----------------------------
# Tournament runner
# ----------------------------

def run_head_to_head(
    agent_A_path: str,
    agent_B_path: str,
    num_games: int = 1000,
    dice_per_player: int = 5,
    out_games_csv: str = "games.csv",
    out_summary_csv: str = "summary.csv",
):

    all_game_rows = []
    all_time_rows = []

    args = []
    gid = 0
    for config in ["A2_B1", "A1_B2"]:
        for _ in range(num_games):
            args.append((
                gid,
                config,
                agent_A_path,
                agent_B_path,
                dice_per_player,
            ))
            gid += 1

    with Pool() as pool:
        for game_row, time_rows in tqdm(
            pool.imap_unordered(run_single_game, args),
            total=len(args),
            desc="Running matches",
        ):
            all_game_rows.append(game_row)
            all_time_rows.extend(time_rows)

    # ----------------------------
    # Save per-game results
    # ----------------------------

    df_games = pd.DataFrame(all_game_rows)
    df_games.to_csv(out_games_csv, index=False)

    # ----------------------------
    # Timing aggregation
    # ----------------------------

    df_time = pd.DataFrame(all_time_rows)

    avg_time = (
        df_time
        .groupby("agent")["time"]
        .mean()
        .reset_index()
        .rename(columns={"time": "avg_move_time"})
    )

    # ----------------------------
    # Placement aggregation (best per game)
    # ----------------------------

    rows = []
    for _, row in df_games.iterrows():
        placements = [
            row["placement_1"],
            row["placement_2"],
            row["placement_3"],
        ]

        seen = {}
        for i, agent in enumerate(placements, start=1):
            seen[agent] = min(seen.get(agent, 3), i)

        for agent, best_place in seen.items():
            rows.append({
                "agent": agent,
                "best_place": best_place,
            })

    df_best = pd.DataFrame(rows)

    mean_place = (
        df_best
        .groupby("agent")["best_place"]
        .mean()
        .reset_index()
        .rename(columns={"best_place": "mean_placement"})
    )

    summary = (
        df_best
        .groupby("agent")["best_place"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={
            1: "first",
            2: "second",
            3: "third",
        })
        .reset_index()
    )

    summary["games"] = summary[["first", "second", "third"]].sum(axis=1)
    summary["win_rate"] = summary["first"] / summary["games"]

    summary = summary.merge(mean_place, on="agent", how="left")
    summary = summary.merge(avg_time, on="agent", how="left")


    summary.to_csv(out_summary_csv, index=False)

    print("\nSaved:")
    print(f"  Per-game results → {out_games_csv}")
    print(f"  Aggregated results → {out_summary_csv}")

if __name__ == "__main__":
    run_head_to_head(
        agent_A_path="liars_dice.agents.ismcts_0_basic:ISMCTSBasicAgent",
        agent_B_path="liars_dice.agents.ismcts_1_heuristic:ISMCTSHeuristicAgent",
        num_games=10,
        dice_per_player=5,
        out_games_csv="head_to_head_games.csv",
        out_summary_csv="head_to_head_summary.csv",
    )
