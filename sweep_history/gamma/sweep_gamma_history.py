import importlib
import random
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from typing import List
from tqdm import tqdm

from liars_dice.core.game import LiarsDiceGame

def load_agent_with_kwargs(path: str, seed: int, **kwargs):
    module, cls = path.split(":")
    mod = importlib.import_module(module)
    Cls = getattr(mod, cls)
    agent = Cls(seed=seed, **kwargs)
    name = getattr(agent, "name", cls)
    return agent, name

def run_single_game(args):
    agent_paths, value, dice_per_player = args

    rng = random.Random()

    agent_specs = [
        (agent_paths[0], {"hist_gamma": value}),   # ISMCTS tuned
        (agent_paths[1], {}),                  # Heuristic
        (agent_paths[2], {}),                  # Random
    ]

    rng.shuffle(agent_specs)

    agents = []
    names = []
    for path, kwargs in agent_specs:
        agent, name = load_agent_with_kwargs(path, seed=rng.randint(0, 10**9), **kwargs)
        agents.append(agent)
        names.append(name)

    game = LiarsDiceGame(num_players=3, dice_per_player=dice_per_player, seed=rng.randint(0, 10**9))

    eliminated = []

    while True:
        pid = game._current
        obs = game.observe(pid)
        action = agents[pid].select_action(game, obs)

        if action not in game.legal_actions():
            raise RuntimeError(f"Illegal action {action} by {names[pid]}")

        info = game.step(action)

        for p in range(3):
            if game._dice_left[p] == 0 and names[p] not in eliminated:
                eliminated.append(names[p])

        if info.get("terminal"):
            winner = info["winner"]
            winner_name = names[winner]
            if winner_name not in eliminated:
                eliminated.append(winner_name)
            break

    placements = eliminated[::-1]

    return placements.index("ISMCTS-History") + 1

def run_c_sweep_parallel(
    values: List[float],
    num_games: int = 1000,
    dice_per_player: int = 5,
    save_csv: str = "sweep_results.csv",
    plot_file: str = "sweep_plot.png",
):
    agent_paths = [
        "liars_dice.agents.ismcts_3_history:ISMCTSHistoryAgent",
        "liars_dice.agents.baseline_heuristic:HeuristicAgent",
        "liars_dice.agents.baseline_random:RandomAgent",
    ]

    results = []

    for v in values:
        print(f"\n=== Running sweep for gamma = {v} ({num_games} games) ===")

        args_list = [(agent_paths, v, dice_per_player) for _ in range(num_games)]

        placement_counts = {1: 0, 2: 0, 3: 0}

        with Pool() as pool:
            for placement in tqdm(
                pool.imap_unordered(run_single_game, args_list),
                total=len(args_list),
                desc=f"v={v}"
            ):
                placement_counts[placement] += 1

        total = sum(placement_counts.values())
        avg = (
            1 * placement_counts[1]
            + 2 * placement_counts[2]
            + 3 * placement_counts[3]
        ) / total

        results.append({
            "value": v,
            "first": placement_counts[1],
            "second": placement_counts[2],
            "third": placement_counts[3],
            "avg": avg
        })

    df = pd.DataFrame(results)
    df.to_csv(save_csv, index=False)
    print("\n=== SWEEP RESULTS ===")
    print(df)

    plt.figure(figsize=(10, 5))
    plt.plot(df["value"], df["avg"], marker="o")

    plt.title("ISMCTS-History Average placement vs History constant gamma")
    plt.xlabel("History constant beta")
    plt.ylabel("Average placement (lower is better)")
    plt.ylim(1, 3)
    plt.yticks([1, 2, 3])
    for v, avg in zip(df["value"], df["avg"]):
        plt.text(
            v, avg,
            f"{avg:.2f}",
            ha="center", va="bottom",
            fontsize=9
        )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

    print(f"\nSaved CSV: {save_csv}")
    print(f"Saved plot: {plot_file}")

    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parallel sweep for ISMCTS-History")
    parser.add_argument("--csv", type=str, default="sweep_results.csv")
    parser.add_argument("--plot", type=str, default="sweep_plot.png")
    parser.add_argument("--games", type=int, default=1000)
    args = parser.parse_args()

    VALUES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

    run_c_sweep_parallel(
        values=VALUES,
        num_games=args.games,
        dice_per_player=5,
        save_csv=args.csv,
        plot_file=args.plot,
    )
