import importlib
import random
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Any, Dict
from tqdm import trange
import argparse

from liars_dice.core.game import LiarsDiceGame

def load_agent_with_kwargs(path: str, seed: int, **kwargs):
    """Load agent from 'module:Class' and pass extra kwargs."""
    module, cls = path.split(":")
    mod = importlib.import_module(module)
    Cls = getattr(mod, cls)
    agent = Cls(seed=seed, **kwargs)
    name = getattr(agent, "name", cls)
    return agent, name

def play_one_game_c_sweep(agent_paths: List[str], sims_count: int, dice_per_player: int, seed: int):
    """
    Plays one 3-player game with:
        Player 0: ISMCTSBasicAgent(c=c_value)
        Player 1: HeuristicAgent
        Player 2: RandomAgent

    Returns the placement (1st, 2nd, 3rd) for ISMCTSBasicAgent.
    """
    rng = random.Random(seed)

    agents_with_params = [
        load_agent_with_kwargs(agent_paths[0], seed=rng.randint(0, 10**9), sims_per_move=sims_count),
        load_agent_with_kwargs(agent_paths[1], seed=rng.randint(0, 10**9)),
        load_agent_with_kwargs(agent_paths[2], seed=rng.randint(0, 10**9)),
    ]

    rng.shuffle(agents_with_params)

    agents = [a for a, _ in agents_with_params]
    names  = [n for _, n in agents_with_params]

    game = LiarsDiceGame(num_players=3, dice_per_player=dice_per_player, seed=seed)

    eliminated = []

    while True:
        pid = game._current
        obs = game.observe(pid)
        action = agents[pid].select_action(game, obs)

        legal = game.legal_actions()
        if action not in legal:
            raise RuntimeError(f"Illegal action {action} by agent {names[pid]}. Legal: {legal}")

        info = game.step(action)

        for player in range(3):
            if game._dice_left[player] == 0 and names[player] not in eliminated:
                eliminated.append(names[player])

        if info.get("terminal"):
            winner = info["winner"]
            if winner not in eliminated:
                eliminated.append(names[winner])
            break

    placements = eliminated[::-1]

    return placements.index("ISMCTS-Basic") + 1

def run_c_sweep(
    s_values: List[int],
    num_games: int = 100,
    dice_per_player: int = 5,
    save_csv: str = "sim_sweep_results.csv",
    plot_file: str = "sim_sweep_placements.png",
):
    agent_paths = [
        "liars_dice.agents.ismcts_0_basic:ISMCTSBasicAgent",
        "liars_dice.agents.baseline_heuristic:HeuristicAgent",
        "liars_dice.agents.baseline_random:RandomAgent",
    ]

    results = []

    for s in s_values:
        placement_counts = {1: 0, 2: 0, 3: 0}

        for g in trange(num_games, desc=f"s={s}"):
            placement = play_one_game_c_sweep(
                agent_paths=agent_paths,
                sims_count=s,
                dice_per_player=dice_per_player,
                seed=random.randint(0, 1_000_000_000),
            )
            placement_counts[placement] += 1

        results.append({
            "s": s,
            "first": placement_counts[1],
            "second": placement_counts[2],
            "third": placement_counts[3],
        })

        print("s: ", s)
        print("first: ", placement_counts[1])
        print("second: ", placement_counts[2])
        print("third: ", placement_counts[3])

    df = pd.DataFrame(results)
    df.to_csv(save_csv, index=False)
    print(df)

    plt.figure(figsize=(10, 5))
    plt.plot(df["s"], df["first"], marker="o", label="1st place")
    plt.plot(df["s"], df["second"], marker="o", label="2nd place")
    plt.plot(df["s"], df["third"], marker="o", label="3rd place")

    plt.title("ISMCTS-Basic Placement distribution vs number of child nodes expanded per move")
    plt.xlabel("Number of child nodes per move")
    plt.ylabel("Number of placements (out of 100 games)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

    print(f"\nSaved CSV to {save_csv}")
    print(f"Saved plot to {plot_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep sim count values for ISMCTS-Basic")
    parser.add_argument("--csv", type=str, default="sims_sweep_results.csv")
    parser.add_argument("--plot", type=str, default="sims_sweep_plot.png")

    args = parser.parse_args()

    S_VALUES = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]

    run_c_sweep(
        s_values=S_VALUES,
        num_games=100,
        dice_per_player=5,
        save_csv=args.csv,
        plot_file=args.plot,
    )
