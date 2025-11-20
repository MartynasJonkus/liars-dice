import importlib
import random
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Any, Dict
from tqdm import trange
import argparse

from liars_dice.core.game import LiarsDiceGame


def load_agent(path: str, seed: int):
    """Load agent from 'module:Class' string and return (instance, pretty_name)."""
    module, cls = path.split(":")
    mod = importlib.import_module(module)
    Cls = getattr(mod, cls)
    agent = Cls(seed=seed)
    name = getattr(agent, "name", cls)
    return agent, name


def play_one_game(agent_paths: List[str], num_players: int, dice_per_player: int, seed: int):
    """
    Plays one game with randomized seat order.
    Returns: winner_name
    """
    rng = random.Random(seed)

    # Shuffle player order BEFORE creating agents
    shuffled = agent_paths[:] 
    rng.shuffle(shuffled)

    # Instantiate agents for this game
    agents_with_names = [
        load_agent(path, seed=rng.randint(0, 10**9)) for path in shuffled
    ]
    agents = [a for a, _ in agents_with_names]
    names  = [n for _, n in agents_with_names]

    # Start game
    game = LiarsDiceGame(
        num_players=num_players,
        dice_per_player=dice_per_player,
        seed=seed
    )

    while True:
        pid = game._current
        obs = game.observe(pid)
        action = agents[pid].select_action(game, obs)

        # Safety check
        legal = game.legal_actions()
        if action not in legal:
            raise RuntimeError(f"Illegal action {action} by agent {names[pid]}. Legal: {legal}")

        info = game.step(action)
        if info.get("terminal"):
            winner_seat = info["winner"]
            return names[winner_seat]


def run_experiment(
    agent_paths: List[str],
    num_games: int = 100,
    num_players: int = 2,
    dice_per_player: int = 5,
    seed: int = 42,
    save_csv: str = "results.csv",
    plot_prefix: str = "plots",
):
    rng = random.Random(seed)

    # Map agent name -> win count
    # Build initial win_counts based on AGENT NAMES (not class names)
    win_counts: Dict[str, int] = {}

    # First, instantiate ONE temporary instance of each agent to read its name
    for path in agent_paths:
        module, cls = path.split(":")
        mod = importlib.import_module(module)
        Cls = getattr(mod, cls)
        temp = Cls(seed=0)
        agent_name = getattr(temp, "name", cls)
        win_counts[agent_name] = 0

    win_history = []  # (game_index, winner_name)

    for g in trange(num_games, desc="Running games"):
        winner_name = play_one_game(
            agent_paths=agent_paths,
            num_players=num_players,
            dice_per_player=dice_per_player,
            seed=rng.randint(0, 10**9),
        )
        win_counts[winner_name] += 1
        win_history.append((g, winner_name))

    # Convert to DataFrame
    df = pd.DataFrame(win_history, columns=["game", "winner"])
    df.to_csv(save_csv, index=False)

    # Compute winrates
    winrates = {
        name: win_counts[name] / num_games * 100
        for name in win_counts
    }

    print("\n=== FINAL RESULTS ===")
    for name, rate in winrates.items():
        print(f"{name:20s}: {rate:.2f}%")

    # Rolling winrates for each agent
    plt.figure(figsize=(10, 5))
    window = max(5, num_games // 20)

    for name in win_counts.keys():
        df_name = (df["winner"] == name).astype(int).rolling(window).mean()
        plt.plot(df_name, label=name)

    plt.title("Rolling Winrate")
    plt.xlabel("Game")
    plt.ylabel(f"Rolling winrate (window={window})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plot_prefix}_rolling_winrate.png")
    plt.close()

    # Final bar chart
    plt.figure(figsize=(6, 4))
    labels = list(winrates.keys())
    values = [winrates[n] for n in labels]
    plt.bar(labels, values)
    plt.title("Final Winrates")
    plt.ylabel("Winrate (%)")
    plt.tight_layout()
    plt.savefig(f"{plot_prefix}_final_winrates.png")
    plt.close()

    print(f"\nSaved CSV: {save_csv}")
    print(f"Saved plots: {plot_prefix}_rolling_winrate.png, {plot_prefix}_final_winrates.png")

    return winrates

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run bulk Liar's Dice experiments")

    parser.add_argument(
        "--p", nargs="+", required=True,
        help="Agent paths like liars_dice.agents.random_agent:RandomAgent"
    )
    parser.add_argument("--num_games", type=int, default=100)
    parser.add_argument("--players", type=int, default=None)
    parser.add_argument("--dice", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=str, default="results.csv")
    parser.add_argument("--prefix", type=str, default="plots")

    args = parser.parse_args()

    # If --players not specified, infer from number of agents
    num_players = args.players if args.players is not None else len(args.p)

    run_experiment(
        agent_paths=args.p,
        num_games=args.num_games,
        num_players=num_players,
        dice_per_player=args.dice,
        seed=args.seed,
        save_csv=args.csv,
        plot_prefix=args.prefix,
    )
