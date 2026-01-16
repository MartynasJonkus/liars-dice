import importlib
import random
from itertools import combinations
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from liars_dice.core.game import LiarsDiceGame


def load_agent(path: str, seed: int):
    module, cls = path.split(":")
    mod = importlib.import_module(module)
    Cls = getattr(mod, cls)
    agent = Cls(seed=seed)
    return agent, agent.name


def run_single_game(args):
    game_id, agent_paths, dice_per_player = args
    rng = random.Random()

    agents = []
    names = []

    specs = [(p, rng.randint(0, 10**9)) for p in agent_paths]
    rng.shuffle(specs)

    for path, seed in specs:
        agent, name = load_agent(path, seed)
        agents.append(agent)
        names.append(name)

    game = LiarsDiceGame(
        num_players=4, dice_per_player=dice_per_player, seed=rng.randint(0, 10**9)
    )

    eliminated = []

    while True:
        pid = game._current
        obs = game.observe(pid)
        action = agents[pid].select_action(game, obs)

        if action not in game.legal_actions():
            raise RuntimeError(f"Illegal action {action} by {names[pid]}")

        info = game.step(action)

        for p in range(4):
            if game._dice_left[p] == 0 and names[p] not in eliminated:
                eliminated.append(names[p])

        if info.get("terminal"):
            winner = info["winner"]
            winner_name = names[winner]
            if winner_name not in eliminated:
                eliminated.append(winner_name)
            break

    placements = eliminated[::-1]

    return {
        "game_id": game_id,
        "p1": placements[0],
        "p2": placements[1],
        "p3": placements[2],
        "p4": placements[3],
    }


def run_round_robin(
    agent_paths,
    games_per_combo=1000,
    dice_per_player=5,
    raw_csv="round_robin_games.csv",
    summary_csv="round_robin_summary.csv",
):
    combos = list(combinations(agent_paths, 4))
    all_results = []
    game_counter = 0

    args = []
    for combo in combos:
        for _ in range(games_per_combo):
            args.append((game_counter, combo, dice_per_player))
            game_counter += 1

    with Pool() as pool:
        for res in tqdm(
            pool.imap_unordered(run_single_game, args),
            total=len(args),
            desc="Running round-robin",
        ):
            all_results.append(res)

    # --------------------------------------------------
    # Raw per-game data
    # --------------------------------------------------

    df_games = pd.DataFrame(all_results)
    df_games.to_csv(raw_csv, index=False)

    # --------------------------------------------------
    # Aggregation
    # --------------------------------------------------

    rows = []
    for _, row in df_games.iterrows():
        for place, col in enumerate(["p1", "p2", "p3", "p4"], start=1):
            rows.append({"agent": row[col], "placement": place})

    df_long = pd.DataFrame(rows)

    summary = (
        df_long.groupby("agent")["placement"]
        .agg(
            first=lambda x: (x == 1).sum(),
            second=lambda x: (x == 2).sum(),
            third=lambda x: (x == 3).sum(),
            fourth=lambda x: (x == 4).sum(),
            games="count",
            mean_placement="mean",
        )
        .reset_index()
    )

    summary["win_rate"] = summary["first"] / summary["games"]

    summary.to_csv(summary_csv, index=False)

    print("\nSaved:")
    print(f"  Raw games: {raw_csv}")
    print(f"  Summary:   {summary_csv}")

    return df_games, summary


if __name__ == "__main__":
    AGENTS = [
        "liars_dice.agents.baseline_heuristic:HeuristicAgent",
        "liars_dice.agents.ismcts_0_basic:ISMCTSBasicAgent",
        "liars_dice.agents.ismcts_1_heuristic:ISMCTSHeuristicAgent",
        "liars_dice.agents.ismcts_2_puct:ISMCTSPUCTAgent",
        "liars_dice.agents.ismcts_3_history:ISMCTSHistoryAgent",
    ]

    run_round_robin(
        agent_paths=AGENTS,
        games_per_combo=1000,
        dice_per_player=5,
    )
