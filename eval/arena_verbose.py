import importlib
import random
from typing import List, Tuple, Any
from liars_dice.core.game import LiarsDiceGame

Action = Tuple[str, Any]

def load_agent(path: str, **kwargs):
    module_name, class_name = path.split(":")
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls(**kwargs)

def play_match(
    agent_paths: List[str],
    num_players: int = 2,
    dice_per_player: int = 5,
    seed: int = 42,
) -> int:
    rng = random.Random(seed)
    game = LiarsDiceGame(num_players=num_players, dice_per_player=dice_per_player, seed=seed)

    agents = [load_agent(p, seed=rng.randint(0, 1_000_000)) for p in agent_paths]

    round_no = 0

    while True:
        showdown_count = sum(1 for h in game._history if isinstance(h, tuple) and h[0] == "showdown")
        if (round_no == showdown_count):
            round_no += 1
            print(f"\n=== ROUND {round_no} START ===")
            for pid, dice in enumerate(game._dice):
                label = getattr(agents[pid], "name", type(agents[pid]).__name__)
                print(f"Player {pid} ({label}) rolled: {dice}")

        pid = game._current
        obs = game.observe(pid)

        action: Action = agents[pid].select_action(game, obs)
        if action not in game.legal_actions():
            raise ValueError(
                f"Player {pid} ({getattr(agents[pid], 'name', type(agents[pid]).__name__)}) "
                f"returned illegal action {action}.\n"
                f"Legal actions were: {game.legal_actions()}"
            )

        info = game.step(action)

        label = getattr(agents[pid], "name", type(agents[pid]).__name__)
        print(f"Player {pid} ({label}) → {action}")

        if action[0] == "liar":
            print(">>> SHOWDOWN <<<")
            showdown = game._history[-1]

            payload = showdown[2] if isinstance(showdown, tuple) and len(showdown) >= 3 else {}
            bid     = payload.get("bid")
            actual  = payload.get("actual")
            loser   = payload.get("loser")
            challenger = payload.get("previous_bidder")
            caller     = payload.get("caller")
            
            q, f = bid

            chall_label = getattr(agents[challenger], "name", type(agents[challenger]).__name__)
            caller_label = getattr(agents[caller], "name", type(agents[caller]).__name__)
            loser_label = getattr(agents[loser], "name", type(agents[loser]).__name__)

            print(f"Player {challenger} ({chall_label}) bid {q} {f}'s")
            print(f"Player {caller} ({caller_label}) called Liar!")
            print(f"Actual amount was {actual} {f}'s")
            print(f"Player {loser} ({loser_label}) loses a die!")
            print(f"Dice left: {game._dice_left}")

        if info.get("terminal"):
            winner = info["winner"]
            label = getattr(agents[winner], "name", type(agents[winner]).__name__)
            print(f"\n*** WINNER: Player {winner} ({label}) ***\n")
            return winner
