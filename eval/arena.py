
import importlib
import random
from typing import List, Tuple, Any
from liars_dice.core.game import LiarsDiceGame

def load_agent(path:str, **kwargs):
    # path like "liars_dice.agents.random_agent:RandomAgent"
    module_name, class_name = path.split(":")
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls(**kwargs)

def play_match(agent_paths: List[str], num_players:int=2, dice_per_player:int=5, seed:int=42, verbose:bool=False):
    rng = random.Random(seed)
    game = LiarsDiceGame(num_players=num_players, dice_per_player=dice_per_player, seed=seed)
    agents = []
    for i, path in enumerate(agent_paths):
        cls = load_agent(path, seed=rng.randint(0, 1_000_000))
        # Give numbered label
        base_name = path.split(":")[1]
        cls.name = f"{base_name} {i}"
        agents.append(cls)
    # wrap to provide legal actions (kept here to keep Agent API simple)
    while True:
        pid = game._current  # accessing for now; could expose method
        obs = game.observe(pid)
        legal = game.legal_actions()

        # basic policy: if RandomAgent error occurs, choose here
        action = None
        try:
            action = agents[pid].select_action(obs)  # if agent uses its own logic
            # sanity check: if action not legal, fall back to random legal
            if action not in legal:
                action = random.choice(legal)
        except Exception:
            action = random.choice(legal)

        info = game.step(action)
        if verbose:
            print(f"P{pid} ({agents[pid].name}) -> {action} | dice_left={game._dice_left}")
            if action[0] == "liar":
                # Last entry in history will be showdown info
                showdown = game._history[-1]
                print(f"   showdown -> {showdown}")
                # Reveal dice
                for i, dice in enumerate(game._dice):
                    print(f"   Player {i} dice: {dice}")


        if info.get("terminal"):
            winner = info["winner"]
            if verbose:
                print(f"Winner: P{winner}")
            return winner
