# debug_run.py
from eval.run_experiments import play_one_game
import random

paths = [
    "liars_dice.agents.heuristic:HeuristicAgent",
    "liars_dice.agents.random:RandomAgent",
    "liars_dice.agents.ismcts_puct:ISMCTSPUCTAgent",
]

print("Starting single debug game...")
winner = play_one_game(paths, num_players=3, dice_per_player=5, seed=123)
print("Winner:", winner)
