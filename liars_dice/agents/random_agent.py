
import random
from typing import Tuple, Any
from liars_dice.core.game import Observation
from liars_dice.agents.base import Agent

class RandomAgent(Agent):
    def __init__(self, seed=None, label="Random"):
        self.rng = random.Random(seed)
        self.name = label

    def select_action(self, obs: Observation) -> Tuple[str, Any]:
        actions = obs.public  # not used; pull from game via legal_actions through runner
        raise RuntimeError("RandomAgent expects the runner to pass legal actions")
