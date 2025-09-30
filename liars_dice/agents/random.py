
import random
from typing import Optional, Tuple, Any
from liars_dice.core.game import Observation, LiarsDiceGame
from liars_dice.agents.base import Agent

class RandomAgent(Agent):
    def __init__(
        self,
        label : str = "Random",
        seed: Optional[int] = None
    ):
        self.name = label
        self.rng = random.Random(seed)
        

    def select_action(self, game: LiarsDiceGame, obs: Observation) -> Tuple[str, Any]:
        legal = game.legal_actions()
        if not legal:
            raise RuntimeError(
                f"{self.name} was asked to act but no legal actions were available."
            )
        return self.rng.choice(legal)
