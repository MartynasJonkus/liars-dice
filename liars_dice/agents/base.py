
from __future__ import annotations
from typing import Tuple, Any, Protocol
from liars_dice.core.game import Observation

class Agent(Protocol):
    name: str
    
    def select_action(self, obs: Observation) -> Tuple[str, Any]:
        ...
    def notify_result(self, obs: Observation, info: dict) -> None:
        ...
