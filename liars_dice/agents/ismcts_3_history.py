from __future__ import annotations
from typing import Tuple, Any, Protocol
from liars_dice.core.game import Observation, LiarsDiceGame

class ISMCTSHistoryAgent(Protocol):
    label: str = "ISMCTS-History",
    name: str

    def select_action(self, game: LiarsDiceGame, obs: Observation) -> Tuple[str, Any]:
        ...
    def notify_result(self, obs: Observation, info: dict) -> None:
        ...
