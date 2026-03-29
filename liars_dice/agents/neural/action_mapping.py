from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

Action = Tuple[str, Any]
Bid = Tuple[int, int]


@dataclass(frozen=True)
class ActionMapper:
    """Maps Liar's Dice actions to a fixed policy index space.

    For the bachelor thesis implementation we assume 4 players with 5 dice each,
    so the maximum total number of dice at any point in the game is 20.
    The output policy space therefore contains:
        - 20 * 6 bid actions  -> (q, f) for q in [1, 20], f in [1, 6]
        - 1 liar action
    Total: 121 actions.
    """

    max_total_dice: int = 20

    def __post_init__(self) -> None:
        if self.max_total_dice <= 0:
            raise ValueError("max_total_dice must be positive")

        bid_actions: List[Action] = []
        action_to_index: Dict[Action, int] = {}
        index_to_action: Dict[int, Action] = {}

        idx = 0
        for q in range(1, self.max_total_dice + 1):
            for face in range(1, 7):
                action = ("bid", (q, face))
                bid_actions.append(action)
                action_to_index[action] = idx
                index_to_action[idx] = action
                idx += 1

        liar_action = ("liar", None)
        action_to_index[liar_action] = idx
        index_to_action[idx] = liar_action

        object.__setattr__(self, "num_actions", idx + 1)
        object.__setattr__(self, "bid_actions", bid_actions)
        object.__setattr__(self, "liar_index", idx)
        object.__setattr__(self, "_action_to_index", action_to_index)
        object.__setattr__(self, "_index_to_action", index_to_action)

    def action_to_index(self, action: Action) -> int:
        try:
            return self._action_to_index[action]
        except KeyError as exc:
            raise KeyError(
                f"Action {action!r} is outside the fixed action space"
            ) from exc

    def index_to_action(self, index: int) -> Action:
        try:
            return self._index_to_action[index]
        except KeyError as exc:
            raise KeyError(f"Invalid policy index {index}") from exc

    def legal_action_mask(self, legal_actions: List[Action]) -> List[float]:
        mask = [0.0] * self.num_actions
        for action in legal_actions:
            if action in self._action_to_index:
                mask[self._action_to_index[action]] = 1.0
        return mask

    def legal_indices(self, legal_actions: List[Action]) -> List[int]:
        return [
            self._action_to_index[a]
            for a in legal_actions
            if a in self._action_to_index
        ]

    def decode_best_legal(
        self, probs: List[float], legal_actions: List[Action]
    ) -> Action:
        legal_indices = self.legal_indices(legal_actions)
        if not legal_indices:
            raise ValueError("No legal actions available to decode")
        best_idx = max(legal_indices, key=lambda i: probs[i])
        return self.index_to_action(best_idx)
