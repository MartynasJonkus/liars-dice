from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

from liars_dice.core.game import Observation


@dataclass(frozen=True)
class ObservationEncoder:
    """
    Encodes a 4-player Liar's Dice observation into a fixed-size feature vector.

    Feature layout:
    - own dice face counts (normalized): 6
    - dice left per player (normalized): 4
    - alive flags per player: 4
    - current player one-hot: 4
    - observing player one-hot: 4
    - last bid quantity (normalized): 1
    - last bid face one-hot: 6
    - has-last-bid flag: 1
    - recent current-round bid history: history_len * history_entry_dim

    Each history entry encodes only a bid event:
    - actor one-hot: 4
    - bid quantity (normalized): 1
    - bid face one-hot: 6
    """

    num_players: int = 4
    max_dice_per_player: int = 5
    max_total_dice: int = 20
    history_len: int = 5

    @property
    def history_entry_dim(self) -> int:
        return self.num_players + 1 + 6  # actor + quantity + face

    @property
    def input_dim(self) -> int:
        base_dim = (
            6  # own dice face counts
            + self.num_players  # dice left
            + self.num_players  # alive flags
            + self.num_players  # current player one-hot
            + self.num_players  # observing player one-hot
            + 1  # last bid quantity
            + 6  # last bid face one-hot
            + 1  # has-last-bid flag
        )
        return base_dim + self.history_len * self.history_entry_dim

    def encode(self, obs: Observation) -> List[float]:
        my_dice = list(obs.private.my_dice)
        dice_left = list(obs.public.dice_left)
        current_player = obs.public.current_player
        my_player = obs.private.my_player
        last_bid = obs.public.last_bid
        history = list(obs.public.history)

        if len(dice_left) != self.num_players:
            raise ValueError(
                f"Encoder configured for {self.num_players} players, "
                f"but observation has {len(dice_left)} players."
            )

        features: List[float] = []

        # Own dice as normalized face counts.
        own_total = max(1, len(my_dice))
        for face in range(1, 7):
            count = sum(1 for d in my_dice if d == face)
            features.append(count / own_total)

        # Public per-player dice counts and alive flags.
        for count in dice_left:
            features.append(count / float(self.max_dice_per_player))
        for count in dice_left:
            features.append(1.0 if count > 0 else 0.0)

        # Turn / seat identity.
        for pid in range(self.num_players):
            features.append(1.0 if pid == current_player else 0.0)
        for pid in range(self.num_players):
            features.append(1.0 if pid == my_player else 0.0)

        # Last bid.
        if last_bid is None:
            features.append(0.0)  # quantity
            features.extend([0.0] * 6)  # face
            features.append(0.0)  # has-last-bid
        else:
            q, face = last_bid
            features.append(q / float(self.max_total_dice))
            for f in range(1, 7):
                features.append(1.0 if f == face else 0.0)
            features.append(1.0)

        # Truncated recent current-round public bid history.
        round_bids = self._current_round_bid_history(history)
        recent = round_bids[-self.history_len :]
        pad = self.history_len - len(recent)

        features.extend([0.0] * (pad * self.history_entry_dim))
        for event in recent:
            features.extend(self._encode_bid_event(event))

        if len(features) != self.input_dim:
            raise AssertionError(
                f"Encoded feature length {len(features)} does not match input_dim {self.input_dim}"
            )

        return features

    def _current_round_bid_history(
        self,
        history: Sequence[Tuple[Any, Any, Any]],
    ) -> List[Tuple[Any, Any, Any]]:
        """
        Returns only bid events from the current round.

        Scans backward until the most recent showdown marker, then keeps only
        player bid events after that point.
        """
        current_round_reversed: List[Tuple[Any, Any, Any]] = []

        for event in reversed(history):
            if isinstance(event, tuple) and len(event) >= 2 and event[0] == "showdown":
                break

            if (
                isinstance(event, tuple)
                and len(event) == 3
                and isinstance(event[0], int)
                and event[1] == "bid"
            ):
                current_round_reversed.append(event)

        current_round_reversed.reverse()
        return current_round_reversed

    def _encode_bid_event(self, event: Tuple[Any, Any, Any]) -> List[float]:
        """
        Encodes a single bid event into a fixed-size history entry.
        Non-bid or malformed events are mapped to zeros defensively.
        """
        out = [0.0] * self.history_entry_dim
        pid, kind, data = event

        if not (
            isinstance(pid, int)
            and 0 <= pid < self.num_players
            and kind == "bid"
            and isinstance(data, tuple)
            and len(data) == 2
        ):
            return out

        qty_offset = self.num_players
        face_offset = qty_offset + 1

        # Actor one-hot.
        out[pid] = 1.0

        q, face = data
        out[qty_offset] = q / float(self.max_total_dice)

        if 1 <= face <= 6:
            out[face_offset + (face - 1)] = 1.0

        return out
