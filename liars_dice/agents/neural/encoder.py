from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

from liars_dice.core.game import Observation


@dataclass(frozen=True)
class ObservationEncoder:
    """Encodes a 4-player Liar's Dice observation into a fixed-size feature vector.

    Feature layout:
        0..5   : own dice face counts, normalized by current own dice count
        6..9   : dice left per player, normalized by max_dice_per_player
        10..13 : alive indicators for each player
        14..17 : current player one-hot
        18..21 : observing player one-hot
        22     : last bid quantity, normalized by max_total_dice
        23..28 : last bid face one-hot (all zeros if no bid)
        29     : has last bid flag
        30..99 : recent history entries (K=5 by default), each entry contributes 14:
                 - actor one-hot (4)
                 - action type one-hot: bid / liar / showdown (3)
                 - bid quantity normalized (1)
                 - bid face one-hot (6)
                 showdown entries are ignored and encoded as zeros except type flag.
    """

    num_players: int = 4
    max_dice_per_player: int = 5
    max_total_dice: int = 20
    history_len: int = 5

    @property
    def history_entry_dim(self) -> int:
        return self.num_players + 3 + 1 + 6

    @property
    def input_dim(self) -> int:
        base_dim = (
            6
            + self.num_players
            + self.num_players
            + self.num_players
            + self.num_players
            + 1
            + 6
            + 1
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
                f"Encoder expects {self.num_players} players, got {len(dice_left)}"
            )

        features: List[float] = []

        # Own dice counts by face as a fraction of total own dice
        own_total = max(1, len(my_dice))
        for face in range(1, 7):
            count = sum(1 for d in my_dice if d == face)
            features.append(count / own_total)

        # Public dice counts as a fraction of maximum dice per player
        for count in dice_left:
            features.append(count / float(self.max_dice_per_player))

        # Indicator of alive players
        for count in dice_left:
            features.append(1.0 if count > 0 else 0.0)

        # Current turn player one-hot
        for pid in range(self.num_players):
            features.append(1.0 if pid == current_player else 0.0)

        # Observing player one-hot
        for pid in range(self.num_players):
            features.append(1.0 if pid == my_player else 0.0)

        # Last bid
        if last_bid is None:
            features.append(0.0)
            features.extend([0.0] * 6)
            features.append(0.0)
        else:
            q, face = last_bid
            features.append(q / float(self.max_total_dice))
            for f in range(1, 7):
                features.append(1.0 if f == face else 0.0)
            features.append(1.0)

        # Truncated recent current-round public bid history.
        round_history = self._current_round_history(history)
        recent = round_history[-self.history_len :]
        pad = self.history_len - len(recent)
        features.extend([0.0] * (pad * self.history_entry_dim))
        for event in recent:
            features.extend(self._encode_history_event(event))

        if len(features) != self.input_dim:
            raise AssertionError(
                f"Unexpected feature length {len(features)} != {self.input_dim}"
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

        kind_offset = self.num_players
        qty_offset = self.num_players + 3
        face_offset = qty_offset + 1

        # Actor one-hot.
        out[pid] = 1.0

        # Event type one-hot. Only bid is used, but kept for stable shape.
        out[kind_offset + 0] = 1.0

        q, face = data
        out[qty_offset] = q / float(self.max_total_dice)

        if 1 <= face <= 6:
            out[face_offset + (face - 1)] = 1.0

        return out
