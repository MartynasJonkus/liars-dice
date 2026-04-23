from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from liars_dice.core.game import Observation

BidEvent = Tuple[int, str, Tuple[int, int]]


@dataclass(frozen=True)
class ObservationEncoder:
    """
    Encodes a 4-player Liar's Dice observation into:
      1. static_features: flat non-sequential features
      2. bid_history: padded current-round bid token sequence
      3. bid_mask: padding mask for the sequence (True = padded token)

    Static features:
    - own dice face counts (normalized): 6
    - dice left per player (normalized): 4
    - alive flags per player: 4
    - current player one-hot: 4
    - observing player one-hot: 4
    - last bid quantity (normalized): 1
    - last bid face one-hot: 6
    - has-last-bid flag: 1

    Bid token features:
    - actor one-hot: 4
    - quantity normalized: 1
    - face one-hot: 6
    - delta quantity normalized: 1
    - same-face flag: 1

    Total token dim = 13
    """

    num_players: int = 4
    max_dice_per_player: int = 5
    max_total_dice: int = 20
    max_bids: int = 40

    @property
    def static_dim(self) -> int:
        return (
            6
            + self.num_players
            + self.num_players
            + self.num_players
            + self.num_players
            + 1
            + 6
            + 1
        )

    @property
    def bid_token_dim(self) -> int:
        return (
            self.num_players + 1 + 6 + 1 + 1
        )  # actor + q + face + delta_q + same_face

    def encode(self, obs: Observation) -> Dict[str, List[Any]]:
        static_features = self.encode_static(obs)
        bid_history, bid_mask = self.encode_bid_history(obs)

        return {
            "static_features": static_features,
            "bid_history": bid_history,
            "bid_mask": bid_mask,
        }

    def encode_static(self, obs: Observation) -> List[float]:
        my_dice = list(obs.private.my_dice)
        dice_left = list(obs.public.dice_left)
        current_player = obs.public.current_player
        my_player = obs.private.my_player
        last_bid = obs.public.last_bid

        if len(dice_left) != self.num_players:
            raise ValueError(
                f"Encoder configured for {self.num_players} players, "
                f"but observation has {len(dice_left)} players."
            )

        features: List[float] = []

        # Own dice as normalized face counts
        own_total = max(1, len(my_dice))
        for face in range(1, 7):
            count = sum(1 for d in my_dice if d == face)
            features.append(count / own_total)

        # Public per-player dice counts
        for count in dice_left:
            features.append(count / float(self.max_dice_per_player))

        # Alive flags
        for count in dice_left:
            features.append(1.0 if count > 0 else 0.0)

        # Current player one-hot
        for pid in range(self.num_players):
            features.append(1.0 if pid == current_player else 0.0)

        # Observing player one-hot
        for pid in range(self.num_players):
            features.append(1.0 if pid == my_player else 0.0)

        # Last bid
        if last_bid is None:
            features.append(0.0)  # quantity
            features.extend([0.0] * 6)  # face one-hot
            features.append(0.0)  # has-last-bid flag
        else:
            q, face = last_bid
            features.append(q / float(self.max_total_dice))
            for f in range(1, 7):
                features.append(1.0 if f == face else 0.0)
            features.append(1.0)

        if len(features) != self.static_dim:
            raise AssertionError(
                f"Static feature length {len(features)} does not match static_dim {self.static_dim}"
            )

        return features

    def encode_bid_history(
        self, obs: Observation
    ) -> Tuple[List[List[float]], List[bool]]:
        history = list(obs.public.history)
        round_bids = self._current_round_bid_history(history)

        tokens: List[List[float]] = []
        prev_bid: Tuple[int, int] | None = None

        for event in round_bids[-self.max_bids :]:
            token = self._encode_bid_token(event, prev_bid)
            tokens.append(token)
            prev_bid = event[2]

        zero_token = [0.0] * self.bid_token_dim

        # Important: transformer cannot handle sequences where all tokens are masked.
        # So if the round history is empty, keep one dummy token unmasked.
        if len(tokens) == 0:
            padded_tokens = [zero_token] + [
                zero_token for _ in range(self.max_bids - 1)
            ]
            bid_mask = [False] + [True for _ in range(self.max_bids - 1)]
            return padded_tokens, bid_mask

        pad_len = self.max_bids - len(tokens)
        padded_tokens = tokens + [zero_token for _ in range(pad_len)]
        bid_mask = [False] * len(tokens) + [True] * pad_len

        return padded_tokens, bid_mask

    def _current_round_bid_history(
        self,
        history: Sequence[Tuple[Any, Any, Any]],
    ) -> List[BidEvent]:
        """
        Returns only bid events from the current round.

        Scans backward until the most recent showdown marker, then keeps only
        player bid events after that point.
        """
        current_round_reversed: List[BidEvent] = []

        for event in reversed(history):
            if isinstance(event, tuple) and len(event) >= 2 and event[0] == "showdown":
                break

            if (
                isinstance(event, tuple)
                and len(event) == 3
                and isinstance(event[0], int)
                and event[1] == "bid"
                and isinstance(event[2], tuple)
                and len(event[2]) == 2
            ):
                current_round_reversed.append(event)  # type: ignore[arg-type]

        current_round_reversed.reverse()
        return current_round_reversed

    def _encode_bid_token(
        self,
        event: BidEvent,
        prev_bid: Tuple[int, int] | None,
    ) -> List[float]:
        pid, kind, data = event

        out = [0.0] * self.bid_token_dim

        if not (
            isinstance(pid, int)
            and 0 <= pid < self.num_players
            and kind == "bid"
            and isinstance(data, tuple)
            and len(data) == 2
        ):
            return out

        q, face = data

        # actor one-hot
        out[pid] = 1.0

        qty_offset = self.num_players
        face_offset = qty_offset + 1
        delta_offset = face_offset + 6
        same_face_offset = delta_offset + 1

        # quantity normalized
        out[qty_offset] = q / float(self.max_total_dice)

        # face one-hot
        if 1 <= face <= 6:
            out[face_offset + (face - 1)] = 1.0

        # delta quantity normalized
        if prev_bid is None:
            delta_q = q
            same_face = 0.0
        else:
            prev_q, prev_face = prev_bid
            delta_q = q - prev_q
            same_face = 1.0 if face == prev_face else 0.0

        out[delta_offset] = delta_q / float(self.max_total_dice)
        out[same_face_offset] = same_face

        return out
