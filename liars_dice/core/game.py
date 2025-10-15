
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

# --- Simple Liar's Dice rules ---
# - N players, each starts with D dice (default 5)
# - Faces 1..6, 1s are wild (count as any face) for showdown
# - On your turn: either raise the bid (quantity, face) or call (liar)
# - A new bid must be strictly higher using (quantity, face) ordering:
#       (q2, f2) > (q1, f1) if q2 > q1 OR (q2 == q1 and f2 > f1)
# - When "liar" is called, count dice that match the face OR are 1s; 
#       if count >= quantity, caller loses a die; else previous bidder loses a die.
# - Round ends after liar, dice are re-rolled for remaining players.
# - Game ends when one player has all other players at 0 dice (last with dice wins).

Bid = Tuple[int, int]  # (quantity, face 1..6)

@dataclass
class PublicState:
    num_players: int
    dice_left: List[int]
    current_player: int
    last_bid: Optional[Bid]
    history: List[Tuple[int, str, Any]]  # (player_id, "bid"/"liar", data)

@dataclass
class PrivateInfo:
    my_player: int
    my_dice: List[int]

@dataclass
class Observation:
    public: PublicState
    private: PrivateInfo

class LiarsDiceGame:
    def __init__(self, num_players:int=2, dice_per_player:int=5, seed:Optional[int]=None):
        assert 2 <= num_players <= 6, "Supported players: 2..6"
        self.num_players = num_players
        self.dice_per_player = dice_per_player
        self.rng = random.Random(seed)

        self._dice: List[List[int]] = [[] for _ in range(num_players)]
        self._dice_left: List[int] = [dice_per_player for _ in range(num_players)]
        self._current: int = 0
        self._last_bid: Optional[Bid] = None
        self._history: List[Tuple[int, str, Any]] = []
        self._round_active: bool = True
        self._roll_all()

    # --- Helpers ---
    def _roll_all(self):
        for pid in range(self.num_players):
            n = self._dice_left[pid]
            self._dice[pid] = [self.rng.randint(1,6) for _ in range(n)]

    @staticmethod
    def _is_higher(bid: Bid, than: Optional[Bid]) -> bool:
        if than is None:
            return True
        (q2, f2) = bid
        (q1, f1) = than
        return (q2 > q1) or (q2 == q1 and f2 > f1)

    def legal_actions(self) -> List[Tuple[str, Any]]:
        actions: List[Tuple[str, Any]] = []
        if self.num_alive() <= 1:
            return actions

        total_dice = sum(self._dice_left)
        start_q = 1 if self._last_bid is None else self._last_bid[0]
        start_f = 1 if self._last_bid is None else self._last_bid[1] + 1

        for q in range(start_q, total_dice + 1):
            f_start = 1 if (self._last_bid is None or q > self._last_bid[0]) else start_f
            for f in range(f_start, 7):
                actions.append(("bid", (q, f)))
        
        if self._last_bid is not None:
            actions.append(("liar", None))
        return actions

    def observe(self, pid:int) -> Observation:
        pub = PublicState(
            num_players = self.num_players,
            dice_left = list(self._dice_left),
            current_player = self._current,
            last_bid = self._last_bid,
            history = list(self._history),
        )
        priv = PrivateInfo(
            my_player = pid,
            my_dice = list(self._dice[pid])
        )
        return Observation(public = pub, private = priv)

    def _count_face(self, face:int) -> int:
        count = 0
        for pid in range(self.num_players):
            for v in self._dice[pid]:
                if v == face or v == 1:  # 1s are wild
                    count += 1
        return count

    def _advance_player(self):
        if self.num_alive() <= 1:
            return
        nxt = (self._current + 1) % self.num_players
        while self._dice_left[nxt] == 0:
            nxt = (nxt + 1) % self.num_players
        self._current = nxt

    def step(self, action: Tuple[str, Any]) -> Dict[str, Any]:
        if self.num_alive() <= 1:
            return {"terminal": True, "winner": self._winner()}
    
        kind, payload = action
        pid = self._current

        if kind == "bid":
            bid: Bid = payload
            assert self._is_higher(bid, self._last_bid), "Illegal bid"
            self._last_bid = bid
            self._history.append((pid, "bid", bid))
            self._advance_player()
            return {"terminal": False}

        elif kind == "liar":
            assert self._last_bid is not None, "Cannot call Liar! before any bid"
            self._history.append((pid, "liar", None))

            q, f = self._last_bid
            actual = self._count_face(f)
            previous_bidder = self._previous_live_player(pid)
            caller = pid

            dice_snapshot = [list(d) for d in self._dice]

            loser = pid if actual >= q else previous_bidder
            self._dice_left[loser] = max(0, self._dice_left[loser] - 1)
            
            self._history.append((
                "showdown",
                "count",
                {
                    "bid": (q, f),
                    "actual": actual,
                    "loser": loser,
                    "dice": dice_snapshot,
                    "previous_bidder": previous_bidder,
                    "caller": caller,
                }
            ))

            if self.num_alive() == 1:
                return {"terminal": True, "winner": self._winner()}
            
            
            self._current = loser if self._dice_left[loser] > 0 else self._next_live_player(loser)
            self._last_bid = None
            self._roll_all()
            return {"terminal": False}

        else:
            raise ValueError(f"Unknown action {kind}")

    def num_alive(self) -> int:
        return sum(1 for n in self._dice_left if n > 0)

    def _winner(self) -> Optional[int]:
        alive = [i for i, n in enumerate(self._dice_left) if n > 0]
        return alive[0] if len(alive) == 1 else None

    def _previous_live_player(self, pid:int) -> int:
        i = (pid - 1) % self.num_players
        while self._dice_left[i] == 0:
            i = (i - 1) % self.num_players
        return i

    def _next_live_player(self, pid:int) -> int:
        i = (pid + 1) % self.num_players
        while self._dice_left[i] == 0:
            i = (i + 1) % self.num_players
        return i
