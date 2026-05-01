from __future__ import annotations
import math
from typing import Tuple, Any, List, Optional
from liars_dice.core.game import LiarsDiceGame, Observation, Bid

class HeuristicAgent:
    def __init__(
        self,
        label: str = "Heuristic",
        liar_threshold: float = 0.25,
        raise_min_prob: float = 0.55,
        aggression: float = 0.0,
        seed: Optional[int] = None
    ):
        self.name = label
        self.liar_threshold = liar_threshold
        self.raise_min_prob = raise_min_prob
        self.aggression = aggression

    def _total_dice(self, dice_left: List[int]) -> int:
        return sum(dice_left)

    def _unknown_dice(self, obs: Observation) -> int:
        return self._total_dice(obs.public.dice_left) - len(obs.private.my_dice)

    def _bid_truth_prob(self, obs: Observation, bid: Bid) -> float:
        q, f = bid
        k_mine = count_my_matches(obs.private.my_dice, f)
        need_from_others = max(0, q - k_mine)
        n_unknown = self._unknown_dice(obs)
        p = success_prob_per_die(f)
        return binom_tail_geq(n_unknown, p, need_from_others)

    def _opening_bid(self, obs: Observation) -> Bid:
        total = self._total_dice(obs.public.dice_left)
        n_unknown = self._unknown_dice(obs)
        my = obs.private.my_dice

        best_face = 2
        best_support = -1
        for f in range(2, 7):
            s = count_my_matches(my, f)
            if s > best_support:
                best_support = s
                best_face = f

        p = success_prob_per_die(best_face)
        exp_others = n_unknown * p

        q = max(1, int(best_support + max(0.0, exp_others - 0.5) + self.aggression))
        q = min(q, total)
        return (q, best_face)

    def select_action(self, game: LiarsDiceGame, obs: Observation) -> Tuple[str, Any]:
        total = self._total_dice(obs.public.dice_left)
        last_bid = obs.public.last_bid

        if last_bid is not None:
            prob_true = self._bid_truth_prob(obs, last_bid)
            if prob_true < self.liar_threshold:
                return ("liar", None)

            legal_bids = enumerate_legal_bids(total, last_bid)
            for b in legal_bids:
                if self._bid_truth_prob(obs, b) >= self.raise_min_prob:
                    return ("bid", b)

            return ("liar", None)

        opening = self._opening_bid(obs)
        return ("bid", opening)

    def notify_result(self, obs: Observation, info: dict) -> None:
        return

def binom_tail_geq(n: int, p: float, k: int) -> float:
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    q = 1.0 - p
    prob = 0.0
    for i in range(k, n + 1):
        prob += math.comb(n, i) * (p ** i) * (q ** (n - i))
    return prob

def count_my_matches(my_dice: List[int], face: int) -> int:
    if face == 1:
        return sum(1 for d in my_dice if d == 1)
    else:
        return sum(1 for d in my_dice if d == face or d == 1)

def success_prob_per_die(face: int) -> float:
    return (1.0 / 6.0) if face == 1 else (2.0 / 6.0)

def enumerate_legal_bids(total_dice: int, last_bid: Optional[Bid]) -> List[Bid]:
    bids: List[Bid] = []
    if last_bid is None:
        start_q = 1
        start_f = 1
    else:
        start_q = last_bid[0]
        start_f = last_bid[1] + 1
    for q in range(start_q, total_dice + 1):
        f_start = 1 if (last_bid is None or q > last_bid[0]) else start_f
        for f in range(f_start, 7):
            bids.append((q, f))
    return bids
