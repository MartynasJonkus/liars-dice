
from __future__ import annotations
"""
Baseline heuristic agent for multi-player Liar's Dice (no MCTS optimizations).

- Deterministic, fast, and simple: good as an external opponent baseline.
- Uses only public info and the acting player's private dice.
- No learning, no history modeling, no randomness (except optional tie-breaking hook).

Public API:
  - class BaselineAgent
  - helper functions used elsewhere (binom_tail_geq, count_my_matches, success_prob_per_die, enumerate_legal_bids)
"""

import math
from typing import Tuple, Any, List, Optional

# Type aliases to avoid a hard import dependency on your package structure.
Bid = Tuple[int, int]  # (quantity, face 1..6)

class BaselineAgent:
    """A minimal, readable heuristic agent.

    Parameters
    ----------
    label : str
        Human-readable name.
    liar_threshold : float
        Call "liar" if the current bid looks this unlikely for the acting player.
    raise_min_prob : float
        Target plausibility for our own raise (choose minimal dominating raise that meets it).
    aggression : float
        Small bump to opening quantity (0..1). Set 0 for neutral play.
    """
    def __init__(
        self,
        label: str = "HeuristicBaseline",
        liar_threshold: float = 0.25,
        raise_min_prob: float = 0.55,
        aggression: float = 0.0,
    ) -> None:
        self.name = label
        self.liar_threshold = float(liar_threshold)
        self.raise_min_prob = float(raise_min_prob)
        self.aggression = float(aggression)

    # --- Public interface expected by the runner ---
    def select_action(self, game, obs) -> Tuple[str, Any]:
        """Return an action tuple: ("bid", (q, f)) or ("liar", None).

        The 'game' and 'obs' objects are assumed to match your engine:
          - obs.public.dice_left : List[int]
          - obs.public.last_bid  : Optional[Bid]
          - obs.private.my_dice  : List[int]
        """
        total = _total_dice(obs.public.dice_left)
        last_bid = obs.public.last_bid

        # If there is a current bid, decide whether to call liar.
        if last_bid is not None:
            prob_true = _bid_truth_prob(obs, last_bid)
            if prob_true < self.liar_threshold:
                return ("liar", None)

            # Otherwise try the minimal dominating raise that meets plausibility
            for b in enumerate_legal_bids(total, last_bid):
                # Skip strictly dominated / impossible bids
                if _is_impossible_bid(obs, b):
                    continue
                if _bid_truth_prob(obs, b) >= self.raise_min_prob:
                    return ("bid", b)

            # No plausible raise found -> Liar!
            return ("liar", None)

        # No previous bid: open with a conservative face and quantity
        opening = _opening_bid(obs, total, self.aggression)
        return ("bid", opening)

    def notify_result(self, obs, info: dict) -> None:
        # Baseline agent does not learn or record outcomes.
        return

# --------- Helper functions (standalone, no engine imports) ---------

def _total_dice(dice_left: List[int]) -> int:
    return sum(dice_left)

def _unknown_dice(obs) -> int:
    return _total_dice(obs.public.dice_left) - len(obs.private.my_dice)

def _opening_bid(obs, total_dice: int, aggression: float) -> Bid:
    """Pick a face we personally support best; set quantity conservatively.

    Prefers non-1 faces (since 1s are wild for non-1 faces in showdown).
    """
    my = obs.private.my_dice
    # Choose the non-1 face we support most (ties broken by lower face value)
    best_face = max(range(2, 7), key=lambda f: count_my_matches(my, f))

    # Expected support from unknown dice for that face
    n_unknown = _unknown_dice(obs)
    p = success_prob_per_die(best_face)  # 2/6 for non-1 faces under wild-1s rule
    exp_from_others = n_unknown * p

    # Quantity: our support + conservative slice of others' expectation + small aggression
    my_support = count_my_matches(my, best_face)
    q = max(1, int(my_support + max(0.0, exp_from_others - 0.5) + aggression))
    q = min(q, total_dice)  # never exceed table dice
    return (q, best_face)

def _bid_truth_prob(obs, bid: Bid) -> float:
    q, f = bid
    k_mine = count_my_matches(obs.private.my_dice, f)
    need_from_others = max(0, q - k_mine)
    n_unknown = _unknown_dice(obs)
    p = success_prob_per_die(f)
    return binom_tail_geq(n_unknown, p, need_from_others)

def _is_impossible_bid(obs, bid: Bid) -> bool:
    """Fast upper bound: can't need more than k_mine + n_unknown."""
    q, f = bid
    k_mine = count_my_matches(obs.private.my_dice, f)
    n_unknown = _unknown_dice(obs)
    return q > (k_mine + n_unknown)

# ---- Math / rules helpers reused elsewhere (kept here for single source of truth) ----

def binom_tail_geq(n: int, p: float, k: int) -> float:
    """P[X >= k] for X ~ Bin(n, p). Naive exact sum is fine for n <= ~25."""
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
    """How many of *my* dice support 'face' under the common wild-1s rule."""
    if face == 1:
        return sum(1 for d in my_dice if d == 1)
    else:
        return sum(1 for d in my_dice if d == face or d == 1)

def success_prob_per_die(face: int) -> float:
    """Probability a single unknown die supports 'face' (1s wild for non-1 faces)."""
    return (1.0 / 6.0) if face == 1 else (2.0 / 6.0)

def enumerate_legal_bids(total_dice: int, last_bid: Optional[Bid]) -> List[Bid]:
    """Enumerate legal bids in engine order (monotone lattice).

    (q2, f2) > (q1, f1) iff q2 > q1 OR (q2 == q1 and f2 > f1).
    Matches the runner's canonical order so agents aren't sensitive to internal differences.
    """
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
