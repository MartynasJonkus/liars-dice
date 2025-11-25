from liars_dice.core.game import LiarsDiceGame, Bid
import math

_precomputed_binom = {}

def _precompute_binomials(max_n=25):
    for n in range(max_n + 1):
        for p in (1.0/6.0, 2.0/6.0):
            key_p = round(p, 3)
            for k in range(max_n + 1):
                if k <= 0:
                    val = 1.0
                elif k > n:
                    val = 0.0
                else:
                    q = 1.0 - p
                    prob = 0.0
                    for i in range(k, n + 1):
                        prob += math.comb(n, i) * (p**i) * (q**(n-i))
                    val = prob
                _precomputed_binom[(n, key_p, k)] = val

_precompute_binomials()

def bid_support_for_actor(game: LiarsDiceGame, actor: int, bid: Bid) -> float:
    q, face = bid
    my_dice = game._dice[actor]

    if face == 1:
        matching_mine = sum(1 for d in my_dice if d == 1)
        p_face = 1.0 / 6.0
    else:
        matching_mine = sum(1 for d in my_dice if d == face or d == 1)
        p_face = 2.0 / 6.0

    need_from_others = max(0, q - matching_mine)
    n_unknown = sum(game._dice_left) - len(my_dice)

    if need_from_others <= 0:
        return 1.0
    if need_from_others > n_unknown:
        return 0.0
    
    return _precomputed_binom[(n_unknown, round(p_face, 3), need_from_others)]
