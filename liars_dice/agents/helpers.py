
from liars_dice.core.game import LiarsDiceGame, Bid
import math

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
    return binom_tail_geq(n_unknown, p_face, need_from_others)

def binom_tail_geq(total_count: int, p_face: float, need_count: int) -> float:
    if need_count <= 0:
        return 1.0
    if need_count > total_count:
        return 0.0
    
    q = 1.0 - p_face
    prob = 0.0
    for i in range(need_count, total_count + 1):
        prob += math.comb(total_count, i) * (p_face ** i) * (q ** (total_count - i))
    return prob