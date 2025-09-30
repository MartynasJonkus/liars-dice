from __future__ import annotations
import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from liars_dice.core.game import LiarsDiceGame, Observation, Bid
from liars_dice.agents.heuristic import (
    count_my_matches,
    success_prob_per_die,
    enumerate_legal_bids,
    binom_tail_geq,
)

Action = Tuple[str, Any]
NodeKey = Tuple[int, Optional[Bid], Tuple[int, ...], int]  # (player, last_bid, dice_left, history_len)

@dataclass
class EdgeStats:
    N: int = 0
    W: float = 0.0
    @property
    def Q(self) -> float:
        return 0.0 if self.N == 0 else self.W / self.N

@dataclass
class Node:
    key: NodeKey
    player: int
    untried: List[Action]
    children: Dict[Action, 'Node'] = field(default_factory=dict)
    edges: Dict[Action, EdgeStats] = field(default_factory=dict)
    N: int = 0

class ISMCTSAgent:
    def __init__(self, label: str = "ISMCTS", sims_per_move: int = 2000,
                 uct_c: float = math.sqrt(2), pw_alpha: float = 0.5, pw_c: float = 1.5,
                 seed: Optional[int] = None):
        self.name = label
        self.sims_per_move = sims_per_move
        self.uct_c = uct_c
        self.pw_alpha = pw_alpha
        self.pw_c = pw_c
        self.rng = random.Random(seed)

    def select_action(self, game: LiarsDiceGame, obs: Observation) -> Action:
        root_player = obs.private.my_player
        root_key = self._node_key_from_obs(obs)
        total_dice = sum(obs.public.dice_left)
        root_untried = self._ordered_actions_by_plausibility(obs, total_dice, obs.public.last_bid)
        root = Node(key=root_key, player=obs.public.current_player, untried=root_untried)

        for _ in range(self.sims_per_move):
            # 1) Determinize from the live game by deep-copying and redrawing others' dice
            g_det = self._determinize_from_game(game, obs)

            # 2) Selection + Progressive Widening
            path: List[Tuple[Node, Action]] = []
            node = root
            while True:
                limit = int(max(1, self.pw_c * (node.N ** self.pw_alpha)))
                if len(node.children) < limit and node.untried:
                    a = node.untried.pop(0)
                    g_det.step(a)
                    child_obs = g_det.observe(g_det._current)
                    child_key = self._node_key_from_obs(child_obs)
                    child_untried = self._ordered_actions_by_plausibility(
                        child_obs, sum(child_obs.public.dice_left), child_obs.public.last_bid
                    )
                    child = Node(key=child_key, player=child_obs.public.current_player, untried=child_untried)
                    node.children[a] = child
                    node.edges.setdefault(a, EdgeStats())
                    path.append((node, a))
                    node = child
                    break
                if not node.children:
                    break
                a = self._select_uct(node)
                path.append((node, a))
                g_det.step(a)
                node = node.children[a]

            # 3) Rollout
            reward = self._rollout_to_terminal(g_det, root_player)

            # 4) Backup
            self._backup(path, reward)

        if not root.children:
            return root_untried[0]
        best_a = max(root.children.keys(), key=lambda a: root.edges[a].N)
        return best_a

    def notify_result(self, obs: Observation, info: dict) -> None:
        return

    # ---- helpers ----
    def _node_key_from_obs(self, obs: Observation) -> NodeKey:
        return (obs.public.current_player, obs.public.last_bid,
                tuple(obs.public.dice_left), len(obs.public.history))

    def _ordered_actions_by_plausibility(self, obs: Observation, total_dice: int,
                                         last_bid: Optional[Bid]) -> List[Action]:
        bids: List[Bid] = enumerate_legal_bids(total_dice, last_bid)
        scored = [ (self._bid_truth_prob(obs, b), b) for b in bids ]
        scored.sort(key=lambda x: (-x[0], x[1][0], x[1][1]))
        actions: List[Action] = [("bid", b) for _, b in scored]
        if last_bid is not None:
            actions.append(("liar", None))
        return actions

    def _bid_truth_prob(self, obs: Observation, bid: Bid) -> float:
        q, f = bid
        k_mine = count_my_matches(obs.private.my_dice, f)
        need_from_others = max(0, q - k_mine)
        n_unknown = sum(obs.public.dice_left) - len(obs.private.my_dice)
        p = success_prob_per_die(f)
        return binom_tail_geq(n_unknown, p, need_from_others)

    def _determinize_from_game(self, game: LiarsDiceGame, obs: Observation) -> LiarsDiceGame:
        g = copy.deepcopy(game)
        for pid in range(g.num_players):
            if pid == obs.private.my_player:
                continue
            n = g._dice_left[pid]
            g._dice[pid] = [self.rng.randint(1, 6) for _ in range(n)]
        return g

    def _select_uct(self, node: Node) -> Action:
        logN = math.log(max(1, node.N))
        best, best_val = None, -1e9
        for a in node.children.keys():
            e = node.edges[a]
            u = e.Q + self.uct_c * math.sqrt(logN / (e.N + 1))
            if u > best_val:
                best_val, best = u, a
        return best

    def _rollout_to_terminal(self, g: LiarsDiceGame, root_player: int) -> float:
        while True:
            if g.num_alive() <= 1:
                winner = g._winner()
                return 1.0 if winner == root_player else 0.0
            
            pid = g._current
            obs = g.observe(pid)
            legal = g.legal_actions()

            if not legal:
                winner = g._winner()
                return 1.0 if winner == root_player else 0.0
            
            last = obs.public.last_bid
            if last is not None:
                if self._bid_truth_prob(obs, last) < 0.25:
                    a = ("liar", None)
                    if a not in legal:
                        a = self.rng.choice(legal)
                else:
                    bids = enumerate_legal_bids(sum(obs.public.dice_left), last)
                    picked = None
                    for b in bids:
                        if ("bid", b) in legal and self._bid_truth_prob(obs, b) >= 0.5:
                            picked = ("bid", b)
                            break
                    a = picked if picked else self.rng.choice(legal)
            else:
                my = obs.private.my_dice
                best_face = max(range(2, 7), key=lambda f: count_my_matches(my, f))
                n_unknown = sum(obs.public.dice_left) - len(my)
                exp = n_unknown * success_prob_per_die(best_face)
                q = max(1, min(sum(obs.public.dice_left),
                        int(count_my_matches(my, best_face) + max(0.0, exp - 0.5))))
                a = ("bid", (q, best_face))
                if a not in legal:
                    a = self.rng.choice(legal)

            info = g.step(a)
            if info.get("terminal"):
                return 1.0 if info["winner"] == root_player else 0.0

    def _backup(self, path: List[Tuple[Node, Action]], reward: float) -> None:
        for node, a in path:
            node.N += 1
            e = node.edges.setdefault(a, EdgeStats())
            e.N += 1
            e.W += reward
