from __future__ import annotations
import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from liars_dice.core.game import LiarsDiceGame, Observation, Bid
from helpers import bid_support_for_actor

Action = Tuple[str, Any]
NodeKey = Tuple[int, Optional[Bid], Tuple[int, ...], int]

@dataclass
class EdgeStats:
    visit_count: int = 0
    value_sum: float = 0.0
    @property
    def avg_value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

@dataclass
class Node:
    key: NodeKey
    player: int
    untried: List[Action]
    children: Dict[Action, 'Node'] = field(default_factory = dict)
    edges: Dict[Action, EdgeStats] = field(default_factory = dict)
    visit_count: int = 0

class ISMCTSHeuristicAgent:
    def __init__(
        self,
        label: str = "ISMCTS-Heuristic",
        sims_per_move: int = 2000,
        uct_c: float = math.sqrt(2.0),
        seed: Optional[int] = None,

        rollout_theta: float = 0.40,   # call if current bid support < theta
        rollout_alpha: float = 0.72,   # target plausibility for own raise
        rollout_eps: float = 0.08,     # small random raise chance
        rollout_max_steps: int = 40,
    ):
        self.name = label
        self.sims_per_move = sims_per_move
        self.uct_c = uct_c
        self.rng = random.Random(seed)

        self.rollout_theta = rollout_theta
        self.rollout_alpha = rollout_alpha
        self.rollout_eps = rollout_eps
        self.rollout_max_steps = rollout_max_steps

    def select_action(self, game: LiarsDiceGame, obs: Observation) -> Action:
        def is_liar(a: Action) -> bool:
            return isinstance(a, tuple) and a[0] == "liar"

        root_player = obs.private.my_player
        root_key = self._node_key_from_obs(obs)
        root_actions = list(game.legal_actions())
        root = Node(key=root_key, player=obs.public.current_player, untried=list(root_actions))

        for _ in range(self.sims_per_move):
            g_det = self._determinize_from_game(game, obs)

            node = root
            path: List[Tuple[Node, Action]] = []
            terminated_in_tree = False

            while True:
                if node.untried:
                    a = self.rng.choice(node.untried)
                    node.untried.remove(a)

                    if is_liar(a):
                        before = list(g_det._dice_left)
                        g_det.step(a)
                        after = g_det._dice_left
                        root_lost = (after[root_player] < before[root_player])
                        reward = 0.0 if root_lost else 1.0

                        path.append((node, a))
                        self._backup(path, reward)
                        terminated_in_tree = True
                        break

                    g_det.step(a)
                    child_obs = g_det.observe(g_det._current)
                    child_key = self._node_key_from_obs(child_obs)
                    child_untried = list(g_det.legal_actions())
                    child = Node(key=child_key, player=child_obs.public.current_player, untried=child_untried)

                    node.children[a] = child
                    node.edges.setdefault(a, EdgeStats())
                    path.append((node, a))
                    node = child
                    break

                if not node.children:
                    break

                a = self._select_uct(node, list(node.children.keys()))

                if is_liar(a):
                    before = list(g_det._dice_left)
                    after = g_det._dice_left
                    root_lost = (after[root_player] < before[root_player])
                    reward = 0.0 if root_lost else 1.0

                    path.append((node, a))
                    self._backup(path, reward)
                    terminated_in_tree = True
                    break

                g_det.step(a)
                node = node.children[a]

            if terminated_in_tree:
                continue

            reward = self._rollout_to_showdown(g_det, root_player)

            self._backup(path, reward)

        legal_now = list(game.legal_actions())

        if root.edges:
            scored = [(a, e.visit_count) for a, e in root.edges.items() if a in legal_now]
            if scored:
                best_visits = max(v for _, v in scored)
                candidates = [a for a, v in scored if v == best_visits]
                return self.rng.choice(candidates)

        if legal_now:
            return self.rng.choice(legal_now)

        return ("liar", None)

    def notify_result(self, obs: Observation, info: dict) -> None:
        return

    def _node_key_from_obs(self, obs: Observation) -> NodeKey:
        return (
            obs.public.current_player,
            obs.public.last_bid,
            tuple(obs.public.dice_left),
            len(obs.public.history),
        )

    def _determinize_from_game(self, game: LiarsDiceGame, obs: Observation) -> LiarsDiceGame:
        g = copy.deepcopy(game)
        for pid in range(g.num_players):
            if pid == obs.private.my_player:
                continue
            n = g._dice_left[pid]
            g._dice[pid] = [self.rng.randint(1, 6) for _ in range(n)]
        return g

    def _select_uct(self, node: Node, legal_children: List[Action]) -> Action:
        logN = math.log(max(1, node.visit_count))
        best, best_val = None, -1e9
        for action in legal_children:
            edge = node.edges[action]
            uct = edge.avg_value + self.uct_c * math.sqrt(logN / (edge.visit_count + 1))
            if uct > best_val:
                best_val, best = uct, action
        return best

    def _backup(self, path: List[Tuple[Node, Action]], reward: float) -> None:
        for node, action in path:
            node.visit_count += 1
            edge = node.edges.setdefault(action, EdgeStats())
            edge.visit_count += 1
            edge.value_sum += reward

    def _rollout_to_showdown(self, game: LiarsDiceGame, root_player: int) -> float:
        if game.num_alive() <= 1:
            winner = game._winner()
            return 1.0 if winner == root_player else 0.0
        
        start_counts = list(game._dice_left)
        steps = 0

        while True:
            steps += 1
            if steps > self.rollout_max_steps:
                winner = game._winner()
                if winner is not None:
                    return 1.0 if winner == root_player else 0.0
                return 0.5

            legal = game.legal_actions()
            if not legal:
                raise RuntimeError("No legal actions available in non-terminal state")

            actor = game._current
            obs = game.observe(actor)
            last_bid = obs.public.last_bid

            if self.rng.random() < self.rollout_eps:
                action = self.rng.choice(legal)
                info = game.step(action)
                if isinstance(action, tuple) and action[0] == "liar":
                    end_counts = game._dice_left
                    root_lost = end_counts[root_player] < start_counts[root_player]
                    return 0.0 if root_lost else 1.0
                if info.get("terminal"):
                    return 1.0 if info.get("winner") == root_player else 0.0
                continue

            if last_bid is not None:
                support_last = bid_support_for_actor(game, actor, last_bid)
                if support_last < self.rollout_theta:
                    action = ("liar", None)
                    info = game.step(action)
                    end_counts = game._dice_left
                    root_lost = end_counts[root_player] < start_counts[root_player]
                    return 0.0 if root_lost else 1.0

                candidate = None
                min_q = None
                for action in legal:
                    if isinstance(action, tuple) and action[0] == "bid":
                        q, f = action[1]
                        support = bid_support_for_actor(game, actor, (q, f))
                        if support >= self.rollout_alpha and (min_q is None or q < min_q):
                            min_q = q
                            candidate = action
                if candidate is None:
                    action = ("liar", None)
                    info = game.step(action)
                    end_counts = game._dice_left
                    root_lost = end_counts[root_player] < start_counts[root_player]
                    return 0.0 if root_lost else 1.0
            else:
                candidate = None
                min_q = None
                for action in legal:
                    if isinstance(action, tuple) and action[0] == "bid":
                        q, f = action[1]
                        support = bid_support_for_actor(game, actor, (q, f))
                        if support >= self.rollout_alpha and (min_q is None or q < min_q):
                            min_q = q
                            candidate = action
                if candidate is None:
                    bids = [action for action in legal if isinstance(action, tuple) and action[0] == "bid"]
                    candidate = bids[0] if bids else self.rng.choice(legal)

            info = game.step(candidate)
            if info.get("terminal"):
                return 1.0 if info.get("winner") == root_player else 0.0
            continue


