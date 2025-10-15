
from __future__ import annotations
import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from liars_dice.core.game import LiarsDiceGame, Observation, Bid

Action = Tuple[str, Any]
NodeKey = Tuple[int, Optional[Bid], Tuple[int, ...], int]  # (player, last_bid, dice_left, history_len)

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

class ISMCTSBasicAgent:
    def __init__(
        self,
        label: str = "ISMCTS-Basic",
        sims_per_move: int = 2000,
        uct_c: float = math.sqrt(2.0),
        seed: Optional[int] = None,
    ):
        self.name = label
        self.sims_per_move = sims_per_move
        self.uct_c = uct_c
        self.rng = random.Random(seed)

    # --- Public API ---
    # def select_action(self, game: LiarsDiceGame, obs: Observation) -> Action:
    #     root_player = obs.private.my_player
    #     root_key = self._node_key_from_obs(obs)
    #     root_untried = list(game.legal_actions())
    #     root = Node(key=root_key, player = obs.public.current_player, untried = root_untried)

    #     for _ in range(self.sims_per_move):
    #         # 1) Determinization
    #         g_det = self._determinize_from_game(game, obs)

    #         # 2) Selection & Expansion
    #         node = root
    #         path: List[Tuple[Node, Action]] = []

    #         while True:
    #             # Expand if we still have untried actions at this information set
    #             if node.untried:
    #                 action = self.rng.choice(node.untried)
    #                 node.untried.remove(action)
    #                 g_det.step(action)

    #                 child_obs = g_det.observe(g_det._current)
    #                 child_key = self._node_key_from_obs(child_obs)
    #                 child_untried = list(g_det.legal_actions())
    #                 child = Node(key=child_key, player=child_obs.public.current_player, untried=child_untried)

    #                 node.children[action] = child
    #                 node.edges.setdefault(action, EdgeStats())
    #                 path.append((node, action))
    #                 node = child
    #                 # Go to rollout from the newly added child
    #                 break  

    #             # Otherwise, select among all already-expanded children with UCT
    #             if not node.children:
    #                 # Nothing to select, go to rollout
    #                 break

    #             action = self._select_uct(node, list(node.children.keys()))
    #             path.append((node, action))
    #             g_det.step(action)
    #             node = node.children[action]

    #         # 3) Rollout
    #         reward = self._rollout_to_showdown(g_det, root_player)

    #         # 4) Backup
    #         self._backup(path, reward)

    #     # Pick the most visited action at root (break ties randomly)
    #     if not root.children:
    #         # If nothing expanded, fall back to a legal action
    #         return root_untried[0]

    #     best_action = max(root.children, key = lambda action: (root.edges[action].visit_count, self.rng.random()))
    #     return best_action

    def select_action(self, game: LiarsDiceGame, obs: Observation) -> Action:
        def is_liar(a: Action) -> bool:
            return isinstance(a, tuple) and a[0] == "liar"

        root_player = obs.private.my_player
        root_key = self._node_key_from_obs(obs)
        root_untried = list(game.legal_actions())
        root = Node(key=root_key, player=obs.public.current_player, untried=root_untried)

        for _ in range(self.sims_per_move):
            # 1) Determinize hidden dice
            g_det = self._determinize_from_game(game, obs)

            # 2) Tree phase (selection/expansion) with 'liar' treated as terminal
            node = root
            path: List[Tuple[Node, Action]] = []
            terminated_in_tree = False

            while True:
                # Expand if possible
                if node.untried:
                    a = self.rng.choice(node.untried)
                    node.untried.remove(a)

                    if is_liar(a):
                        # Resolve showdown immediately; no child created
                        before = list(g_det._dice_left)
                        info = g_det.step(a)
                        after = g_det._dice_left
                        root_lost = (after[root_player] < before[root_player])
                        reward = 0.0 if root_lost else 1.0

                        path.append((node, a))
                        self._backup(path, reward)
                        terminated_in_tree = True
                        break

                    # Normal expansion
                    g_det.step(a)
                    child_obs = g_det.observe(g_det._current)
                    child_key = self._node_key_from_obs(child_obs)
                    child_untried = list(g_det.legal_actions())
                    child = Node(key=child_key, player=child_obs.public.current_player, untried=child_untried)

                    node.children[a] = child
                    node.edges.setdefault(a, EdgeStats())
                    path.append((node, a))
                    node = child
                    break  # rollout starts from this new child

                # Otherwise, select among existing children
                if not node.children:
                    break  # nothing expanded yet -> go to rollout

                a = self._select_uct(node, list(node.children.keys()))

                if is_liar(a):
                    # Resolve showdown immediately; do not descend to a child
                    before = list(g_det._dice_left)
                    info = g_det.step(a)
                    after = g_det._dice_left
                    root_lost = (after[root_player] < before[root_player])
                    reward = 0.0 if root_lost else 1.0

                    path.append((node, a))
                    self._backup(path, reward)
                    terminated_in_tree = True
                    break

                # Normal selection step
                g_det.step(a)
                node = node.children[a]

            if terminated_in_tree:
                # We already backed up; skip rollout for this simulation
                continue

            # 3) Rollout to SHOWDOWN (first 'liar' or true terminal)
            reward = self._rollout_to_showdown(g_det, root_player)

            # 4) Backup
            self._backup(path, reward)

        # Pick the most visited action at root (break ties randomly)
        if not root.children:
            # If nothing expanded, fall back to a legal action
            return root_untried[0]

        best_action = max(root.children, key = lambda action: (root.edges[action].visit_count, self.rng.random()))
        return best_action


    def notify_result(self, obs: Observation, info: dict) -> None:
        return

    # --- Internals ---
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
    
    def _rollout_to_showdown(self, g: LiarsDiceGame, root_player: int) -> float:
        start_counts = list(g._dice_left)

        while True:
            legal = g.legal_actions()
            if not legal:
                raise RuntimeError("No legal actions available in non-terminal state")

            action = random.choice(legal)
            info = g.step(action)

            if isinstance(action, tuple) and action[0] == "liar":
                end_counts = g._dice_left
                root_lost = end_counts[root_player] < start_counts[root_player]
                return 0.0 if root_lost else 1.0

            if info.get("terminal"):
                return 1.0 if info.get("winner") == root_player else 0.0

    def _backup(self, path: List[Tuple[Node, Action]], reward: float) -> None:
        for node, action in path:
            node.visit_count += 1
            edge = node.edges.setdefault(action, EdgeStats())
            edge.visit_count += 1
            edge.value_sum += reward
