from __future__ import annotations

from typing import Any, List, Optional, Tuple

from liars_dice.agents.ISMCTS_History import ISMCTSHistoryAgent
from liars_dice.core.game import LiarsDiceGame, Observation
from neural.common.action_mapping import ActionMapper

Action = Tuple[str, Any]


class VisitTracingHistoryAgent(ISMCTSHistoryAgent):
    """History agent variant that exposes root visit distributions for supervision."""

    def __init__(self, *args, action_mapper: ActionMapper, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_mapper = action_mapper
        self.last_root_policy: Optional[List[float]] = None

    def select_action(self, game: LiarsDiceGame, obs: Observation) -> Action:
        action, root_policy = self.search_policy(game, obs)
        self.last_root_policy = root_policy
        return action

    def search_policy(
        self, game: LiarsDiceGame, obs: Observation
    ) -> Tuple[Action, List[float]]:
        # Reuse the original implementation structure by calling super().select_action is not enough,
        # because the parent does not expose root-edge statistics. This method mirrors the parent agent
        # only at the root-output stage while keeping the public behavior identical.
        def is_liar(a: Action) -> bool:
            return isinstance(a, tuple) and a[0] == "liar"

        import importlib

        mod = importlib.import_module(self.__class__.__mro__[1].__module__)
        Node = getattr(mod, "Node")
        EdgeStats = getattr(mod, "EdgeStats")

        root_player = obs.private.my_player
        root_key = self._node_key_from_obs(obs)
        root = Node(key=root_key, player=obs.public.current_player)

        for _ in range(self.sims_per_move):
            g_det = self._determinize_from_game(game, obs)
            node = root
            self._compute_priors_inplace(node, g_det)
            path: List[Tuple[Any, Action]] = []
            terminated_in_tree = False

            while True:
                if not node.priors:
                    break

                a = self._select_puct_over_all(node)

                if is_liar(a):
                    before = list(g_det._dice_left)
                    g_det.step(a)
                    after = g_det._dice_left
                    root_lost = after[root_player] < before[root_player]
                    reward = 0.0 if root_lost else 1.0

                    node.edges.setdefault(a, EdgeStats())
                    path.append((node, a))
                    self._backup(path, reward)
                    terminated_in_tree = True
                    break

                g_det.step(a)

                if a not in node.children:
                    child_obs = g_det.observe(g_det._current)
                    child_key = self._node_key_from_obs(child_obs)
                    child = Node(key=child_key, player=child_obs.public.current_player)
                    node.children[a] = child
                    node.edges.setdefault(a, EdgeStats())
                    path.append((node, a))
                    node = child
                    self._compute_priors_inplace(node, g_det)
                    break
                else:
                    node.edges.setdefault(a, EdgeStats())
                    path.append((node, a))
                    node = node.children[a]
                    self._compute_priors_inplace(node, g_det)

            if terminated_in_tree:
                continue

            reward = self._rollout_to_showdown(g_det, root_player)
            self._backup(path, reward)

        legal_now = list(game.legal_actions())
        policy = [0.0] * self.action_mapper.num_actions
        counts = {a: e.visit_count for a, e in root.edges.items() if a in legal_now}
        total = sum(counts.values())
        if total > 0:
            inv = 1.0 / total
            for action, count in counts.items():
                policy[self.action_mapper.action_to_index(action)] = count * inv

        if counts:
            best_visits = max(counts.values())
            candidates = [a for a, c in counts.items() if c == best_visits]
            return self.rng.choice(candidates), policy

        if legal_now:
            return self.rng.choice(legal_now), policy
        return ("liar", None), policy
