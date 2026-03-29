from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from action_mapping import ActionMapper
from encoder import ObservationEncoder
from nn_model import PolicyNetwork

from liars_dice.agents.helpers import bid_support_for_actor
from liars_dice.core.game import Bid, LiarsDiceGame, Observation

Action = Tuple[str, Any]
NodeKey = Tuple[int, Optional[Bid], Tuple[int, ...], int]


@dataclass
class EdgeStats:
    visit_count: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


@dataclass
class Node:
    key: NodeKey
    player: int
    children: Dict[Action, "Node"] = field(default_factory=dict)
    edges: Dict[Action, EdgeStats] = field(default_factory=dict)
    priors: Dict[Action, float] = field(default_factory=dict)
    visit_count: int = 0


class NeuralISMCTSPUCTAgent:
    """PUCT agent whose priors come from a policy network.

    The search structure intentionally mirrors the handcrafted ISMCTSPUCTAgent so
    that evaluation stays comparable.
    """

    def __init__(
        self,
        model: PolicyNetwork,
        encoder: ObservationEncoder,
        action_mapper: ActionMapper,
        label: str = "ISMCTS-NeuralPolicy",
        sims_per_move: int = 1000,
        seed: Optional[int] = None,
        puct_c: float = 1.5,
        device: str = "cpu",
        prior_floor: float = 1e-8,
        rollout_theta: float = 0.40,
        rollout_alpha: float = 0.70,
        rollout_eps: float = 0.15,
        rollout_max_steps: int = 40,
    ):
        self.name = label
        self.model = model.to(device)
        self.model.eval()
        self.encoder = encoder
        self.action_mapper = action_mapper
        self.sims_per_move = sims_per_move
        self.puct_c = puct_c
        self.rng = random.Random(seed)
        self.device = device
        self.prior_floor = prior_floor

        # Keep heuristic rollout unchanged for now; only priors are neural.
        self.rollout_theta = rollout_theta
        self.rollout_alpha = rollout_alpha
        self.rollout_eps = rollout_eps
        self.rollout_max_steps = rollout_max_steps

    def select_action(self, game: LiarsDiceGame, obs: Observation) -> Action:
        action, _ = self.search_policy(game, obs)
        return action

    def search_policy(
        self, game: LiarsDiceGame, obs: Observation
    ) -> Tuple[Action, Dict[Action, float]]:
        def is_liar(a: Action) -> bool:
            return isinstance(a, tuple) and a[0] == "liar"

        root_player = obs.private.my_player
        root_key = self._node_key_from_obs(obs)
        root = Node(key=root_key, player=obs.public.current_player)

        for _ in range(self.sims_per_move):
            g_det = self._determinize_from_game(game, obs)

            node = root
            self._compute_priors_inplace(node, g_det)

            path: List[Tuple[Node, Action]] = []
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
        root_policy = self._root_visit_distribution(root, legal_now)

        if root_policy:
            best_prob = max(root_policy.values())
            candidates = [a for a, p in root_policy.items() if p == best_prob]
            return self.rng.choice(candidates), root_policy

        if legal_now:
            return self.rng.choice(legal_now), {}
        return ("liar", None), {}

    def notify_result(self, obs: Observation, info: dict) -> None:
        return

    def _node_key_from_obs(self, obs: Observation) -> NodeKey:
        return (
            obs.public.current_player,
            obs.public.last_bid,
            tuple(obs.public.dice_left),
            len(obs.public.history),
        )

    def _determinize_from_game(
        self, game: LiarsDiceGame, obs: Observation
    ) -> LiarsDiceGame:
        g = game.clone_for_determinization()
        for pid in range(g.num_players):
            if pid == obs.private.my_player:
                continue
            n = g._dice_left[pid]
            g._dice[pid] = [self.rng.randint(1, 6) for _ in range(n)]
        return g

    def _select_puct_over_all(self, node: Node) -> Action:
        sqrt_N = math.sqrt(max(1, node.visit_count))
        best, best_val = None, -1e18
        for action, prior in node.priors.items():
            edge = node.edges.get(action)
            n_a = 0 if edge is None else edge.visit_count
            q_a = 0.0 if edge is None else edge.mean_value
            u = q_a + self.puct_c * prior * (sqrt_N / (1.0 + n_a))
            u += 1e-12 * self.rng.random()
            if u > best_val:
                best_val, best = u, action
        return best

    @torch.no_grad()
    def _compute_priors_inplace(self, node: Node, g_det: LiarsDiceGame) -> None:
        legal = list(g_det.legal_actions())
        if not legal:
            node.priors = {}
            return

        obs = g_det.observe(node.player)
        encoded = torch.tensor(
            self.encoder.encode(obs), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        logits = self.model(encoded).squeeze(0)

        mask = torch.tensor(
            self.action_mapper.legal_action_mask(legal),
            dtype=torch.float32,
            device=self.device,
        )
        probs = torch.softmax(logits, dim=-1) * mask
        mass = probs.sum()

        if float(mass) <= 0.0:
            probs = mask / mask.sum().clamp_min(1.0)
        else:
            probs = probs / mass

        priors: Dict[Action, float] = {}
        for action in legal:
            idx = self.action_mapper.action_to_index(action)
            priors[action] = max(self.prior_floor, float(probs[idx].item()))

        norm = sum(priors.values())
        if norm > 0.0:
            inv = 1.0 / norm
            for action in list(priors.keys()):
                priors[action] *= inv
        node.priors = priors

    def _root_visit_distribution(
        self, root: Node, legal_now: List[Action]
    ) -> Dict[Action, float]:
        counts = {a: e.visit_count for a, e in root.edges.items() if a in legal_now}
        total = sum(counts.values())
        if total <= 0:
            return {}
        inv = 1.0 / total
        return {a: c * inv for a, c in counts.items()}

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
                    game.step(action)
                    end_counts = game._dice_left
                    root_lost = end_counts[root_player] < start_counts[root_player]
                    return 0.0 if root_lost else 1.0

                candidate = None
                min_q = None
                for action in legal:
                    if isinstance(action, tuple) and action[0] == "bid":
                        q, f = action[1]
                        support = bid_support_for_actor(game, actor, (q, f))
                        if support >= self.rollout_alpha and (
                            min_q is None or q < min_q
                        ):
                            min_q = q
                            candidate = action
                if candidate is None:
                    action = ("liar", None)
                    game.step(action)
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
                        if support >= self.rollout_alpha and (
                            min_q is None or q < min_q
                        ):
                            min_q = q
                            candidate = action
                if candidate is None:
                    bids = [
                        action
                        for action in legal
                        if isinstance(action, tuple) and action[0] == "bid"
                    ]
                    candidate = bids[0] if bids else self.rng.choice(legal)

            info = game.step(candidate)
            if info.get("terminal"):
                return 1.0 if info.get("winner") == root_player else 0.0
