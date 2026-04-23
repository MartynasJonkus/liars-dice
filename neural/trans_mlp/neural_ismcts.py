from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from liars_dice.agents.helpers import bid_support_for_actor
from liars_dice.core.game import Bid, LiarsDiceGame, Observation
from neural.action_mapping import ActionMapper
from neural.trans_mlp.encoder import ObservationEncoder
from neural.trans_mlp.nn_model import PolicyNetwork

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
    """
    ISMCTS + PUCT where action priors come from a neural policy network.

    Important design choices:
    - The network operates on the acting player's observation, not on hidden state.
    - Root priors are computed once per real move.
    - Deeper-node priors are recomputed per simulation path, because the same
      node key may be visited under different histories / determinizations.
    """

    def __init__(
        self,
        model: PolicyNetwork,
        encoder: ObservationEncoder,
        action_mapper: ActionMapper,
        label: str = "Neural-ISMCTS",
        sims_per_move: int | None = 500,
        time_limit_s: float | None = None,
        puct_c: float = 1.5,
        seed: Optional[int] = None,
        device: str = "cpu",
        prior_floor: float = 1e-6,
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
        self.time_limit_s = time_limit_s
        self._last_sim_count = 0

        self.puct_c = puct_c
        self.rng = random.Random(seed)
        self.device = device
        self.prior_floor = prior_floor

        self.rollout_theta = rollout_theta
        self.rollout_alpha = rollout_alpha
        self.rollout_eps = rollout_eps
        self.rollout_max_steps = rollout_max_steps

    def select_action(self, game: LiarsDiceGame, obs: Observation) -> Action:
        action, _, sim_count = self.search_policy(game, obs)
        self._last_sim_count = sim_count
        return action

    def search_policy(
        self,
        game: LiarsDiceGame,
        obs: Observation,
    ) -> Tuple[Action, Dict[Action, float], int]:
        def is_liar(a: Action) -> bool:
            return isinstance(a, tuple) and a[0] == "liar"

        root_player = obs.private.my_player
        root_key = self._node_key_from_obs(obs)
        root = Node(key=root_key, player=obs.public.current_player)

        prior_cache: Dict[Any, Dict[Action, float]] = {}

        self._compute_priors_from_obs_inplace(
            root,
            obs,
            list(game.legal_actions()),
            prior_cache,
        )

        start_time = time.perf_counter()
        sim_count = 0

        while self._should_continue(sim_count, start_time):
            g_det = self._determinize_from_game(game, obs)

            node = root
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

                    self._compute_priors_from_obs_inplace(
                        node,
                        child_obs,
                        list(g_det.legal_actions()),
                        prior_cache,
                    )
                    break
                else:
                    node.edges.setdefault(a, EdgeStats())
                    path.append((node, a))
                    node = node.children[a]

                    child_obs = g_det.observe(g_det._current)
                    self._compute_priors_from_obs_inplace(
                        node,
                        child_obs,
                        list(g_det.legal_actions()),
                        prior_cache,
                    )

            sim_count += 1

            if terminated_in_tree:
                continue

            reward = self._rollout_to_showdown(g_det, root_player)
            self._backup(path, reward)

        legal_now = list(game.legal_actions())
        root_policy = self._root_visit_distribution(root, legal_now)

        if root_policy:
            best_prob = max(root_policy.values())
            candidates = [a for a, p in root_policy.items() if p == best_prob]
            return self.rng.choice(candidates), root_policy, sim_count

        if legal_now:
            return self.rng.choice(legal_now), {}, sim_count

        return ("liar", None), {}, sim_count

    def notify_result(self, obs: Observation, info: dict) -> None:
        return

    def _should_continue(self, sim_count: int, start_time: float) -> bool:
        if self.sims_per_move is None and self.time_limit_s is None:
            raise ValueError(
                "At least one of sims_per_move or time_limit_s must be set"
            )

        if self.sims_per_move is not None and sim_count >= self.sims_per_move:
            return False

        if (
            self.time_limit_s is not None
            and (time.perf_counter() - start_time) >= self.time_limit_s
        ):
            return False

        return True

    def _node_key_from_obs(self, obs: Observation) -> NodeKey:
        return (
            obs.public.current_player,
            obs.public.last_bid,
            tuple(obs.public.dice_left),
            len(obs.public.history),
        )

    def _determinize_from_game(
        self,
        game: LiarsDiceGame,
        obs: Observation,
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

        for a, prior in node.priors.items():
            e = node.edges.get(a)
            n_a = 0 if e is None else e.visit_count
            q_a = 0.0 if e is None else e.mean_value

            u = q_a + self.puct_c * prior * (sqrt_N / (1.0 + n_a))
            u += 1e-12 * self.rng.random()

            if u > best_val:
                best_val, best = u, a

        if best is None:
            raise RuntimeError("PUCT selection failed: no action selected")

        return best

    def _make_prior_cache_key(self, encoded: Dict[str, Any]) -> Any:
        static_key = tuple(encoded["static_features"])
        history_key = tuple(tuple(token) for token in encoded["bid_history"])
        mask_key = tuple(encoded["bid_mask"])
        return (static_key, history_key, mask_key)

    @torch.inference_mode()
    def _compute_priors_from_obs_inplace(
        self,
        node: Node,
        obs: Observation,
        legal_actions: List[Action],
        prior_cache: Dict[Any, Dict[Action, float]],
    ) -> None:
        if not legal_actions:
            node.priors = {}
            return

        encoded = self.encoder.encode(obs)
        cache_key = self._make_prior_cache_key(encoded)

        cached = prior_cache.get(cache_key)
        if cached is not None:
            node.priors = {a: cached[a] for a in legal_actions if a in cached}
            return

        static_x = torch.tensor(
            encoded["static_features"],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        bid_history = torch.tensor(
            encoded["bid_history"],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        bid_mask = torch.tensor(
            encoded["bid_mask"],
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)

        logits = self.model(static_x, bid_history, bid_mask).squeeze(0)

        legal_mask = torch.tensor(
            self.action_mapper.legal_action_mask(legal_actions),
            dtype=torch.float32,
            device=self.device,
        )

        probs = torch.softmax(logits, dim=-1) * legal_mask
        mass = probs.sum()

        if float(mass.item()) <= 0.0:
            legal_indices = self.action_mapper.legal_indices(legal_actions)
            probs = torch.zeros_like(probs)
            if legal_indices:
                uniform_prob = 1.0 / len(legal_indices)
                for idx in legal_indices:
                    probs[idx] = uniform_prob
        else:
            probs = probs / mass

        priors: Dict[Action, float] = {}
        for action in legal_actions:
            idx = self.action_mapper.action_to_index(action)
            priors[action] = max(self.prior_floor, float(probs[idx].item()))

        total = sum(priors.values())
        if total <= 0.0:
            u = 1.0 / len(legal_actions)
            for action in legal_actions:
                priors[action] = u
        else:
            inv_total = 1.0 / total
            for action in list(priors.keys()):
                priors[action] *= inv_total

        prior_cache[cache_key] = dict(priors)
        node.priors = priors

    def _root_visit_distribution(
        self,
        root: Node,
        legal_now: List[Action],
    ) -> Dict[Action, float]:
        counts = {a: e.visit_count for a, e in root.edges.items() if a in legal_now}
        total = sum(counts.values())
        if total <= 0:
            return {}
        inv_total = 1.0 / total
        return {a: c * inv_total for a, c in counts.items()}

    def _backup(self, path: List[Tuple[Node, Action]], reward: float) -> None:
        for node, action in path:
            node.visit_count += 1
            e = node.edges.setdefault(action, EdgeStats())
            e.visit_count += 1
            e.value_sum += reward

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
