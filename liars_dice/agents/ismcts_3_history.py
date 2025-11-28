from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    # We keep children/edges and a prior table over the FULL legal set for THIS determinization
    children: Dict[Action, "Node"] = field(default_factory=dict)
    edges: Dict[Action, EdgeStats] = field(default_factory=dict)
    priors: Dict[Action, float] = field(
        default_factory=dict
    )  # π(a | I) for THIS determinization
    visit_count: int = 0


class ISMCTSHistoryAgent:
    def __init__(
        self,
        label: str = "ISMCTS-History",
        sims_per_move: int = 1000,
        seed: Optional[int] = None,
        puct_c: float = 1.5,
        prior_tau: float = 1.0,  # soften S(q,f) -> prior; <1 sharp, >1 flat
        liar_exp: float = 0.5,  # prior_liar ~ (1 - S(last_bid)) ** liar_exp
        prior_floor: float = 1e-6,  # tiny floor so priors never zero
        hist_beta: float = 0.5,
        hist_gamma: float = 1.0,
        rollout_theta: float = 0.40,
        rollout_alpha: float = 0.70,
        rollout_eps: float = 0.15,
        rollout_max_steps: int = 40,
    ):
        self.name = label
        self.sims_per_move = sims_per_move
        self.puct_c = puct_c
        self.rng = random.Random(seed)

        self.prior_tau = prior_tau
        self.liar_exp = liar_exp
        self.prior_floor = prior_floor

        self.hist_beta = hist_beta
        self.hist_gamma = hist_gamma

        self.rollout_theta = rollout_theta
        self.rollout_alpha = rollout_alpha
        self.rollout_eps = rollout_eps
        self.rollout_max_steps = rollout_max_steps

    def select_action(self, game: LiarsDiceGame, obs: Observation) -> Action:
        def is_liar(a: Action) -> bool:
            return isinstance(a, tuple) and a[0] == "liar"

        root_player = obs.private.my_player
        root_key = self._node_key_from_obs(obs)

        g_debug = self._determinize_from_game(game, obs, debug=True)

        # Root node (lazy expansion: no untried list; we always pick by PUCT over priors)
        root = Node(
            key=root_key,
            player=obs.public.current_player,
        )

        for _ in range(self.sims_per_move):
            g_det = self._determinize_from_game(game, obs)

            node = root
            self._compute_priors_inplace(node, g_det)  # priors for THIS determinization

            path: List[Tuple[Node, Action]] = []
            terminated_in_tree = False

            while True:
                # If no legal moves (shouldn't happen mid-round), break to rollout
                if not node.priors:
                    break

                # AlphaGo Zero-style: select argmax over ALL legal actions (expanded or not)
                a = self._select_puct_over_all(node)

                if is_liar(a):
                    before = list(g_det._dice_left)
                    g_det.step(a)
                    after = g_det._dice_left
                    root_lost = after[root_player] < before[root_player]
                    reward = 0.0 if root_lost else 1.0

                    # ensure edge exists for backup
                    node.edges.setdefault(a, EdgeStats())
                    path.append((node, a))
                    self._backup(path, reward)
                    terminated_in_tree = True
                    break

                g_det.step(a)

                # Lazily expand child if not present
                if a not in node.children:
                    child_obs = g_det.observe(g_det._current)
                    child_key = self._node_key_from_obs(child_obs)
                    child = Node(key=child_key, player=child_obs.public.current_player)
                    node.children[a] = child
                    node.edges.setdefault(a, EdgeStats())
                    path.append((node, a))
                    node = child
                    # compute priors for child in THIS determinization and rollout from here
                    self._compute_priors_inplace(node, g_det)
                    break  # rollout will start from this new child
                else:
                    # already expanded: descend and recompute priors at the child
                    node.edges.setdefault(a, EdgeStats())
                    path.append((node, a))
                    node = node.children[a]
                    self._compute_priors_inplace(node, g_det)

            if terminated_in_tree:
                continue

            reward = self._rollout_to_showdown(g_det, root_player)

            self._backup(path, reward)

        legal_now = list(game.legal_actions())

        if root.edges:
            scored = [
                (a, e.visit_count) for a, e in root.edges.items() if a in legal_now
            ]
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

    # def _determinize_from_game(
    #     self, game: LiarsDiceGame, obs: Observation
    # ) -> LiarsDiceGame:
    #     g = game.clone_for_determinization()
    #     for pid in range(g.num_players):
    #         if pid == obs.private.my_player:
    #             continue
    #         n = g._dice_left[pid]
    #         g._dice[pid] = [self.rng.randint(1, 6) for _ in range(n)]
    #     return g

    # def weighted_sample(self, probs):
    #     r = self.rng.random()
    #     acc = 0
    #     for f,p in enumerate(probs, start=1):
    #         acc += p
    #         if r <= acc:
    #             return f
    #     return 6

    # def _determinize_from_game(self, game: LiarsDiceGame, obs: Observation) -> LiarsDiceGame:
    #     g = game.clone_for_determinization()

    #     # Collect bid history once
    #     hist = g._history
    #     face_freq = {pid: [0]*7 for pid in range(g.num_players)}
    #     qty_max  = {pid: 1 for pid in range(g.num_players)}

    #     for pid, kind, data in hist:
    #         if kind == "bid":
    #             q, f = data
    #             face_freq[pid][f] += 1
    #             qty_max[pid] = max(qty_max[pid], q)

    #     # Sample dice using history-biased priors
    #     for pid in range(g.num_players):
    #         if pid == obs.private.my_player:
    #             continue

    #         dice_cnt = g._dice_left[pid]
    #         hf = face_freq[pid]
    #         total_bids = sum(hf)
    #         base = [1/6]*7

    #         probs = []
    #         for f in range(1,6+1):
    #             freq = (hf[f] / total_bids) if total_bids > 0 else 0.0
    #             p = base[f] * (1 + self.hist_beta * freq)  # β is tunable
    #             probs.append(p)

    #         # Player bid large quantities → inflate chance of matching faces
    #         scale = 1 + self.hist_gamma * (qty_max[pid] / sum(g._dice_left))
    #         probs = [p * scale for p in probs]

    #         # Normalize
    #         Z = sum(probs)
    #         probs = [p/Z for p in probs]

    #         # Finally sample dice for this opponent
    #         g._dice[pid] = [self.weighted_sample(probs) for _ in range(dice_cnt)]

    #     return g

    def weighted_sample(self, probs):
        r = self.rng.random()
        acc = 0
        for f, p in enumerate(probs, start=1):
            acc += p
            if r <= acc:
                return f
        return 6  # fallback

    def _determinize_from_game(self, game, obs, debug=False):
        g = game.clone_for_determinization()

        # Keep only actions since the most recent "showdown"
        hist = []
        for event in reversed(g._history):
            if isinstance(event, tuple) and event[0] == "showdown":
                break
            hist.append(event)
        hist.reverse()

        face_freq = {pid: [0] * 7 for pid in range(g.num_players)}
        qty_max = {pid: 1 for pid in range(g.num_players)}

        # --- Extract history ---
        for pid, kind, data in hist:
            if kind == "bid":
                q, f = data
                face_freq[pid][f] += 1
                qty_max[pid] = max(qty_max[pid], q)

        if debug:
            print("\n=== HISTORY DETECTED ===")
            for pid in range(g.num_players):
                print(
                    f"Player {pid} bid faces:", face_freq[pid], "max qty:", qty_max[pid]
                )

        # --- Determinization per opponent ---
        for pid in range(g.num_players):
            if pid == obs.private.my_player:
                continue

            dice_cnt = g._dice_left[pid]
            hf = face_freq[pid]
            total_bids = sum(hf)

            # Base uniform distribution
            base = [1 / 6] * 7

            # Compute β-adjusted face weights
            probs = []
            for face in range(1, 7):
                freq = (hf[face] / total_bids) if total_bids > 0 else 0
                p = base[face] * (1 + self.hist_beta * freq)
                probs.append(p)

            # γ scaling for aggression
            scale = 1 + self.hist_gamma * (qty_max[pid] / sum(g._dice_left))
            probs = [p * scale for p in probs]

            # Normalize
            Z = sum(probs)
            probs = [p / Z for p in probs]

            if debug:
                print(f"\nOpponent {pid}:")
                print(" Raw face freq:", hf[1:])
                print(" Base uniform: [1/6,...]")
                print(
                    f" β={self.hist_beta} → freq-weighted probs:",
                    [round(x, 4) for x in probs],
                )
                print(
                    f" γ={self.hist_gamma} → scaled probs:",
                    [round(x, 4) for x in probs],
                )
                print(" Normalized probs:", [round(x, 3) for x in probs])

            # Sample dice
            g._dice[pid] = [self.weighted_sample(probs) for _ in range(dice_cnt)]

            if debug:
                print(" Sampled dice:", g._dice[pid])

        return g

    def _select_puct_over_all(self, node: Node) -> Action:
        """
        score(a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        Iterate over ALL legal actions available at this node in THIS determinization
        (i.e., node.priors keys). If an action has never been visited, Q=0, N=0.
        """
        # Use max(1, N) so we have non-zero exploration at virgin nodes
        sqrt_N = math.sqrt(max(1, node.visit_count))
        best, best_val = None, -1e18
        for a, prior in node.priors.items():
            e = node.edges.get(a)  # may be None if unexpanded
            n_a = 0 if e is None else e.visit_count
            q_a = 0.0 if e is None else e.mean_value
            u = q_a + self.puct_c * prior * (sqrt_N / (1.0 + n_a))
            # random tie-break to avoid deterministic ties
            u += 1e-12 * self.rng.random()
            if u > best_val:
                best_val, best = u, a
        return best  # type: ignore

    def _compute_priors_inplace(self, node: Node, g_det: LiarsDiceGame) -> None:
        """
        Compute actor-centric priors π(a|I) for legal actions at this node in THIS determinization.
        - For bids (q,f): π ∝ S_actor(q,f)^(1/τ)
        - For liar: π ∝ (1 - S_actor(last_bid))^liar_exp
        Normalize to sum 1 (with floor) among legal actions.
        """
        legal = list(g_det.legal_actions())
        priors: Dict[Action, float] = {}

        actor = node.player
        # Bids
        for a in legal:
            if isinstance(a, tuple) and a[0] == "bid":
                q, f = a[1]
                S = bid_support_for_actor(g_det, actor, (q, f))
                base = max(self.prior_floor, S ** (1.0 / max(1e-6, self.prior_tau)))
                priors[a] = base

        # Liar (only if legal)
        if any(isinstance(x, tuple) and x[0] == "liar" for x in legal):
            last_bid = g_det._last_bid  # public
            if last_bid is not None:
                S_last = bid_support_for_actor(g_det, actor, last_bid)
                p_liar = max(
                    self.prior_floor, (1.0 - S_last) ** max(1e-6, self.liar_exp)
                )
            else:
                p_liar = self.prior_floor
            priors[("liar", None)] = p_liar

        # Normalize over legal
        s = sum(priors.values())
        if s <= 0.0:
            u = 1.0 / max(1, len(legal))
            for a in legal:
                priors[a] = u
        else:
            inv = 1.0 / s
            for a in list(priors.keys()):
                priors[a] *= inv

        node.priors = priors

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
                        if support >= self.rollout_alpha and (
                            min_q is None or q < min_q
                        ):
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
            continue
