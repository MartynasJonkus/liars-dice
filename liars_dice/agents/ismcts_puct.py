from __future__ import annotations
import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from liars_dice.core.game import LiarsDiceGame, Observation, Bid

Action = Tuple[str, Any]
NodeKey = Tuple[int, Optional[Bid], Tuple[int, ...], int]  # (player_to_act, last_bid, dice_left[], history_len)

# --------------------------
# Stats containers
# --------------------------
@dataclass
class EdgeStats:
    visits: int = 0            # N(s,a)
    value_sum: float = 0.0     # sum of returns from root's perspective
    @property
    def mean_value(self) -> float:  # Q(s,a)
        return 0.0 if self.visits == 0 else self.value_sum / self.visits

@dataclass
class Node:
    key: NodeKey
    player: int
    # We keep children/edges and a prior table over the FULL legal set for THIS determinization
    children: Dict[Action, 'Node'] = field(default_factory=dict)
    edges: Dict[Action, EdgeStats] = field(default_factory=dict)
    priors: Dict[Action, float] = field(default_factory=dict)  # π(a | I) for THIS determinization
    visit_count: int = 0  # N(s) = sum_b N(s,b)

# --------------------------
# ISMCTS + PUCT (no PW), showdown-terminal rollouts
# --------------------------
class ISMCTSPUCTAgent:
    def __init__(
        self,
        label: str = "ISMCTS-PUCT",
        sims_per_move: int = 20000,
        puct_c: float = 0.5,            # softer prior pull than 1.0
        seed: Optional[int] = None,
        # --- prior shaping knobs (domain prior only) ---
        prior_tau: float = 1.5,         # soften S(q,f) -> prior; <1 sharp, >1 flat
        liar_exp: float = 1.25,         # prior_liar ~ (1 - S(last_bid)) ** liar_exp
        prior_floor: float = 1e-6,      # tiny floor so priors never zero
        # --- heuristic rollout knobs (tuned to reduce over-escalation) ---
        rollout_theta: float = 0.50,    # call if current bid support < theta
        rollout_alpha: float = 0.80,    # target plausibility for own raise
        rollout_eps: float = 0.05,      # small random raise chance
        rollout_max_steps: int = 40,    # safety cap
    ):
        self.name = label
        self.sims_per_move = sims_per_move
        self.puct_c = puct_c
        self.rng = random.Random(seed)

        self.prior_tau = prior_tau
        self.liar_exp = liar_exp
        self.prior_floor = prior_floor

        self.rollout_theta = rollout_theta
        self.rollout_alpha = rollout_alpha
        self.rollout_eps = rollout_eps
        self.rollout_max_steps = rollout_max_steps

    # ----------------------
    # Public API
    # ----------------------
    def select_action(self, game: LiarsDiceGame, obs: Observation) -> Action:
        def is_liar(a: Action) -> bool:
            return isinstance(a, tuple) and a[0] == "liar"

        root_player = obs.private.my_player
        root_key = self._node_key_from_obs(obs)

        # Root node (lazy expansion: no untried list; we always pick by PUCT over priors)
        root = Node(
            key=root_key,
            player=obs.public.current_player,
        )

        for _ in range(self.sims_per_move):
            # 1) Determinize hidden dice (opponents only)
            g_det = self._determinize_from_game(game, obs)

            # 2) Selection with PUCT over ALL legal actions; expand lazily
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

                # If 'liar' → resolve immediately (no child)
                if is_liar(a):
                    before = list(g_det._dice_left)
                    info = g_det.step(a)
                    after = g_det._dice_left
                    root_lost = (after[root_player] < before[root_player])
                    reward = 0.0 if root_lost else 1.0
                    # ensure edge exists for backup
                    node.edges.setdefault(a, EdgeStats())
                    path.append((node, a))
                    self._backup(path, reward)
                    terminated_in_tree = True
                    break

                # Otherwise step the game
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
                continue  # go to next simulation

            # 3) Rollout: heuristic to the NEXT showdown (or true terminal)
            reward = self._rollout_to_showdown(g_det, root_player)

            # 4) Backup along the path
            self._backup(path, reward)

        # 5) Root decision: most visits among edges that are currently legal
        legal_now = list(game.legal_actions())
        if root.edges:
            scored = [(a, e.visits) for a, e in root.edges.items() if a in legal_now]
            if scored:
                best_visits = max(v for _, v in scored)
                candidates = [a for a, v in scored if v == best_visits]
                return self.rng.choice(candidates)

        # Fallback: any legal move (rare)
        if legal_now:
            return self.rng.choice(legal_now)
        return ("liar", None)

    def notify_result(self, obs: Observation, info: dict) -> None:
        return

    # ----------------------
    # Internals
    # ----------------------
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

    # -------- PUCT selection over ALL legal actions (AlphaGo Zero) --------
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
            n_a = 0 if e is None else e.visits
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
                S = self._bid_support_for_actor(g_det, actor, (q, f))
                base = max(self.prior_floor, S ** (1.0 / max(1e-6, self.prior_tau)))
                priors[a] = base

        # Liar (only if legal)
        if any(isinstance(x, tuple) and x[0] == "liar" for x in legal):
            last_bid = g_det._last_bid  # public
            if last_bid is not None:
                S_last = self._bid_support_for_actor(g_det, actor, last_bid)
                p_liar = max(self.prior_floor, (1.0 - S_last) ** max(1e-6, self.liar_exp))
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

    # -------- Heuristic rollout to SHOWDOWN ----------
    def _rollout_to_showdown(self, g: LiarsDiceGame, root_player: int) -> float:
        """
        Heuristic rollout until the next showdown ('liar'), information-safe:
        each actor uses only their private dice in THIS determinization + public info.
        Returns 1.0 if root does NOT lose the showdown, else 0.0.
        """
        if g.num_alive() <= 1:
            winner = g._winner()
            return 1.0 if winner == root_player else 0.0

        start_counts = list(g._dice_left)
        steps = 0

        while True:
            steps += 1
            if steps > self.rollout_max_steps:
                winner = g._winner()
                if winner is not None:
                    return 1.0 if winner == root_player else 0.0
                return 0.5

            legal = g.legal_actions()
            if not legal:
                winner = g._winner()
                return 1.0 if winner == root_player else (0.0 if winner is not None else 0.5)

            actor = g._current
            obs = g.observe(actor)
            last_bid = obs.public.last_bid

            # ε-exploration
            if self.rng.random() < self.rollout_eps:
                action = self.rng.choice(legal)
                info = g.step(action)
                if isinstance(action, tuple) and action[0] == "liar":
                    end_counts = g._dice_left
                    root_lost = end_counts[root_player] < start_counts[root_player]
                    return 0.0 if root_lost else 1.0
                if info.get("terminal"):
                    return 1.0 if info.get("winner") == root_player else 0.0
                continue

            # Heuristic decision
            if last_bid is not None:
                # Consider calling
                support_last = self._bid_support_for_actor(g, actor, last_bid)
                if support_last < self.rollout_theta:
                    action = ("liar", None)
                    info = g.step(action)
                    end_counts = g._dice_left
                    root_lost = end_counts[root_player] < start_counts[root_player]
                    return 0.0 if root_lost else 1.0

                # Try minimal dominating raise with support >= alpha
                candidate = None
                min_q = None
                for a in legal:
                    if isinstance(a, tuple) and a[0] == "bid":
                        q, f = a[1]
                        s = self._bid_support_for_actor(g, actor, (q, f))
                        if s >= self.rollout_alpha and (min_q is None or q < min_q):
                            min_q = q
                            candidate = a
                if candidate is None:
                    # No plausible raise -> call
                    action = ("liar", None)
                    info = g.step(action)
                    end_counts = g._dice_left
                    root_lost = end_counts[root_player] < start_counts[root_player]
                    return 0.0 if root_lost else 1.0

                info = g.step(candidate)
                if info.get("terminal"):
                    return 1.0 if info.get("winner") == root_player else 0.0
                continue

            else:
                # Opening: minimal plausible opening bid; else smallest legal bid
                candidate = None
                min_q = None
                for a in legal:
                    if isinstance(a, tuple) and a[0] == "bid":
                        q, f = a[1]
                        s = self._bid_support_for_actor(g, actor, (q, f))
                        if s >= self.rollout_alpha and (min_q is None or q < min_q):
                            min_q = q
                            candidate = a
                if candidate is None:
                    bids = [a for a in legal if isinstance(a, tuple) and a[0] == "bid"]
                    candidate = bids[0] if bids else self.rng.choice(legal)

                info = g.step(candidate)
                if info.get("terminal"):
                    return 1.0 if info.get("winner") == root_player else 0.0
                continue

    # -------- Support utilities ----------
    def _bid_support_for_actor(self, g: LiarsDiceGame, actor: int, bid: Bid) -> float:
        """
        P[ total count for face >= q | actor's private dice + public info ].
        Assumes 'ones are wild' for non-1 faces (p=2/6), and p=1/6 for face==1.
        """
        q, face = bid
        my = g._dice[actor]

        if face == 1:
            k_mine = sum(1 for d in my if d == 1)
            p_face = 1.0 / 6.0
        else:
            k_mine = sum(1 for d in my if d == face or d == 1)
            p_face = 2.0 / 6.0

        total_left = sum(g._dice_left)
        need_from_others = max(0, q - k_mine)
        n_unknown = total_left - len(my)

        if need_from_others <= 0:
            return 1.0
        if need_from_others > n_unknown:
            return 0.0
        return self._binom_tail_geq(n_unknown, p_face, need_from_others)

    def _binom_tail_geq(self, n: int, p: float, k: int) -> float:
        """P[X >= k] for X ~ Bin(n, p). Small n (<=25) is fine with direct sum."""
        if k <= 0:
            return 1.0
        if k > n:
            return 0.0
        q = 1.0 - p
        prob = 0.0
        for i in range(k, n + 1):
            prob += math.comb(n, i) * (p ** i) * (q ** (n - i))
        return prob

    # -------- Backup ----------
    def _backup(self, path: List[Tuple[Node, Action]], reward: float) -> None:
        for node, action in path:
            node.visit_count += 1
            e = node.edges.setdefault(action, EdgeStats())
            e.visits += 1
            e.value_sum += reward
