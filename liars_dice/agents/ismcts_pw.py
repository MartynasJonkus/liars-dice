from __future__ import annotations
import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

from liars_dice.core.game import LiarsDiceGame, Observation, Bid

Action = Tuple[str, Any]
# NodeKey: public info only (actor seat, last bid, dice left, history length)
NodeKey = Tuple[int, Optional[Bid], Tuple[int, ...], int]


# --------------------------
# Stats containers
# --------------------------
@dataclass
class EdgeStats:
    visits: int = 0          # N(s,a)
    value_sum: float = 0.0   # sum of returns from root's perspective
    @property
    def mean_value(self) -> float:  # Q(s,a)
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


@dataclass
class Node:
    key: NodeKey
    player: int
    # Progressive widening state
    all_bids: List[Action] = field(default_factory=list)  # all ("bid",(q,f)) legal at this info set
    active_bids: Set[Action] = field(default_factory=set) # admitted bids (subset of all_bids)
    # Expansion / children
    untried: List[Action] = field(default_factory=list)   # actions not yet expanded (liar only when legal, plus admitted bids)
    children: Dict[Action, 'Node'] = field(default_factory=dict)  # non-liar actions only have children
    edges: Dict[Action, EdgeStats] = field(default_factory=dict)  # stats for ALL actions taken from this node
    visit_count: int = 0  # N(s)


# --------------------------
# ISMCTS with BL-PW + showdown-terminal rollouts
# --------------------------
class ISMCTSPWAgent:
    def __init__(
        self,
        label: str = "ISMCTS-ProgressiveWidening",
        sims_per_move: int = 2000,
        uct_c: float = math.sqrt(2.0),
        seed: Optional[int] = None,
        # --- heuristic rollout knobs ---
        rollout_theta: float = 0.40,   # call if current bid support < theta
        rollout_alpha: float = 0.72,   # target plausibility for own raise
        rollout_eps: float = 0.08,     # small random raise chance
        rollout_max_steps: int = 40,   # safety cap
        # --- progressive widening knobs ---
        pw_K0: int = 1,        # base number of admitted raises
        pw_k: float = 1.0,     # scale
        pw_alpha: float = 0.4, # growth exponent (0<alpha<1)
    ):
        self.name = label
        self.sims_per_move = sims_per_move
        self.uct_c = uct_c
        self.rng = random.Random(seed)

        self.rollout_theta = rollout_theta
        self.rollout_alpha = rollout_alpha
        self.rollout_eps = rollout_eps
        self.rollout_max_steps = rollout_max_steps

        self.pw_K0 = pw_K0
        self.pw_k = pw_k
        self.pw_alpha = pw_alpha

    # ----------------------
    # Public API
    # ----------------------
    def select_action(self, game: LiarsDiceGame, obs: Observation) -> Action:
        def is_liar(a: Action) -> bool:
            return isinstance(a, tuple) and a[0] == "liar"

        root_player = obs.private.my_player
        root_key = self._node_key_from_obs(obs)

        # Initialize root node (PW: split bids, keep 'liar' admissible ONLY if legal)
        root_actions = list(game.legal_actions())
        root = Node(
            key=root_key,
            player=obs.public.current_player,
            untried=[],
        )
        root.all_bids = [a for a in root_actions if isinstance(a, tuple) and a[0] == "bid"]
        # Add 'liar' only if legal now
        if any(isinstance(a, tuple) and a[0] == "liar" for a in root_actions):
            root.untried.append(("liar", None))

        for _ in range(self.sims_per_move):
            # 1) Determinize hidden dice (opponents only)
            g_det = self._determinize_from_game(game, obs)

            # 2) Tree phase: selection/expansion with 'liar' terminal and PW
            node = root
            path: List[Tuple[Node, Action]] = []
            terminated_in_tree = False

            while True:
                # Admit raises per BL-PW for THIS determinization and node visit count
                self._ensure_pw_actions(node, g_det)

                # Expand if possible (random among admissible untried)
                if node.untried:
                    a = self.rng.choice(node.untried)
                    node.untried.remove(a)

                    if is_liar(a):
                        # Resolve showdown immediately, back up, no child
                        before = list(g_det._dice_left)
                        info = g_det.step(a)
                        after = g_det._dice_left
                        root_lost = (after[root_player] < before[root_player])
                        reward = 0.0 if root_lost else 1.0
                        path.append((node, a))
                        self._backup(path, reward)
                        terminated_in_tree = True
                        break

                    # Normal expansion to child
                    g_det.step(a)
                    child_obs = g_det.observe(g_det._current)
                    child_key = self._node_key_from_obs(child_obs)

                    child_actions = list(g_det.legal_actions())
                    child = Node(key=child_key, player=child_obs.public.current_player, untried=[])
                    child.all_bids = [x for x in child_actions if isinstance(x, tuple) and x[0] == "bid"]
                    # Add 'liar' only if legal at the child
                    if any(isinstance(x, tuple) and x[0] == "liar" for x in child_actions):
                        child.untried.append(("liar", None))

                    node.children[a] = child
                    node.edges.setdefault(a, EdgeStats())
                    path.append((node, a))
                    node = child
                    break  # rollout from newly created child

                # Otherwise select among existing children with UCT
                if not node.children:
                    break  # nothing to select -> rollout

                a = self._select_uct(node, list(node.children.keys()))

                if is_liar(a):
                    # Resolve showdown immediately, back up, no child
                    before = list(g_det._dice_left)
                    info = g_det.step(a)
                    after = g_det._dice_left
                    root_lost = (after[root_player] < before[root_player])
                    reward = 0.0 if root_lost else 1.0
                    path.append((node, a))
                    self._backup(path, reward)
                    terminated_in_tree = True
                    break

                # Normal selection descend
                g_det.step(a)
                node = node.children[a]

            if terminated_in_tree:
                continue  # next simulation

            # 3) Rollout: heuristic to the NEXT showdown (or true terminal)
            reward = self._rollout_to_showdown(g_det, root_player)

            # 4) Backup
            self._backup(path, reward)

        # 5) Root decision: choose among actions with edge stats (includes 'liar')
        legal_now = list(game.legal_actions())
        if root.edges:
            scored = [(a, e.visits) for a, e in root.edges.items() if a in legal_now]
            if scored:
                best_visits = max(v for _, v in scored)
                candidates = [a for a, v in scored if v == best_visits]
                return self.rng.choice(candidates)

        # Fallback: pick any legal move (rare)
        if legal_now:
            return self.rng.choice(legal_now)

        # Absolute fallback (shouldn't happen)
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

    def _select_uct(self, node: Node, actions: List[Action]) -> Action:
        logN = math.log(max(1, node.visit_count))
        best, best_val = None, -1e18
        for a in actions:
            e = node.edges[a]
            u = e.mean_value + self.uct_c * math.sqrt(logN / (e.visits + 1))
            if u > best_val:
                best_val, best = u, a
        return best  # type: ignore

    # -------- Progressive Widening (BL-PW) ----------
    def _ensure_pw_actions(self, node: Node, g_det: LiarsDiceGame) -> None:
        """
        Admit up to K raises into node.active_bids (and node.untried if not yet expanded),
        ranked by the acting player's bid support in THIS determinization.
        'liar' is handled separately (only if legal).
        """
        # target K
        K = int(self.pw_K0 + self.pw_k * (node.visit_count ** self.pw_alpha))
        if K < 1:
            K = 1

        # Seed with minimal dominating raise if none admitted
        if not node.active_bids and node.all_bids:
            first_bid = node.all_bids[0]  # engine's minimal dominating raise
            node.active_bids.add(first_bid)
            if first_bid not in node.children and first_bid not in node.untried:
                node.untried.append(first_bid)

        # Admit more up to K
        if len(node.active_bids) < K:
            locked = [a for a in node.all_bids if a not in node.active_bids]
            if locked:
                actor = node.player

                def score_bid(a: Action) -> Tuple[float, int, int]:
                    (q, f) = a[1]
                    s = self._bid_support_for_actor(g_det, actor, (q, f))
                    # tie-break: prefer smaller q, then smaller face
                    return (s, -q, -f)

                locked.sort(key=score_bid, reverse=True)
                for a in locked:
                    if len(node.active_bids) >= K:
                        break
                    node.active_bids.add(a)
                    if a not in node.children and a not in node.untried:
                        node.untried.append(a)

        # Keep 'liar' present ONLY if it's legal in THIS state.
        legal = g_det.legal_actions()
        liar_legal = any(isinstance(x, tuple) and x[0] == "liar" for x in legal)
        if liar_legal:
            if ("liar", None) not in node.children and ("liar", None) not in node.untried:
                node.untried.append(("liar", None))
        else:
            # If somehow present from a previous determinization, remove it
            try:
                node.untried.remove(("liar", None))
            except ValueError:
                pass

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
                # Safety fallback
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
