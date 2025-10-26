
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

class ISMCTSHeuristicAgent:
    def __init__(
        self,
        label: str = "ISMCTS-Heuristic",
        sims_per_move: int = 20000,
        uct_c: float = math.sqrt(2.0),
        seed: Optional[int] = None,
         # --- rollout knobs (heuristic rollout) ---
        rollout_theta: float = 0.40,   # call if current bid support < theta
        rollout_alpha: float = 0.72,   # target plausibility for own raise
        rollout_eps: float = 0.08,     # small random raise chance
        rollout_max_steps: int = 40,   # hard cap; safety only
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

        # 5) Choose action at root
        legal_now = list(game.legal_actions())

        # Prefer the most visited action among edges that are still legal now.
        if root.edges:
            # Filter to legal actions with stats
            scored = [(a, e.visit_count) for a, e in root.edges.items() if a in legal_now]
            if scored:
                best_visits = max(v for _, v in scored)
                candidates = [a for a, v in scored if v == best_visits]
                return self.rng.choice(candidates)

        # Fallback: pick any legal move (should be rare)
        if legal_now:
            return self.rng.choice(legal_now)

        # Absolute fallback (shouldn't happen): raise or return a safe default
        return ("liar", None)

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
        """
        Heuristic rollout until the next showdown ('liar' is called).
        - On each turn, the acting simulated player uses only their private dice + public info.
        - Policy: threshold caller + minimal plausible raise, with small epsilon exploration.
        Returns 1.0 if root_player does NOT lose the showdown, else 0.0.
        """
        # Terminal guard (rare but cheap)
        if g.num_alive() <= 1:
            winner = g._winner()
            return 1.0 if winner == root_player else 0.0

        start_counts = list(g._dice_left)

        steps = 0
        while True:
            steps += 1
            if steps > self.rollout_max_steps:
                # Safety fallback if something loops (should be rare):
                winner = g._winner()
                if winner is not None:
                    return 1.0 if winner == root_player else 0.0
                return 0.5  # neutral fallback

            legal = g.legal_actions()
            if not legal:
                winner = g._winner()
                return 1.0 if winner == root_player else (0.0 if winner is not None else 0.5)

            # Observe from the acting player's seat to get public last_bid
            actor = g._current
            obs = g.observe(actor)
            last_bid = obs.public.last_bid

            # ε-exploration: occasionally take a random legal raise to keep diversity
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

            # Heuristic decision for the actor
            if last_bid is not None:
                # Decide whether to call or raise
                support_last = self._bid_support_for_actor(g, actor, last_bid)
                if support_last < self.rollout_theta:
                    action = ("liar", None)
                    info = g.step(action)
                    end_counts = g._dice_left
                    root_lost = end_counts[root_player] < start_counts[root_player]
                    return 0.0 if root_lost else 1.0

                # Try to find the minimal dominating raise with support >= alpha
                # legal is a list like [("liar", None), ("bid",(q,f)), ...]
                candidate = None
                min_q = None
                for a in legal:
                    if isinstance(a, tuple) and a[0] == "bid":
                        q, f = a[1]
                        s = self._bid_support_for_actor(g, actor, (q, f))
                        if s >= self.rollout_alpha:
                            if min_q is None or q < min_q:
                                min_q = q
                                candidate = a
                if candidate is None:
                    # No plausible raise found -> call
                    action = ("liar", None)
                    info = g.step(action)
                    end_counts = g._dice_left
                    root_lost = end_counts[root_player] < start_counts[root_player]
                    return 0.0 if root_lost else 1.0

                # Take the minimal plausible raise
                info = g.step(candidate)
                if info.get("terminal"):
                    return 1.0 if info.get("winner") == root_player else 0.0
                continue

            else:
                # Opening: pick the minimal plausible opening bid; if none, pick the minimal legal bid
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
                    # Fall back to the smallest legal bid (by engine ordering)
                    # Note: legal includes 'liar' only when last_bid != None, so safe here.
                    bids = [a for a in legal if isinstance(a, tuple) and a[0] == "bid"]
                    candidate = bids[0] if bids else self.rng.choice(legal)

                info = g.step(candidate)
                if info.get("terminal"):
                    return 1.0 if info.get("winner") == root_player else 0.0
                continue


    def _backup(self, path: List[Tuple[Node, Action]], reward: float) -> None:
        for node, action in path:
            node.visit_count += 1
            edge = node.edges.setdefault(action, EdgeStats())
            edge.visit_count += 1
            edge.value_sum += reward

    def _bid_support_for_actor(self, g: LiarsDiceGame, actor: int, bid: Bid) -> float:
        """
        P[ total count for face >= q | actor's private dice + public info ].
        Assumes 'ones are wild' for non-1 faces (p=2/6), and p=1/6 for face==1.
        """
        q, face = bid
        my = g._dice[actor]
        # Count matches in my dice (1's are wild for non-1 faces)
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
        # P[X >= k] for X~Bin(n,p). n<=25-ish in this game; direct sum is fine.
        if k <= 0:
            return 1.0
        if k > n:
            return 0.0
        q = 1.0 - p
        prob = 0.0
        for i in range(k, n + 1):
            prob += math.comb(n, i) * (p ** i) * (q ** (n - i))
        return prob

