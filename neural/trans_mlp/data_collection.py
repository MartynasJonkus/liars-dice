from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from liars_dice.core.game import LiarsDiceGame, Observation
from neural.action_mapping import ActionMapper
from neural.trans_mlp.encoder import ObservationEncoder


@dataclass
class PolicySample:
    static_features: List[float]
    bid_history: List[List[float]]
    bid_mask: List[bool]
    target_policy: List[float]


class VisitTracingHistoryAgent:
    """
    Thin wrapper around ISMCTSHistoryAgent that exposes the root visit
    distribution for supervised training.

    We instantiate the real history agent internally rather than subclassing it
    directly here, so this file stays decoupled from the agent implementation.
    """

    def __init__(
        self,
        action_mapper: ActionMapper,
        seed: int | None = None,
        **history_agent_kwargs,
    ):
        self.action_mapper = action_mapper
        self.last_root_policy: List[float] | None = None

        module = importlib.import_module("liars_dice.agents.ismcts_3_history")
        AgentCls = getattr(module, "ISMCTSHistoryAgent")
        NodeCls = getattr(module, "Node")
        EdgeStatsCls = getattr(module, "EdgeStats")

        self._agent = AgentCls(seed=seed, **history_agent_kwargs)
        self._Node = NodeCls
        self._EdgeStats = EdgeStatsCls

        # Mirror expected public attribute
        self.name = self._agent.name

    def notify_result(self, obs: Observation, info: dict) -> None:
        return self._agent.notify_result(obs, info)

    def select_action(self, game: LiarsDiceGame, obs: Observation) -> Tuple[str, Any]:
        action, root_policy = self.search_policy(game, obs)
        self.last_root_policy = root_policy
        return action

    def search_policy(
        self,
        game: LiarsDiceGame,
        obs: Observation,
    ) -> Tuple[Tuple[str, Any], List[float]]:
        """
        Reimplements the public search loop of ISMCTSHistoryAgent so we can
        extract root visit counts as a full fixed-size policy target.
        """
        rng = self._agent.rng

        def is_liar(a: Tuple[str, Any]) -> bool:
            return isinstance(a, tuple) and a[0] == "liar"

        root_player = obs.private.my_player
        root_key = self._agent._node_key_from_obs(obs)
        root = self._Node(
            key=root_key,
            player=obs.public.current_player,
        )

        for _ in range(self._agent.sims_per_move):
            g_det = self._agent._determinize_from_game(game, obs)

            node = root
            self._agent._compute_priors_inplace(node, g_det)

            path: List[Tuple[Any, Tuple[str, Any]]] = []
            terminated_in_tree = False

            while True:
                if not node.priors:
                    break

                a = self._agent._select_puct_over_all(node)

                if is_liar(a):
                    before = list(g_det._dice_left)
                    g_det.step(a)
                    after = g_det._dice_left
                    root_lost = after[root_player] < before[root_player]
                    reward = 0.0 if root_lost else 1.0

                    node.edges.setdefault(a, self._EdgeStats())
                    path.append((node, a))
                    self._agent._backup(path, reward)
                    terminated_in_tree = True
                    break

                g_det.step(a)

                if a not in node.children:
                    child_obs = g_det.observe(g_det._current)
                    child_key = self._agent._node_key_from_obs(child_obs)
                    child = self._Node(
                        key=child_key,
                        player=child_obs.public.current_player,
                    )
                    node.children[a] = child
                    node.edges.setdefault(a, self._EdgeStats())
                    path.append((node, a))
                    node = child
                    self._agent._compute_priors_inplace(node, g_det)
                    break
                else:
                    node.edges.setdefault(a, self._EdgeStats())
                    path.append((node, a))
                    node = node.children[a]
                    self._agent._compute_priors_inplace(node, g_det)

            if terminated_in_tree:
                continue

            reward = self._agent._rollout_to_showdown(g_det, root_player)
            self._agent._backup(path, reward)

        legal_now = list(game.legal_actions())

        # Build fixed-size root visit target
        target_policy = [0.0] * self.action_mapper.num_actions
        total_visits = 0

        for action, edge in root.edges.items():
            if action in legal_now:
                total_visits += edge.visit_count

        if total_visits > 0:
            inv_total = 1.0 / total_visits
            for action, edge in root.edges.items():
                if action in legal_now:
                    idx = self.action_mapper.action_to_index(action)
                    target_policy[idx] = edge.visit_count * inv_total
        else:
            # fallback uniform over legal actions
            legal_indices = self.action_mapper.legal_indices(legal_now)
            if legal_indices:
                u = 1.0 / len(legal_indices)
                for idx in legal_indices:
                    target_policy[idx] = u

        # Root action by visit count, matching your earlier agents
        if root.edges:
            scored = [
                (a, e.visit_count) for a, e in root.edges.items() if a in legal_now
            ]
            if scored:
                best_visits = max(v for _, v in scored)
                candidates = [a for a, v in scored if v == best_visits]
                return rng.choice(candidates), target_policy

        if legal_now:
            return rng.choice(legal_now), target_policy

        return ("liar", None), target_policy


class SupervisedSelfPlayCollector:
    """
    Collects supervised policy targets from teacher self-play.

    Each decision point becomes one sample:
      observation encoding -> root visit distribution
    """

    def __init__(
        self,
        teacher: VisitTracingHistoryAgent,
        encoder: ObservationEncoder,
        action_mapper: ActionMapper,
        num_players: int = 4,
        dice_per_player: int = 5,
        seed: int | None = None,
    ):
        self.teacher = teacher
        self.encoder = encoder
        self.action_mapper = action_mapper
        self.num_players = num_players
        self.dice_per_player = dice_per_player
        self.seed = seed

    def collect_games(self, num_games: int) -> List[PolicySample]:
        samples: List[PolicySample] = []

        for game_idx in range(num_games):
            game_seed = None if self.seed is None else self.seed + game_idx

            game = LiarsDiceGame(
                num_players=self.num_players,
                dice_per_player=self.dice_per_player,
                seed=game_seed,
            )

            while True:
                if game.num_alive() <= 1:
                    break

                pid = game._current
                obs = game.observe(pid)

                encoded = self.encoder.encode(obs)
                action = self.teacher.select_action(game, obs)

                if self.teacher.last_root_policy is None:
                    raise RuntimeError("Teacher did not expose root policy")

                samples.append(
                    PolicySample(
                        static_features=encoded["static_features"],
                        bid_history=encoded["bid_history"],
                        bid_mask=encoded["bid_mask"],
                        target_policy=list(self.teacher.last_root_policy),
                    )
                )

                info = game.step(action)
                if info.get("terminal"):
                    break

        return samples

    def save_samples_jsonl(
        self,
        samples: List[PolicySample],
        path: str | Path,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            for sample in samples:
                obj = {
                    "static_features": sample.static_features,
                    "bid_history": sample.bid_history,
                    "bid_mask": sample.bid_mask,
                    "target_policy": sample.target_policy,
                }
                f.write(json.dumps(obj) + "\n")

    def load_samples_jsonl(
        self,
        path: str | Path,
    ) -> List[PolicySample]:
        path = Path(path)
        samples: List[PolicySample] = []

        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} in {path}"
                    ) from exc

                samples.append(
                    PolicySample(
                        static_features=obj["static_features"],
                        bid_history=obj["bid_history"],
                        bid_mask=obj["bid_mask"],
                        target_policy=obj["target_policy"],
                    )
                )

        return samples
