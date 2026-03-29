from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from liars_dice.agents.ismcts_3_history import ISMCTSHistoryAgent
from liars_dice.agents.neural.action_mapping import ActionMapper
from liars_dice.agents.neural.encoder import ObservationEncoder
from liars_dice.agents.neural.nn_model import PolicyNetwork
from liars_dice.core.game import LiarsDiceGame, Observation

Action = Tuple[str, Any]


@dataclass
class PolicySample:
    features: List[float]
    target_policy: List[float]


class PolicyDataset(Dataset):
    def __init__(self, samples: List[PolicySample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        x = torch.tensor(sample.features, dtype=torch.float32)
        y = torch.tensor(sample.target_policy, dtype=torch.float32)
        return x, y


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

        root_player = obs.private.my_player
        root_key = self._node_key_from_obs(obs)
        root = type("NodeShim", (), {})()
        root.key = root_key
        root.player = obs.public.current_player
        root.children = {}
        root.edges = {}
        root.priors = {}
        root.visit_count = 0

        # The parent code is not easily reusable without exposing internals. To avoid modifying the original
        # source file, import the dataclass definitions dynamically from the module object at runtime.
        import importlib

        mod = importlib.import_module(self.__class__.__mro__[1].__module__)
        Node = getattr(mod, "Node")
        EdgeStats = getattr(mod, "EdgeStats")
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


class SupervisedSelfPlayCollector:
    def __init__(
        self,
        teacher: VisitTracingHistoryAgent,
        encoder: ObservationEncoder,
        action_mapper: ActionMapper,
        num_players: int = 4,
        dice_per_player: int = 5,
        seed: Optional[int] = None,
    ):
        self.teacher = teacher
        self.encoder = encoder
        self.action_mapper = action_mapper
        self.num_players = num_players
        self.dice_per_player = dice_per_player
        self.rng = random.Random(seed)

    def collect_games(self, num_games: int) -> List[PolicySample]:
        samples: List[PolicySample] = []
        for game_idx in range(num_games):
            game = LiarsDiceGame(
                num_players=self.num_players,
                dice_per_player=self.dice_per_player,
                seed=self.rng.randint(0, 10**9),
            )

            while True:
                if game.num_alive() <= 1:
                    break

                actor = game._current
                obs = game.observe(actor)
                features = self.encoder.encode(obs)
                action = self.teacher.select_action(game, obs)
                target_policy = self.teacher.last_root_policy
                if target_policy is None:
                    raise RuntimeError("Teacher did not expose root visit distribution")

                samples.append(
                    PolicySample(features=features, target_policy=target_policy)
                )
                info = game.step(action)
                if info.get("terminal"):
                    break
        return samples

    def save_samples_jsonl(self, samples: List[PolicySample], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for sample in samples:
                record = {
                    "features": sample.features,
                    "target_policy": sample.target_policy,
                }
                f.write(json.dumps(record) + "\n")


@dataclass
class TrainingConfig:
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    device: str = "cpu"


def train_policy_network(
    model: PolicyNetwork,
    samples: List[PolicySample],
    config: TrainingConfig,
) -> Dict[str, List[float]]:
    if not samples:
        raise ValueError("No training samples provided")

    dataset = PolicyDataset(samples)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = model.to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: Dict[str, List[float]] = {"loss": []}
    model.train()

    for epoch in range(config.epochs):
        running_loss = 0.0
        batches = 0
        for x, target_policy in loader:
            x = x.to(config.device)
            target_policy = target_policy.to(config.device)

            logits = model(x)
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = -(target_policy * log_probs).sum(dim=-1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batches += 1

        history["loss"].append(running_loss / max(1, batches))

    return history


def save_model_checkpoint(
    model: PolicyNetwork,
    encoder: ObservationEncoder,
    action_mapper: ActionMapper,
    path: str | Path,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "encoder_config": {
            "num_players": encoder.num_players,
            "max_dice_per_player": encoder.max_dice_per_player,
            "max_total_dice": encoder.max_total_dice,
            "history_len": encoder.history_len,
        },
        "action_mapper_config": {
            "max_total_dice": action_mapper.max_total_dice,
        },
    }
    torch.save(payload, Path(path))
