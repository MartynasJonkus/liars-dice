from __future__ import annotations

import torch

from liars_dice.agents.neural.action_mapping import ActionMapper
from liars_dice.agents.neural.encoder import ObservationEncoder
from liars_dice.agents.neural.neural_ismcts import NeuralISMCTSPUCTAgent
from liars_dice.agents.neural.nn_model import PolicyNetwork
from liars_dice.core.game import LiarsDiceGame
from liars_dice.training.training_pipeline import (
    SupervisedSelfPlayCollector,
    TrainingConfig,
    VisitTracingHistoryAgent,
    save_model_checkpoint,
    train_policy_network,
)


def main() -> None:
    action_mapper = ActionMapper(max_total_dice=20)
    encoder = ObservationEncoder(
        num_players=4,
        max_dice_per_player=5,
        max_total_dice=20,
        history_len=5,
    )

    teacher = VisitTracingHistoryAgent(
        action_mapper=action_mapper,
        sims_per_move=300,
        puct_c=1.5,
        hist_beta=1.0,
        seed=42,
    )

    collector = SupervisedSelfPlayCollector(
        teacher=teacher,
        encoder=encoder,
        action_mapper=action_mapper,
        num_players=4,
        dice_per_player=5,
        seed=42,
    )

    print("Collecting supervised samples...")
    samples = collector.collect_games(num_games=20)
    print(f"Collected {len(samples)} decision samples")

    model = PolicyNetwork(
        input_dim=encoder.input_dim,
        num_actions=action_mapper.num_actions,
        hidden_dim=256,
        dropout=0.1,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TrainingConfig(
        batch_size=128,
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=10,
        device=device,
    )

    print(f"Training on {device}...")
    history = train_policy_network(model, samples, config)
    print("Loss history:", history["loss"])

    save_model_checkpoint(
        model, encoder, action_mapper, "/mnt/data/neural_policy_checkpoint.pt"
    )
    collector.save_samples_jsonl(samples, "/mnt/data/supervised_policy_samples.jsonl")

    agent = NeuralISMCTSPUCTAgent(
        model=model,
        encoder=encoder,
        action_mapper=action_mapper,
        sims_per_move=300,
        puct_c=1.5,
        device=device,
        seed=123,
    )

    game = LiarsDiceGame(num_players=4, dice_per_player=5, seed=123)
    obs = game.observe(game._current)
    action = agent.select_action(game, obs)
    print("Example neural action:", action)


if __name__ == "__main__":
    main()
