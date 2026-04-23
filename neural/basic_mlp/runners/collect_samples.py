from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from neural.action_mapping import ActionMapper
from neural.basic_mlp.data_collection import (
    SupervisedSelfPlayCollector,
    VisitTracingHistoryAgent,
)
from neural.basic_mlp.encoder import ObservationEncoder

NUM_GAMES = 1000
WORKERS = max(1, (os.cpu_count() or 2) - 1)
OUTPUT_DIR = "artifacts/data_basic"
MERGED_NAME = "supervised_samples.jsonl"
BASE_SEED = 12345

NUM_PLAYERS = 4
DICE_PER_PLAYER = 5
HISTORY_LEN = 5

SIMS_PER_MOVE = 500
PUCT_C = 1.5
PRIOR_TAU = 1.0
LIAR_EXP = 0.5
PRIOR_FLOOR = 1e-6
HIST_BETA = 1.0
HIST_GAMMA = 0.0
ROLLOUT_THETA = 0.40
ROLLOUT_ALPHA = 0.70
ROLLOUT_EPS = 0.15
ROLLOUT_MAX_STEPS = 40


# =====================
# WORKER
# =====================
def _collect_worker(config: Dict) -> Dict:
    worker_id = int(config["worker_id"])
    num_games = int(config["num_games"])
    output_dir = Path(config["output_dir"])
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    action_mapper = ActionMapper(max_total_dice=int(config["max_total_dice"]))
    encoder = ObservationEncoder(
        num_players=int(config["num_players"]),
        max_dice_per_player=int(config["dice_per_player"]),
        max_total_dice=int(config["max_total_dice"]),
        history_len=int(config["history_len"]),
    )

    teacher = VisitTracingHistoryAgent(
        action_mapper=action_mapper,
        label=f"History-Teacher-W{worker_id}",
        sims_per_move=int(config["sims_per_move"]),
        seed=int(config["base_seed"]) + worker_id,
        puct_c=float(config["puct_c"]),
        prior_tau=float(config["prior_tau"]),
        liar_exp=float(config["liar_exp"]),
        prior_floor=float(config["prior_floor"]),
        hist_beta=float(config["hist_beta"]),
        hist_gamma=float(config["hist_gamma"]),
        rollout_theta=float(config["rollout_theta"]),
        rollout_alpha=float(config["rollout_alpha"]),
        rollout_eps=float(config["rollout_eps"]),
        rollout_max_steps=int(config["rollout_max_steps"]),
    )

    collector = SupervisedSelfPlayCollector(
        teacher=teacher,
        encoder=encoder,
        action_mapper=action_mapper,
        num_players=int(config["num_players"]),
        dice_per_player=int(config["dice_per_player"]),
        seed=int(config["base_seed"]) + 10_000 + worker_id,
    )

    samples = collector.collect_games(num_games=num_games)
    shard_path = shard_dir / f"samples_worker_{worker_id:02d}.jsonl"
    collector.save_samples_jsonl(samples, shard_path)

    return {
        "worker_id": worker_id,
        "num_games": num_games,
        "num_samples": len(samples),
        "shard_path": str(shard_path),
    }


# =====================
# MERGE
# =====================
def _merge_shards(shard_paths: List[Path], merged_path: Path) -> int:
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    total_lines = 0

    with merged_path.open("w", encoding="utf-8") as fout:
        for shard_path in shard_paths:
            with shard_path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    total_lines += 1

    return total_lines


# =====================
# MAIN
# =====================
def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_total_dice = NUM_PLAYERS * DICE_PER_PLAYER

    worker_count = min(WORKERS, NUM_GAMES)
    games_per_worker = math.ceil(NUM_GAMES / worker_count)

    worker_configs: List[Dict] = []
    remaining_games = NUM_GAMES

    for worker_id in range(worker_count):
        this_worker_games = min(games_per_worker, remaining_games)
        if this_worker_games <= 0:
            break
        remaining_games -= this_worker_games

        worker_configs.append(
            {
                "worker_id": worker_id,
                "num_games": this_worker_games,
                "output_dir": str(output_dir),
                "base_seed": BASE_SEED,
                "num_players": NUM_PLAYERS,
                "dice_per_player": DICE_PER_PLAYER,
                "max_total_dice": max_total_dice,
                "history_len": HISTORY_LEN,
                "sims_per_move": SIMS_PER_MOVE,
                "puct_c": PUCT_C,
                "prior_tau": PRIOR_TAU,
                "liar_exp": LIAR_EXP,
                "prior_floor": PRIOR_FLOOR,
                "hist_beta": HIST_BETA,
                "hist_gamma": HIST_GAMMA,
                "rollout_theta": ROLLOUT_THETA,
                "rollout_alpha": ROLLOUT_ALPHA,
                "rollout_eps": ROLLOUT_EPS,
                "rollout_max_steps": ROLLOUT_MAX_STEPS,
            }
        )

    print(f"Collecting {NUM_GAMES} games using {len(worker_configs)} workers...")

    results: List[Dict] = []

    with ProcessPoolExecutor(max_workers=len(worker_configs)) as ex:
        futures = [ex.submit(_collect_worker, cfg) for cfg in worker_configs]

        # Progress bar for workers
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Workers finished"
        ):
            result = future.result()
            results.append(result)

    results.sort(key=lambda x: x["worker_id"])
    shard_paths = [Path(r["shard_path"]) for r in results]

    print("Merging shards...")
    merged_path = output_dir / MERGED_NAME
    merged_samples = _merge_shards(shard_paths, merged_path)

    print(f"Done. Total samples: {merged_samples}")
    print(f"Saved to: {merged_path}")


if __name__ == "__main__":
    main()
