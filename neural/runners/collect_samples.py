from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from pathlib import Path
from queue import Empty
from typing import Any, Dict, Iterable, List

from tqdm import tqdm

from liars_dice.core.game import LiarsDiceGame

from neural.common.action_mapping import ActionMapper
from neural.common.tracing_agent import VisitTracingHistoryAgent

from neural.basic_mlp.encoder_mlp import ObservationEncoder as MLPObservationEncoder
from neural.basic_mlp.data_collection_mlp import make_policy_sample as make_mlp_sample

from neural.trans_mlp.encoder_trans import ObservationEncoder as TransObservationEncoder
from neural.trans_mlp.data_collection_trans import (
    make_policy_sample as make_trans_sample,
)

# =====================
# CONFIG
# =====================
NUM_GAMES = 10
WORKERS = max(1, (os.cpu_count() or 2) - 1)

OUTPUT_DIR = Path("artifacts/data")
MLP_DIR = OUTPUT_DIR / "mlp"
TRANS_DIR = OUTPUT_DIR / "trans"
SHARD_DIR = OUTPUT_DIR / "shards"

MERGED_NAME = "supervised_samples.jsonl"

BASE_SEED = 12345

NUM_PLAYERS = 4
DICE_PER_PLAYER = 5
MAX_TOTAL_DICE = NUM_PLAYERS * DICE_PER_PLAYER

# Keep both models on the same bid-history window.
HISTORY_LEN = 10
MAX_BIDS = 10

SIMS_PER_MOVE = 1000

PUCT_C = 1.5
PRIOR_TAU = 1.0
LIAR_EXP = 0.5
PRIOR_FLOOR = 1e-6

HIST_BETA = 1.0

ROLLOUT_THETA = 0.30
ROLLOUT_ALPHA = 0.60
ROLLOUT_EPS = 0.20
ROLLOUT_MAX_STEPS = 40


# =====================
# IO HELPERS
# =====================
def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            count += 1

    return count


def merge_jsonl(shard_paths: List[Path], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for shard_path in shard_paths:
            with shard_path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    total += 1

    return total


def make_worker_configs() -> List[Dict[str, Any]]:
    worker_count = min(WORKERS, NUM_GAMES)
    games_per_worker = math.ceil(NUM_GAMES / worker_count)

    configs: List[Dict[str, Any]] = []
    remaining_games = NUM_GAMES

    for worker_id in range(worker_count):
        this_worker_games = min(games_per_worker, remaining_games)
        if this_worker_games <= 0:
            break

        remaining_games -= this_worker_games

        configs.append(
            {
                "worker_id": worker_id,
                "num_games": this_worker_games,
                "base_seed": BASE_SEED,
                "shard_dir": str(SHARD_DIR),
                "num_players": NUM_PLAYERS,
                "dice_per_player": DICE_PER_PLAYER,
                "max_total_dice": MAX_TOTAL_DICE,
                "history_len": HISTORY_LEN,
                "max_bids": MAX_BIDS,
                "sims_per_move": SIMS_PER_MOVE,
                "puct_c": PUCT_C,
                "prior_tau": PRIOR_TAU,
                "liar_exp": LIAR_EXP,
                "prior_floor": PRIOR_FLOOR,
                "hist_beta": HIST_BETA,
                "rollout_theta": ROLLOUT_THETA,
                "rollout_alpha": ROLLOUT_ALPHA,
                "rollout_eps": ROLLOUT_EPS,
                "rollout_max_steps": ROLLOUT_MAX_STEPS,
            }
        )

    return configs


# =====================
# COLLECTION LOGIC
# =====================
def collect_joint_samples(config: Dict[str, Any], progress_queue) -> Dict[str, Any]:
    worker_id = int(config["worker_id"])
    num_games = int(config["num_games"])
    base_seed = int(config["base_seed"])
    sims_per_move = int(config["sims_per_move"])

    if sims_per_move <= 0:
        raise ValueError("sims_per_move must be a positive integer for data collection")

    shard_dir = Path(config["shard_dir"])
    shard_dir.mkdir(parents=True, exist_ok=True)

    action_mapper = ActionMapper(max_total_dice=int(config["max_total_dice"]))

    mlp_encoder = MLPObservationEncoder(
        num_players=int(config["num_players"]),
        max_dice_per_player=int(config["dice_per_player"]),
        max_total_dice=int(config["max_total_dice"]),
        history_len=int(config["history_len"]),
    )

    trans_encoder = TransObservationEncoder(
        num_players=int(config["num_players"]),
        max_dice_per_player=int(config["dice_per_player"]),
        max_total_dice=int(config["max_total_dice"]),
        max_bids=int(config["max_bids"]),
    )

    teacher = VisitTracingHistoryAgent(
        action_mapper=action_mapper,
        label=f"History-Teacher-W{worker_id}",
        sims_per_move=sims_per_move,
        time_limit_s=None,
        seed=base_seed + worker_id,
        puct_c=float(config["puct_c"]),
        prior_tau=float(config["prior_tau"]),
        liar_exp=float(config["liar_exp"]),
        prior_floor=float(config["prior_floor"]),
        hist_beta=float(config["hist_beta"]),
        rollout_theta=float(config["rollout_theta"]),
        rollout_alpha=float(config["rollout_alpha"]),
        rollout_eps=float(config["rollout_eps"]),
        rollout_max_steps=int(config["rollout_max_steps"]),
    )

    mlp_rows: List[Dict[str, Any]] = []
    trans_rows: List[Dict[str, Any]] = []

    sample_id = 0

    for game_idx in range(num_games):
        game_seed = base_seed + 10_000 + worker_id * 1_000_000 + game_idx

        game = LiarsDiceGame(
            num_players=int(config["num_players"]),
            dice_per_player=int(config["dice_per_player"]),
            seed=game_seed,
        )

        while True:
            pid = game._current
            obs = game.observe(pid)

            # Run the teacher once for this exact pre-action observation.
            # The teacher should search using cloned/determinized games and must not
            # mutate the real game state.
            action = teacher.select_action(game, obs)
            target_policy = teacher.last_root_policy

            if target_policy is None:
                raise RuntimeError("Teacher did not expose last_root_policy")

            target_policy = list(target_policy)
            target_mass = sum(float(x) for x in target_policy)

            # A valid MCTS visit distribution should have positive mass.
            # If this ever fails, skip the training sample but still play the action.
            if target_mass > 0.0:
                common_meta = {
                    "worker_id": worker_id,
                    "game_idx": game_idx,
                    "sample_id": sample_id,
                    "player": pid,
                }

                mlp_sample = make_mlp_sample(
                    encoder=mlp_encoder,
                    obs=obs,
                    target_policy=target_policy,
                )

                trans_sample = make_trans_sample(
                    encoder=trans_encoder,
                    obs=obs,
                    target_policy=target_policy,
                )

                mlp_rows.append(
                    {
                        **common_meta,
                        **mlp_sample.to_json_dict(),
                    }
                )

                trans_rows.append(
                    {
                        **common_meta,
                        **trans_sample.to_json_dict(),
                    }
                )

                sample_id += 1

            info = game.step(action)
            if info.get("terminal"):
                break

        # Report one completed game to the main process.
        progress_queue.put(1)

    mlp_shard = shard_dir / f"mlp_worker_{worker_id:02d}.jsonl"
    trans_shard = shard_dir / f"trans_worker_{worker_id:02d}.jsonl"

    mlp_count = write_jsonl(mlp_shard, mlp_rows)
    trans_count = write_jsonl(trans_shard, trans_rows)

    if mlp_count != trans_count:
        raise RuntimeError(
            f"Worker {worker_id}: MLP and Transformer sample counts differ: "
            f"{mlp_count} vs {trans_count}"
        )

    return {
        "worker_id": worker_id,
        "num_games": num_games,
        "num_samples": mlp_count,
        "mlp_shard": str(mlp_shard),
        "trans_shard": str(trans_shard),
    }


# =====================
# MAIN
# =====================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MLP_DIR.mkdir(parents=True, exist_ok=True)
    TRANS_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    worker_configs = make_worker_configs()

    print(f"Collecting {NUM_GAMES} games using {len(worker_configs)} workers...")
    print(f"MLP output:   {MLP_DIR / MERGED_NAME}")
    print(f"Trans output: {TRANS_DIR / MERGED_NAME}")

    results: List[Dict[str, Any]] = []

    with Manager() as manager:
        progress_queue = manager.Queue()

        with ProcessPoolExecutor(max_workers=len(worker_configs)) as ex:
            futures = [
                ex.submit(collect_joint_samples, cfg, progress_queue)
                for cfg in worker_configs
            ]

            completed_games = 0

            with tqdm(total=NUM_GAMES, desc="Games collected") as pbar:
                while completed_games < NUM_GAMES:
                    # Surface worker errors instead of hanging forever.
                    for future in futures:
                        if future.done():
                            exc = future.exception()
                            if exc is not None:
                                raise exc

                    try:
                        progress_queue.get(timeout=1.0)
                    except Empty:
                        continue

                    completed_games += 1
                    pbar.update(1)

            results = [future.result() for future in futures]

    results.sort(key=lambda r: r["worker_id"])

    mlp_shards = [Path(r["mlp_shard"]) for r in results]
    trans_shards = [Path(r["trans_shard"]) for r in results]

    print("Merging MLP shards...")
    mlp_total = merge_jsonl(mlp_shards, MLP_DIR / MERGED_NAME)

    print("Merging Transformer shards...")
    trans_total = merge_jsonl(trans_shards, TRANS_DIR / MERGED_NAME)

    if mlp_total != trans_total:
        raise RuntimeError(
            f"Merged sample count mismatch: MLP={mlp_total}, Transformer={trans_total}"
        )

    total_worker_samples = sum(int(r["num_samples"]) for r in results)
    if total_worker_samples != mlp_total:
        raise RuntimeError(
            f"Worker sample total does not match merged count: "
            f"{total_worker_samples} vs {mlp_total}"
        )

    summary = {
        "num_games": NUM_GAMES,
        "workers": len(worker_configs),
        "samples": mlp_total,
        "num_players": NUM_PLAYERS,
        "dice_per_player": DICE_PER_PLAYER,
        "max_total_dice": MAX_TOTAL_DICE,
        "history_len": HISTORY_LEN,
        "max_bids": MAX_BIDS,
        "sims_per_move": SIMS_PER_MOVE,
        "puct_c": PUCT_C,
        "prior_tau": PRIOR_TAU,
        "liar_exp": LIAR_EXP,
        "prior_floor": PRIOR_FLOOR,
        "hist_beta": HIST_BETA,
        "rollout_theta": ROLLOUT_THETA,
        "rollout_alpha": ROLLOUT_ALPHA,
        "rollout_eps": ROLLOUT_EPS,
        "rollout_max_steps": ROLLOUT_MAX_STEPS,
        "mlp_output": str(MLP_DIR / MERGED_NAME),
        "trans_output": str(TRANS_DIR / MERGED_NAME),
    }

    summary_path = OUTPUT_DIR / "collection_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Samples: {mlp_total}")
    print(f"Saved MLP data to: {MLP_DIR / MERGED_NAME}")
    print(f"Saved Transformer data to: {TRANS_DIR / MERGED_NAME}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
