from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from liars_dice.agents.neural.action_mapping import ActionMapper
from liars_dice.agents.neural.encoder import ObservationEncoder
from liars_dice.training.data_collection import (
    SupervisedSelfPlayCollector,
    VisitTracingHistoryAgent,
)


def _collect_worker(config: Dict) -> Dict:
    """
    Worker process entrypoint.

    Each worker creates its own teacher + collector, collects a shard of games,
    writes that shard to disk as JSONL, and returns only lightweight metadata.
    This avoids sending large sample lists back through multiprocessing.
    """
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


def _save_manifest(manifest_path: Path, payload: Dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect supervised policy targets from parallel History-agent self-play."
    )

    parser.add_argument(
        "--num-games", type=int, default=400, help="Total number of games to collect."
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/supervised_data")
    parser.add_argument("--merged-name", type=str, default="supervised_samples.jsonl")
    parser.add_argument("--base-seed", type=int, default=12345)

    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--dice-per-player", type=int, default=5)
    parser.add_argument("--history-len", type=int, default=5)

    parser.add_argument("--sims-per-move", type=int, default=400)
    parser.add_argument("--puct-c", type=float, default=1.5)
    parser.add_argument("--prior-tau", type=float, default=1.0)
    parser.add_argument("--liar-exp", type=float, default=0.5)
    parser.add_argument("--prior-floor", type=float, default=1e-6)
    parser.add_argument("--hist-beta", type=float, default=1.0)
    parser.add_argument("--hist-gamma", type=float, default=0.0)
    parser.add_argument("--rollout-theta", type=float, default=0.40)
    parser.add_argument("--rollout-alpha", type=float, default=0.70)
    parser.add_argument("--rollout-eps", type=float, default=0.15)
    parser.add_argument("--rollout-max-steps", type=int, default=40)

    args = parser.parse_args()

    if args.num_players != 4:
        raise ValueError("This runner is intended for the fixed 4-player thesis setup.")

    if args.num_games <= 0:
        raise ValueError("--num-games must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_total_dice = args.num_players * args.dice_per_player

    worker_count = min(args.workers, args.num_games)
    games_per_worker = math.ceil(args.num_games / worker_count)

    worker_configs: List[Dict] = []
    remaining_games = args.num_games
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
                "base_seed": args.base_seed,
                "num_players": args.num_players,
                "dice_per_player": args.dice_per_player,
                "max_total_dice": max_total_dice,
                "history_len": args.history_len,
                "sims_per_move": args.sims_per_move,
                "puct_c": args.puct_c,
                "prior_tau": args.prior_tau,
                "liar_exp": args.liar_exp,
                "prior_floor": args.prior_floor,
                "hist_beta": args.hist_beta,
                "hist_gamma": args.hist_gamma,
                "rollout_theta": args.rollout_theta,
                "rollout_alpha": args.rollout_alpha,
                "rollout_eps": args.rollout_eps,
                "rollout_max_steps": args.rollout_max_steps,
            }
        )

    print(
        f"Collecting {args.num_games} games using {len(worker_configs)} worker processes..."
    )

    results: List[Dict] = []
    with ProcessPoolExecutor(max_workers=len(worker_configs)) as ex:
        futures = [ex.submit(_collect_worker, cfg) for cfg in worker_configs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"Worker {result['worker_id']:02d} finished: "
                f"{result['num_games']} games, {result['num_samples']} samples"
            )

    results.sort(key=lambda x: x["worker_id"])
    shard_paths = [Path(r["shard_path"]) for r in results]

    merged_path = output_dir / args.merged_name
    merged_samples = _merge_shards(shard_paths, merged_path)

    manifest = {
        "num_games": args.num_games,
        "workers": len(worker_configs),
        "sims_per_move": args.sims_per_move,
        "num_players": args.num_players,
        "dice_per_player": args.dice_per_player,
        "history_len": args.history_len,
        "max_total_dice": max_total_dice,
        "merged_samples": merged_samples,
        "merged_path": str(merged_path),
        "shards": results,
    }
    _save_manifest(output_dir / "manifest.json", manifest)

    print(f"Done. Merged {merged_samples} samples into: {merged_path}")
    print(f"Manifest written to: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
