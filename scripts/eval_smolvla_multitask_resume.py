#!/usr/bin/env python3
"""Resumable single-environment LIBERO Object evaluation for a SmolVLA checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def refresh(result: dict) -> None:
    episodes = []
    for row in result["task_results"].values():
        row["successes"] = sum(bool(item["success"]) for item in row["per_episode"])
        row["episodes"] = len(row["per_episode"])
        row["success_rate"] = row["successes"] / row["episodes"] if row["episodes"] else 0.0
        episodes.extend(row["per_episode"])
    result["overall_successes"] = sum(bool(item["success"]) for item in episodes)
    result["overall_episodes"] = len(episodes)
    result["overall_success_rate"] = result["overall_successes"] / len(episodes) if episodes else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=42020)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Missing SmolVLA checkpoint config: {checkpoint}")
    dataset_root = Path(os.environ["LEROBOT_DATASET_ROOT"])
    task_table = pd.read_parquet(dataset_root / "meta/tasks.parquet")
    instructions = {str(task): str(text) for task, text in enumerate(task_table.index.tolist())}
    output = Path(args.output_dir).resolve()
    result_path = output / "instrumentation.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        result = {
            "status": "in_progress", "policy_type": "smolvla", "checkpoint": str(checkpoint),
            "n_action_steps": args.n_action_steps, "episodes_per_task": args.episodes_per_task,
            "seed_rule": f"{args.seed_base} + 10 * task + episode", "videos_saved": False,
            "task_results": {str(task): {"instruction": instructions[str(task)], "per_episode": []} for task in range(10)},
        }

    for task in range(10):
        row = result["task_results"][str(task)]
        finished = {episode["seed"] for episode in row["per_episode"]}
        for episode_index in range(args.episodes_per_task):
            seed = args.seed_base + 10 * task + episode_index
            if seed in finished:
                continue
            run_dir = output / "runs" / f"task{task}" / f"seed_{seed}"
            metrics_path = output / "episodes" / f"task{task}" / f"seed_{seed}.json"
            environment = os.environ | {"LEROBOT_EVAL_NO_VIDEOS": "1", "ACT_SMOKE_EVAL_METRICS_PATH": str(metrics_path), "PYTHONHASHSEED": str(seed)}
            command = [
                sys.executable, str(PROJECT_ROOT / "scripts/eval_policy_instrumented.py"),
                f"--policy.path={checkpoint}", f"--policy.n_action_steps={args.n_action_steps}",
                "--env.type=libero", "--env.task=libero_object", f"--env.task_ids=[{task}]",
                '--env.camera_name_mapping={"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}',
                "--env.observation_height=256", "--env.observation_width=256", "--env.max_parallel_tasks=1",
                "--eval.batch_size=1", "--eval.n_episodes=1", "--eval.recording=false", f"--seed={seed}", f"--output_dir={run_dir}",
            ]
            subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=environment, stdin=subprocess.DEVNULL)
            metrics = json.loads(metrics_path.read_text())
            if metrics.get("status") != "completed" or len(metrics.get("per_episode", [])) != 1:
                raise RuntimeError(f"Incomplete episode: {metrics_path}")
            episode = metrics["per_episode"][0]
            row["per_episode"].append({"task": task, "seed": seed, "instruction": row["instruction"], "success": bool(episode["success"])})
            row["per_episode"].sort(key=lambda item: item["seed"])
            refresh(result)
            write_json(result_path, result)

    refresh(result)
    result["status"] = "completed"
    write_json(result_path, result)
    print(json.dumps({key: result[key] for key in ("overall_successes", "overall_episodes", "overall_success_rate")}, indent=2))


if __name__ == "__main__":
    main()
