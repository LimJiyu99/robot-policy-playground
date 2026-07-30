#!/usr/bin/env python3
"""Render five representative successful LIBERO Object rollouts from π0.5 6K."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


CHECKPOINT = Path("outputs/pi05_libero_base_expert_only_6k/checkpoints/006000/pretrained_model")
ROOT = Path("outputs/pi05_libero_base_expert_only_6k_demo_videos5")
RESULT_PATH = ROOT / "instrumentation.json"
DATASET_ROOT = Path("/workspace/jy/datasets/lerobot/libero_object_image")
TASKS = (0, 1, 2, 3, 4)


def seed_for(task: int) -> int:
    # First seed for each task from the completed 100-seed evaluation.
    return 42020 + 10 * task


def write_json(data: dict) -> None:
    temporary = RESULT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(RESULT_PATH)


def task_instructions() -> dict[str, str]:
    tasks = pd.read_parquet(DATASET_ROOT / "meta" / "tasks.parquet")
    if tasks.index.name != "task" or len(tasks) != 10:
        raise ValueError("Expected the ten LIBERO Object task instructions")
    return {str(index): str(value) for index, value in enumerate(tasks.index.tolist())}


def run_task(task: int, seed: int) -> dict:
    run_dir = ROOT / "runs" / f"task{task}" / f"seed_{seed}"
    metrics_path = ROOT / "episodes" / f"task{task}" / f"seed_{seed}.json"
    environment = os.environ | {
        "ACT_SMOKE_EVAL_METRICS_PATH": str(metrics_path),
        "PYTHONHASHSEED": str(seed),
    }
    command = [
        sys.executable,
        "scripts/eval_act_smoke_instrumented.py",
        f"--policy.path={CHECKPOINT}",
        "--policy.n_action_steps=10",
        "--env.type=libero",
        "--env.task=libero_object",
        f"--env.task_ids=[{task}]",
        "--env.control_mode=relative",
        '--env.camera_name_mapping={"agentview_image":"image","robot0_eye_in_hand_image":"image2"}',
        "--env.observation_height=256",
        "--env.observation_width=256",
        "--env.max_parallel_tasks=1",
        "--eval.batch_size=1",
        "--eval.n_episodes=1",
        # This uses the evaluator's direct render callback; no recording dataset
        # is created and the resulting main-camera MP4 is returned in metrics.
        "--eval.recording=false",
        f"--seed={seed}",
        f"--output_dir={run_dir}",
    ]
    subprocess.run(command, check=True, env=environment, stdin=subprocess.DEVNULL)
    metrics = json.loads(metrics_path.read_text())
    episodes = metrics.get("per_episode", [])
    videos = [Path(path) for path in metrics.get("video_paths", [])]
    if metrics.get("status") != "completed" or len(episodes) != 1 or len(videos) != 1:
        raise RuntimeError(f"Incomplete video rollout: {metrics_path}")
    if not videos[0].is_file():
        raise FileNotFoundError(f"Recorded MP4 missing: {videos[0]}")
    episode = episodes[0]
    outcome = "success" if episode["success"] else "failure"
    destination = ROOT / "videos" / f"task{task}_seed{seed}_{outcome}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite video: {destination}")
    videos[0].replace(destination)
    return {
        "task": task,
        "seed": seed,
        "success": bool(episode["success"]),
        "sum_reward": episode.get("sum_reward"),
        "max_reward": episode.get("max_reward"),
        "video_path": str(destination),
        "metrics_path": str(metrics_path),
    }


def main() -> None:
    if not CHECKPOINT.is_dir():
        raise FileNotFoundError(f"Missing π0.5 6K checkpoint: {CHECKPOINT}")
    if ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {ROOT}")
    instructions = task_instructions()
    ROOT.mkdir(parents=True)
    result = {
        "status": "in_progress",
        "policy_type": "pi05",
        "checkpoint": str(CHECKPOINT),
        "source_evaluation": "outputs/pi05_libero_base_vs_expert_only_6k_eval100/instrumentation.json",
        "n_action_steps": 10,
        "env_control_mode": "relative",
        "episodes": [],
    }
    for task in TASKS:
        row = run_task(task, seed_for(task))
        row["instruction"] = instructions[str(task)]
        result["episodes"].append(row)
        write_json(result)
    result["status"] = "completed"
    result["successes"] = sum(row["success"] for row in result["episodes"])
    result["total_episodes"] = len(result["episodes"])
    write_json(result)


if __name__ == "__main__":
    main()
