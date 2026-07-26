#!/usr/bin/env python3
"""Resumable single-environment LIBERO sanity check for the official π0.5 policy."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("outputs/pi05_official_libero_finetuned_sanity30")
RESULT_PATH = ROOT / "instrumentation.json"
DATASET_ROOT = Path("/workspace/jy/datasets/lerobot/libero_object_image")
POLICY_PATH = "lerobot/pi05_libero_finetuned"
EPISODES_PER_TASK = 3


def write_result(result: dict[str, Any]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    temporary = RESULT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(RESULT_PATH)


def load_instructions() -> dict[str, str]:
    tasks = pd.read_parquet(DATASET_ROOT / "meta" / "tasks.parquet")
    if tasks.index.name != "task" or len(tasks) != 10:
        raise ValueError("Expected ten LIBERO Object task strings in meta/tasks.parquet")
    return {str(index): str(task) for index, task in enumerate(tasks.index.tolist())}


def make_result(instructions: dict[str, str]) -> dict[str, Any]:
    return {
        "status": "in_progress",
        "policy_type": "pi05",
        "policy_path": POLICY_PATH,
        "evaluation_protocol": "single-environment; one fresh process per task/seed",
        "episodes_per_task": EPISODES_PER_TASK,
        "seed_rule": "task t, episode e: 42000 + 100*t + e",
        "n_action_steps_override": 10,
        "env_control_mode": "relative",
        "camera_name_mapping": {
            "agentview_image": "image",
            "robot0_eye_in_hand_image": "image2",
        },
        "videos_saved": False,
        "task_results": {
            str(task): {"instruction": instructions[str(task)], "per_episode": []} for task in range(10)
        },
    }


def refresh_totals(result: dict[str, Any]) -> None:
    episodes = []
    for task_result in result["task_results"].values():
        task_result["successes"] = sum(bool(item["success"]) for item in task_result["per_episode"])
        task_result["episodes"] = len(task_result["per_episode"])
        episodes.extend(task_result["per_episode"])
    result["overall_successes"] = sum(bool(item["success"]) for item in episodes)
    result["overall_episodes"] = len(episodes)
    result["overall_success_rate"] = result["overall_successes"] / len(episodes) if episodes else 0.0


def validate_existing_result(result: dict[str, Any]) -> None:
    if result.get("policy_path") != POLICY_PATH or result.get("env_control_mode") != "relative":
        raise ValueError(f"Refusing to mix a different sanity protocol into {RESULT_PATH}")


def run_episode(task: int, seed: int) -> bool:
    run_dir = ROOT / "runs" / f"task{task}" / f"seed_{seed}"
    metrics_path = ROOT / "episodes" / f"task{task}" / f"seed_{seed}.json"
    environment = os.environ | {
        "LEROBOT_EVAL_NO_VIDEOS": "1",
        "ACT_SMOKE_EVAL_METRICS_PATH": str(metrics_path),
        "PYTHONHASHSEED": str(seed),
    }
    command = [
        sys.executable,
        "scripts/eval_act_smoke_instrumented.py",
        f"--policy.path={POLICY_PATH}",
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
        "--eval.recording=false",
        f"--seed={seed}",
        f"--output_dir={run_dir}",
    ]
    subprocess.run(command, check=True, env=environment)
    raw = json.loads(metrics_path.read_text())
    episodes = raw.get("per_episode", [])
    if raw.get("status") != "completed" or len(episodes) != 1:
        raise RuntimeError(f"Incomplete episode metrics: {metrics_path}")
    return bool(episodes[0]["success"])


def main() -> None:
    instructions = load_instructions()
    result = json.loads(RESULT_PATH.read_text()) if RESULT_PATH.exists() else make_result(instructions)
    validate_existing_result(result)
    for task in range(10):
        task_result = result["task_results"][str(task)]
        completed = {item["seed"] for item in task_result["per_episode"]}
        for episode in range(EPISODES_PER_TASK):
            seed = 42000 + 100 * task + episode
            if seed in completed:
                continue
            task_result["per_episode"].append(
                {"task": task, "seed": seed, "success": run_episode(task, seed)}
            )
            task_result["per_episode"].sort(key=lambda item: item["seed"])
            refresh_totals(result)
            write_result(result)
    refresh_totals(result)
    result["status"] = "completed"
    write_result(result)


if __name__ == "__main__":
    main()
