#!/usr/bin/env python3
"""Resumable, single-environment π0.5 checkpoint ablation for LIBERO Object.

Each checkpoint/task/seed is evaluated in a fresh process.  This makes the
environment seed and the process-level torch RNG (used by PI05's inference
noise) start from the same condition for every checkpoint comparison.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("outputs/pi05_batch2_checkpoint_ablation_action_steps_10")
RESULT_PATH = ROOT / "instrumentation.json"
DATASET_ROOT = Path("/workspace/jy/datasets/lerobot/libero_object_image")
TRAIN_ROOT = Path("outputs/pi05_libero_object_multitask_batch2_10k")
CHECKPOINTS = {"2.5k": "002500", "5k": "005000", "7.5k": "007500", "10k": "010000"}
EPISODES_PER_TASK = 3


def write_result(result: dict[str, Any]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    temporary = RESULT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(RESULT_PATH)


def checkpoint_path(step: str) -> Path:
    return TRAIN_ROOT / "checkpoints" / step / "pretrained_model"


def load_checkpoint_configs() -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for label, step in CHECKPOINTS.items():
        path = checkpoint_path(step)
        config_path = path / "config.json"
        if not path.is_dir() or not config_path.is_file():
            raise FileNotFoundError(f"Missing π0.5 checkpoint/config for {label}: {path}")
        config = json.loads(config_path.read_text())
        if config.get("type") != "pi05":
            raise ValueError(f"{config_path} is not a pi05 checkpoint: {config.get('type')!r}")
        if config.get("chunk_size", 0) < 10:
            raise ValueError(f"{config_path} chunk settings cannot support n_action_steps=10")
        configs[label] = {
            "checkpoint": str(path),
            "config_path": str(config_path),
            "chunk_size": config["chunk_size"],
            "checkpoint_n_action_steps": config["n_action_steps"],
            "normalization_mapping": config["normalization_mapping"],
            "use_relative_actions": config["use_relative_actions"],
        }
    return configs


def load_instructions() -> dict[str, str]:
    tasks = pd.read_parquet(DATASET_ROOT / "meta" / "tasks.parquet")
    if tasks.index.name != "task" or len(tasks) != 10:
        raise ValueError("Expected ten LIBERO Object task strings in meta/tasks.parquet")
    return {str(index): str(task) for index, task in enumerate(tasks.index.tolist())}


def make_result(instructions: dict[str, str], configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "in_progress",
        "policy_type": "pi05",
        "evaluation_protocol": "single-environment; one fresh process per checkpoint/task/seed",
        "episodes_per_task": EPISODES_PER_TASK,
        "seed_rule": "task t, episode e: 42000 + 100*t + e",
        "policy_inference_rng_rule": "fresh process and --seed equal to environment seed for every episode",
        "videos_saved": False,
        "n_action_steps_override": 10,
        "checkpoints": {
            label: {
                "training_steps": int(step),
                **configs[label],
                "task_results": {
                    str(task): {"instruction": instructions[str(task)], "per_episode": []}
                    for task in range(10)
                },
            }
            for label, step in CHECKPOINTS.items()
        },
    }


def refresh_totals(result: dict[str, Any]) -> None:
    candidates = []
    for label, checkpoint in result["checkpoints"].items():
        all_episodes = []
        for task_result in checkpoint["task_results"].values():
            task_result["successes"] = sum(bool(item["success"]) for item in task_result["per_episode"])
            task_result["episodes"] = len(task_result["per_episode"])
            all_episodes.extend(task_result["per_episode"])
        checkpoint["overall_successes"] = sum(bool(item["success"]) for item in all_episodes)
        checkpoint["overall_episodes"] = len(all_episodes)
        checkpoint["overall_success_rate"] = (
            checkpoint["overall_successes"] / checkpoint["overall_episodes"]
            if checkpoint["overall_episodes"]
            else 0.0
        )
        if checkpoint["overall_episodes"] == 30:
            candidates.append((label, checkpoint))
    result["best_checkpoint"] = (
        max(candidates, key=lambda item: item[1]["overall_success_rate"])[0] if candidates else None
    )


def validate_existing_result(result: dict[str, Any]) -> None:
    if result.get("policy_type") != "pi05" or result.get("n_action_steps_override") != 10:
        raise ValueError(f"Refusing to mix a non-π0.5/action_steps=10 result into {RESULT_PATH}")
    if set(result.get("checkpoints", {})) != set(CHECKPOINTS):
        raise ValueError(f"Checkpoint set in {RESULT_PATH} does not match this ablation")


def run_episode(label: str, task: int, seed: int) -> bool:
    run_dir = ROOT / "runs" / label / f"task{task}" / f"seed_{seed}"
    metrics_path = ROOT / "episodes" / label / f"task{task}" / f"seed_{seed}.json"
    environment = os.environ | {
        "HF_HUB_OFFLINE": "1",
        "LEROBOT_EVAL_NO_VIDEOS": "1",
        "ACT_SMOKE_EVAL_METRICS_PATH": str(metrics_path),
        "PYTHONHASHSEED": str(seed),
    }
    command = [
        sys.executable,
        "scripts/eval_act_smoke_instrumented.py",
        f"--policy.path={checkpoint_path(CHECKPOINTS[label])}",
        "--policy.n_action_steps=10",
        "--env.type=libero",
        "--env.task=libero_object",
        f"--env.task_ids=[{task}]",
        '--env.camera_name_mapping={"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}',
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
    configs = load_checkpoint_configs()
    if RESULT_PATH.exists():
        result = json.loads(RESULT_PATH.read_text())
        validate_existing_result(result)
    else:
        result = make_result(instructions, configs)

    for label in CHECKPOINTS:
        for task in range(10):
            task_result = result["checkpoints"][label]["task_results"][str(task)]
            completed = {item["seed"] for item in task_result["per_episode"]}
            for episode in range(EPISODES_PER_TASK):
                seed = 42000 + 100 * task + episode
                if seed in completed:
                    continue
                success = run_episode(label, task, seed)
                task_result["per_episode"].append(
                    {
                        "checkpoint": label,
                        "task": task,
                        "seed": seed,
                        "environment_seed": seed,
                        "inference_seed": seed,
                        "success": success,
                    }
                )
                task_result["per_episode"].sort(key=lambda item: item["seed"])
                refresh_totals(result)
                write_result(result)

    refresh_totals(result)
    result["status"] = "completed"
    write_result(result)


if __name__ == "__main__":
    main()
