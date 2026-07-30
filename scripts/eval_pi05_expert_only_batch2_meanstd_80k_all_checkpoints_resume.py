#!/usr/bin/env python3
"""Resumable π0.5 evaluation across all saved expert-only MEAN_STD checkpoints."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("outputs/pi05_expert_only_batch2_meanstd_80k_all_checkpoints_action_steps_10")
RESULT_PATH = ROOT / "instrumentation.json"
SOURCE_RESULT = Path("outputs/pi05_expert_only_batch2_meanstd_40k_checkpoint_ablation_action_steps_10/instrumentation.json")
DATASET_ROOT = Path("/workspace/jy/datasets/lerobot/libero_object_image")
CHECKPOINTS = {
    "10k": Path("outputs/pi05_libero_object_multitask_batch2_meanstd_40k/checkpoints/010000/pretrained_model"),
    "20k": Path("outputs/pi05_libero_object_multitask_batch2_meanstd_40k/checkpoints/020000/pretrained_model"),
    "30k": Path("outputs/pi05_libero_object_multitask_batch2_meanstd_40k/checkpoints/030000/pretrained_model"),
    "40k": Path("outputs/pi05_libero_object_multitask_batch2_meanstd_40k/checkpoints/040000/pretrained_model"),
    "60k": Path("outputs/pi05_libero_object_multitask_batch2_meanstd_80k_resume40k/checkpoints/060000/pretrained_model"),
    "80k": Path("outputs/pi05_libero_object_multitask_batch2_meanstd_80k_resume40k/checkpoints/080000/pretrained_model"),
}
REUSE_LABELS = {"10k", "20k", "40k"}
EPISODES_PER_TASK = 3


def write_result(result: dict[str, Any]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    temporary = RESULT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(RESULT_PATH)


def expected_seeds(task: int) -> list[int]:
    return [42000 + 100 * task + episode for episode in range(EPISODES_PER_TASK)]


def checkpoint_config(label: str, path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    weight_path = path / "model.safetensors"
    if not config_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(f"Missing complete checkpoint for {label}: {path}")
    config = json.loads(config_path.read_text())
    expected_norm = {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
    if (
        config.get("type") != "pi05"
        or config.get("chunk_size") != 50
        or config.get("n_action_steps") != 10
        or config.get("normalization_mapping") != expected_norm
        or config.get("use_peft")
        or not config.get("train_expert_only")
    ):
        raise ValueError(f"Unexpected expert-only MEAN_STD config: {config_path}")
    return {
        "checkpoint": str(path),
        "config_path": str(config_path),
        "weight_path": str(weight_path),
        "chunk_size": config["chunk_size"],
        "checkpoint_n_action_steps": config["n_action_steps"],
        "normalization_mapping": config["normalization_mapping"],
        "use_relative_actions": config["use_relative_actions"],
    }


def instructions() -> dict[str, str]:
    tasks = pd.read_parquet(DATASET_ROOT / "meta" / "tasks.parquet")
    if tasks.index.name != "task" or len(tasks) != 10:
        raise ValueError("Expected ten LIBERO Object task strings")
    return {str(i): str(task) for i, task in enumerate(tasks.index.tolist())}


def task_results(task_text: dict[str, str]) -> dict[str, Any]:
    return {str(task): {"instruction": task_text[str(task)], "per_episode": []} for task in range(10)}


def validate_reuse(source: dict[str, Any]) -> None:
    if source.get("status") != "completed" or source.get("n_action_steps_override") != 10:
        raise ValueError(f"Source evaluation is incomplete or incompatible: {SOURCE_RESULT}")
    for label in REUSE_LABELS:
        for task in range(10):
            episodes = source["checkpoints"][label]["task_results"][str(task)]["per_episode"]
            if [episode["seed"] for episode in episodes] != expected_seeds(task):
                raise ValueError(f"Unexpected source seeds for {label}, task {task}")


def make_result() -> dict[str, Any]:
    source = json.loads(SOURCE_RESULT.read_text())
    validate_reuse(source)
    task_text = instructions()
    checkpoints: dict[str, Any] = {}
    for label, path in CHECKPOINTS.items():
        metadata = checkpoint_config(label, path)
        if label in REUSE_LABELS:
            checkpoint = source["checkpoints"][label]
            checkpoint["reused_from"] = str(SOURCE_RESULT)
            checkpoint.update(metadata)
            checkpoints[label] = checkpoint
        else:
            checkpoints[label] = {
                "training_steps": int(label.removesuffix("k")) * 1000,
                **metadata,
                "task_results": task_results(task_text),
            }
    return {
        "status": "in_progress",
        "policy_type": "pi05",
        "evaluation_protocol": "single-environment; one fresh process per checkpoint/task/seed",
        "episodes_per_task": EPISODES_PER_TASK,
        "seed_rule": "task t, episode e: 42000 + 100*t + e",
        "seed_protocol": "matches previous π0.5 30-episode ablations",
        "policy_inference_rng_rule": "fresh process and --seed equal to environment seed for every episode",
        "videos_saved": False,
        "n_action_steps_override": 10,
        "reused_results": {"10k_20k_40k": str(SOURCE_RESULT)},
        "checkpoints": checkpoints,
    }


def refresh_totals(result: dict[str, Any]) -> None:
    complete = []
    for label, checkpoint in result["checkpoints"].items():
        episodes = []
        for task_result in checkpoint["task_results"].values():
            task_result["successes"] = sum(bool(item["success"]) for item in task_result["per_episode"])
            task_result["episodes"] = len(task_result["per_episode"])
            episodes.extend(task_result["per_episode"])
        checkpoint["overall_successes"] = sum(bool(item["success"]) for item in episodes)
        checkpoint["overall_episodes"] = len(episodes)
        checkpoint["overall_success_rate"] = checkpoint["overall_successes"] / len(episodes) if episodes else 0.0
        if len(episodes) == 30:
            complete.append((label, checkpoint))
    result["best_checkpoint"] = max(complete, key=lambda item: item[1]["overall_success_rate"])[0] if complete else None


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
        sys.executable, "scripts/eval_act_smoke_instrumented.py",
        f"--policy.path={CHECKPOINTS[label]}", "--policy.n_action_steps=10",
        "--env.type=libero", "--env.task=libero_object", f"--env.task_ids=[{task}]",
        '--env.camera_name_mapping={"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}',
        "--env.observation_height=256", "--env.observation_width=256", "--env.max_parallel_tasks=1",
        "--eval.batch_size=1", "--eval.n_episodes=1", "--eval.recording=false",
        f"--seed={seed}", f"--output_dir={run_dir}",
    ]
    subprocess.run(command, check=True, env=environment)
    raw = json.loads(metrics_path.read_text())
    episodes = raw.get("per_episode", [])
    if raw.get("status") != "completed" or len(episodes) != 1:
        raise RuntimeError(f"Incomplete episode metrics: {metrics_path}")
    return bool(episodes[0]["success"])


def main() -> None:
    result = json.loads(RESULT_PATH.read_text()) if RESULT_PATH.exists() else make_result()
    if set(result.get("checkpoints", {})) != set(CHECKPOINTS):
        raise ValueError(f"Checkpoint set in {RESULT_PATH} does not match this evaluation")
    for label in CHECKPOINTS:
        if label in REUSE_LABELS:
            continue
        for task in range(10):
            task_result = result["checkpoints"][label]["task_results"][str(task)]
            completed = {item["seed"] for item in task_result["per_episode"]}
            for seed in expected_seeds(task):
                if seed in completed:
                    continue
                task_result["per_episode"].append({
                    "checkpoint": label, "task": task, "seed": seed,
                    "environment_seed": seed, "inference_seed": seed,
                    "success": run_episode(label, task, seed),
                })
                task_result["per_episode"].sort(key=lambda item: item["seed"])
                refresh_totals(result)
                write_result(result)
    refresh_totals(result)
    result["status"] = "completed"
    write_result(result)


if __name__ == "__main__":
    main()
