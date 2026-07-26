#!/usr/bin/env python3
"""Resumable same-seed comparison of lerobot/pi05_libero_base and local expert-only π0.5 80K."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from lerobot.configs.policies import PreTrainedConfig
# Register the ``pi05`` Draccus choice before policy_metadata calls from_pretrained.
# Rollout loading remains delegated to LeRobot's official --policy.path loader.
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401


ROOT = Path("outputs/pi05_libero_base_vs_expert_only_80k_eval30")
RESULT_PATH = ROOT / "instrumentation.json"
DATASET_ROOT = Path("/workspace/jy/datasets/lerobot/libero_object_image")
POLICIES = {
    "pi05_libero_base": "lerobot/pi05_libero_base",
    "expert_only_80k": "outputs/pi05_libero_object_multitask_batch2_meanstd_80k_resume40k/checkpoints/080000/pretrained_model",
}
EPISODES_PER_TASK = 3


def seed_for(task: int, episode: int) -> int:
    """First three seeds from each task's 10-seed block within 42020–42119."""
    return 42020 + 10 * task + episode


def write_result(result: dict[str, Any]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    temporary = RESULT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(RESULT_PATH)


def camera_mapping(label: str, config: PreTrainedConfig) -> dict[str, str]:
    if label == "pi05_libero_base":
        # Official LIBERO base checkpoint uses image/image2 rather than image/wrist_image.
        return {"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}
    visual_keys = [key for key, feature in config.input_features.items() if feature.type.value == "VISUAL"]
    main_keys = [key for key in visual_keys if "wrist" not in key and "hand" not in key]
    wrist_keys = [key for key in visual_keys if "wrist" in key or "hand" in key]
    if len(main_keys) != 1 or len(wrist_keys) != 1:
        raise ValueError(f"Cannot unambiguously map main/wrist cameras from model config: {visual_keys}")
    return {
        "agentview_image": main_keys[0].split(".")[-1],
        "robot0_eye_in_hand_image": wrist_keys[0].split(".")[-1],
    }


def policy_metadata(label: str, policy_path: str) -> dict[str, Any]:
    config = PreTrainedConfig.from_pretrained(policy_path)
    if config.type != "pi05":
        raise ValueError(f"Expected π0.5 config for {label}, got {config.type}")
    mapping = camera_mapping(label, config)
    return {
        "policy_path": policy_path,
        "model_type": config.type,
        "checkpoint_chunk_size": config.chunk_size,
        "checkpoint_n_action_steps": config.n_action_steps,
        "normalization_mapping": config.normalization_mapping,
        "input_features": {key: {"type": value.type.value, "shape": list(value.shape)} for key, value in config.input_features.items()},
        "camera_name_mapping": mapping,
    }


def task_instructions() -> dict[str, str]:
    tasks = pd.read_parquet(DATASET_ROOT / "meta" / "tasks.parquet")
    if tasks.index.name != "task" or len(tasks) != 10:
        raise ValueError("Expected ten LIBERO Object task strings")
    return {str(index): str(task) for index, task in enumerate(tasks.index.tolist())}


def new_result() -> dict[str, Any]:
    task_text = task_instructions()
    return {
        "status": "in_progress", "policy_type": "pi05", "episodes_per_task": EPISODES_PER_TASK,
        "seed_rule": "task t, episode e: 42020 + 10*t + e", "seed_range": "42020–42119",
        "policy_inference_rng_rule": "fresh process and --seed equal to environment seed",
        "videos_saved": False, "n_action_steps_override": 10,
        "policies": {
            label: {
                **policy_metadata(label, path),
                "task_results": {str(task): {"instruction": task_text[str(task)], "per_episode": []} for task in range(10)},
            }
            for label, path in POLICIES.items()
        },
    }


def refresh(result: dict[str, Any]) -> None:
    complete = []
    for label, policy in result["policies"].items():
        episodes = []
        for row in policy["task_results"].values():
            row["successes"] = sum(bool(item["success"]) for item in row["per_episode"])
            row["episodes"] = len(row["per_episode"])
            episodes.extend(row["per_episode"])
        policy["overall_successes"] = sum(bool(item["success"]) for item in episodes)
        policy["overall_episodes"] = len(episodes)
        policy["overall_success_rate"] = policy["overall_successes"] / len(episodes) if episodes else 0.0
        if len(episodes) == 30:
            complete.append((label, policy))
    if len(complete) == 2:
        base = {(task, item["seed"]): bool(item["success"]) for task, row in result["policies"]["pi05_libero_base"]["task_results"].items() for item in row["per_episode"]}
        expert = {(task, item["seed"]): bool(item["success"]) for task, row in result["policies"]["expert_only_80k"]["task_results"].items() for item in row["per_episode"]}
        result["paired_comparison"] = {
            "pi05_libero_base_only_success": sum(base[key] and not expert[key] for key in base),
            "expert_only_80k_only_success": sum(not base[key] and expert[key] for key in base),
            "both_success": sum(base[key] and expert[key] for key in base),
            "both_failure": sum(not base[key] and not expert[key] for key in base),
        }
        result["best_policy"] = max(complete, key=lambda item: item[1]["overall_success_rate"])[0]


def run_episode(label: str, metadata: dict[str, Any], task: int, seed: int) -> bool:
    run_dir = ROOT / "runs" / label / f"task{task}" / f"seed_{seed}"
    metrics_path = ROOT / "episodes" / label / f"task{task}" / f"seed_{seed}.json"
    env = os.environ | {
        "LEROBOT_EVAL_NO_VIDEOS": "1", "ACT_SMOKE_EVAL_METRICS_PATH": str(metrics_path),
        "PYTHONHASHSEED": str(seed),
    }
    command = [
        sys.executable, "scripts/eval_act_smoke_instrumented.py",
        f"--policy.path={metadata['policy_path']}", "--policy.n_action_steps=10",
        "--env.type=libero", "--env.task=libero_object", f"--env.task_ids=[{task}]",
        f"--env.camera_name_mapping={json.dumps(metadata['camera_name_mapping'], separators=(',', ':'))}",
        "--env.observation_height=256", "--env.observation_width=256", "--env.max_parallel_tasks=1",
        "--eval.batch_size=1", "--eval.n_episodes=1", "--eval.recording=false",
        f"--seed={seed}", f"--output_dir={run_dir}",
    ]
    subprocess.run(command, check=True, env=env)
    raw = json.loads(metrics_path.read_text())
    episodes = raw.get("per_episode", [])
    if raw.get("status") != "completed" or len(episodes) != 1:
        raise RuntimeError(f"Incomplete episode metrics: {metrics_path}")
    return bool(episodes[0]["success"])


def main() -> None:
    result = json.loads(RESULT_PATH.read_text()) if RESULT_PATH.exists() else new_result()
    if set(result.get("policies", {})) != set(POLICIES) or result.get("n_action_steps_override") != 10:
        raise ValueError(f"Incompatible existing result: {RESULT_PATH}")
    for label, metadata in result["policies"].items():
        for task in range(10):
            row = metadata["task_results"][str(task)]
            completed = {item["seed"] for item in row["per_episode"]}
            for episode in range(EPISODES_PER_TASK):
                seed = seed_for(task, episode)
                if seed in completed:
                    continue
                row["per_episode"].append({
                    "policy": label, "task": task, "seed": seed, "environment_seed": seed,
                    "inference_seed": seed, "success": run_episode(label, metadata, task, seed),
                })
                row["per_episode"].sort(key=lambda item: item["seed"])
                refresh(result)
                write_result(result)
    refresh(result)
    result["status"] = "completed"
    write_result(result)


if __name__ == "__main__":
    main()
