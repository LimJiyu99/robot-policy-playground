#!/usr/bin/env python3
"""Resumable 100-seed paired evaluation: raw LIBERO base vs 6K adaptation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from lerobot.configs.policies import PreTrainedConfig
# Register the pi05 Draccus choice before reading checkpoint metadata.
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(
    os.environ.get(
        "PI05_PAIRED_EVAL_OUTPUT",
        PROJECT_ROOT / "outputs/pi05_libero_base_vs_expert_only_6k_eval100",
    )
)
RESULT_PATH = ROOT / "instrumentation.json"
SUMMARY_PATH = ROOT / "summary.json"
DATASET_ROOT = Path(os.environ["LEROBOT_DATASET_ROOT"])
POLICIES = {
    "pi05_libero_base_untrained": "lerobot/pi05_libero_base",
    "pi05_libero_base_expert_only_6k": (
        str(PROJECT_ROOT / "outputs/pi05_libero_base_expert_only_6k/checkpoints/006000/pretrained_model")
    ),
}
EPISODES_PER_TASK = 10
CAMERA_MAPPING = {"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}


def seed_for(task: int, episode: int) -> int:
    """Ten contiguous seeds per task: 42020–42119 over all ten tasks."""
    return 42020 + 10 * task + episode


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def policy_metadata(policy_path: str) -> dict[str, Any]:
    config = PreTrainedConfig.from_pretrained(policy_path)
    if config.type != "pi05":
        raise ValueError(f"Expected π0.5 config for {policy_path}, got {config.type}")
    visual_keys = [key for key, value in config.input_features.items() if value.type.value == "VISUAL"]
    required_visual_keys = ["observation.images.image", "observation.images.image2"]
    extra_visual_keys = visual_keys[len(required_visual_keys) :]
    if visual_keys[: len(required_visual_keys)] != required_visual_keys or any(
        not key.startswith("observation.images.empty_camera_") for key in extra_visual_keys
    ):
        raise ValueError(
            f"Expected image/image2 plus optional empty camera features, got {visual_keys}: {policy_path}"
        )
    return {
        "policy_path": policy_path,
        "checkpoint_chunk_size": config.chunk_size,
        "checkpoint_n_action_steps": config.n_action_steps,
        "normalization_mapping": config.normalization_mapping,
        "input_features": {
            key: {"type": value.type.value, "shape": list(value.shape)}
            for key, value in config.input_features.items()
        },
        "camera_name_mapping": CAMERA_MAPPING,
    }


def instructions() -> dict[str, str]:
    tasks = pd.read_parquet(DATASET_ROOT / "meta/tasks.parquet")
    if tasks.index.name != "task" or len(tasks) != 10:
        raise ValueError("Expected exactly ten LIBERO Object task instructions")
    return {str(index): str(task) for index, task in enumerate(tasks.index.tolist())}


def new_result() -> dict[str, Any]:
    task_text = instructions()
    return {
        "status": "in_progress",
        "policy_type": "pi05",
        "episodes_per_task": EPISODES_PER_TASK,
        "seed_rule": "42020 + 10 * task + episode",
        "seed_range": "42020–42119",
        "policy_inference_rng_rule": "fresh process and --seed equal to environment seed",
        "n_action_steps_override": 10,
        "env_control_mode": "relative",
        "videos_saved": False,
        "policies": {
            label: {
                **policy_metadata(path),
                "task_results": {
                    str(task): {"instruction": task_text[str(task)], "per_episode": []}
                    for task in range(10)
                },
            }
            for label, path in POLICIES.items()
        },
    }


def refresh(result: dict[str, Any]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    complete = True
    for label, policy in result["policies"].items():
        all_episodes: list[dict[str, Any]] = []
        task_summary: dict[str, Any] = {}
        for task, row in policy["task_results"].items():
            row["successes"] = sum(bool(item["success"]) for item in row["per_episode"])
            row["episodes"] = len(row["per_episode"])
            row["success_rate"] = row["successes"] / row["episodes"] if row["episodes"] else 0.0
            all_episodes.extend(row["per_episode"])
            task_summary[task] = {
                "instruction": row["instruction"],
                "successes": row["successes"],
                "episodes": row["episodes"],
                "success_rate": row["success_rate"],
            }
        policy["overall_successes"] = sum(bool(item["success"]) for item in all_episodes)
        policy["overall_episodes"] = len(all_episodes)
        policy["overall_success_rate"] = (
            policy["overall_successes"] / policy["overall_episodes"] if policy["overall_episodes"] else 0.0
        )
        complete = complete and policy["overall_episodes"] == 100
        summaries[label] = {
            "overall_successes": policy["overall_successes"],
            "overall_episodes": policy["overall_episodes"],
            "overall_success_rate": policy["overall_success_rate"],
            "task_results": task_summary,
        }

    summary: dict[str, Any] = {"policies": summaries}
    if complete:
        base = {
            (int(task), item["seed"]): bool(item["success"])
            for task, row in result["policies"]["pi05_libero_base_untrained"]["task_results"].items()
            for item in row["per_episode"]
        }
        adapted = {
            (int(task), item["seed"]): bool(item["success"])
            for task, row in result["policies"]["pi05_libero_base_expert_only_6k"]["task_results"].items()
            for item in row["per_episode"]
        }
        paired = {
            "base_only_success": sum(base[key] and not adapted[key] for key in base),
            "adapted_only_success": sum(not base[key] and adapted[key] for key in base),
            "both_success": sum(base[key] and adapted[key] for key in base),
            "both_failure": sum(not base[key] and not adapted[key] for key in base),
        }
        result["paired_comparison"] = paired
        result["best_policy"] = max(
            result["policies"], key=lambda label: result["policies"][label]["overall_success_rate"]
        )
        summary["paired_comparison"] = paired
        summary["best_policy"] = result["best_policy"]
    write_json(RESULT_PATH, result)
    write_json(SUMMARY_PATH, summary)
    return summary


def run_episode(label: str, policy_path: str, task: int, seed: int) -> dict[str, Any]:
    run_dir = ROOT / "runs" / label / f"task{task}" / f"seed_{seed}"
    metrics_path = ROOT / "episodes" / label / f"task{task}" / f"seed_{seed}.json"
    environment = os.environ | {
        "LEROBOT_EVAL_NO_VIDEOS": "1",
        "ACT_SMOKE_EVAL_METRICS_PATH": str(metrics_path),
        "PYTHONHASHSEED": str(seed),
    }
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/eval_policy_instrumented.py"),
        f"--policy.path={policy_path}",
        "--policy.n_action_steps=10",
        "--env.type=libero",
        "--env.task=libero_object",
        f"--env.task_ids=[{task}]",
        "--env.control_mode=relative",
        f"--env.camera_name_mapping={json.dumps(CAMERA_MAPPING, separators=(',', ':'))}",
        "--env.observation_height=256",
        "--env.observation_width=256",
        "--env.max_parallel_tasks=1",
        "--eval.batch_size=1",
        "--eval.n_episodes=1",
        "--eval.recording=false",
        f"--seed={seed}",
        f"--output_dir={run_dir}",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=environment, stdin=subprocess.DEVNULL)
    raw = json.loads(metrics_path.read_text())
    episodes = raw.get("per_episode", [])
    if raw.get("status") != "completed" or len(episodes) != 1:
        raise RuntimeError(f"Incomplete episode metrics: {metrics_path}")
    episode = episodes[0]
    return {
        "policy": label,
        "task": task,
        "seed": seed,
        "environment_seed": seed,
        "inference_seed": seed,
        "success": bool(episode["success"]),
        "sum_reward": episode.get("sum_reward"),
        "max_reward": episode.get("max_reward"),
        "metrics_path": str(metrics_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Validate paths/configs but do not evaluate.")
    args = parser.parse_args()
    if not Path(POLICIES["pi05_libero_base_expert_only_6k"]).is_dir():
        raise FileNotFoundError(f"6K checkpoint is missing: {POLICIES['pi05_libero_base_expert_only_6k']}")
    if args.dry_run:
        print(json.dumps(new_result(), indent=2, ensure_ascii=False))
        return

    result = json.loads(RESULT_PATH.read_text()) if RESULT_PATH.exists() else new_result()
    if (
        set(result.get("policies", {})) != set(POLICIES)
        or result.get("episodes_per_task") != EPISODES_PER_TASK
        or result.get("n_action_steps_override") != 10
        or result.get("seed_range") != "42020–42119"
    ):
        raise ValueError(f"Incompatible existing output: {RESULT_PATH}")

    for label, policy in result["policies"].items():
        for task in range(10):
            row = policy["task_results"][str(task)]
            completed = {item["seed"] for item in row["per_episode"]}
            for episode in range(EPISODES_PER_TASK):
                seed = seed_for(task, episode)
                if seed in completed:
                    continue
                row["per_episode"].append(run_episode(label, policy["policy_path"], task, seed))
                row["per_episode"].sort(key=lambda item: item["seed"])
                refresh(result)

    refresh(result)
    result["status"] = "completed"
    summary = refresh(result)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
