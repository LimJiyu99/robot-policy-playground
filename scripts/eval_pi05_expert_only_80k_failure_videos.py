#!/usr/bin/env python3
"""Save one labelled rollout video for each LIBERO Object task from π0.5 80K."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


CHECKPOINT = Path("outputs/pi05_libero_object_multitask_batch2_meanstd_80k_resume40k/checkpoints/080000/pretrained_model")
# Keep the failed LeRobotDataset-recording attempt untouched.
ROOT = Path("outputs/pi05_expert_only_batch2_meanstd_80k_render_videos_eval10")
RESULT_PATH = ROOT / "instrumentation.json"
DATASET_ROOT = Path("/workspace/jy/datasets/lerobot/libero_object_image")


def seed_for(task: int) -> int:
    return 42000 + 100 * task


def write_result(result: dict[str, Any]) -> None:
    temporary = RESULT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(RESULT_PATH)


def instructions() -> dict[str, str]:
    tasks = pd.read_parquet(DATASET_ROOT / "meta" / "tasks.parquet")
    if tasks.index.name != "task" or len(tasks) != 10:
        raise ValueError("Expected ten LIBERO Object task strings")
    return {str(i): str(task) for i, task in enumerate(tasks.index.tolist())}


def verify_checkpoint() -> None:
    config_path = CHECKPOINT / "config.json"
    if not (CHECKPOINT / "model.safetensors").is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Missing 80K checkpoint: {CHECKPOINT}")
    config = json.loads(config_path.read_text())
    if config.get("type") != "pi05" or config.get("chunk_size") != 50 or config.get("n_action_steps") != 10:
        raise ValueError(f"Unexpected 80K π0.5 configuration: {config_path}")


def run_task(task: int) -> tuple[bool, Path, Path]:
    seed = seed_for(task)
    run_dir = ROOT / "runs" / f"task{task}" / f"seed_{seed}"
    metrics_path = ROOT / "episodes" / f"task{task}" / f"seed_{seed}.json"
    environment = os.environ | {"HF_HUB_OFFLINE": "1", "ACT_SMOKE_EVAL_METRICS_PATH": str(metrics_path), "PYTHONHASHSEED": str(seed)}
    command = [
        sys.executable, "scripts/eval_act_smoke_instrumented.py",
        f"--policy.path={CHECKPOINT}", "--policy.n_action_steps=10",
        "--env.type=libero", "--env.task=libero_object", f"--env.task_ids=[{task}]",
        '--env.camera_name_mapping={"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}',
        "--env.observation_height=256", "--env.observation_width=256", "--env.max_parallel_tasks=1",
        # recording=false selects LeRobot eval_policy's direct render-callback video path,
        # avoiding LeRobotDataset.create while still writing the main-camera mp4.
        "--eval.batch_size=1", "--eval.n_episodes=1", "--eval.recording=false",
        f"--seed={seed}", f"--output_dir={run_dir}",
    ]
    subprocess.run(command, check=True, env=environment)
    metrics = json.loads(metrics_path.read_text())
    episodes = metrics.get("per_episode", [])
    if metrics.get("status") != "completed" or len(episodes) != 1:
        raise RuntimeError(f"Incomplete episode metrics: {metrics_path}")
    videos = [Path(path) for path in metrics.get("video_paths", [])]
    if len(videos) != 1 or not videos[0].is_file():
        raise RuntimeError(f"Expected exactly one recorded video in {metrics_path}: {videos}")
    success = bool(episodes[0]["success"])
    outcome = "success" if success else "failure"
    labelled_video = ROOT / "videos" / f"task{task}_seed{seed}_{outcome}.mp4"
    labelled_video.parent.mkdir(parents=True, exist_ok=True)
    if labelled_video.exists():
        raise FileExistsError(f"Refusing to overwrite video: {labelled_video}")
    videos[0].replace(labelled_video)
    return success, labelled_video, metrics_path


def main() -> None:
    if ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {ROOT}")
    verify_checkpoint()
    ROOT.mkdir(parents=True)
    task_text = instructions()
    result: dict[str, Any] = {
        "status": "in_progress", "policy_type": "pi05", "checkpoint": str(CHECKPOINT),
        "episodes_per_task": 1, "seed_rule": "task t: 42000 + 100*t", "n_action_steps_override": 10,
        "videos_saved": True, "task_results": {},
    }
    for task in range(10):
        success, video_path, metrics_path = run_task(task)
        result["task_results"][str(task)] = {
            "task": task, "seed": seed_for(task), "instruction": task_text[str(task)],
            "success": success, "video_path": str(video_path), "episode_metrics_path": str(metrics_path),
        }
        write_result(result)
    result["overall_successes"] = sum(bool(row["success"]) for row in result["task_results"].values())
    result["overall_episodes"] = 10
    result["status"] = "completed"
    write_result(result)


if __name__ == "__main__":
    main()
