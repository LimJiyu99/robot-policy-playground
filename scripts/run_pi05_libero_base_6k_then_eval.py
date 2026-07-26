#!/usr/bin/env python3
"""Train π0.5 from the LIBERO base checkpoint, then resumably evaluate it.

This workflow deliberately uses a fresh output root and never resumes or
modifies the prior pi05_base expert-only 80K experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/workspace/jy/datasets/lerobot/libero_object_image")
ROOT = REPO / "outputs/pi05_libero_base_expert_only_6k"
TEMP_ROOT = REPO / "outputs/.runlogs/pi05_libero_base_expert_only_6k_workflow"
CHECKPOINT_STEP = 6000
CHECKPOINT = ROOT / "checkpoints/006000/pretrained_model"
EVAL_ROOT = ROOT / "eval"
INSTRUMENTATION_PATH = EVAL_ROOT / "instrumentation.json"
SUMMARY_PATH = EVAL_ROOT / "summary.json"
STATUS_PATH = ROOT / "workflow_status.json"
TEMP_STATUS_PATH = TEMP_ROOT / "workflow_status.json"
TEMP_LOG_PATH = TEMP_ROOT / "workflow.log"

POLICY_SOURCE = "lerobot/pi05_libero_base"
POLICY_NAME = "pi05_libero_base_expert_only_6k"
EPISODES_PER_TASK = 3
CAMERA_RENAME_MAP = {"observation.images.wrist_image": "observation.images.image2"}
CAMERA_MAPPING = {"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_log(message: str) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    with TEMP_LOG_PATH.open("a") as handle:
        handle.write(line + "\n")
    if ROOT.exists():
        with (ROOT / "workflow.log").open("a") as handle:
            handle.write(line + "\n")


def write_status(stage: str, **extra: Any) -> None:
    status = {"updated_at": utc_now(), "stage": stage, **extra}
    write_json(TEMP_STATUS_PATH, status)
    if ROOT.exists():
        write_json(STATUS_PATH, status)


def sync_workflow_log() -> None:
    if ROOT.exists() and TEMP_LOG_PATH.exists() and not (ROOT / "workflow.log").exists():
        shutil.copy2(TEMP_LOG_PATH, ROOT / "workflow.log")


def source_config_path() -> Path:
    cache = Path("/workspace/jy/cache/huggingface/hub/models--lerobot--pi05_libero_base/snapshots")
    matches = sorted(cache.glob("*/config.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one cached pi05_libero_base config, found: {matches}")
    return matches[0]


def validate_contract() -> dict[str, Any]:
    """Static validation only; it neither instantiates a model nor starts training."""
    dataset_info = json.loads((DATASET_ROOT / "meta/info.json").read_text())
    dataset_stats = json.loads((DATASET_ROOT / "meta/stats.json").read_text())
    source_cfg_path = source_config_path()
    source_cfg = json.loads(source_cfg_path.read_text())

    features = dataset_info["features"]
    expected_dataset = {
        "observation.images.image": [256, 256, 3],
        "observation.images.wrist_image": [256, 256, 3],
        "observation.state": [8],
        "action": [7],
    }
    for key, shape in expected_dataset.items():
        actual = features.get(key, {}).get("shape")
        if actual != shape:
            raise ValueError(f"Dataset feature mismatch for {key}: expected {shape}, got {actual}")

    expected_source = {
        "observation.images.image": [3, 256, 256],
        "observation.images.image2": [3, 256, 256],
        "observation.state": [8],
    }
    for key, shape in expected_source.items():
        actual = source_cfg.get("input_features", {}).get(key, {}).get("shape")
        if actual != shape:
            raise ValueError(f"Source policy feature mismatch for {key}: expected {shape}, got {actual}")
    if source_cfg.get("output_features", {}).get("action", {}).get("shape") != [7]:
        raise ValueError("pi05_libero_base action shape is not 7D")
    for key in ("observation.state", "action"):
        if not {"mean", "std"}.issubset(dataset_stats.get(key, {})):
            raise ValueError(f"Dataset stats for {key} do not support MEAN_STD normalization")

    return {
        "source_config": str(source_cfg_path),
        "source_input_features": source_cfg["input_features"],
        "source_output_features": source_cfg["output_features"],
        "dataset_feature_shapes": {key: features[key]["shape"] for key in expected_dataset},
        "dataset_total_episodes": dataset_info["total_episodes"],
        "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
        "rename_map": CAMERA_RENAME_MAP,
        "camera_mapping": CAMERA_MAPPING,
    }


def train_command() -> list[str]:
    """Values are copied from the prior expert-only 80K train config."""
    return [
        "lerobot-train",
        "--dataset.repo_id=lerobot/libero_object_image",
        f"--dataset.root={DATASET_ROOT}",
        "--dataset.return_uint8=true",
        f"--policy.path={POLICY_SOURCE}",
        "--policy.device=cuda",
        "--policy.dtype=bfloat16",
        "--policy.use_peft=false",
        "--policy.train_expert_only=true",
        "--policy.freeze_vision_encoder=true",
        "--policy.gradient_checkpointing=true",
        "--policy.compile_model=false",
        "--policy.use_relative_actions=false",
        "--policy.chunk_size=50",
        "--policy.n_action_steps=10",
        "--policy.push_to_hub=false",
        '--policy.normalization_mapping={"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}',
        "--policy.optimizer_lr=2.5e-5",
        "--policy.optimizer_betas=[0.9,0.95]",
        "--policy.optimizer_eps=1e-8",
        "--policy.optimizer_weight_decay=0.01",
        "--policy.optimizer_grad_clip_norm=1.0",
        "--policy.scheduler_warmup_steps=1000",
        "--policy.scheduler_decay_steps=30000",
        "--policy.scheduler_decay_lr=2.5e-6",
        f"--rename_map={json.dumps(CAMERA_RENAME_MAP, separators=(',', ':'))}",
        f"--output_dir={ROOT}",
        f"--job_name={POLICY_NAME}",
        "--batch_size=2",
        f"--steps={CHECKPOINT_STEP}",
        "--seed=42",
        "--num_workers=8",
        "--log_freq=50",
        "--env_eval_freq=0",
        "--save_checkpoint=true",
        f"--save_freq={CHECKPOINT_STEP}",
        "--save_checkpoint_to_hub=false",
        "--wandb.enable=false",
        "--wandb.mode=disabled",
    ]


def run_command(command: list[str], stage: str, environment: dict[str, str] | None = None) -> None:
    append_log(f"{stage} command: {subprocess.list2cmdline(command)}")
    process = subprocess.Popen(
        command,
        cwd=REPO,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        append_log(f"{stage}: {line.rstrip()}")
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"{stage} subprocess failed with exit code {returncode}")


def load_instructions() -> dict[str, str]:
    tasks = pd.read_parquet(DATASET_ROOT / "meta/tasks.parquet")
    if tasks.index.name != "task" or len(tasks) != 10:
        raise ValueError("Expected ten LIBERO Object task strings in meta/tasks.parquet")
    return {str(index): str(task) for index, task in enumerate(tasks.index.tolist())}


def new_instrumentation(instructions: dict[str, str]) -> dict[str, Any]:
    return {
        "status": "in_progress",
        "policy": POLICY_NAME,
        "policy_path": str(CHECKPOINT),
        "checkpoint_step": CHECKPOINT_STEP,
        "policy_type": "pi05",
        "evaluation_protocol": "single-environment; one fresh process per task/seed",
        "episodes_per_task": EPISODES_PER_TASK,
        "seed_rule": "42020 + 10 * task + episode",
        "n_action_steps_override": 10,
        "env_control_mode": "relative",
        "camera_name_mapping": CAMERA_MAPPING,
        "videos_saved": False,
        "task_results": {
            str(task): {"instruction": instructions[str(task)], "per_episode": []} for task in range(10)
        },
    }


def validate_instrumentation(result: dict[str, Any]) -> None:
    if (
        result.get("policy_path") != str(CHECKPOINT)
        or result.get("checkpoint_step") != CHECKPOINT_STEP
        or result.get("n_action_steps_override") != 10
        or result.get("camera_name_mapping") != CAMERA_MAPPING
    ):
        raise ValueError(f"Refusing to mix a different evaluation protocol into {INSTRUMENTATION_PATH}")


def refresh_results(result: dict[str, Any]) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    simple_tasks: dict[str, Any] = {}
    for task, row in result["task_results"].items():
        row["successes"] = sum(bool(item["success"]) for item in row["per_episode"])
        row["episodes"] = len(row["per_episode"])
        row["success_rate"] = row["successes"] / row["episodes"] if row["episodes"] else 0.0
        episodes.extend(row["per_episode"])
        simple_tasks[task] = {
            "instruction": row["instruction"],
            "successes": row["successes"],
            "episodes": row["episodes"],
            "success_rate": row["success_rate"],
        }
    successes = sum(bool(item["success"]) for item in episodes)
    result["overall_successes"] = successes
    result["overall_episodes"] = len(episodes)
    result["overall_success_rate"] = successes / len(episodes) if episodes else 0.0
    summary = {
        "policy": POLICY_NAME,
        "checkpoint_step": CHECKPOINT_STEP,
        "overall_successes": successes,
        "overall_episodes": len(episodes),
        "overall_success_rate": result["overall_success_rate"],
        "successful_tasks": sum(row["successes"] > 0 for row in result["task_results"].values()),
        "task_results": simple_tasks,
    }
    write_json(INSTRUMENTATION_PATH, result)
    write_json(SUMMARY_PATH, summary)
    return summary


def eval_command(task: int, seed: int, episode_metrics_path: Path) -> list[str]:
    run_dir = EVAL_ROOT / "runs" / f"task{task}" / f"seed_{seed}"
    return [
        sys.executable,
        "scripts/eval_act_smoke_instrumented.py",
        f"--policy.path={CHECKPOINT}",
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


def completed_episode_from_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    episodes = raw.get("per_episode", [])
    if raw.get("status") != "completed" or len(episodes) != 1:
        return None
    return episodes[0]


def evaluate() -> dict[str, Any]:
    instructions = load_instructions()
    result = json.loads(INSTRUMENTATION_PATH.read_text()) if INSTRUMENTATION_PATH.exists() else new_instrumentation(instructions)
    validate_instrumentation(result)
    write_status("evaluating", checkpoint=str(CHECKPOINT))

    for task in range(10):
        row = result["task_results"][str(task)]
        completed = {item["seed"] for item in row["per_episode"]}
        for episode in range(EPISODES_PER_TASK):
            seed = 42020 + 10 * task + episode
            if seed in completed:
                append_log(f"evaluation: skip completed task={task} seed={seed}")
                continue
            metrics_path = EVAL_ROOT / "episodes" / f"task{task}" / f"seed_{seed}.json"
            raw_episode = completed_episode_from_metrics(metrics_path)
            if raw_episode is None:
                environment = os.environ | {
                    "LEROBOT_EVAL_NO_VIDEOS": "1",
                    "ACT_SMOKE_EVAL_METRICS_PATH": str(metrics_path),
                    "PYTHONHASHSEED": str(seed),
                }
                run_command(eval_command(task, seed, metrics_path), f"eval task={task} seed={seed}", environment)
                raw_episode = completed_episode_from_metrics(metrics_path)
            if raw_episode is None:
                raise RuntimeError(f"Missing or incomplete episode metrics: {metrics_path}")
            row["per_episode"].append(
                {
                    "task": task,
                    "seed": seed,
                    "success": bool(raw_episode["success"]),
                    "sum_reward": raw_episode.get("sum_reward"),
                    "max_reward": raw_episode.get("max_reward"),
                    "metrics_path": str(metrics_path),
                }
            )
            row["per_episode"].sort(key=lambda item: item["seed"])
            refresh_results(result)

    result["status"] = "completed"
    summary = refresh_results(result)
    write_status("completed", checkpoint=str(CHECKPOINT), summary=summary)
    append_log(
        f"completed: {summary['overall_successes']}/{summary['overall_episodes']} "
        f"({summary['overall_success_rate']:.1%})"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Validate contracts and print commands only.")
    args = parser.parse_args()

    contract = validate_contract()
    checkpoint_exists = CHECKPOINT.is_dir()
    if args.dry_run:
        print(json.dumps({"contract": contract, "checkpoint_exists": checkpoint_exists}, indent=2, ensure_ascii=False))
        if not checkpoint_exists:
            print("TRAIN COMMAND:\n" + subprocess.list2cmdline(train_command()))
        print("EVAL COMMAND EXAMPLE:\n" + subprocess.list2cmdline(eval_command(0, 42020, EVAL_ROOT / "episodes/task0/seed_42020.json")))
        return

    try:
        if not checkpoint_exists:
            if ROOT.exists():
                raise FileExistsError(
                    f"{ROOT} exists without the final 6K checkpoint; refusing to overwrite a partial run."
                )
            write_status("training_started", contract=contract)
            run_command(train_command(), "training", os.environ | {"PYTHONUNBUFFERED": "1"})
            if not CHECKPOINT.is_dir():
                raise FileNotFoundError(f"Training succeeded but final checkpoint is missing: {CHECKPOINT}")
            sync_workflow_log()
            write_status("training_completed", checkpoint=str(CHECKPOINT))
        else:
            append_log(f"training: final checkpoint exists; skipping training: {CHECKPOINT}")
            write_status("training_skipped_existing_checkpoint", checkpoint=str(CHECKPOINT))

        evaluate()
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_status("failed", error=error)
        append_log(f"failed: {error['type']}: {error['message']}")
        raise


if __name__ == "__main__":
    main()
