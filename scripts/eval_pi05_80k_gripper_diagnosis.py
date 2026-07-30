#!/usr/bin/env python3
"""Trace π0.5 gripper actions without modifying LeRobot core code.

The parent process evaluates all requested action-step settings.  Each child uses
LeRobot's normal evaluator with a wrapped rollout that records the three action
representations around its existing processor calls.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts import lerobot_eval


CHECKPOINT = Path("outputs/pi05_libero_object_multitask_batch2_meanstd_80k_resume40k/checkpoints/080000/pretrained_model")
DATASET_ROOT = Path("/workspace/jy/datasets/lerobot/libero_object_image")
ROOT = Path("outputs/pi05_expert_only_batch2_meanstd_80k_gripper_diagnosis")
RESULT_PATH = ROOT / "instrumentation.json"
ACTION_STEPS = (10, 5, 2, 1)
TRACE_CSV = Path(os.environ.get("PI05_GRIPPER_TRACE_CSV", ""))
METRICS_PATH = Path(os.environ.get("PI05_GRIPPER_METRICS_PATH", ""))


def seed_for(task: int) -> int:
    return 42000 + task


def as_gripper(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return float(array.reshape(-1)[6])


def gripper_qpos(observation: Any) -> tuple[float | None, float | None]:
    try:
        value = observation["robot_state"]["gripper"]["qpos"]
        array = np.asarray(value).reshape(-1)
        return float(array[0]), float(array[1]) if array.size > 1 else None
    except (KeyError, IndexError, TypeError, ValueError):
        return None, None


def write_child_metrics(metrics: dict[str, Any]) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str) + "\n")


def single_episode() -> None:
    """Run one standard LeRobot eval while observing its existing rollout boundaries."""
    trace: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {"status": "started"}
    official_make_policy = lerobot_eval.make_policy
    official_make_processors = lerobot_eval.make_pre_post_processors
    official_make_env_processors = lerobot_eval.make_env_pre_post_processors
    official_eval_policy = lerobot_eval.eval_policy
    official_rollout = lerobot_eval.rollout

    def traced_make_policy(*args: Any, **kwargs: Any):
        policy = official_make_policy(*args, **kwargs)
        original_select_action = policy.select_action

        def traced_select_action(batch: dict[str, Any]):
            new_inference = len(getattr(policy, "_action_queue", ())) == 0
            action = original_select_action(batch)
            trace.append(
                {
                    "timestep": len(trace),
                    "new_policy_inference": bool(new_inference),
                    "normalized_action_6": as_gripper(action),
                    "denormalized_action_6": None,
                    "env_action_6": None,
                    "gripper_qpos_0": None,
                    "gripper_qpos_1": None,
                }
            )
            return action

        policy.select_action = traced_select_action
        return policy

    def traced_make_processors(*args: Any, **kwargs: Any):
        preprocessor, postprocessor = official_make_processors(*args, **kwargs)

        def traced_postprocessor(action: Any):
            output = postprocessor(action)
            if trace:
                trace[-1]["denormalized_action_6"] = as_gripper(output)
            return output

        return preprocessor, traced_postprocessor

    def traced_make_env_processors(*args: Any, **kwargs: Any):
        env_preprocessor, env_postprocessor = official_make_env_processors(*args, **kwargs)

        def traced_env_postprocessor(action_transition: dict[str, Any]):
            output = env_postprocessor(action_transition)
            if trace:
                trace[-1]["env_action_6"] = as_gripper(output["action"])
            return output

        return env_preprocessor, traced_env_postprocessor

    class StepTraceEnv:
        def __init__(self, env: Any):
            self._env = env

        def __getattr__(self, name: str) -> Any:
            return getattr(self._env, name)

        def step(self, action: Any):
            result = self._env.step(action)
            if trace:
                q0, q1 = gripper_qpos(result[0])
                trace[-1]["gripper_qpos_0"] = q0
                trace[-1]["gripper_qpos_1"] = q1
            return result

    def traced_rollout(*args: Any, **kwargs: Any):
        if args:
            args = (StepTraceEnv(args[0]), *args[1:])
        else:
            kwargs["env"] = StepTraceEnv(kwargs["env"])
        return official_rollout(*args, **kwargs)

    def traced_eval_policy(*args: Any, **kwargs: Any):
        result = official_eval_policy(*args, **kwargs)
        metrics["per_episode"] = result["per_episode"]
        metrics["video_paths"] = result.get("video_paths", [])
        metrics["aggregated"] = result["aggregated"]
        return result

    lerobot_eval.make_policy = traced_make_policy
    lerobot_eval.make_pre_post_processors = traced_make_processors
    lerobot_eval.make_env_pre_post_processors = traced_make_env_processors
    lerobot_eval.eval_policy = traced_eval_policy
    lerobot_eval.rollout = traced_rollout
    try:
        lerobot_eval.main()
    except BaseException as error:
        metrics.update({"status": "failed", "error_type": type(error).__name__, "error_message": str(error)})
        raise
    else:
        metrics["status"] = "completed"
    finally:
        if TRACE_CSV:
            TRACE_CSV.parent.mkdir(parents=True, exist_ok=True)
            with TRACE_CSV.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=[
                    "timestep", "new_policy_inference", "normalized_action_6", "denormalized_action_6",
                    "env_action_6", "gripper_qpos_0", "gripper_qpos_1",
                ])
                writer.writeheader()
                writer.writerows(trace)
        metrics["trace_csv"] = str(TRACE_CSV)
        metrics["trace_steps"] = len(trace)
        if METRICS_PATH:
            write_child_metrics(metrics)


def action6_dataset_stats() -> dict[str, Any]:
    dataset = LeRobotDataset("lerobot/libero_object_image", root=DATASET_ROOT, return_uint8=True)
    action6 = np.asarray([np.asarray(action)[6] for action in dataset.hf_dataset["action"]], dtype=np.float32)
    counts, edges = np.histogram(action6, bins=40)
    histogram_path = ROOT / "dataset_action6_histogram.csv"
    with histogram_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["bin_left", "bin_right", "count"])
        writer.writerows(zip(edges[:-1], edges[1:], counts, strict=True))
    stats = {
        "count": int(action6.size), "min": float(action6.min()), "max": float(action6.max()),
        "mean": float(action6.mean()),
        "quantiles": {str(q): float(np.quantile(action6, q)) for q in (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)},
        "histogram_csv": str(histogram_path),
    }
    (ROOT / "dataset_action6_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def write_result(result: dict[str, Any]) -> None:
    temporary = RESULT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(RESULT_PATH)


def rename_video(metrics: dict[str, Any], target: Path) -> Path:
    videos = [Path(path) for path in metrics.get("video_paths", [])]
    if len(videos) != 1 or not videos[0].is_file():
        raise RuntimeError(f"Expected one direct-render video, got: {videos}")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite video: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    videos[0].replace(target)
    return target


def plot_task_traces(task: int, result: dict[str, Any]) -> None:
    figure, axis = plt.subplots(figsize=(11, 4))
    plotted = False
    for steps in ACTION_STEPS:
        row = result["action_steps"][str(steps)]["tasks"].get(str(task))
        if not row:
            continue
        data = pd.read_csv(row["trace_csv"])
        axis.plot(data["timestep"], data["env_action_6"], label=f"π0.5 steps={steps}")
        plotted = True
    # Optional only: an ACT rollout trace is used if a future compatible CSV exists.
    candidates = list(Path("outputs").glob("**/*act*gripper*trace*.csv"))
    if candidates:
        act = pd.read_csv(candidates[0])
        if {"timestep", "env_action_6"}.issubset(act.columns):
            axis.plot(act["timestep"], act["env_action_6"], "--", label="ACT success rollout")
    if plotted:
        axis.set(title=f"Task {task}: gripper command delivered to LIBERO", xlabel="env timestep", ylabel="action[6]")
        axis.legend()
        figure.tight_layout()
        output = ROOT / "plots" / f"task{task}_gripper_env_action6.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150)
    plt.close(figure)


def outer_main() -> None:
    if ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {ROOT}")
    if not (CHECKPOINT / "model.safetensors").is_file():
        raise FileNotFoundError(CHECKPOINT)
    ROOT.mkdir(parents=True)
    result: dict[str, Any] = {
        "status": "in_progress", "checkpoint": str(CHECKPOINT), "policy_type": "pi05",
        "n_action_steps": list(ACTION_STEPS), "seed_rule": "task t: 42000 + t", "episodes_per_task": 1,
        "dataset_action6": action6_dataset_stats(), "action_steps": {},
    }
    for steps in ACTION_STEPS:
        step_result = {"tasks": {}}
        result["action_steps"][str(steps)] = step_result
        for task in range(10):
            seed = seed_for(task)
            case_root = ROOT / f"action_steps_{steps}" / f"task{task}" / f"seed_{seed}"
            trace_csv = case_root / "gripper_trace.csv"
            metrics_path = case_root / "instrumentation.json"
            command = [
                sys.executable, str(Path(__file__).resolve()), "--single",
                f"--policy.path={CHECKPOINT}", f"--policy.n_action_steps={steps}",
                "--env.type=libero", "--env.task=libero_object", f"--env.task_ids=[{task}]",
                '--env.camera_name_mapping={"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}',
                "--env.observation_height=256", "--env.observation_width=256", "--env.max_parallel_tasks=1",
                "--eval.batch_size=1", "--eval.n_episodes=1", "--eval.recording=false",
                f"--seed={seed}", f"--output_dir={case_root / 'eval'}",
            ]
            env = os.environ | {
                "HF_HUB_OFFLINE": "1", "PYTHONHASHSEED": str(seed),
                "PI05_GRIPPER_TRACE_CSV": str(trace_csv), "PI05_GRIPPER_METRICS_PATH": str(metrics_path),
            }
            subprocess.run(command, check=True, env=env)
            metrics = json.loads(metrics_path.read_text())
            episode = metrics.get("per_episode", [])
            if metrics.get("status") != "completed" or len(episode) != 1:
                raise RuntimeError(f"Incomplete evaluation: {metrics_path}")
            success = bool(episode[0]["success"])
            outcome = "success" if success else "failure"
            video = rename_video(metrics, ROOT / "videos" / f"task{task}_seed{seed}_steps{steps}_{outcome}.mp4")
            step_result["tasks"][str(task)] = {
                "seed": seed, "success": success, "trace_csv": str(trace_csv),
                "video_path": str(video), "instrumentation_path": str(metrics_path),
            }
            write_result(result)
    for task in range(10):
        plot_task_traces(task, result)
    result["status"] = "completed"
    write_result(result)


if __name__ == "__main__":
    if "--single" in sys.argv:
        sys.argv.remove("--single")
        single_episode()
    else:
        outer_main()
