#!/usr/bin/env python3
"""Evaluate a local ACT checkpoint through LeRobot's normal LIBERO evaluator.

The module only wraps public functions from ``lerobot.scripts.lerobot_eval``
to collect smoke-test evidence.  Environment creation, processor application,
rollout, success computation, and video writing remain in LeRobot 0.6.1.
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from lerobot.scripts import lerobot_eval


METRICS_PATH = Path(os.environ.get("ACT_SMOKE_EVAL_METRICS_PATH", "outputs/act_smoke_eval/instrumentation.json"))


def shape_summary(batch: dict[str, Any]) -> dict[str, Any]:
    """Return JSON-safe tensor shape/type summaries without retaining rollout data."""
    summary: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            summary[key] = {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
        elif isinstance(value, dict):
            summary[key] = {"type": "dict", "keys": sorted(value)}
        elif isinstance(value, (list, tuple)):
            summary[key] = {"type": type(value).__name__, "length": len(value)}
        else:
            summary[key] = {"type": type(value).__name__}
    return summary


metrics: dict[str, Any] = {
    "status": "started",
    "torch_cuda_available": torch.cuda.is_available(),
    "select_action_seconds": [],
    "action_generation_seconds": [],
}

official_make_policy = lerobot_eval.make_policy
official_make_env_pre_post_processors = lerobot_eval.make_env_pre_post_processors
official_make_pre_post_processors = lerobot_eval.make_pre_post_processors
official_eval_policy = lerobot_eval.eval_policy
official_rollout = lerobot_eval.rollout


def timed_make_policy(*args: Any, **kwargs: Any):
    """Record successful checkpoint loading and ACT inference timing."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    started = time.perf_counter()
    policy = official_make_policy(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    metrics["checkpoint_load_seconds"] = time.perf_counter() - started
    metrics["checkpoint_loaded"] = True
    metrics["policy_device"] = str(next(policy.parameters()).device)
    metrics["policy_parameter_count"] = sum(parameter.numel() for parameter in policy.parameters())

    official_select_action = policy.select_action

    def timed_select_action(batch: dict[str, Any]):
        # ACT computes a 100-action chunk only when its queue is empty.
        generates_chunk = len(getattr(policy, "_action_queue", ())) == 0
        if "policy_input_shapes" not in metrics:
            metrics["policy_input_shapes"] = shape_summary(batch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        action = official_select_action(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - started

        metrics["select_action_seconds"].append(latency)
        if generates_chunk:
            metrics["action_generation_seconds"].append(latency)
        if "policy_action_shape" not in metrics:
            metrics["policy_action_shape"] = list(action.shape)
        return action

    policy.select_action = timed_select_action
    return policy


def instrumented_make_env_pre_post_processors(*args: Any, **kwargs: Any):
    """Capture the LIBERO image/state tensors before and after its processor."""
    env_preprocessor, env_postprocessor = official_make_env_pre_post_processors(*args, **kwargs)

    def capture_env_preprocessor(observation: dict[str, Any]):
        if "environment_observation_shapes" not in metrics:
            metrics["environment_observation_shapes"] = shape_summary(observation)
        processed = env_preprocessor(observation)
        if "libero_processed_observation_shapes" not in metrics:
            metrics["libero_processed_observation_shapes"] = shape_summary(processed)
        return processed

    return capture_env_preprocessor, env_postprocessor


def instrumented_make_pre_post_processors(*args: Any, **kwargs: Any):
    """Capture the normalized policy input, while preserving LeRobot processors."""
    preprocessor, postprocessor = official_make_pre_post_processors(*args, **kwargs)

    def capture_preprocessor(observation: dict[str, Any]):
        processed = preprocessor(observation)
        if "normalized_policy_input_shapes" not in metrics:
            metrics["normalized_policy_input_shapes"] = shape_summary(processed)
        return processed

    return capture_preprocessor, postprocessor


def instrumented_eval_policy(*args: Any, **kwargs: Any):
    """Keep LeRobot's per-episode termination and success result verbatim."""
    if os.environ.get("LEROBOT_EVAL_NO_VIDEOS") == "1":
        kwargs["max_episodes_rendered"] = 0
    result = official_eval_policy(*args, **kwargs)
    prior = metrics.setdefault("per_episode", [])
    for episode in result["per_episode"]:
        episode = dict(episode)
        episode["episode_ix"] = len(prior)
        prior.append(episode)
    metrics.setdefault("eval_policy_batches", []).append(result["aggregated"])
    metrics["aggregated"] = result["aggregated"]
    metrics.setdefault("video_paths", []).extend(result.get("video_paths", []))
    return result


def instrumented_rollout(*args: Any, **kwargs: Any):
    """Record action/step counts and the first done index from the official rollout."""
    result = official_rollout(*args, **kwargs)
    done = result["done"]
    done_indices = torch.argmax(done.to(torch.int64), dim=1)
    rollout = {
        "action_tensor_shape": list(result["action"].shape),
        "rollout_steps": int(done.shape[1]),
        "first_done_step_zero_based": done_indices.tolist(),
        "first_done_step_one_based": (done_indices + 1).tolist(),
        "final_done_values": done[:, -1].tolist(),
        "success_seen_during_rollout": torch.any(result["success"].bool(), dim=1).tolist(),
    }
    metrics.setdefault("rollouts", []).append(rollout)
    metrics["rollout"] = rollout
    return result


def finalize_metrics() -> None:
    """Persist evidence even when the evaluator raises an exception."""
    latencies = metrics["select_action_seconds"]
    generation_latencies = metrics["action_generation_seconds"]
    if latencies:
        metrics["select_action_calls"] = len(latencies)
        metrics["mean_select_action_latency_seconds"] = sum(latencies) / len(latencies)
        metrics["max_select_action_latency_seconds"] = max(latencies)
        metrics["p95_select_action_latency_seconds"] = sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1]
    if generation_latencies:
        metrics["action_generation_calls"] = len(generation_latencies)
        metrics["mean_action_generation_latency_seconds"] = sum(generation_latencies) / len(generation_latencies)
        metrics["max_action_generation_latency_seconds"] = max(generation_latencies)
        metrics["p95_action_generation_latency_seconds"] = sorted(generation_latencies)[
            math.ceil(0.95 * len(generation_latencies)) - 1
        ]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        metrics["torch_cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
        metrics["torch_cuda_memory_allocated_at_exit_bytes"] = torch.cuda.memory_allocated()

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str) + "\n")


def main() -> int:
    lerobot_eval.make_policy = timed_make_policy
    lerobot_eval.make_env_pre_post_processors = instrumented_make_env_pre_post_processors
    lerobot_eval.make_pre_post_processors = instrumented_make_pre_post_processors
    lerobot_eval.eval_policy = instrumented_eval_policy
    lerobot_eval.rollout = instrumented_rollout
    try:
        lerobot_eval.main()
    except BaseException as error:
        metrics["status"] = "failed"
        metrics["error_type"] = type(error).__name__
        metrics["error_message"] = str(error)
        metrics["traceback"] = traceback.format_exc()
        raise
    else:
        metrics["status"] = "completed"
        return 0
    finally:
        finalize_metrics()


if __name__ == "__main__":
    raise SystemExit(main())
