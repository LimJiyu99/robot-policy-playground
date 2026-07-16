#!/usr/bin/env python3
"""Run the official LeRobot evaluator while collecting PI05 timing metrics.

This file does not replace LeRobot's evaluation logic.  It wraps two public
evaluation functions, records timings and tensor shapes, and then calls the
same ``lerobot_eval.main()`` entry point used by ``lerobot-eval``.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from lerobot.scripts import lerobot_eval


METRICS_PATH = Path(
    os.environ.get(
        "PI05_METRICS_PATH",
        "outputs/pi05_task0_smoke/instrumentation.json",
    )
)


def shape_summary(batch: dict[str, Any]) -> dict[str, Any]:
    """Return only lightweight shape/type information from a policy batch."""
    summary: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "shape"):
            summary[key] = list(value.shape)
        elif isinstance(value, (list, tuple)):
            summary[key] = {
                "length": len(value),
                "example": str(value[0]) if value else None,
            }
        else:
            summary[key] = type(value).__name__
    return summary


metrics: dict[str, Any] = {
    "status": "started",
    "select_action_seconds": [],
    "action_generation_seconds": [],
}

# Preserve the official functions so the wrappers below can delegate to them.
official_make_policy = lerobot_eval.make_policy
official_eval_policy_all = lerobot_eval.eval_policy_all


def timed_make_policy(*args: Any, **kwargs: Any):
    """Time checkpoint construction/loading and instrument select_action()."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    started = time.perf_counter()
    policy = official_make_policy(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    metrics["model_loading_seconds"] = time.perf_counter() - started
    metrics["cuda_allocated_after_load_bytes"] = (
        torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    )
    metrics["parameter_count"] = sum(parameter.numel() for parameter in policy.parameters())

    official_select_action = policy.select_action

    def timed_select_action(batch: dict[str, Any]):
        # PI05 generates a complete action chunk only when this queue is empty.
        generates_chunk = len(getattr(policy, "_action_queue", ())) == 0
        if "policy_input_shapes" not in metrics:
            metrics["policy_input_shapes"] = shape_summary(batch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        action_started = time.perf_counter()
        action = official_select_action(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - action_started

        metrics["select_action_seconds"].append(elapsed)
        if generates_chunk:
            metrics["action_generation_seconds"].append(elapsed)
        if "policy_action_shape" not in metrics:
            metrics["policy_action_shape"] = list(action.shape)
        return action

    policy.select_action = timed_select_action
    return policy


def timed_eval_policy_all(*args: Any, **kwargs: Any):
    """Time only the requested episode rollout and retain its success metrics."""
    started = time.perf_counter()
    result = official_eval_policy_all(*args, **kwargs)
    metrics["episode_seconds"] = time.perf_counter() - started
    metrics["eval_result"] = result
    return result


def finalize_metrics() -> None:
    """Write metrics even when evaluation raises an exception."""
    latencies = metrics.get("select_action_seconds", [])
    generation_latencies = metrics.get("action_generation_seconds", [])
    if latencies:
        metrics["mean_select_action_latency_seconds"] = sum(latencies) / len(latencies)
        metrics["select_action_calls"] = len(latencies)
    if generation_latencies:
        metrics["mean_action_generation_latency_seconds"] = sum(generation_latencies) / len(
            generation_latencies
        )
        metrics["action_generation_calls"] = len(generation_latencies)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        metrics["torch_cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
        metrics["torch_cuda_memory_allocated_at_exit_bytes"] = torch.cuda.memory_allocated()

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str) + "\n")


def main() -> int:
    """Install timing wrappers and invoke the unmodified LeRobot CLI parser."""
    lerobot_eval.make_policy = timed_make_policy
    lerobot_eval.eval_policy_all = timed_eval_policy_all
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
