#!/usr/bin/env python3
"""Create LIBERO-Object task 0, take one zero-action step, and save RGB."""

from pathlib import Path

import numpy as np
from PIL import Image

import libero.libero as libero_core
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "outputs" / "smoke_test.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    # hf-libero 0.1.4 otherwise ignores config.yaml for its initial assets probe
    # and falls back to ~/.cache/libero/assets.
    assets = Path("/workspace/jy/datasets/libero_assets")
    if not assets.is_dir():
        raise FileNotFoundError(f"LIBERO assets not found: {assets}")
    libero_core._assets_path_cache = str(assets)

    benchmark = get_benchmark("libero_object")(task_order_index=0)
    task = benchmark.get_task(0)
    bddl_path = benchmark.get_task_bddl_file_path(0)
    print(f"suite=libero_object task_id=0 task={task.name}")
    print(f"bddl={bddl_path}")

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
    )
    try:
        observation = env.reset()
        print(f"reset=ok observation_keys={sorted(observation)}")

        action = np.zeros(env.env.action_dim, dtype=np.float32)
        observation, reward, done, info = env.step(action)
        print(
            f"step=ok action_shape={action.shape} reward={reward} "
            f"done={done} info={info}"
        )

        # Match LeRobot's LiberoEnv.render(): raw robosuite observations are
        # rotated 180 degrees for display.
        frame = observation["agentview_image"][::-1, ::-1]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        Image.fromarray(frame).save(output)
        print(f"frame={output} shape={frame.shape} dtype={frame.dtype}")
    finally:
        env.close()
        print("close=ok")


if __name__ == "__main__":
    main()
