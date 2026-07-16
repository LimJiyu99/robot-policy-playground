#!/usr/bin/env python3
"""Inspect the local LIBERO-Object subset and create a beginner-friendly plot.

This script only reads data.  It does not train a policy and does not contact
the Hugging Face Hub when the selected files are already present locally.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as arrow_dataset
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# Pinning the revision makes this inspection reproducible even if Hub main
# changes later.  The dataset itself lives outside the Git repository.
REPO_ID = "lerobot/libero"
REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
DATASET_ROOT = Path("/workspace/jy/datasets/lerobot/libero")
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "dataset_sample.png"

# In lerobot/libero, task indices 20 through 29 are the ten LIBERO-Object
# language tasks.  Other indices belong to Goal, Long, or Spatial suites.
LIBERO_OBJECT_TASK_IDS = set(range(20, 30))
CAMERA_KEYS = ["observation.images.image", "observation.images.image2"]


def find_libero_object_episodes() -> list[int]:
    """Return episode ids whose parquet rows use a LIBERO-Object task id."""

    # Reading only two integer columns keeps this scan fast and memory-light.
    table = arrow_dataset.dataset(DATASET_ROOT / "data", format="parquet").to_table(
        columns=["episode_index", "task_index"]
    )
    episode_tasks = table.to_pandas().drop_duplicates()
    selected = episode_tasks[episode_tasks["task_index"].isin(LIBERO_OBJECT_TASK_IDS)]
    return sorted(selected["episode_index"].astype(int).tolist())


def image_for_matplotlib(value: torch.Tensor) -> np.ndarray:
    """Convert a LeRobot CHW tensor into the HWC layout matplotlib expects."""

    image = value.detach().cpu()
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = image.permute(1, 2, 0)
    array = image.numpy()

    # return_uint8=True normally gives 0..255.  This fallback also supports a
    # future LeRobot version returning normalized 0..1 floating-point images.
    if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
        array = np.clip(array, 0.0, 1.0)
    return array


def print_structure(dataset: LeRobotDataset, episodes: list[int], sample: dict) -> None:
    """Print the requested dataset facts in a compact, explicit format."""

    task_table = pd.read_parquet(DATASET_ROOT / "meta" / "tasks.parquet").reset_index()
    object_tasks = task_table[task_table["task_index"].isin(LIBERO_OBJECT_TASK_IDS)]

    print(f"dataset repo: {REPO_ID}@{REVISION}")
    print(f"local root: {DATASET_ROOT}")
    print(f"dataset feature keys: {list(dataset.features.keys())}")
    print(f"selected episode count: {len(episodes)}")
    print(f"selected task count: {len(object_tasks)}")
    print(f"fps: {dataset.meta.fps}")
    print(f"observation.state shape: {tuple(sample['observation.state'].shape)}")
    print(f"action shape: {tuple(sample['action'].shape)}")
    for key in CAMERA_KEYS:
        stored_shape = tuple(dataset.features[key]["shape"])
        tensor_shape = tuple(sample[key].shape)
        print(f"camera image shape ({key}): stored HWC={stored_shape}, loaded CHW={tensor_shape}")
    print(f"language instruction example: {sample['task']}")


def save_first_episode_plot(dataset: LeRobotDataset, first_episode_length: int) -> None:
    """Plot three times from both cameras plus a small action/state summary."""

    # The selected dataset begins at the first requested episode, so these are
    # local frame offsets inside that first episode.
    offsets = [0, first_episode_length // 2, first_episode_length - 1]
    samples = [dataset[offset] for offset in offsets]
    times = np.arange(first_episode_length, dtype=np.float32) / dataset.meta.fps

    # State/action values are stored in parquet, so collecting one episode is
    # cheap.  Camera frames are decoded only for the three offsets above.
    actions = torch.stack([dataset.reader.hf_dataset[i]["action"] for i in range(first_episode_length)]).numpy()
    states = torch.stack(
        [dataset.reader.hf_dataset[i]["observation.state"] for i in range(first_episode_length)]
    ).numpy()

    figure, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for column, (offset, sample) in enumerate(zip(offsets, samples, strict=True)):
        axes[0, column].imshow(image_for_matplotlib(sample[CAMERA_KEYS[0]]))
        axes[0, column].set_title(f"Main camera · frame {offset}")
        axes[1, column].imshow(image_for_matplotlib(sample[CAMERA_KEYS[1]]))
        axes[1, column].set_title(f"Wrist camera · frame {offset}")
        axes[0, column].axis("off")
        axes[1, column].axis("off")

    # Plot only the first three dimensions to keep the introductory figure
    # readable.  The complete vectors are still available in the dataset.
    axes[2, 0].plot(times, actions[:, :3])
    axes[2, 0].set_title("Action dimensions 0–2")
    axes[2, 0].set_xlabel("seconds")
    axes[2, 0].set_ylabel("control value")
    axes[2, 0].legend(["a0", "a1", "a2"], fontsize=8)

    axes[2, 1].plot(times, states[:, :3])
    axes[2, 1].set_title("State dimensions 0–2")
    axes[2, 1].set_xlabel("seconds")
    axes[2, 1].set_ylabel("state value")
    axes[2, 1].legend(["s0", "s1", "s2"], fontsize=8)

    axes[2, 2].axis("off")
    axes[2, 2].text(
        0.0,
        1.0,
        "Language instruction\n"
        f"{samples[0]['task']}\n\n"
        f"First state[:4]\n{states[0, :4]}\n\n"
        f"First action[:4]\n{actions[0, :4]}",
        va="top",
        wrap=True,
        fontsize=10,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150)
    plt.close(figure)
    print(f"saved visualization: {OUTPUT_PATH}")


def main() -> None:
    """Load the selected subset, print its schema, and save one figure."""

    if not (DATASET_ROOT / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Dataset metadata not found under {DATASET_ROOT}")

    episodes = find_libero_object_episodes()
    dataset = LeRobotDataset(
        REPO_ID,
        root=DATASET_ROOT,
        episodes=episodes,
        revision=REVISION,
        return_uint8=True,
    )
    first_sample = dataset[0]
    print_structure(dataset, episodes, first_sample)

    episode_table = pd.read_parquet(DATASET_ROOT / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    first_length = int(episode_table.loc[episode_table["episode_index"] == episodes[0], "length"].iloc[0])
    save_first_episode_plot(dataset, first_length)


if __name__ == "__main__":
    main()
