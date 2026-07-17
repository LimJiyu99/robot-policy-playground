#!/usr/bin/env bash
# Train a small ACT smoke run on a locally cached LIBERO-Object image subset.
#
# This script deliberately trains for only 200 updates.  It uses episodes 0-6
# from the requested Hub dataset, whose two local parquet shards are enough for
# a functional dataloader and action-chunk test without downloading all 8.6 GiB.

set -uo pipefail

ACT_PROJECT_ROOT="/workspace/jy/projects/lerobot-libero-benchmark"
ACT_DATASET_ROOT="/workspace/jy/datasets/lerobot/libero_object_image"
ACT_DATASET_REVISION="e1e080d7df1d0a359dff5c86c222e047549f447f"
ACT_OUTPUT_DIR="${ACT_PROJECT_ROOT}/outputs/act_smoke"
ACT_MONITOR_DIR="${ACT_PROJECT_ROOT}/outputs/act_smoke_monitor"

# Set ACT_SMOKE_BATCH_SIZE=4 only if the initial batch-size-8 attempt reports
# a CUDA out-of-memory error.  The default is the requested first attempt.
ACT_BATCH_SIZE="${ACT_SMOKE_BATCH_SIZE:-8}"

if [[ "${ACT_BATCH_SIZE}" != "8" && "${ACT_BATCH_SIZE}" != "4" ]]; then
    echo "ACT_SMOKE_BATCH_SIZE must be 8 (default) or 4 (OOM fallback)." >&2
    exit 2
fi

# LeRobot refuses to overwrite an existing training directory.  Keeping this
# guard makes an accidental re-run explicit and preserves smoke-test evidence.
if [[ -e "${ACT_OUTPUT_DIR}" ]]; then
    echo "Refusing to overwrite existing output: ${ACT_OUTPUT_DIR}" >&2
    exit 2
fi

# The project activation script selects the explicit Conda prefix and stores
# Hugging Face, Torch, and pip caches under /workspace/jy rather than $HOME.
source "${ACT_PROJECT_ROOT}/scripts/activate_lerobot.sh"
mkdir -p "${ACT_MONITOR_DIR}"

# Save baseline GPU state before starting the child process.
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits > "${ACT_MONITOR_DIR}/gpu_before.csv"

# These are the LeRobot 0.6.1 CLI names verified with `lerobot-train --help`.
# `dataset.episodes` confines this smoke run to data already present locally.
ACT_TRAIN_ARGS=(
    --dataset.repo_id=lerobot/libero_object_image
    --dataset.root="${ACT_DATASET_ROOT}"
    --dataset.revision="${ACT_DATASET_REVISION}"
    --dataset.episodes='[0,1,2,3,4,5,6]'
    --dataset.return_uint8=true
    --policy.type=act
    --policy.device=cuda
    --policy.push_to_hub=false
    --output_dir="${ACT_OUTPUT_DIR}"
    --job_name=act_libero_object_smoke
    --steps=200
    --batch_size="${ACT_BATCH_SIZE}"
    --num_workers=2
    --log_freq=10
    --env_eval_freq=0
    --save_checkpoint=true
    --save_freq=100
    --save_checkpoint_to_hub=false
    --wandb.enable=false
    --wandb.mode=disabled
)

# /usr/bin/time records process peak RSS and elapsed time.  nvidia-smi polling
# records GPU VRAM peak without modifying the LeRobot source tree.
/usr/bin/time -v -o "${ACT_MONITOR_DIR}/process_time.txt" \
    lerobot-train "${ACT_TRAIN_ARGS[@]}" \
    > "${ACT_MONITOR_DIR}/train.log" 2>&1 &
ACT_TRAIN_PID=$!

printf 'unix_time\tgpu_used_mib\n' > "${ACT_MONITOR_DIR}/gpu_samples.tsv"
while kill -0 "${ACT_TRAIN_PID}" 2>/dev/null; do
    ACT_GPU_USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR==1 {gsub(/ /, ""); print; exit}')"
    printf '%s\t%s\n' "$(date +%s.%N)" "${ACT_GPU_USED_MIB}" >> "${ACT_MONITOR_DIR}/gpu_samples.tsv"
    sleep 0.25
done

wait "${ACT_TRAIN_PID}"
ACT_STATUS=$?

# Retain the final GPU state and process exit code for the smoke-test report.
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits > "${ACT_MONITOR_DIR}/gpu_after.csv"
printf '%s\n' "${ACT_STATUS}" > "${ACT_MONITOR_DIR}/exit_status.txt"
exit "${ACT_STATUS}"
