#!/usr/bin/env bash
# Train the task_index=0 (orange juice) ACT baseline for 5,000 optimizer updates.

set -uo pipefail

ACT_PROJECT_ROOT="/workspace/jy/projects/lerobot-libero-benchmark"
ACT_DATASET_ROOT="/workspace/jy/datasets/lerobot/libero_object_image"
ACT_DATASET_REVISION="e1e080d7df1d0a359dff5c86c222e047549f447f"
ACT_OUTPUT_DIR="${ACT_PROJECT_ROOT}/outputs/act_task9_5k"
ACT_MONITOR_DIR="${ACT_PROJECT_ROOT}/outputs/act_task9_5k_monitor"
ACT_BATCH_SIZE="${ACT_TASK9_BATCH_SIZE:-32}"
ACT_NUM_WORKERS="${ACT_TASK9_NUM_WORKERS:-8}"
TASK0_EPISODES='[0,22,25,28,30,41,47,59,63,73,91,116,119,172,206,234,236,237,238,239,240,242,243,266,277,286,287,307,314,315,332,339,348,350,352,353,365,366,368,370,390,393,400,411,420]'

if [[ "${ACT_BATCH_SIZE}" != "8" && "${ACT_BATCH_SIZE}" != "16" && "${ACT_BATCH_SIZE}" != "32" ]]; then
    echo "ACT_TASK9_BATCH_SIZE must be 8, 16, or 32." >&2
    exit 2
fi
if ! [[ "${ACT_NUM_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ACT_TASK9_NUM_WORKERS must be a positive integer." >&2
    exit 2
fi
if [[ -e "${ACT_OUTPUT_DIR}" ]]; then
    echo "Refusing to overwrite existing output: ${ACT_OUTPUT_DIR}" >&2
    echo "For a later run, resume from a checkpoint with lerobot-train --resume=true --config_path=<checkpoint>/pretrained_model/train_config.json." >&2
    exit 2
fi

source "${ACT_PROJECT_ROOT}/scripts/activate_lerobot.sh"
export HF_HUB_DISABLE_XET=1
python "${ACT_PROJECT_ROOT}/scripts/prepare_act_task9_dataset.py"
mkdir -p "${ACT_MONITOR_DIR}"
date --iso-8601=seconds > "${ACT_MONITOR_DIR}/started_at.txt"
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits > "${ACT_MONITOR_DIR}/gpu_before.csv"

# save_freq=500 additionally writes 1,000, 2,500, and 5,000; LeRobot always saves at the final step.
/usr/bin/time -v -o "${ACT_MONITOR_DIR}/process_time.txt" \
    lerobot-train \
    --dataset.repo_id=lerobot/libero_object_image \
    --dataset.root="${ACT_DATASET_ROOT}" \
    --dataset.revision="${ACT_DATASET_REVISION}" \
    --dataset.episodes="${TASK0_EPISODES}" \
    --dataset.return_uint8=true \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir="${ACT_OUTPUT_DIR}" \
    --job_name=act_task9_5k \
    --seed=42 \
    --steps=5000 \
    --batch_size="${ACT_BATCH_SIZE}" \
    --num_workers="${ACT_NUM_WORKERS}" \
    --log_freq=50 \
    --env_eval_freq=0 \
    --save_checkpoint=true \
    --save_freq=500 \
    --save_checkpoint_to_hub=false \
    --wandb.enable=false \
    --wandb.mode=disabled \
    > "${ACT_MONITOR_DIR}/train.log" 2>&1 &
ACT_TRAIN_PID=$!

printf 'unix_time\tgpu_used_mib\n' > "${ACT_MONITOR_DIR}/gpu_samples.tsv"
while kill -0 "${ACT_TRAIN_PID}" 2>/dev/null; do
    ACT_GPU_USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR==1 {gsub(/ /, ""); print; exit}')"
    printf '%s\t%s\n' "$(date +%s.%N)" "${ACT_GPU_USED_MIB}" >> "${ACT_MONITOR_DIR}/gpu_samples.tsv"
    sleep 0.25
done

wait "${ACT_TRAIN_PID}"
ACT_TRAIN_STATUS=$?
date --iso-8601=seconds > "${ACT_MONITOR_DIR}/finished_at.txt"
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits > "${ACT_MONITOR_DIR}/gpu_after.csv"
printf '%s\n' "${ACT_TRAIN_STATUS}" > "${ACT_MONITOR_DIR}/exit_status.txt"
exit "${ACT_TRAIN_STATUS}"
