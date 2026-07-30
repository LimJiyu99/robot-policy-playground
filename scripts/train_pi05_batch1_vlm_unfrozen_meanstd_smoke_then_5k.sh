#!/usr/bin/env bash
# π0.5: train the Action Expert plus the VLM language path; keep vision frozen.

set -euo pipefail

source scripts/activate_lerobot.sh

SMOKE_OUTPUT="outputs/pi05_libero_object_multitask_batch1_vlm_unfrozen_meanstd_smoke20"
FULL_OUTPUT="outputs/pi05_libero_object_multitask_batch1_vlm_unfrozen_meanstd_5k"
LOG_DIR="outputs/.runlogs/pi05_batch1_vlm_unfrozen_meanstd_smoke_then_5k"
GPU_MONITOR_PID=""

for output_dir in "${SMOKE_OUTPUT}" "${FULL_OUTPUT}"; do
    if [[ -e "${output_dir}" ]]; then
        echo "Refusing to overwrite existing training output: ${output_dir}" >&2
        exit 2
    fi
done
if [[ -e "${LOG_DIR}" && -n "$(find "${LOG_DIR}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite existing temporary logs: ${LOG_DIR}" >&2
    exit 2
fi
mkdir -p "${LOG_DIR}"

stop_gpu_monitor() {
    if [[ -n "${GPU_MONITOR_PID}" ]]; then
        kill "${GPU_MONITOR_PID}" 2>/dev/null || true
        wait "${GPU_MONITOR_PID}" 2>/dev/null || true
        GPU_MONITOR_PID=""
    fi
}
trap stop_gpu_monitor EXIT INT TERM

start_gpu_monitor() {
    local output_path="$1"
    printf 'timestamp,gpu_memory_used_mib\n' > "${output_path}"
    (
        while true; do
            nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits | head -n 1
            sleep 1
        done
    ) >> "${output_path}" &
    GPU_MONITOR_PID=$!
}

COMMON_ARGS=(
    --dataset.repo_id=lerobot/libero_object_image
    --dataset.root=/workspace/jy/datasets/lerobot/libero_object_image
    --dataset.return_uint8=true
    --policy.type=pi05
    --policy.pretrained_path=lerobot/pi05_base
    --policy.device=cuda
    --policy.dtype=bfloat16
    --policy.train_expert_only=false
    --policy.freeze_vision_encoder=true
    --policy.gradient_checkpointing=true
    --policy.compile_model=false
    --policy.use_relative_actions=false
    --policy.chunk_size=50
    --policy.n_action_steps=10
    --policy.push_to_hub=false
    '--policy.normalization_mapping={"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'
    --batch_size=1
    --seed=42
    --num_workers=8
    --log_freq=50
    --env_eval_freq=0
    --save_checkpoint_to_hub=false
    --wandb.enable=false
    --wandb.mode=disabled
)

run_training() {
    local output_dir="$1"
    local job_name="$2"
    local steps="$3"
    local save_checkpoint="$4"
    local save_freq="$5"
    local log_path="$6"
    local time_path="$7"
    local gpu_path="$8"

    local status=0
    start_gpu_monitor "${gpu_path}"
    /usr/bin/time -v -o "${time_path}" \
        lerobot-train "${COMMON_ARGS[@]}" \
        --output_dir="${output_dir}" \
        --job_name="${job_name}" \
        --steps="${steps}" \
        --save_checkpoint="${save_checkpoint}" \
        --save_freq="${save_freq}" \
        > "${log_path}" 2>&1 || status=$?
    stop_gpu_monitor
    return "${status}"
}

SMOKE_LOG="${LOG_DIR}/smoke_train.log"
if ! run_training \
    "${SMOKE_OUTPUT}" \
    "pi05_libero_object_batch1_vlm_unfrozen_meanstd_smoke20" \
    20 false 20 \
    "${SMOKE_LOG}" "${LOG_DIR}/smoke_training_time.txt" "${LOG_DIR}/smoke_gpu_usage.csv"; then
    echo "Smoke training failed; full training was not started. See ${SMOKE_LOG}" >&2
    exit 1
fi
if rg -qi '(^|[^[:alpha:]])nan([^[:alpha:]]|$)' "${SMOKE_LOG}"; then
    echo "Smoke training reported NaN; full training was not started. See ${SMOKE_LOG}" >&2
    exit 1
fi

FULL_LOG="${LOG_DIR}/train.log"
run_training \
    "${FULL_OUTPUT}" \
    "pi05_libero_object_batch1_vlm_unfrozen_meanstd_5k" \
    5000 true 2500 \
    "${FULL_LOG}" "${LOG_DIR}/training_time.txt" "${LOG_DIR}/gpu_usage.csv"

for step in 002500 005000; do
    checkpoint="${FULL_OUTPUT}/checkpoints/${step}/pretrained_model"
    [[ -d "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}" >&2; exit 1; }
done
echo "Smoke and 5K training completed successfully."
