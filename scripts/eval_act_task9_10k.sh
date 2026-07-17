#!/usr/bin/env bash
# Evaluate the final task-9 ACT checkpoint on 20 deterministic LIBERO rollouts.

set -uo pipefail

ACT_PROJECT_ROOT="/workspace/jy/projects/lerobot-libero-benchmark"
ACT_CHECKPOINT="${ACT_PROJECT_ROOT}/outputs/act_task9_5k/checkpoints/010000/pretrained_model"
ACT_OUTPUT_DIR="${ACT_PROJECT_ROOT}/outputs/act_task9_10k_eval"
ACT_EVAL_SEED=42000
ACT_EVAL_EPISODES=20
ACT_EVAL_BATCH_SIZE=5

if [[ ! -f "${ACT_CHECKPOINT}/config.json" || ! -f "${ACT_CHECKPOINT}/model.safetensors" ]]; then
    echo "Final ACT checkpoint is incomplete or missing: ${ACT_CHECKPOINT}" >&2
    exit 2
fi
if [[ -e "${ACT_OUTPUT_DIR}" ]]; then
    echo "Refusing to overwrite existing output: ${ACT_OUTPUT_DIR}" >&2
    exit 2
fi

source "${ACT_PROJECT_ROOT}/scripts/activate_lerobot.sh"
export HF_HUB_OFFLINE=1
export ACT_SMOKE_EVAL_METRICS_PATH="${ACT_OUTPUT_DIR}/instrumentation.json"
mkdir -p "${ACT_OUTPUT_DIR}"
printf '%s\n' "${ACT_EVAL_SEED}..$((ACT_EVAL_SEED + ACT_EVAL_EPISODES - 1))" > "${ACT_OUTPUT_DIR}/seed_range.txt"
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits > "${ACT_OUTPUT_DIR}/gpu_before.csv"

# LeRobot expands start_seed sequentially, so the fixed seed list is exactly 42000 through 42019.
/usr/bin/time -v -o "${ACT_OUTPUT_DIR}/process_time.txt" \
    python "${ACT_PROJECT_ROOT}/scripts/eval_act_smoke_instrumented.py" \
    --policy.path="${ACT_CHECKPOINT}" \
    --env.type=libero \
    --env.task=libero_object \
    --env.task_ids='[9]' \
    --env.camera_name_mapping='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}' \
    --env.observation_height=256 \
    --env.observation_width=256 \
    --env.max_parallel_tasks=1 \
    --eval.batch_size="${ACT_EVAL_BATCH_SIZE}" \
    --eval.n_episodes="${ACT_EVAL_EPISODES}" \
    --eval.recording=false \
    --seed="${ACT_EVAL_SEED}" \
    --output_dir="${ACT_OUTPUT_DIR}/eval" \
    > "${ACT_OUTPUT_DIR}/eval.log" 2>&1 &
ACT_EVAL_PID=$!

printf 'unix_time\tgpu_used_mib\n' > "${ACT_OUTPUT_DIR}/gpu_samples.tsv"
while kill -0 "${ACT_EVAL_PID}" 2>/dev/null; do
    ACT_GPU_USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR==1 {gsub(/ /, ""); print; exit}')"
    printf '%s\t%s\n' "$(date +%s.%N)" "${ACT_GPU_USED_MIB}" >> "${ACT_OUTPUT_DIR}/gpu_samples.tsv"
    sleep 0.25
done

wait "${ACT_EVAL_PID}"
ACT_EVAL_STATUS=$?
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits > "${ACT_OUTPUT_DIR}/gpu_after.csv"
printf '%s\n' "${ACT_EVAL_STATUS}" > "${ACT_OUTPUT_DIR}/exit_status.txt"
exit "${ACT_EVAL_STATUS}"
