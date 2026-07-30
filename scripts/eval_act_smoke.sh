#!/usr/bin/env bash
# Run three ACT smoke rollouts on the trained LIBERO-Object orange-juice task.

set -uo pipefail

ACT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACT_CHECKPOINT="${ACT_PROJECT_ROOT}/outputs/act_smoke/checkpoints/000200/pretrained_model"
ACT_OUTPUT_DIR="${ACT_PROJECT_ROOT}/outputs/act_smoke_eval"

if [[ ! -f "${ACT_CHECKPOINT}/config.json" || ! -f "${ACT_CHECKPOINT}/model.safetensors" ]]; then
    echo "ACT checkpoint is incomplete or missing: ${ACT_CHECKPOINT}" >&2
    exit 2
fi

source "${ACT_PROJECT_ROOT}/scripts/activate_lerobot.sh"
mkdir -p "${ACT_OUTPUT_DIR}"

# The task mapping was verified against metadata: dataset episode 0 has task_index
# 0 and language "pick up the orange juice ..."; LIBERO-Object uses suite task_id 9
# for that same language.  The three vector-env slots select init states 0, 1, and 2.
export HF_HUB_OFFLINE=1
export ACT_SMOKE_EVAL_METRICS_PATH="${ACT_OUTPUT_DIR}/instrumentation.json"

nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits > "${ACT_OUTPUT_DIR}/gpu_before.csv"

/usr/bin/time -v -o "${ACT_OUTPUT_DIR}/process_time.txt" \
    python "${ACT_PROJECT_ROOT}/scripts/eval_policy_instrumented.py" \
    --policy.path="${ACT_CHECKPOINT}" \
    --env.type=libero \
    --env.task=libero_object \
    --env.task_ids='[9]' \
    --env.camera_name_mapping='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}' \
    --env.observation_height=256 \
    --env.observation_width=256 \
    --env.max_parallel_tasks=1 \
    --eval.batch_size=3 \
    --eval.n_episodes=3 \
    --eval.recording=false \
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
