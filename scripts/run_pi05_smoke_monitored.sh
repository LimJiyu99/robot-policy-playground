#!/usr/bin/env bash
# Run exactly one official LIBERO-Object task-0 episode and sample host/GPU use.

set -uo pipefail

OUTPUT_DIR="outputs/pi05_task0_smoke"
REVISION="dbf8a3f794a9c4297b44f40b752712f50073d945"
SNAPSHOT="/workspace/jy/cache/huggingface/hub/models--lerobot--pi05_libero_finetuned_v044/snapshots/${REVISION}"
mkdir -p "${OUTPUT_DIR}"

# Prevent any network fallback: the pinned snapshot must already be in HF_HOME.
export HF_HUB_OFFLINE=1
export PI05_METRICS_PATH="${OUTPUT_DIR}/instrumentation.json"

# Store machine-wide memory immediately before evaluation.
free -b > "${OUTPUT_DIR}/ram_before.txt"
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits > "${OUTPUT_DIR}/gpu_before.csv"

# /usr/bin/time adds the evaluated process's peak resident-set size.
/usr/bin/time -v -o "${OUTPUT_DIR}/process_time.txt" \
    python scripts/pi05_eval_instrumented.py \
    --policy.path="${SNAPSHOT}" \
    --env.type=libero \
    --env.task=libero_object \
    --env.task_ids='[0]' \
    --env.max_parallel_tasks=1 \
    --eval.batch_size=1 \
    --eval.n_episodes=1 \
    --eval.recording=false \
    --output_dir="${OUTPUT_DIR}/eval" \
    > "${OUTPUT_DIR}/eval.log" 2>&1 &
eval_pid=$!

# Poll global system RAM and GPU VRAM while the single evaluation is alive.
printf 'unix_time\tram_used_bytes\tgpu_used_mib\n' > "${OUTPUT_DIR}/resource_samples.tsv"
while kill -0 "${eval_pid}" 2>/dev/null; do
    mem_total_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
    mem_available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    gpu_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
    printf '%s\t%s\t%s\n' "$(date +%s.%N)" \
        "$(( (mem_total_kib - mem_available_kib) * 1024 ))" \
        "${gpu_used_mib// /}" >> "${OUTPUT_DIR}/resource_samples.tsv"
    sleep 0.25
done

wait "${eval_pid}"
status=$?

# Always retain the post-run state, including when the single allowed run fails.
free -b > "${OUTPUT_DIR}/ram_after.txt"
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits > "${OUTPUT_DIR}/gpu_after.csv"
printf '%s\n' "${status}" > "${OUTPUT_DIR}/exit_status.txt"
exit "${status}"
