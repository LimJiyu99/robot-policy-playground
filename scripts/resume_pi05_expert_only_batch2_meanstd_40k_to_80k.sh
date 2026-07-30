#!/usr/bin/env bash
set -euo pipefail

cd /workspace/jy/projects/lerobot-libero-benchmark
source scripts/activate_lerobot.sh

RESUME_CONFIG="outputs/pi05_libero_object_multitask_batch2_meanstd_40k/checkpoints/040000/pretrained_model/train_config.json"
OUT="outputs/pi05_libero_object_multitask_batch2_meanstd_80k_resume40k"
RUNLOG="outputs/.runlogs/pi05_libero_object_multitask_batch2_meanstd_80k_resume40k"

test -f "$RESUME_CONFIG" || { echo "오류: 40K resume config가 없습니다: $RESUME_CONFIG"; exit 1; }

# Keep the original 40K run immutable; LeRobot creates this new output directory itself.
if [[ -e "$OUT" ]]; then
    echo "오류: 출력 폴더가 이미 존재합니다: $OUT"
    exit 1
fi
if [[ -e "$RUNLOG" ]]; then
    echo "오류: 임시 로그 폴더가 이미 존재합니다: $RUNLOG"
    exit 1
fi
mkdir -p "$RUNLOG"

nvidia-smi \
  --query-gpu=timestamp,memory.used,utilization.gpu,power.draw \
  --format=csv \
  -l 1 > "$RUNLOG/gpu_usage.csv" &
GPU_MONITOR_PID=$!

cleanup() {
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --steps is the global target, so this resumes at 40K and stops at 80K.
# save_freq=20K emits new checkpoints at 60K and 80K; the original 40K remains intact.
PYTHONUNBUFFERED=1 /usr/bin/time -v -o "$RUNLOG/training_time.txt" \
lerobot-train \
  --resume=true \
  --config_path="$RESUME_CONFIG" \
  --output_dir="$OUT" \
  --job_name=pi05_libero_object_multitask_batch2_meanstd_80k_resume40k \
  --steps=80000 \
  --save_freq=20000 \
  --log_freq=50 \
  --env_eval_freq=0 \
  --wandb.enable=false \
  --wandb.mode=disabled \
  2>&1 | tee "$RUNLOG/train.log"

cleanup
trap - EXIT INT TERM

cp "$RUNLOG/train.log" "$OUT/train.log"
cp "$RUNLOG/training_time.txt" "$OUT/training_time.txt"
cp "$RUNLOG/gpu_usage.csv" "$OUT/gpu_usage.csv"

test -d "$OUT/checkpoints/060000/pretrained_model"
test -d "$OUT/checkpoints/080000/pretrained_model"

echo "학습 완료: $OUT"
