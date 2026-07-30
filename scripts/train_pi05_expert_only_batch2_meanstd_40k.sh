#!/usr/bin/env bash
set -euo pipefail

cd /workspace/jy/projects/lerobot-libero-benchmark
source scripts/activate_lerobot.sh

OUT="outputs/pi05_libero_object_multitask_batch2_meanstd_40k"
RUNLOG="outputs/.runlogs/pi05_libero_object_multitask_batch2_meanstd_40k"

# Keep LeRobot's output directory absent until training starts; temporary logs live separately.
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

PYTHONUNBUFFERED=1 /usr/bin/time -v -o "$RUNLOG/training_time.txt" \
lerobot-train \
  --dataset.repo_id=lerobot/libero_object_image \
  --dataset.root=/workspace/jy/datasets/lerobot/libero_object_image \
  --dataset.return_uint8=true \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.use_peft=false \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.use_relative_actions=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=10 \
  --policy.push_to_hub=false \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --output_dir="$OUT" \
  --job_name=pi05_libero_object_multitask_batch2_meanstd_40k \
  --batch_size=2 \
  --steps=40000 \
  --seed=42 \
  --num_workers=8 \
  --log_freq=50 \
  --env_eval_freq=0 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --save_checkpoint_to_hub=false \
  --wandb.enable=false \
  --wandb.mode=disabled \
  2>&1 | tee "$RUNLOG/train.log"

cleanup
trap - EXIT INT TERM

cp "$RUNLOG/train.log" "$OUT/train.log"
cp "$RUNLOG/training_time.txt" "$OUT/training_time.txt"
cp "$RUNLOG/gpu_usage.csv" "$OUT/gpu_usage.csv"

test -d "$OUT/checkpoints/010000/pretrained_model"
test -d "$OUT/checkpoints/020000/pretrained_model"
test -d "$OUT/checkpoints/040000/pretrained_model"

echo "학습 완료: $OUT"
