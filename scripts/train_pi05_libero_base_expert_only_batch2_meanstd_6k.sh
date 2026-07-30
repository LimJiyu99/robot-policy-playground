#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/activate_lerobot.sh"

: "${LEROBOT_DATASET_ROOT:?Set LEROBOT_DATASET_ROOT to the local libero_object_image dataset.}"

# This is a fresh, controlled initialization experiment. It never resumes the
# pi05_base 80K run and writes into a separate adaptation root.
OUT="${PI05_OUTPUT_DIR:-$PROJECT_ROOT/outputs/pi05_libero_base_expert_only_batch2_meanstd_6k}"
RUNLOG="${PI05_RUNLOG_DIR:-$PROJECT_ROOT/outputs/.runlogs/pi05_libero_base_expert_only_batch2_meanstd_6k}"

if [[ -e "$OUT" ]]; then
    echo "오류: 출력 폴더가 이미 존재합니다: $OUT"
    exit 1
fi
if [[ -e "$RUNLOG" ]]; then
    echo "오류: 임시 로그 폴더가 이미 존재합니다: $RUNLOG"
    exit 1
fi

# save_freq only supports a fixed interval. At 1K it includes the required
# 1K, 3K, and 6K resumable checkpoints (and also 2K, 4K, and 5K).
# Six full pi05 checkpoints require about 65 GiB; leave a small write margin.
required_bytes=$((75 * 1024 * 1024 * 1024))
available_bytes=$(df -PB1 "$PWD" | awk 'NR == 2 { print $4 }')
if (( available_bytes < required_bytes )); then
    echo "오류: 1K 간격 checkpoint 저장에는 최소 75 GiB 여유 공간이 필요합니다."
    echo "현재 여유 공간: $((available_bytes / 1024 / 1024 / 1024)) GiB"
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

# No rename_map is needed here: unlike --policy.path, --policy.pretrained_path
# keeps the existing dataset-derived policy feature names (image, wrist_image).
# The two camera tensors retain the pretrained checkpoint's main/wrist order.
PYTHONUNBUFFERED=1 /usr/bin/time -v -o "$RUNLOG/training_time.txt" \
lerobot-train \
  --dataset.repo_id=lerobot/libero_object_image \
  --dataset.root="$LEROBOT_DATASET_ROOT" \
  --dataset.return_uint8=true \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_libero_base \
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
  --job_name=pi05_libero_base_expert_only_batch2_meanstd_6k \
  --batch_size=2 \
  --steps=6000 \
  --seed=42 \
  --num_workers=8 \
  --log_freq=50 \
  --env_eval_freq=0 \
  --save_checkpoint=true \
  --save_freq=1000 \
  --save_checkpoint_to_hub=false \
  --wandb.enable=false \
  --wandb.mode=disabled \
  2>&1 | tee "$RUNLOG/train.log"

cleanup
trap - EXIT INT TERM

cp "$RUNLOG/train.log" "$OUT/train.log"
cp "$RUNLOG/training_time.txt" "$OUT/training_time.txt"
cp "$RUNLOG/gpu_usage.csv" "$OUT/gpu_usage.csv"

test -d "$OUT/checkpoints/001000/pretrained_model"
test -d "$OUT/checkpoints/003000/pretrained_model"
test -d "$OUT/checkpoints/006000/pretrained_model"

echo "학습 완료: $OUT"
