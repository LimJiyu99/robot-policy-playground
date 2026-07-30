#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/scripts/activate_lerobot.sh"
: "${LEROBOT_DATASET_ROOT:?Set LEROBOT_DATASET_ROOT to libero_object_image.}"
OUT="${SMOLVLA_OUTPUT_DIR:-$PROJECT_ROOT/outputs/smolvla_libero_object_multitask_batch4_20k}"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite: $OUT" >&2; exit 2; }

# Full 454-episode multi-task training; options follow the initial batch-4 checkpoint config.
lerobot-train --dataset.repo_id=lerobot/libero_object_image --dataset.root="$LEROBOT_DATASET_ROOT" --dataset.return_uint8=true \
  --policy.type=smolvla --policy.pretrained_path=lerobot/smolvla_base --policy.device=cuda --policy.chunk_size=40 --policy.n_action_steps=15 \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --policy.freeze_vision_encoder=true --policy.train_expert_only=true --policy.train_state_proj=true --policy.optimizer_lr=0.0001 --policy.scheduler_warmup_steps=1000 --policy.scheduler_decay_steps=30000 --policy.scheduler_decay_lr=2.5e-06 --policy.push_to_hub=false \
  --output_dir="$OUT" --job_name=smolvla_libero_object_multitask_batch4_20k --batch_size=4 --steps=20000 --seed=42 --num_workers=8 --log_freq=50 --env_eval_freq=0 --save_checkpoint=true --save_freq=5000 --save_checkpoint_to_hub=false --wandb.enable=false --wandb.mode=disabled
