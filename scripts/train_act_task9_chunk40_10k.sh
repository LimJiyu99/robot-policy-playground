#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/scripts/activate_lerobot.sh"
: "${LEROBOT_DATASET_ROOT:?Set LEROBOT_DATASET_ROOT to libero_object_image.}"
OUT="${ACT_OUTPUT_DIR:-$PROJECT_ROOT/outputs/act_task9_chunk40_10k}"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite: $OUT" >&2; exit 2; }

# Final ACT task-9 configuration, reconstructed from the 10K checkpoint's train_config.json.
EPISODES='[0,22,25,28,30,41,47,59,63,73,91,116,119,172,206,234,236,237,238,239,240,242,243,266,277,286,287,307,314,315,332,339,348,350,352,353,365,366,368,370,390,393,400,411,420]'
lerobot-train --dataset.repo_id=lerobot/libero_object_image --dataset.root="$LEROBOT_DATASET_ROOT" --dataset.episodes="$EPISODES" --dataset.return_uint8=true \
  --policy.type=act --policy.device=cuda --policy.chunk_size=40 --policy.n_action_steps=20 \
  --policy.normalization_mapping='{"VISUAL":"MEAN_STD","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --policy.vision_backbone=resnet18 --policy.optimizer_lr=1e-05 --policy.optimizer_lr_backbone=1e-05 --policy.optimizer_weight_decay=0.0001 --policy.kl_weight=10 --policy.push_to_hub=false \
  --output_dir="$OUT" --job_name=act_task9_chunk40_10k --batch_size=8 --steps=10000 --seed=42 --num_workers=8 --log_freq=50 --env_eval_freq=0 --save_checkpoint=true --save_freq=2000 --save_checkpoint_to_hub=false --wandb.enable=false --wandb.mode=disabled
