#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/scripts/activate_lerobot.sh"
: "${LEROBOT_DATASET_ROOT:?Set LEROBOT_DATASET_ROOT to libero_object_image.}"
OUT="${DIFFUSION_OUTPUT_DIR:-$PROJECT_ROOT/outputs/diffusion_task9_h40_a15_10k}"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite: $OUT" >&2; exit 2; }

# Exact task-9 episode list and key hyperparameters from the completed 10K config.
EPISODES='[0,22,25,28,30,41,47,59,63,73,91,116,119,172,206,234,236,237,238,239,240,242,243,266,277,286,287,307,314,315,332,339,348,350,352,353,365,366,368,370,390,393,400,411,420]'
lerobot-train --dataset.repo_id=lerobot/libero_object_image --dataset.root="$LEROBOT_DATASET_ROOT" --dataset.episodes="$EPISODES" --dataset.return_uint8=true \
  --policy.type=diffusion --policy.device=cuda --policy.horizon=40 --policy.n_action_steps=15 --policy.n_obs_steps=2 \
  --policy.normalization_mapping='{"VISUAL":"MEAN_STD","STATE":"MIN_MAX","ACTION":"MIN_MAX"}' \
  --policy.use_separate_rgb_encoder_per_camera=true --policy.optimizer_lr=0.0001 --policy.scheduler_name=cosine --policy.scheduler_warmup_steps=500 \
  --policy.push_to_hub=false --output_dir="$OUT" --job_name=diffusion_task9_h40_a15_10k --batch_size=8 --steps=10000 --seed=42 --num_workers=8 --log_freq=50 --env_eval_freq=0 --save_checkpoint=true --save_freq=2000 --save_checkpoint_to_hub=false --wandb.enable=false --wandb.mode=disabled
