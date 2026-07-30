# ACT LIBERO-Object training smoke report

Date: 2026-07-17 (Asia/Seoul)

## Result

The ACT training smoke test completed successfully on CUDA with batch size 8.
It ran exactly 200 optimizer updates, printed loss every 10 steps, and saved
checkpoints at steps 100 and 200.  No CUDA out-of-memory error occurred, so
the batch-size-4 fallback was not used.

## Environment and CLI

- Project: `/workspace/jy/projects/lerobot-libero-benchmark`
- Conda prefix: `/workspace/jy/conda/envs/lerobot_libero`
- LeRobot: `0.6.1` editable source
- PyTorch: `2.11.0+cu128`
- CUDA device: NVIDIA GeForce RTX 4090
- `lerobot-train --help` was checked before constructing the command.

The first train attempt stopped before dataset/model construction because
LeRobot 0.6.1 requires the `accelerate` package for `lerobot-train`.  The
minimal official remedy was applied:

```bash
source scripts/activate_lerobot.sh
python -m pip install -e './lerobot[training]'
```

This installed `accelerate==1.14.0` and the training extra's compatible
`wandb==0.27.2`; W&B remained disabled for the actual training run.  PyTorch
and its CUDA build were not changed.

## Dataset and ACT compatibility

The requested dataset is `lerobot/libero_object_image` at fixed revision
`e1e080d7df1d0a359dff5c86c222e047549f447f`.  The full dataset is 454 episodes
and 66,984 frames.  For this short smoke test, only local episodes 0–6 were
selected (981 frames, first two parquet shards) rather than downloading the
entire 8.615 GiB dataset.

| Feature | Dataset metadata | Runtime tensor | ACT use |
|---|---:|---:|---|
| `observation.images.image` | HWC `(256, 256, 3)` | CHW `(3, 256, 256)` | first visual camera |
| `observation.images.wrist_image` | HWC `(256, 256, 3)` | CHW `(3, 256, 256)` | second visual camera |
| `observation.state` | `(8,)` float32 | `(8,)` float32 | optional proprioceptive input |
| `action` | `(7,)` float32 | `(7,)` float32 | required policy output |

ACT supports one or more `observation.images.*` inputs only when their image
shapes match; both cameras are 256×256×3, so they are compatible.  Its state
input is optional but the available 8-D state is accepted, and the required
`action` key is present with 7 dimensions.  The default ACT action chunk size
is 100; each selected episode is at least 124 frames, leaving valid temporal
training samples.

## Reproducible command

Run the executable script below from the project root.  It refuses to
overwrite an existing `outputs/act_smoke` directory and records resource
samples under `outputs/act_smoke_monitor`.

```bash
source scripts/activate_lerobot.sh
scripts/train_act_smoke.sh
```

The effective LeRobot command was:

```bash
lerobot-train \
  --dataset.repo_id=lerobot/libero_object_image \
  --dataset.root=/workspace/jy/datasets/lerobot/libero_object_image \
  --dataset.revision=e1e080d7df1d0a359dff5c86c222e047549f447f \
  --dataset.episodes='[0,1,2,3,4,5,6]' \
  --dataset.return_uint8=true \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=outputs/act_smoke --job_name=act_libero_object_smoke \
  --steps=200 --batch_size=8 --num_workers=2 --log_freq=10 \
  --env_eval_freq=0 --save_checkpoint=true --save_freq=100 \
  --save_checkpoint_to_hub=false --wandb.enable=false --wandb.mode=disabled
```

If a future run at batch size 8 reports a CUDA OOM, use the script's explicit
fallback without changing its other settings:

```bash
ACT_SMOKE_BATCH_SIZE=4 scripts/train_act_smoke.sh
```

## Measured results

| Check | Result |
|---|---:|
| Training process exit code | 0 |
| Updates | 200 |
| Effective batch size | 8 |
| Loss at step 10 | 42.921 |
| Loss at step 100 | 4.074 |
| Loss at step 200 | 3.406 |
| Whole process elapsed time | 4 min 14.51 s |
| Peak GPU VRAM (`nvidia-smi` polling) | 2,316 MiB (2.262 GiB) |
| LeRobot logged GPU allocation | about 1.34–1.41 GiB |
| Peak process RSS | 2,523,108 KiB (2.406 GiB) |
| W&B | disabled |
| Hub push / checkpoint push | disabled / disabled |

The default ACT ResNet-18 ImageNet weights were downloaded once (44.7 MB) to
`/workspace/jy/cache/torch/hub/checkpoints/`, which is the configured Torch
cache.  No model or dataset cache was created under the home directory.

## Output files

- Final checkpoint: `outputs/act_smoke/checkpoints/000200/`
- Intermediate checkpoint: `outputs/act_smoke/checkpoints/000100/`
- Final policy weights: `outputs/act_smoke/checkpoints/000200/pretrained_model/model.safetensors`
- Full training log: `outputs/act_smoke_monitor/train.log`
- GPU samples: `outputs/act_smoke_monitor/gpu_samples.tsv`
- Process timing: `outputs/act_smoke_monitor/process_time.txt`
