# ACT refined action-steps evaluation

All evaluations use LIBERO `libero_object` task id 9, batch size 5, 256x256 observations, one parallel task, and the fixed seed sequence beginning at 42000. Both checkpoints retain `chunk_size=100`; only `policy.n_action_steps` changes. Existing 10k results for 100, 20, 10, and 5 action steps were reused without rerunning them.

## 10k checkpoint: 20 episodes

Each row uses seeds 42000--42019. Runtime is `/usr/bin/time -v` wall-clock time; GPU memory is the maximum of 0.25-second `nvidia-smi` samples. Mean and P95 latency include queue hits and action-chunk generation calls.

| `n_action_steps` | Success rate | Runtime | Peak GPU memory | Mean / P95 `select_action` latency |
|---:|---:|---:|---:|---:|
| 100 (existing) | 0/20 (0.0%) | 38.41 s | 3,442 MiB | 4.091 / 0.731 ms |
| 5 (existing) | 2/20 (10.0%) | 42.37 s | 3,540 MiB | 5.380 / 7.486 ms |
| 10 (existing) | 3/20 (15.0%) | 43.71 s | 3,540 MiB | 4.911 / 6.566 ms |
| 15 | 11/20 (55.0%) | 51.84 s | 3,539 MiB | 4.535 / 5.101 ms |
| 20 (existing) | 10/20 (50.0%) | 51.34 s | 3,540 MiB | 4.496 / 1.579 ms |
| 25 | 7/20 (35.0%) | 49.47 s | 3,536 MiB | 4.432 / 0.991 ms |
| 30 | 2/20 (10.0%) | 43.50 s | 3,535 MiB | 4.463 / 1.010 ms |
| 40 | 0/20 (0.0%) | 41.18 s | 3,539 MiB | 4.404 / 0.904 ms |

The 20-episode ranking selected 15 and 20 action steps for the requested 100-episode confirmation.

## Top-two confirmation: 100 episodes

Both rows use seeds 42000--42099. The longer evaluation reverses the close 20-episode ordering: 20 action steps reaches 62/100, ahead of 15 action steps at 44/100.

| `n_action_steps` | Success rate | Runtime | Peak GPU memory | Mean / P95 `select_action` latency |
|---:|---:|---:|---:|---:|
| 15 | 44/100 (44.0%) | 3:06.26 | 3,540 MiB | 1.738 / 5.446 ms |
| 20 | 62/100 (62.0%) | 3:24.23 | 3,540 MiB | 1.622 / 4.225 ms |

## 5k versus 10k at 20 action steps

Both rows use 20 episodes and seeds 42000--42019. The 10k checkpoint doubles the observed success count at this selected action queue length.

| Checkpoint | Success rate | Runtime | Peak GPU memory | Mean / P95 `select_action` latency |
|---|---:|---:|---:|---:|
| 5k (`005000`) | 5/20 (25.0%) | 46.30 s | 3,535 MiB | 4.539 / 1.286 ms |
| 10k (`010000`) | 10/20 (50.0%) | 51.34 s | 3,540 MiB | 4.496 / 1.579 ms |

## Videos and observations

Every new setting completed with exit status 0 and emitted 10 MP4 files (`eval_episode_0.mp4` through `eval_episode_9.mp4`) under its `eval/videos/libero_object_9/` directory. New video roots are:

- `outputs/act_task9_10k_action_steps_{15,25,30,40}_eval_refined20/`
- `outputs/act_task9_5k_action_steps_20_eval_refined20/`
- `outputs/act_task9_10k_action_steps_{15,20}_eval_refined100/`

The 100-episode result makes `n_action_steps=20` the preferred tested setting: it has the highest confirmed success rate (62.0%) while keeping the trained 100-action chunk size. The shorter 15-step queue is competitive in the initial 20-episode sample but does not retain that lead at 100 episodes.

Re-run the new evaluations with `scripts/eval_act_task9_refined_action_steps.sh`, setting `ACT_CHECKPOINT_STEP`, `ACT_ACTION_STEPS_LIST`, `ACT_EVAL_EPISODES`, and `ACT_OUTPUT_TAG`. It preserves `chunk_size=100`, writes separate output directories, and refuses to overwrite existing output.
