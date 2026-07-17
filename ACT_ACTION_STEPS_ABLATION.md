# ACT 10k action-steps ablation

The ACT checkpoint at `outputs/act_task9_5k/checkpoints/010000/pretrained_model` was evaluated on LIBERO `libero_object` task id 9 with the same configuration as the existing 10k result: 20 episodes, seeds 42000--42019, batch size 5, 256x256 observations, and a single parallel task. The checkpoint's `chunk_size` remained 100. Only `policy.n_action_steps` was overridden to 20, 10, or 5; the existing 100-step result was reused without rerunning it.

| `n_action_steps` | Success rate | Wall-clock runtime | Sampled peak GPU memory | Mean / P95 `select_action` latency | Action-chunk generations |
|---:|---:|---:|---:|---:|---:|
| 100 (existing) | 0/20 (0.0%) | 38.41 s | 3,442 MiB | 4.091 / 0.731 ms | 12 |
| 20 | 10/20 (50.0%) | 51.34 s | 3,540 MiB | 4.496 / 1.579 ms | 56 |
| 10 | 3/20 (15.0%) | 43.71 s | 3,540 MiB | 4.911 / 6.566 ms | 112 |
| 5 | 2/20 (10.0%) | 42.37 s | 3,540 MiB | 5.380 / 7.486 ms | 224 |

PyTorch peak allocated memory was unchanged at 1,391,104,512 bytes for all four settings. Runtime is `/usr/bin/time -v` wall-clock time; GPU memory is the maximum of 0.25-second `nvidia-smi` samples. Latency includes cache hits and action-chunk generations, so shorter execution queues increase the number of expensive generations during the same 1,120 control steps.

## Videos

Each new setting produced 10 MP4 files (`eval_episode_0.mp4` through `eval_episode_9.mp4`) in the following directories, matching the existing evaluator's video output convention:

- `n_action_steps=100`: `outputs/act_task9_10k_eval/eval/videos/libero_object_9/`
- `n_action_steps=20`: `outputs/act_task9_10k_action_steps_20_eval_run2/eval/videos/libero_object_9/`
- `n_action_steps=10`: `outputs/act_task9_10k_action_steps_10_eval_run2/eval/videos/libero_object_9/`
- `n_action_steps=5`: `outputs/act_task9_10k_action_steps_5_eval_run2/eval/videos/libero_object_9/`

## Observations

Reducing `n_action_steps` from 100 to 20 materially improved fixed-seed task success (0/20 to 10/20). Reducing it further to 10 or 5 lowered success to 3/20 and 2/20, respectively. Shorter queues also required more action-chunk generations (56, 112, and 224 versus 12), increasing mean and tail latency and making the end-to-end runs slower than the reused 100-step baseline. This ablation identifies 20 as the strongest tested setting under the stated deterministic evaluation.

Re-run the new settings with `scripts/eval_act_task9_10k_action_steps.sh`. The script preserves the checkpoint's 100-action chunk size and changes only `n_action_steps`; it writes each setting to its own output directory and refuses to overwrite it.
