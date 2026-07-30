# ACT task-9 10k evaluation

The 10,000-update checkpoint (`outputs/act_task9_5k/checkpoints/010000/pretrained_model`) was evaluated with the same deterministic configuration as the 5k baseline: LIBERO `libero_object` task id 9, 20 episodes, seeds 42000--42019, batch size 5, 256x256 observations, and no recording requested. The completed evaluation output is `outputs/act_task9_10k_eval`; it exited with status 0 and contains all 20 seeded episode records.

All 20 episodes reached the 280-step LIBERO-Object limit with `success=false`, for a success rate of 0/20 (0.0%). The evaluator's rollout time was 31.762 s (1.588 s/episode); end-to-end wall-clock runtime measured by `/usr/bin/time -v` was 38.41 s. Sampled peak GPU memory was 3,442 MiB, and PyTorch peak allocated memory was 1,391,104,512 bytes. Across 1,120 `select_action` calls, mean latency was 4.091 ms and p95 was 0.731 ms; the mean includes the first cold action-chunk generation.

| Metric | 5k (`005000`) | 10k (`010000`) | Change (10k - 5k) |
|---|---:|---:|---:|
| Success rate | 0/20 (0.0%) | 0/20 (0.0%) | 0 episodes |
| Wall-clock runtime | 41.78 s | 38.41 s | -3.37 s (-8.1%) |
| Sampled peak GPU memory | 3,450 MiB | 3,442 MiB | -8 MiB |
| Mean `select_action` latency | 4.291 ms | 4.091 ms | -0.200 ms (-4.7%) |
| P95 `select_action` latency | 0.766 ms | 0.731 ms | -0.035 ms (-4.6%) |

The additional 5,000 updates did not improve task-9 success under this fixed 20-seed evaluation. Runtime, sampled GPU use, and action latency were slightly lower in this run; these are run-level measurements and should not be interpreted as a training-quality improvement.

Re-run evaluation: `scripts/eval_act_task9_10k.sh`. The script is a copy of the 5k evaluator with only the checkpoint changed to `010000` and a separate output directory.
