# ACT smoke rollout evaluation report

## Result

The 200-step ACT smoke checkpoint completed a three-rollout LIBERO evaluation without error (process exit status `0`). This is a pipeline smoke test, not a performance measurement. All three rollouts returned `success: false` (0.0%), which is expectedly inconclusive for a checkpoint trained for only 200 updates on 981 frames.

- Checkpoint: `outputs/act_smoke/checkpoints/000200/pretrained_model`
- Evaluated suite/task: `libero_object`, suite `task_id=9`
- Task: `pick up the orange juice and place it in the basket`
- Rollouts: 3 parallel initial states (seeds 1000, 1001, 1002)
- Image resolution: 256 x 256, matching the ACT training data
- LeRobot: `0.6.1` editable (`lerobot` source commit `3f2179f3`)

## Interface validation

Before implementation, the repository structure and Git status were inspected, and `lerobot-eval --help` was run in the project activation environment. The installed evaluator supports the options used here: `--policy.path`, `--env.type=libero`, `--env.task`, `--env.task_ids`, `--env.camera_name_mapping`, `--env.observation_height`, `--env.observation_width`, `--eval.batch_size`, `--eval.n_episodes`, and `--output_dir`.

The implementation delegates all environment creation, preprocessing, rollout logic, success checks, and video writing to the installed `lerobot.scripts.lerobot_eval` implementation. The local Python file only wraps its public functions to record timing, tensor shapes, and per-episode results; no external LeRobot or site-packages files were modified.

## Dataset task mapping

The smoke-training selection is not a single task. Dataset metadata for episodes 0--6 has the following language-task mapping. The dataset `task_index` is an ordinal for this converted dataset and is **not** the same as the installed LIBERO suite task id. The suite mapping was verified by querying the installed `libero_object` benchmark task list by language.

| Training episode | Dataset `task_index` | Metadata task language | `libero_object` suite task id |
| --- | ---: | --- | ---: |
| 0 | 0 | pick up the orange juice and place it in the basket | 9 |
| 1 | 1 | pick up the ketchup and place it in the basket | 4 |
| 2 | 1 | pick up the ketchup and place it in the basket | 4 |
| 3 | 2 | pick up the cream cheese and place it in the basket | 1 |
| 4 | 3 | pick up the bbq sauce and place it in the basket | 3 |
| 5 | 3 | pick up the bbq sauce and place it in the basket | 3 |
| 6 | 4 | pick up the alphabet soup and place it in the basket | 0 |

The evaluation deliberately chooses suite task 9 (orange juice), so it evaluates a task actually represented in the smoke checkpoint's training subset. `--eval.batch_size=3` creates three environments for that one task and evaluates three distinct deterministic initial-state slots; it does not reinterpret dataset episode indices as simulator task ids.

## Pipeline evidence

| Check | Observed result |
| --- | --- |
| Checkpoint loading | Loaded normally in 0.337 s; 51,600,263 parameters |
| CUDA | Available; policy parameters and normalized inputs on `cuda:0` |
| Environment observation images | `image` and `wrist_image`: `(3, 3, 256, 256)` = batch, CHW |
| State after LIBERO processor | `observation.state`: `(3, 8)` |
| Policy action | `(3, 7)` |
| Environment stepping | 280 actions applied per environment; rollout action tensor `(3, 280, 7)` |
| Episode termination | All three first reached `done` at step 280 (0-based index 279); final done values all `true` |
| Success signal | `success_seen_during_rollout`: `[false, false, false]`; LIBERO evaluator success rate 0.0% |
| Inference | 280 `select_action` calls; three ACT chunk generations; mean call 14.64 ms, mean chunk generation 1.332 s (first cold generation 3.988 s) |
| GPU VRAM | PyTorch peak allocated: 1,313.15 MiB (1.282 GiB); sampled total GPU-use peak: 2,488 MiB (288 MiB idle baseline) |
| Total wall time | 19.90 s for the evaluated process; LeRobot rollout aggregate 13.52 s (4.51 s/episode) |

The LIBERO-Object adapter has a 280-step episode limit. No rollout succeeded early, so LeRobot's rollout completion occurred at that configured limit; this verifies normal step execution and clean terminal handling, but does not indicate task competence.

## Per-episode results

| Eval episode | Seed | Sum reward | Max reward | LIBERO success |
| ---: | ---: | ---: | ---: | --- |
| 0 | 1000 | 0.0 | 0.0 | false |
| 1 | 1001 | 0.0 | 0.0 | false |
| 2 | 1002 | 0.0 | 0.0 | false |

## Reproduction

From the project root:

```bash
source scripts/activate_lerobot.sh
scripts/eval_act_smoke.sh
```

The wrapper uses the local checkpoint, runs offline with `HF_HUB_OFFLINE=1`, captures process/GPU samples, and invokes `scripts/eval_act_smoke_instrumented.py`. The latter forwards the configured arguments into the installed `lerobot-eval` entry point.

## Output files

- Evaluation log: `outputs/act_smoke_eval/eval.log`
- Instrumented metrics: `outputs/act_smoke_eval/instrumentation.json`
- Official aggregate result: `outputs/act_smoke_eval/eval/eval_info.json`
- Resource samples: `outputs/act_smoke_eval/gpu_samples.tsv` and `outputs/act_smoke_eval/process_time.txt`
- Rollout videos:
  - `outputs/act_smoke_eval/eval/videos/libero_object_9/eval_episode_0.mp4`
  - `outputs/act_smoke_eval/eval/videos/libero_object_9/eval_episode_1.mp4`
  - `outputs/act_smoke_eval/eval/videos/libero_object_9/eval_episode_2.mp4`
