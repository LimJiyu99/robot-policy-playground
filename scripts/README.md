# Scripts

| Script | Purpose |
| --- | --- |
| `smoke_test_libero.py` | Validate the LIBERO installation and rendering. |
| `eval_policy_instrumented.py` | Evaluate a local LeRobot checkpoint with episode JSON instrumentation. |
| `train_act_smoke.sh` | Short ACT training smoke test. |
| `eval_act_smoke.sh` | Three-rollout ACT smoke evaluation with instrumentation. |
| `train_act_task9_5k.sh` | ACT task-9 5K training run used by the historical reports. |
| `eval_act_task9_{5k,10k,10k_action_steps}.sh` | Deterministic ACT task-9 evaluation and action-step ablation. |
| `analyze_act_gripper.py` | Compare ACT action predictions and ground-truth gripper commands. |
| `train_pi05_libero_base_expert_only_batch2_meanstd_6k.sh` | Fresh π0.5 expert-only 6K adaptation from `lerobot/pi05_libero_base`. |
| `eval_pi05_libero_base_vs_expert_only_6k_eval100.py` | Same-seed raw-base versus 6K, 100-episode comparison. |
| `eval_pi05_libero_base_expert_only_6k_demo_videos.py` | Render five π0.5 6K representative rollouts. |

Set `LEROBOT_DATASET_ROOT` to the local `libero_object_image` dataset before running dataset-dependent scripts. Checkpoints and outputs are intentionally excluded from Git.
