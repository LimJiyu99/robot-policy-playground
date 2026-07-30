# ACT 10K gripper analysis

Action semantics were verified from the 7D LIBERO EEF convention used by this task: `action[0:3]` is end-effector delta position, `action[3:6]` is delta axis-angle rotation, and `action[6]` is the gripper command. Positive/negative `action[6]` are reported as open/close directions.

All 415 task-9 training actions have gripper range -1.000 to 1.000: positive/open 233 (56.1%), negative/close 182 (43.9%), zero 0.

The 10K policy was run on 415 frames from representative training episodes [0, 22, 25]. Per-dimension MAE (`action[0]`..`action[6]`) is 0.7381, 0.5018, 0.4243, 0.2735, 0.5117, 0.5897, 0.1450. Gripper MAE is 0.1450; open/close direction agreement is 98.1%.

The three GT/prediction plots are `outputs/act_gripper_analysis/episode_{0,22,25}_gripper.png`; machine-readable metrics are `outputs/act_gripper_analysis/metrics.json`.
