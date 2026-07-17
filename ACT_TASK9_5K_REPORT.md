# ACT task-9 5k baseline

Task-index 0 metadata selected 45 episodes (IDs: 0,22,25,28,30,41,47,59,63,73,91,116,119,172,206,234,236,237,238,239,240,242,243,266,277,286,287,307,314,315,332,339,348,350,352,353,365,366,368,370,390,393,400,411,420), totaling 6,291 frames; mean length 139.8. This maps by language to LIBERO `libero_object` task id 9 (orange juice).

LeRobot 0.6.1 (`3f2179f3`) trained ACT for 5,000 CUDA updates, seed 42, with W&B and Hub push disabled. Checkpoints were saved every 500 steps, including `001000`, `002500`, and final `005000`.

Batch probes (30 real updates, two workers) all completed without OOM: batch 8/16/32 had sampled peaks 2,361/3,281/5,015 MiB. Decode-limited end-to-end step times were about 1.23/2.57/4.86 s, so batch 8 was selected. The final run used eight workers to remove that host bottleneck; it achieved roughly 21--26 samples/s while retaining the same batch size. With batch 8, steps per epoch is `ceil(6291/8)=787`; 5,000 steps is about 6.35 epochs.

Training completed successfully in 27:58.98. Sampled max GPU use was 2,395 MiB. Loss fell from 14.044 (step 50) to 3.013 (250), 1.287 (2,000), 0.603 (4,000), and 0.405 (5,000). Logs: `outputs/act_task9_5k_monitor/train.log`; final checkpoint: `outputs/act_task9_5k/checkpoints/005000/pretrained_model`.

The final checkpoint was evaluated on fixed seeds 42000--42019 in four stable batches of five simulator environments. Episodes 0--19 (seeds 42000--42019) all recorded `success=false` and terminated at the 280-step LIBERO-Object limit; success rate was therefore 0/20 (0.0%) and mean episode length was 280. Mean `select_action` latency was 4.291 ms and p95 was 0.766 ms (the mean includes cold action-chunk generations). Evaluation took 41.78 s, sampled max GPU use was 3,450 MiB, and PyTorch peak allocation was 1,391,104,512 bytes. Videos and metrics are under `outputs/act_task9_5k_eval`.

Re-run training: `scripts/train_act_task9_5k.sh`. Re-run evaluation: `scripts/eval_act_task9_5k.sh`. To resume to 10,000 steps: `source scripts/activate_lerobot.sh && lerobot-train --resume=true --config_path=outputs/act_task9_5k/checkpoints/005000/pretrained_model/train_config.json --steps=10000 --save_freq=500`.
