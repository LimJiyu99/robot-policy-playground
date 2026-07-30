# 재현 안내

모든 실험은 RTX 4090 24GB 한 장과 LeRobot 0.6.1에서 수행했다. checkpoint, dataset, 대용량 로그는 저장소에 포함하지 않는다.

```bash
conda env create -f environment.yml
conda activate lerobot-libero
export LEROBOT_DATASET_ROOT=/absolute/path/to/libero_object_image
export LIBERO_ASSETS_PATH=/absolute/path/to/libero_assets
source scripts/activate_lerobot.sh
```

## π0.5 6K adaptation

```bash
bash scripts/train_pi05_libero_base_expert_only_batch2_meanstd_6k.sh
python scripts/eval_pi05_libero_base_vs_expert_only_6k_eval100.py
python scripts/eval_pi05_libero_base_expert_only_6k_demo_videos.py
```

첫 명령은 `lerobot/pi05_libero_base`에서 expert-only 6K 학습을 한다. 둘째 명령은 raw base와 6K checkpoint를 task별 10개의 같은 seed로 평가하며, episode 결과를 즉시 JSON으로 갱신해 재개할 수 있다. 셋째 명령은 공개용 대표 rollout 다섯 개를 저장한다.

ACT, Diffusion Policy, SmolVLA checkpoint도 [`scripts/eval_policy_instrumented.py`](../scripts/eval_policy_instrumented.py)에 checkpoint 경로와 policy별 config를 전달해 동일한 LeRobot evaluator로 실행할 수 있다.

## ACT / Diffusion Policy task 9

두 정책은 같은 45개 task-9 demonstration으로 학습했다. 아래 명령은 기존 output을 덮어쓰지 않는다.

```bash
bash scripts/train_act_task9_chunk40_10k.sh
python scripts/eval_policy_instrumented.py --policy.path=outputs/act_task9_chunk40_10k/checkpoints/010000/pretrained_model --policy.n_action_steps=15 --env.type=libero --env.task=libero_object --env.task_ids='[9]' --env.control_mode=relative --env.camera_name_mapping='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}' --env.observation_height=256 --env.observation_width=256 --env.max_parallel_tasks=1 --eval.batch_size=5 --eval.n_episodes=100 --eval.recording=false --seed=42000 --output_dir=outputs/act_task9_reproduction_eval100

bash scripts/train_diffusion_task9_10k.sh
python scripts/eval_policy_instrumented.py --policy.path=outputs/diffusion_task9_h40_a15_10k/checkpoints/010000/pretrained_model --policy.n_action_steps=10 --env.type=libero --env.task=libero_object --env.task_ids='[9]' --env.control_mode=relative --env.camera_name_mapping='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}' --env.observation_height=256 --env.observation_width=256 --env.max_parallel_tasks=1 --eval.batch_size=5 --eval.n_episodes=100 --eval.recording=false --seed=42020 --output_dir=outputs/diffusion_task9_reproduction_eval100
```

## SmolVLA multi-task

SmolVLA는 모든 454 episodes와 task language를 사용한다. task별 10개 seed의 재개형 평가는 instruction, seed, success를 `instrumentation.json`에 즉시 기록한다.

```bash
bash scripts/train_smolvla_multitask_batch4_20k.sh
python scripts/eval_smolvla_multitask_resume.py --checkpoint outputs/smolvla_libero_object_multitask_batch4_20k/checkpoints/020000/pretrained_model --output-dir outputs/smolvla_multitask_reproduction_eval100 --episodes-per-task 10 --n-action-steps 10
```

## 제외하는 산출물

checkpoint/model weight, optimizer state, dataset, Hugging Face cache, 평가 원본 mp4, 대용량 log 및 `.incomplete` 파일은 Git에서 제외한다. README에 연결한 `assets/videos/`의 짧은 대표 영상만 예외로 포함한다.
