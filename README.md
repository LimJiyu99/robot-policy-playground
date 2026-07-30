# LeRobot LIBERO Benchmark

LIBERO Object에서 ACT, Diffusion Policy, SmolVLA, π0.5를 비교한 재현 가능한 로봇 imitation-learning 실험 모음이다. 모든 결과는 단일 RTX 4090 24GB에서 수행했다.

## 핵심 결과

| Policy | 설정 | 평가 | 성공률 |
| --- | --- | ---: | ---: |
| ACT | chunk 40, action steps 15, 10K | task 9, 동일 100 seeds | 94/100 (94%) |
| Diffusion Policy | horizon 40, action steps 10, 10K | task 9, 동일 100 seeds | 92/100 (92%) |
| SmolVLA | batch 4, 20K, action steps 10 | LIBERO Object 10 tasks | 87/100 (87%) |
| π0.5 | `pi05_libero_base` → expert-only 6K | LIBERO Object 10 tasks, 동일 100 seeds | 100/100 (100%) |

π0.5의 100/100은 raw `pi05_libero_base`와 동일한 100개 seed에서 얻은 결과이며, OOD 또는 일반화 성능을 뜻하지 않는다. 실패한 π0.5 경로(MIN_MAX, MEAN_STD, LoRA, expert-only 80K)는 [실패 분석](docs/pi05_experiment_failures.md)에 별도로 기록했다.

## 대표 rollout

| Policy | 성공 rollout |
| --- | --- |
| ACT | [task 9 MP4](assets/videos/act_task9_success.mp4) |
| Diffusion Policy | [task 9 MP4](assets/videos/diffusion_task9_success.mp4) |
| SmolVLA batch4 20K | [task 5 MP4](assets/videos/smolvla_task5_success.mp4) |
| π0.5 6K | [task 0 MP4](assets/videos/pi05_task0_success.mp4) |

## 빠른 시작

```bash
git clone https://github.com/LimJiyu99/robot-policy-playground.git
cd robot-policy-playground
conda env create -f environment.yml
source scripts/activate_lerobot.sh
```

LIBERO 데이터셋 위치를 지정한다.

```bash
export LEROBOT_DATASET_ROOT=/path/to/libero_object_image
export LIBERO_ASSETS_PATH=/path/to/libero_assets
python scripts/smoke_test_libero.py
```

π0.5 6K 학습과 대표 영상 재생성 예시는 다음과 같다. checkpoint와 output은 Git에 포함되지 않는다.

```bash
bash scripts/train_pi05_libero_base_expert_only_batch2_meanstd_6k.sh
python scripts/eval_pi05_libero_base_expert_only_6k_demo_videos.py
```

정확한 설정·명령은 [재현 가이드](docs/REPRODUCIBILITY.md), 결과표는 [RESULTS](docs/RESULTS.md), 기계 판독용 요약은 [results JSON](results/pi05_libero_object_public_summary.json)에 있다.

## 범위와 한계

- LIBERO Object는 10개 단일 객체 조작 task, 454 episodes를 사용한다.
- checkpoint, dataset, raw output, 대량 로그는 Git에 포함하지 않는다.
- 서로 다른 batch size·RNG·vectorized environment protocol에서 얻은 결과는 paired comparison으로 합치지 않는다.
