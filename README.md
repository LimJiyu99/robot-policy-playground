# LeRobot + LIBERO benchmark environment

ACT, Diffusion Policy, SmolVLA를 동일한 LIBERO 조건에서 비교하기 위한 개발 환경이다. 현재 단계에는 환경과 시뮬레이터 smoke test만 포함하며 학습 데이터, pretrained 모델, 학습 결과는 포함하지 않는다.

## ACT LIBERO Object task 9 최신 결과

동일한 LIBERO Object task 9 평가 조건(100 episodes, seeds 42000--42099, 두 카메라, batch size 5)의 최신 비교다.

| Checkpoint | `chunk_size` | `n_action_steps` | 성공률 |
|---|---:|---:|---:|
| ACT 10K | 100 | 20 | 62/100 (62%) |
| ACT 10K | 40 | 20 | 84/100 (84%) |
| ACT 10K | 40 | 15 | **94/100 (94%)** |

chunk size를 100에서 40으로 줄여 성공률이 62%에서 84%로 상승했고, action steps를 20에서 15로 줄여 94%까지 상승했다. 최종 ACT 설정은 **10K checkpoint, `chunk_size=40`, `n_action_steps=15`**다.

| Chunk 40 action-step ablation (20 episodes) | 성공률 |
|---|---:|
| 10 | 19/20 (95%) |
| 15 | 19/20 (95%) |
| 25 | 10/20 (50%) |

20-step 동률은 20에 가까운 15를 선택했다. Temporal ensemble(10K, 1 step)은 0/20이었고 action scaling 0.8은 60/100으로 baseline 62/100을 넘지 못해 유의미한 개선이 없었다.

최종 설정 영상/JSON: `outputs/act_task9_chunk40_10k_action_steps_15_eval_ablation100/eval/videos/libero_object_9/`, `outputs/act_task9_chunk40_10k_action_steps_15_eval_ablation100/instrumentation.json`.

## Diffusion Policy 및 동일-seed 최종 비교

Diffusion Policy 10K (`horizon=40`)의 20-episode action-step ablation 결과다.

| `n_action_steps` | 성공률 |
|---:|---:|
| 10 | 19/20 (95%) |
| 15 | 17/20 (85%) |
| 20 | 15/20 (75%) |
| 25 | 15/20 (75%) |

`n_action_steps=10`은 새 seed 42020--42119에서 92/100 (92%)을 기록했다. 같은 100개 초기조건의 최종 비교는 다음과 같다.

| Policy | 설정 | 성공률 | 상대 episode 결과 |
|---|---|---:|---:|
| ACT 10K | `chunk_size=40`, `n_action_steps=15` | **94/100 (94%)** | ACT 성공 / Diffusion 실패: 7 |
| Diffusion Policy 10K | `horizon=40`, `n_action_steps=10` | 92/100 (92%) | ACT 실패 / Diffusion 성공: 5 |

ACT는 chunk size와 replanning interval 조정으로 62%에서 94%까지 향상됐다. Diffusion Policy는 더 짧은 실행 구간에서 92%를 달성했다. 동일 초기조건에서 ACT가 2회 더 성공했지만 두 정책 모두 90% 이상의 유사한 성능을 보였고, 단일 객체 조작에서 Diffusion Policy가 ACT보다 반드시 우수하지는 않았다. 이후 실험은 multi-task 및 language-conditioned VLA로 확장할 예정이다.

## SmolVLA Multi-Task VLA

SmolVLA는 LIBERO Object의 10개 task와 language instruction을 함께 입력으로 사용하는 multi-task policy다. `train_config.json`과 checkpoint config에서 확인한 기본 설정은 `lerobot/smolvla_base` 초기화, `policy.type=smolvla`, `chunk_size=40`, 학습 시 `n_action_steps=15`, `freeze_vision_encoder=true`, VLM `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`다. 데이터셋은 `lerobot/libero_object_image`(454 episodes)이며, 평가는 checkpoint의 action chunk 설정을 유지한 채 `n_action_steps=10`으로 수행했다.

### Checkpoint ablation (3 episodes/task, 30 episodes)

아래 batch-1 checkpoint ablation은 task별 seed `42000 + 100 × task + {0,1,2}`를 사용한 기록된 30-episode 결과다. 100-episode final 결과와는 별도 프로토콜이므로 합산하거나 paired 결과로 해석하지 않는다.

| Training steps | 5K | 10K | 15K | 20K | 30K | 40K | 50K | 60K | 70K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Batch size 1 | 0/30 (0.0%) | 13/30 (43.3%) | 16/30 (53.3%) | 18/30 (60.0%) | 20/30 (66.7%) | 19/30 (63.3%) | 13/30 (43.3%) | 11/30 (36.7%) | 11/30 (36.7%) |

batch size 4의 동일 30-episode ablation은 다음과 같다. 각 행의 `T0`–`T9`는 task별 성공 횟수(/3)다.

| Training steps | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Batch 4, 5K | 1 | 2 | 1 | 1 | 3 | 0 | 3 | 2 | 1 | 0 | 14/30 (46.7%) |
| Batch 4, 10K | 3 | 1 | 1 | 2 | 1 | 3 | 2 | 1 | 3 | 0 | 17/30 (56.7%) |
| Batch 4, 15K | 1 | 2 | 2 | 2 | 2 | 2 | 3 | 2 | 3 | 1 | 20/30 (66.7%) |
| Batch 4, 20K | 2 | 2 | 1 | 3 | 3 | 2 | 3 | 2 | 3 | 3 | **24/30 (80.0%)** |
| Batch 4, 25K | 2 | 2 | 2 | 2 | 3 | 2 | 3 | 1 | 3 | 1 | 21/30 (70.0%) |

같은 30개 seed로 기록된 batch-1 checkpoint와 비교하면 batch4 5K/10K는 각각 14 vs 18, 17 vs 19로 낮았고, batch4 15K는 batch1 60K의 기록된 11보다 20으로 높았다. 이 비교의 batch4-only / batch1-only 성공은 각각 5/9, 7/9, 11/2 episodes다. 60K의 과거 요약 수치가 아니라 원본 instrumentation의 seed-level 결과를 사용했다.

### Final 100-episode evaluation (10 episodes/task)

최종 비교는 task별 10개 seed, 총 100 episodes에서 별도로 수행했다. `T0`–`T9`는 task별 성공 횟수(/10)이며, 아래 행끼리만 같은 100개 초기조건을 공유한다.

| Policy / checkpoint | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Batch 1, 70K | 7 | 4 | 9 | 6 | 6 | 3 | 9 | 5 | 7 | 7 | 63/100 (63.0%) |
| Batch 4, 15K | 8 | 7 | 8 | 8 | 8 | 9 | 8 | 8 | 10 | 9 | 83/100 (83.0%) |
| Batch 4, 20K | 9 | 9 | 8 | 10 | 10 | 4 | 9 | 9 | 10 | 9 | **87/100 (87.0%)** |

Batch4 15K와 20K의 동일 100 seed 비교에서는 20K만 성공 14, 15K만 성공 10, 둘 다 성공 73, 둘 다 실패 3 episodes였다. 따라서 최종 최고 평균 성공률 checkpoint는 **batch4 20K**다.

### Task 5 controlled analysis

Task 5(`pick up the tomato sauce and place it in the basket`)는 전체 final 표와 분리한 controlled evaluation이다. 같은 50개 seed를 batch-1 single-environment protocol로 두 checkpoint에 적용했다. 15K는 34/50 (68.0%), 20K는 25/50 (50.0%)였고, 15K만 성공 9, 20K만 성공 0, 둘 다 성공 25, 둘 다 실패 16 episodes였다. 첫 10개 기존 seed와 새 40개 seed 모두에서 20K가 뒤져 task-specific regression이 재현됐다.

Tomato sauce can의 원통형 물체 형상이나 다른 물체에 대한 편향, multi-task negative transfer가 가능한 가설이지만 원인은 확정되지 않았다. 저장된 영상을 이용한 failure-mode 분석이 추가로 필요하다.

SmolVLA는 inference 시 `torch.normal`로 noise를 샘플링하며 checkpoint의 `inference_seed`는 `null`이다. 환경 seed가 같아도 batch size, vectorized environment 실행 방식, process별 RNG 호출 순서가 다르면 rollout 결과가 달라질 수 있다. 따라서 batch-3 전체 평가의 15K Task 5 결과와 batch-1 controlled evaluation의 15K 결과를 직접 paired 비교하지 않았다. 반면 controlled protocol 내부의 15K vs 20K 비교는 유효하다.

**핵심 결론:** batch size를 1에서 4로 늘리면서 multi-task 성능이 크게 향상됐고, batch4 20K가 최종 100-episode 평가에서 87.0%로 최고 성능을 기록했다. 25K의 소규모 ablation 성능은 20K보다 하락해 20K를 최종 checkpoint로 선택했다. 전체 평균은 개선됐지만 Task 5에서는 재현 가능한 regression이 관찰됐으며, negative transfer 또는 object geometry bias 가능성은 후속 영상 분석으로 검증할 과제다.

## 빠른 시작

```bash
cd /workspace/jy/projects/lerobot-libero-benchmark
source scripts/activate_lerobot.sh
python scripts/smoke_test_libero.py
```

성공하면 `outputs/smoke_test.png`가 생성되고 마지막 로그가 `close=ok`로 끝난다.

활성화 스크립트는 explicit-prefix Conda 환경을 활성화하고 다음을 설정한다.

- `HF_HOME=/workspace/jy/cache/huggingface`
- `TORCH_HOME=/workspace/jy/cache/torch`
- `PIP_CACHE_DIR=/workspace/jy/cache/pip`
- `MUJOCO_GL=egl`
- `LIBERO_CONFIG_PATH=<project>/.libero`
- `LEROBOT_DATA_HOME=/workspace/jy/datasets`
- `LEROBOT_OUTPUT_HOME=/workspace/jy/checkpoints`

ROS Python 3.10 패키지가 Python 3.12 환경으로 섞이지 않도록 `PYTHONPATH`를 해제하고 `PYTHONNOUSERSITE=1`도 설정한다. 전역 `~/.bashrc`는 수정하지 않았다.

## 새로 재현하기

공식 LeRobot 소스가 `lerobot/`에 있어야 한다.

```bash
git clone https://github.com/huggingface/lerobot.git lerobot
CONDA_PKGS_DIRS=/workspace/jy/conda/pkgs \
  /workspace/jy/miniforge3/bin/conda env create \
  -p /workspace/jy/conda/envs/lerobot_libero \
  -f environment.yml
```

LIBERO simulator assets는 `/workspace/jy/datasets/libero_assets`에 둔다. 데모 데이터셋과는 별개이며 smoke test에 필요하다. 현재 `hf-libero 0.1.4`의 기본 downloader는 assets 경로를 `~/.cache/libero/assets`로 강제하는 문제가 있으므로, 다운로드가 필요하면 아래처럼 목적지를 명시한다.

```bash
source scripts/activate_lerobot.sh
python -c "from libero.libero.utils.download_utils import download_assets_from_huggingface; download_assets_from_huggingface('/workspace/jy/datasets/libero_assets')"
```

## 점검 명령

```bash
source scripts/activate_lerobot.sh
python --version
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
lerobot-info
lerobot-train --help
lerobot-eval --help
python -m pip check
```

정확한 설치 버전과 해결 내역은 [SETUP_REPORT.md](SETUP_REPORT.md)를 참고한다.

## 1단계: LIBERO 데이터 구조 확인

이번 단계에서는 공식 [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero) v3.0 변환본을 선택했다. 전체 저장소는 약 1.80 GiB이며, 비교 후보였던 `HuggingFaceVLA/libero`와 `physical-intelligence/libero`는 각각 약 32.53 GiB와 32.54 GiB이다. 최신 변환본은 이미지가 MP4로 분리돼 suite별 선택 다운로드가 가능하고 현재 LeRobot key 규칙과 바로 맞기 때문에 구조 조사에 적합하다.

`libero_object`는 전체 dataset의 task index 20–29이다. 선택된 subset은 다음과 같다.

| 항목 | 값 |
|---|---:|
| task | 10 |
| episode | 454 |
| frame | 66,984 |
| fps | 10 |
| state | `observation.state`, float32, `(8,)` |
| action | `action`, float32, `(7,)` |
| main camera | `observation.images.image`, 256×256 RGB |
| wrist camera | `observation.images.image2`, 256×256 RGB |

로컬 위치는 `/workspace/jy/datasets/lerobot/libero`이다. 전체 state/action parquet 약 19.3 MiB와 LIBERO-Object가 참조하는 두 카메라 MP4 20개 약 559.11 MiB만 받았다. 다른 suite 영상과 모델 가중치는 받지 않았다.

LeRobot v3 dataset의 역할은 다음처럼 나뉜다.

- `meta/info.json`: fps, 전체 episode/task 수, feature key와 shape.
- `meta/tasks.parquet`: `task_index`와 language instruction 대응표.
- `meta/episodes/...parquet`: 각 episode의 frame 범위와 video timestamp 범위.
- `data/...parquet`: frame별 state, action, timestamp, episode/task index.
- `videos/<camera-key>/...mp4`: 카메라별 RGB frame 묶음. parquet timestamp와 동기화된다.

검사와 시각화:

```bash
source scripts/activate_lerobot.sh
python scripts/inspect_libero_dataset.py
```

스크립트는 feature keys, episode/task 수, fps, state/action/image shape, instruction 예시를 출력하고 첫 episode의 두 카메라 및 action/state 일부를 `outputs/dataset_sample.png`에 저장한다. pretrained 후보와 아직 실행하지 않은 평가 명령은 [PRETRAINED_MODELS.md](PRETRAINED_MODELS.md)에 정리돼 있다.
