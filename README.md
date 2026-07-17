# LeRobot + LIBERO benchmark environment

ACT, Diffusion Policy, SmolVLA를 동일한 LIBERO 조건에서 비교하기 위한 개발 환경이다. 현재 단계에는 환경과 시뮬레이터 smoke test만 포함하며 학습 데이터, pretrained 모델, 학습 결과는 포함하지 않는다.

## ACT task 9: 5K/10K 평가 비교

동일 조건(LIBERO `libero_object` task 9, 20 episodes, seeds 42000--42019, batch size 5)에서의 결과이다.

| Checkpoint | 성공률 | Runtime | Peak GPU memory | Mean / P95 latency |
|---|---:|---:|---:|---:|
| 5K (`005000`) | 0/20 (0.0%) | 41.78 s | 3,450 MiB | 4.291 / 0.766 ms |
| 10K (`010000`) | 0/20 (0.0%) | 38.41 s | 3,442 MiB | 4.091 / 0.731 ms |

상세 측정과 재실행 방법은 [ACT_TASK9_10K_REPORT.md](ACT_TASK9_10K_REPORT.md)를 참고한다.

## ACT 10K action-steps ablation

`chunk_size=100` 고정, task 9·20 episodes·seeds 42000--42019 조건의 결과이다.

| `n_action_steps` | 성공률 | Runtime | Peak GPU memory | Mean / P95 latency |
|---:|---:|---:|---:|---:|
| 100 | 0/20 (0.0%) | 38.41 s | 3,442 MiB | 4.091 / 0.731 ms |
| 20 | 10/20 (50.0%) | 51.34 s | 3,540 MiB | 4.496 / 1.579 ms |
| 10 | 3/20 (15.0%) | 43.71 s | 3,540 MiB | 4.911 / 6.566 ms |
| 5 | 2/20 (10.0%) | 42.37 s | 3,540 MiB | 5.380 / 7.486 ms |

상세 결과와 영상 경로는 [ACT_ACTION_STEPS_ABLATION.md](ACT_ACTION_STEPS_ABLATION.md)를 참고한다.

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
