# Setup report

작성일: 2026-07-16 (Asia/Seoul)

## 결과

환경 설치와 LIBERO-Object task 0 smoke test가 완료됐다. 학습, LIBERO 데모 데이터 전체 다운로드, pretrained VLA 다운로드는 수행하지 않았다.

| 항목 | 실제 결과 |
|---|---|
| 프로젝트 | `/workspace/jy/projects/lerobot-libero-benchmark` |
| 최초 가용 공간 | 214 GB (`/dev/nvme0n1p4`, 50% 사용) |
| 기존 Conda | 22.9.0, `/home/jy/anaconda3` (변경하지 않음) |
| 별도 Miniforge | 26.3.2, `/workspace/jy/miniforge3` |
| 환경 prefix | `/workspace/jy/conda/envs/lerobot_libero` |
| Python | 3.12.13 |
| ffmpeg | 7.1.1 (conda-forge) |
| LeRobot | 0.6.1, editable source |
| LeRobot commit | `3f2179f3b69708b6ad009b2e7685dd9d05269ee1` |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| hf-libero | 0.1.4 |
| MuJoCo | 3.8.1 |
| transformers | 5.5.4 |
| datasets | 4.8.5 |
| NumPy | 2.2.6 |
| GPU | NVIDIA GeForce RTX 4090 |
| VRAM | 24,564 MiB |
| NVIDIA driver | 575.64.03 |

기존 Conda 22.9.0은 Python 3.12 출시 이전 버전이고 현재 공식 LeRobot도 Miniforge/Python >=3.12를 권장한다. 기존 Anaconda를 업데이트하지 않고 별도 Miniforge를 설치했다. 설치 파일 SHA-256은 공식 릴리스 값 `42260ffe3830fb953d5eee1bbb32229ff06aa7c3833c1ed7a9a0420a95685d94`와 일치했다.

## extras 확인

clone 직후 최신 `lerobot/pyproject.toml`을 직접 확인했다.

- LIBERO: `libero` extra 존재, 공식 최소 설치는 `.[libero]`.
- Diffusion Policy: `diffusion` extra 존재. 이번에는 설치하지 않음.
- SmolVLA: `smolvla` extra 존재. 이번에는 설치하지 않음.
- ACT: 별도 extra가 없고 기본 LeRobot 정책 코드에 포함됨.

이번 단계는 `.[libero]`만 설치했다. 정책별 추가 패키지는 실제 학습 단계에서 동일 base environment를 기준으로 별도 검증하는 편이 충돌 원인 추적에 유리하다.

## 검증 결과

| 검증 | 결과 |
|---|---|
| `python --version` | 성공, 3.12.13 |
| PyTorch/CUDA | 성공, 2.11.0+cu128 / `True` |
| `torch.cuda.get_device_name(0)` | 성공, RTX 4090 |
| GPU VRAM | 성공, 24,564 MiB |
| `import lerobot` | 성공 |
| `import libero` | 성공 |
| `import mujoco` | 성공 |
| `lerobot-info` | 성공 |
| `lerobot-train --help` | 성공 (3,852 lines) |
| `lerobot-eval --help` | 성공 (3,117 lines) |
| `python -m pip check` | 성공 (`PYTHONNOUSERSITE=1`) |
| LIBERO-Object task 0 생성 | 성공 |
| reset | 성공 |
| zero action 1 step | 성공, action shape `(7,)`, reward `0.0`, done `False` |
| offscreen EGL RGB | 성공, 256x256 RGB PNG |
| close | 성공 |

결과 이미지는 `outputs/smoke_test.png`에 있다.

## 실패와 해결

1. PyPI 기본 resolver가 PyTorch 2.11의 CUDA 13.0 wheel을 선택했다. 설치를 즉시 중단하고 LeRobot `pyproject.toml`의 공식 Linux source 설정과 같은 PyTorch CUDA 12.8 index에서 `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`을 먼저 설치했다.
2. 첫 `pip check`는 현재 셸의 ROS/user-site 패키지 `generate-parameter-library-py`가 노출돼 `typeguard` 경고를 냈다. 환경 자체 결함은 아니었으며 `PYTHONPATH` 해제와 `PYTHONNOUSERSITE=1`에서 재검증해 `No broken requirements found`를 확인했다.
3. 첫 LIBERO import는 `~/.libero/config.yaml`을 만들기 위한 대화형 프롬프트에서 EOF로 중단됐다. 홈 설정을 만들지 않고 프로젝트의 `.libero/config.yaml`과 `LIBERO_CONFIG_PATH`를 사용하도록 했다.
4. `hf-libero 0.1.4`가 config의 assets 경로를 초기 probe에서 무시하고 기본 `~/.cache/libero/assets`로 다운로드했다. 중단 시점까지 약 397 MB의 시뮬레이터 assets가 이 위치에 생성됐다. 사용자의 기존 홈 파일은 삭제하거나 이동하지 않았고, 생성된 assets를 `/workspace/jy/datasets/libero_assets`로 복사해 smoke test가 workspace 복사본만 사용하도록 고정했다. 이는 데모 데이터셋이나 pretrained 모델 다운로드가 아니다.
5. raw robosuite camera frame은 표시 방향이 뒤집혀 있어 LeRobot `LiberoEnv.render()`와 동일한 180도 보정을 적용했다.

`~/.bashrc`에는 점검 당시 `HF_HOME`, `TORCH_HOME`, `PIP_CACHE_DIR`, `MUJOCO_GL` 등록이 없었고 이번 작업에서도 수정하지 않았다. sudo, apt, 드라이버 변경, 기존 Conda 환경 삭제는 수행하지 않았다.
