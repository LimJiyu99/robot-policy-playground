# Official pretrained LIBERO policy candidates

조사일: 2026-07-16. 아래 용량은 Hugging Face Hub API가 반환한 저장소 내 모든 파일 크기의 합이며, 실제 cache 사용량은 중복 blob과 filesystem 단위 때문에 조금 다를 수 있다. 이번 단계에서는 config와 model card만 확인했고 가중치는 다운로드하지 않았다.

## 우선순위

1. **공식 재현 기준: `lerobot/pi05_libero_finetuned_v044`** — 현재 데이터 및 환경과 입력 key/shape가 정확히 맞고, LeRobot 공식 문서에 LIBERO-Object 99% 결과가 공개돼 있다.
2. **가벼운 호환성 점검 후보: `lerobot/smolvla_libero`** — 가장 작지만 현재 checkpoint config가 구형 6차원 state 및 `camera1/2/3` key를 요구한다. 현재 8차원 state와 `image/image2` 환경에 바로 평가하면 안 되며 adapter/호환 checkpoint 확인이 먼저다.
3. **연구 비교 후보: π0, VLA-JEPA** — 더 큰 GPU 메모리와 각 policy extra가 필요하다.

## 후보 표

| 모델 | Hub 용량 | 표시 파라미터 | 정책 종류 | 현재 LIBERO 입력 호환성 |
|---|---:|---:|---|---|
| [`lerobot/pi05_libero_finetuned_v044`](https://huggingface.co/lerobot/pi05_libero_finetuned_v044) | 6.96 GiB | 4B | π0.5, flow matching VLA | `state=(8,)`, `image/image2`, `action=(7,)`; 일치 |
| [`lerobot/smolvla_libero`](https://huggingface.co/lerobot/smolvla_libero) | 0.84 GiB | 0.5B | SmolVLA | config가 `state=(6,)`, `camera1/2/3`; 현재 환경과 불일치 |
| [`lerobot/pi0_libero_finetuned_v044`](https://huggingface.co/lerobot/pi0_libero_finetuned_v044) | 6.53 GiB | 4B | π0, flow matching VLA | `state=(8,)`, `image/image2`, `action=(7,)`; 일치 |
| [`lerobot/VLA-JEPA-LIBERO`](https://huggingface.co/lerobot/VLA-JEPA-LIBERO) | 5.74 GiB | 3B | VLA-JEPA + DiT action head | key/shape 일치, model card에 공식 평가 결과 없음 |
| [`lerobot/pi0fast-libero`](https://huggingface.co/lerobot/pi0fast-libero) | 7.20 GiB | 미표기 | π0-FAST | model card 정보가 부족하여 첫 평가에서는 제외 |

공식 LeRobot 계정에서는 LIBERO용 ACT 또는 Diffusion Policy checkpoint를 찾지 못했다. 따라서 이후 ACT/Diffusion 비교는 동일 데이터로 직접 학습한 checkpoint를 사용해야 하며, 이번 단계에서는 학습하지 않는다.

## 평가 명령

아래 명령은 준비용이며 **이번 단계에서 실행하지 않았다**. 실행하면 policy 가중치가 Hugging Face cache로 다운로드된다. 빠른 파이프라인 확인은 task 0, episode 1개로 시작하고, 최종 비교는 task당 10 episode로 늘린다.

### π0.5 — 권장 첫 baseline

필요 extra: `pip install -e "./lerobot[pi]"`

```bash
source scripts/activate_lerobot.sh
lerobot-eval \
  --policy.path=lerobot/pi05_libero_finetuned_v044 \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids='[0]' \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --output_dir=/workspace/jy/checkpoints/eval/pi05_libero_object_smoke
```

### SmolVLA — checkpoint adapter 확인 후 사용

필요 extra: `pip install -e "./lerobot[smolvla]"`

```bash
source scripts/activate_lerobot.sh
lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids='[0]' \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --output_dir=/workspace/jy/checkpoints/eval/smolvla_libero_object_smoke
```

이 명령의 구조는 맞지만 현재 checkpoint의 state/camera schema가 환경과 다르므로 그대로 실행하기 전에 rename/state adapter 또는 갱신된 checkpoint가 필요하다.

### π0

필요 extra: `pip install -e "./lerobot[pi]"`

```bash
source scripts/activate_lerobot.sh
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned_v044 \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids='[0]' \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --output_dir=/workspace/jy/checkpoints/eval/pi0_libero_object_smoke
```

### VLA-JEPA

필요 extra: `pip install -e "./lerobot[vla_jepa]"`

```bash
source scripts/activate_lerobot.sh
lerobot-eval \
  --policy.path=lerobot/VLA-JEPA-LIBERO \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids='[0]' \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --output_dir=/workspace/jy/checkpoints/eval/vla_jepa_libero_object_smoke
```

## 공정한 최종 평가 설정

최종 성공률 비교에서는 모든 정책에 동일한 task, 초기 상태, episode 수를 사용한다. 공식 권장 프로토콜은 suite별 10개 task × task당 10 episode이다. 단, 정책 checkpoint가 요구하는 `relative`/`absolute` control mode와 normalization schema는 학습 설정에 맞춰야 한다.
