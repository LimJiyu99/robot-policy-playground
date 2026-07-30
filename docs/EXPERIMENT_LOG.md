# π0.5 LIBERO Object 실험 로그

이 문서는 완료된 local JSON과 checkpoint config를 기준으로 작성했다. checkpoint weight, optimizer state, raw rollout 및 영상은 공개 저장소에 포함하지 않는다.

## 실험 범위

- 데이터: `lerobot/libero_object_image`, LIBERO Object 10 tasks, 454 episodes
- 하드웨어: single RTX 4090 24GB
- 공통 평가: LIBERO relative control, 256×256 agentview/wrist images, batch size 1, `n_action_steps=10`, task별 독립 process
- 30-episode 비교 seed: task `t`, episode `e`에 `42020 + 10*t + e` (`e=0,1,2`)

## 완료된 실험

| ID | 초기 checkpoint | 학습 범위 | 평가 | 검증된 결과 | 근거 |
|---|---|---|---|---|---|
| P1 | `lerobot/pi05_base` | expert-only, batch 2, 40K 후 80K까지 resume | 10×3 | 5/30 (16.7%) | `outputs/pi05_libero_base_vs_expert_only_80k_eval30/instrumentation.json` |
| P2 | `lerobot/pi05_libero_base` | 추가 학습 없음 | 10×3 | 0/30 (0.0%) | 같은 JSON의 `pi05_libero_base` 항목 |
| P3 | `lerobot/pi05_libero_finetuned_v044` | 추가 학습 없음 | task 0×3 | 3/3 (100.0%) | `outputs/pi05_finetuned_v044_positive_control/eval_info.json` |
| P4 | `lerobot/pi05_libero_base` | expert-only, batch 2, 6K | 10×3 | 30/30 (100.0%) | `outputs/pi05_libero_base_expert_only_6k/eval/instrumentation.json` |
| P5 | raw `pi05_libero_base` vs P4 | 같은 task별 10 seed, policy당 100 episodes | 10×10 / policy | raw 0/100, 6K 100/100 | `outputs/pi05_libero_base_vs_expert_only_6k_eval100/instrumentation.json` |

P1의 task별 성공(/3)은 `0,0,0,1,1,0,1,1,1,0`이다. P4는 모든 task가 3/3이지만 sample 수가 작으므로 preliminary result로만 기록한다.

## Checkpoint와 설정 관계

- P1은 `pi05_base`에서 시작한 40K run의 checkpoint를 optimizer/scheduler state와 함께 80K까지 재개했다. `train_expert_only=true`, `freeze_vision_encoder=true`, STATE/ACTION `MEAN_STD`, `chunk_size=50`, `n_action_steps=10`, batch 2, seed 42를 사용했다.
- P4는 `pi05_libero_base`를 초기화로 사용했고 위 expert-only, batch, normalization, action-horizon 설정을 유지했다. source feature contract에 맞추기 위해 dataset `observation.images.wrist_image`를 `observation.images.image2`로 rename했다.
- 두 run의 initialization이 다르므로 P1과 P4는 scratch 학습의 공정한 step-to-step 비교가 아니다. `pi05_libero_base`는 이미 LIBERO 학습 이력이 있는 checkpoint다.

## Positive control과 성공 판정

P3는 task 0에서 3/3을 기록했다. 이는 현재 LeRobot/LIBERO environment, relative control, camera mapping 및 성공 판정 경로가 적어도 이 task의 세 rollout에서 동작했다는 positive control이다. 전체 10 tasks의 end-to-end 검증은 아니다.

custom `scripts/eval_act_smoke_instrumented.py`는 공식 `eval_policy`를 호출하고, first done step까지 success가 한 번이라도 true인 episode를 성공으로 집계한다. 공식 `lerobot-eval`과 동일한 집계 기준이다.

## 실패 결과와 해석

**Observed**

- P1의 80K expert-only run은 5/30에 머물렀다.
- 같은 30 seed에서 raw `pi05_libero_base`는 0/30이었다.

**Verified**

- P3 positive control 3/3과 공식 evaluator와의 동일한 success aggregation을 확인했다.
- P4 checkpoint config에는 `image/image2` visual feature와 `wrist_image → image2` rename map이 기록돼 있다.

**Hypotheses**

- P1의 낮은 성공률은 small batch, Action Expert만 학습한 범위, normalization, 또는 base checkpoint의 converted-weight adaptation 특성과 관련될 수 있다.
- P4의 빠른 개선은 LIBERO-specific initialization 및 feature/processor contract 정합성과 관련될 수 있다.

**Not verified**

- 어느 가설이 주원인인지는 통제 실험으로 분리하지 않았다.
- 6K의 동일 task·seed 100-episode 재현성은 확인했지만, 다른 환경 seed와 OOD object/task 일반화는 검증하지 않았다.

## Completed paired evaluation

`outputs/pi05_libero_base_vs_expert_only_6k_eval100/instrumentation.json`은 raw base와 6K adaptation을 task별 10 seed로 paired 평가한 총 200-rollout 결과다. JSON status는 `completed`이며 raw base는 0/100, 6K adaptation은 100/100을 기록했다. 동일 seed 기준 6K만 성공은 100, raw base만 성공은 0이었다.
