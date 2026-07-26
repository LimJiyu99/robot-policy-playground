# π0.5 LIBERO Object 결과

공개용 요약이다. 서로 다른 evaluation protocol은 합산하거나 paired comparison으로 해석하지 않는다.

| 실험 | 평가 protocol | 성공 | 성공률 | 상태 |
|---|---|---:|---:|---|
| `pi05_base` → expert-only 80K | 10 tasks × 3, seed `42020–42119`의 task별 첫 3개 | 5/30 | 16.7% | 완료 |
| raw `pi05_libero_base` | 같은 30 episodes | 0/30 | 0.0% | 완료 |
| `pi05_libero_finetuned_v044` | task 0 × 3 | 3/3 | 100.0% | positive control |
| `pi05_libero_base` → expert-only 6K | 10 tasks × 3, 같은 30-episode protocol | 30/30 | 100.0% | 완료, preliminary |
| raw base vs 6K paired | task별 10, 총 200 rollouts 계획 | — | — | Paused / Incomplete |

## 읽는 방법

- 6K의 30/30은 task당 세 rollout 결과다. 100-seed 최종 결과, 일반화 또는 OOD 성능으로 확대 해석하지 않는다.
- 80K와 6K는 각각 `pi05_base`, `pi05_libero_base`에서 시작했으므로 학습 step만으로 비교할 수 없다.
- positive control은 task 0에서만 수행됐다. evaluation pipeline이 대체로 정상이라는 근거이지만 전체 task 보증은 아니다.

기계 판독용 수치는 [`results/pi05_libero_object_public_summary.json`](../results/pi05_libero_object_public_summary.json)에, 원본 provenance는 `docs/EXPERIMENT_LOG.md`에 있다.
