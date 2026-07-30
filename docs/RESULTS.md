# 결과 인덱스

각 수치는 Git에서 제외한 원본 `outputs/**/instrumentation.json`에서 검증했다. 서로 다른 task·seed·batch/RNG protocol의 결과를 합산하거나 paired comparison으로 해석하지 않는다.

| Policy | 최종 선택 설정 | 평가 범위 | 결과 | 상세 |
| --- | --- | --- | ---: | --- |
| ACT | 10K, chunk 40, action steps 15 | task 9, 100 episodes | 94/100 (94%) | [README](../README.md#act-libero-object-task-9-최신-결과) · [action-step 기록](../ACT_REFINED_ACTION_STEPS_EVALUATION.md) |
| Diffusion Policy | 10K, horizon 40, action steps 10 | task 9, 100 episodes | 92/100 (92%) | [README](../README.md#diffusion-policy-및-동일-seed-최종-비교) |
| SmolVLA | batch4 20K, action steps 10 | 10 tasks, 100 episodes | 87/100 (87%) | [README](../README.md#smolvla-멀티태스크-vla) |
| π0.5 | `pi05_libero_base` → expert-only 6K | 10 tasks, same 100 seeds | 100/100 (100%) | [README](../README.md#π05-libero-object-fine-tuning-검증) |

π0.5의 100/100은 동일 task·seed 집합의 결과일 뿐 OOD/generalization 성능을 뜻하지 않는다. 실패한 MIN_MAX, MEAN_STD, LoRA와 `pi05_base` expert-only 경로는 [실패 분석](pi05_experiment_failures.md)에 별도 보존했다. 기계 판독용 π0.5 수치는 [`results/pi05_libero_object_public_summary.json`](../results/pi05_libero_object_public_summary.json)에 있다.
