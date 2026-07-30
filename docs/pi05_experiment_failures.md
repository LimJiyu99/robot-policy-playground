# π0.5 실패 실험 정리

아래 수치는 완료된 `instrumentation.json`과 각 checkpoint의 `train_config.json`에서 확인했다. 이 문서는 재현·원인 분석에 필요한 설정, 결과 JSON, 학습 로그만 남기는 용도이며 checkpoint weight와 optimizer는 Git 추적 대상이 아니다.

| 실험 | 정규화 | batch / steps | 학습 범위 | 평가 결과 | 확인된 실패 원인 |
| --- | --- | --- | --- | --- | --- |
| Expert-only MIN_MAX | STATE/ACTION `MIN_MAX` | 2 / 10K | Action Expert만; vision encoder 고정 | 2.5K 3/30 (10.0%), 5K 3/30 (10.0%), 7.5K 0/30 (0.0%), 10K 1/30 (3.3%) | 동일 30-seed 평가에서 낮은 성공률과 학습 step 증가에 따른 개선 부재 |
| Expert-only MEAN_STD | STATE/ACTION `MEAN_STD` | 2 / 5K | Action Expert만; vision encoder 고정 | 2.5K 0/30 (0.0%), 5K 0/30 (0.0%) | MIN_MAX 대비 정규화 변경만으로 개선되지 않음 |
| LoRA + Action Expert | STATE/ACTION `MEAN_STD` | 1 / 10K | language attention projection LoRA(72개)와 Action Expert; vision encoder 고정 | 5K 0/30 (0.0%), 10K 1/30 (3.3%) | adapter는 저장·PEFT 로딩이 확인됐지만 성공률 개선 없음 |
| 초기 LoRA smoke | 해당 없음 | 1 / 20 | LoRA 주입 시도 | 평가 없음 | 잘못된 target module로 `No modules were targeted for adaptation` 오류 |
| VLM 일부 학습 smoke | STATE/ACTION `MEAN_STD` | 1 / 20 | Action Expert와 vision 외 VLM 학습 | 평가 없음 | optimizer state 초기화 중 RTX 4090 24GB CUDA OOM |

## 실패 경로와 성공 경로의 구분

- `lerobot/pi05_base` expert-only 80K는 5/30 (16.7%)에 머물렀고, MIN_MAX·MEAN_STD·LoRA 변형도 최대 3/30 이하로 이 경로의 저성능을 해소하지 못했다.
- 이는 `lerobot/pi05_libero_base` 초기화의 6K expert-only adaptation과 동일한 실험이 아니다. 후자는 raw 0/100 대 adaptation 100/100의 완료된 paired 결과를 보였으며, 초기 checkpoint가 다르므로 위 실패 경로와 학습 step만으로 직접 비교할 수 없다.

## 보존하는 근거 파일

- 결과: `outputs/pi05_batch2_checkpoint_ablation_action_steps_10/instrumentation.json`, `outputs/pi05_batch2_meanstd_checkpoint_ablation_action_steps_10/instrumentation.json`, `outputs/pi05_lora_checkpoint_ablation_action_steps_10/instrumentation.json`
- 설정: 각 checkpoint의 `pretrained_model/{train_config.json,config.json,policy_preprocessor.json,policy_postprocessor.json}`
- 로그: `outputs/pi05_libero_object_multitask_batch2_meanstd_5k/{train.log,gpu_usage.csv,training_time.txt}`, `outputs/.runlogs/pi05_lora_batch1_meanstd_smoke_then_10k/`, `outputs/.runlogs/pi05_lora_batch1_meanstd_smoke_then_5k_failed_20260725_025834/`, `outputs/.runlogs/pi05_batch1_vlm_unfrozen_meanstd_smoke_then_5k/`

## 삭제 범위 메모

- MIN_MAX expert-only checkpoint의 대형 model/optimizer weight는 이미 존재하지 않는다. config, processor, training-state 메타데이터만 남아 있다.
- LoRA의 `adapter_model.safetensors`는 checkpoint당 약 1.52 GiB로 소형 adapter가 아니며, Action Expert의 `modules_to_save`를 포함한다. adapter 로딩 검증은 완료됐으므로 삭제 승인 시 두 checkpoint의 adapter weight와 optimizer를 삭제할 수 있다. JSON/config/processor/log는 유지한다.
- `outputs/pi05_libero_object_multitask_smoke50`은 50-step checkpoint가 완전하게 존재하지만 완료 실패 여부를 확인할 근거가 없어 삭제 후보에서 제외한다.
