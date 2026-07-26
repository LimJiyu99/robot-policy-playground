# π0.5 LIBERO Object 재현 안내

## 전제

- GPU: RTX 4090 24GB 한 장
- dataset root: `/workspace/jy/datasets/lerobot/libero_object_image`
- 실제 평가 결과는 output JSON을 읽어야 하며 checkpoint와 dataset은 저장소에 포함하지 않는다.

## 6K adaptation 후 30-episode 평가

아래 workflow는 `lerobot/pi05_libero_base`에서 expert-only 6K 학습을 수행한 뒤, final checkpoint가 성공적으로 저장된 경우에만 10 tasks × 3 평가를 시작한다. WandB는 비활성화돼 있으며 stdin을 사용하지 않는다.

```bash
source scripts/activate_lerobot.sh && python scripts/run_pi05_libero_base_6k_then_eval.py
```

출력은 `outputs/pi05_libero_base_expert_only_6k/` 아래 checkpoint, `eval/instrumentation.json`, `eval/summary.json`, `workflow_status.json`에 저장된다. checkpoint가 이미 있으면 학습을 건너뛰고, 기록된 episode는 건너뛰어 평가를 재개한다.

## raw base 대 6K, 100-seed paired evaluation

다음 평가는 task별 10 seed(`42020–42119`)로 두 policy를 각각 100회 실행한다. 현재 결과는 중단 상태이므로 완료 전 수치를 최종 성능으로 사용하지 않는다.

```bash
source scripts/activate_lerobot.sh && python scripts/eval_pi05_libero_base_vs_expert_only_6k_eval100.py
```

결과는 `outputs/pi05_libero_base_vs_expert_only_6k_eval100/{instrumentation.json,summary.json}`에 episode마다 즉시 기록된다.

## 80K와 raw-base 30-episode 확인

기존 80K checkpoint와 raw base의 같은 30 seed 비교는 다음 script로 재개할 수 있다.

```bash
source scripts/activate_lerobot.sh && python scripts/eval_pi05_libero_base_vs_expert80k_resume.py
```

## Positive control

공식 fine-tuned checkpoint의 task 0 positive control은 다음과 같다.

```bash
lerobot-eval --policy.path=lerobot/pi05_libero_finetuned_v044 --policy.n_action_steps=10 --env.type=libero --env.task=libero_object --env.task_ids='[0]' --env.control_mode=relative --eval.batch_size=1 --eval.n_episodes=3 --env.max_parallel_tasks=1 --output_dir=outputs/pi05_finetuned_v044_positive_control
```

## 저장소에 포함하지 않는 산출물

checkpoint/model weight, optimizer state, dataset, Hugging Face cache, mp4, 대용량 log 및 `.incomplete` 파일은 `.gitignore` 대상이다. 원본 output JSON 경로는 문서에 provenance로만 기록한다.
