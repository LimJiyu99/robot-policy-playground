# π0.5 LIBERO-Object task 0 smoke test 보고서

작성일: 2026-07-16 (Asia/Seoul)

## 결론

Hugging Face 인증을 갱신한 뒤 tokenizer만 추가로 받아 동일 설정의
1 episode rollout을 완료했다. LIBERO-Object task 0은 143 step에서
성공했으며 성공률은 100%다.

- 모델 weight 로드: 성공 (모든 state-dict key 일치)
- LIBERO-Object task 0 환경 생성: 성공
- processor/tokenizer 생성: 성공
- policy inference: 성공 (`select_action` 143회, action chunk 생성 3회)
- episode rollout: 성공, reward 1.0, success `true`
- rollout 영상: 생성됨
- OOM, 프로세스 강제 종료, shape mismatch: 발생하지 않음

## 소스와 CLI 확인

- LeRobot: `0.6.1`, editable source
- source commit: `3f2179f3b69708b6ad009b2e7685dd9d05269ee1`
- PyTorch: `2.11.0+cu128` (설치 전후 동일)
- `lerobot-eval --help`에서 실제 확인한 옵션:
  `--policy.path`, `--policy.pretrained_revision`, `--env.type`,
  `--env.task`, `--env.task_ids`, `--env.max_parallel_tasks`,
  `--eval.batch_size`, `--eval.n_episodes`, `--eval.recording`,
  `--output_dir`

`pip install -e "./lerobot[pi]"`만 사용해 `pi` extra를 확인·설치했다.
`PI05Policy` import와 `pip check`가 통과했으며 기존 PyTorch/cu128은
변경되지 않았다.

## checkpoint와 다운로드

- Hub model: `lerobot/pi05_libero_finetuned_v044`
- 고정 revision: `dbf8a3f794a9c4297b44f40b752712f50073d945`
- Hub metadata: 9개 파일, `7,473,127,862 bytes` (`6.960 GiB`)
- weight: `model.safetensors`, `7,473,096,344 bytes`
- 실제 snapshot:
  `/workspace/jy/cache/huggingface/hub/models--lerobot--pi05_libero_finetuned_v044/snapshots/dbf8a3f794a9c4297b44f40b752712f50073d945`
- 불완전 다운로드 파일: 0개

processor config가 추가로 요구한 tokenizer는
`google/paligemma-3b-pt-224` revision
`35e4f46485b4d07967e7e9935bc3786aad50687c`이다. 전체 PaliGemma weight는
대상에서 제외했고, metadata로 확인한 tokenizer 관련 6개 파일만 합쳐
`21,855,253 bytes` (`20.843 MiB`)였다. 이 파일의 다운로드는 gated
repository 미인증으로 `401 Unauthorized`가 발생했다.

## 평가 대상과 명령

- suite: `libero_object`
- task id: `0`만 선택
- task name: `pick_up_the_alphabet_soup_and_place_it_in_the_basket`
- instruction: `pick up the alphabet soup and place it in the basket`
- batch size: 1
- requested episodes: 1
- parallel tasks: 1

고정 snapshot을 로컬 경로로 지정한 이유는 LeRobot 0.6.1 parser가
Hub repo의 `main/config.json`을 CLI revision override보다 먼저 읽기
때문이다. 실제 계측 실행은 다음 명령과 동일하다.

```bash
source scripts/activate_lerobot.sh
export HF_HUB_OFFLINE=1
python scripts/pi05_eval_instrumented.py \
  --policy.path=/workspace/jy/cache/huggingface/hub/models--lerobot--pi05_libero_finetuned_v044/snapshots/dbf8a3f794a9c4297b44f40b752712f50073d945 \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids='[0]' \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval.recording=false \
  --output_dir=outputs/pi05_task0_smoke/eval
```

`scripts/pi05_eval_instrumented.py`는 공식
`lerobot.scripts.lerobot_eval.main()`을 그대로 호출하고, 모델 로딩과
`select_action` 호출 주위에 계측만 추가한다. 실제 실행에는
`scripts/run_pi05_smoke_monitored.sh`를 사용했다.

## shape

checkpoint shape와 성공한 rollout에서 실제 수집한 batch shape는 다음과
같다. LIBERO 원본 image는 360×360이며 policy 내부에서 checkpoint 입력
해상도로 전처리된다.

| 항목 | checkpoint shape | runtime batch shape |
|---|---:|---:|
| `observation.state` | `(8,)` | `(1, 8)` |
| `action` | `(7,)` | `(1, 7)` |
| `observation.images.image` | `(3, 256, 256)` | `(1, 3, 360, 360)` |
| `observation.images.image2` | `(3, 256, 256)` | `(1, 3, 360, 360)` |
| 자동 추가 empty camera | `(3, 224, 224)` | policy 내부 생성 |
| language tokens | max length 200 | `(1, 200)` |
| language attention mask | max length 200 | `(1, 200)` |

## 실제 자원 계측

시스템 RAM 사용량은 `/proc/meminfo`의 `MemTotal - MemAvailable`, GPU
VRAM은 `nvidia-smi`, process peak RSS는 `/usr/bin/time -v`로 측정했다.

| 항목 | 결과 |
|---|---:|
| 시스템 RAM 평가 전 | 3,777,183,744 bytes (3.517 GiB) |
| 시스템 RAM 관측 최대 | 21,821,329,408 bytes (20.323 GiB) |
| 시스템 RAM 평가 후 | 3,613,835,264 bytes (3.366 GiB) |
| 평가 process 최대 RSS | 18,356,444 KiB (17.506 GiB) |
| GPU VRAM 평가 전 | 278 MiB |
| GPU VRAM 관측 최대 | 10,690 MiB (10.439 GiB) |
| GPU VRAM 평가 후 | 277 MiB |
| `torch.cuda.max_memory_allocated()` | 9,573,197,312 bytes (8.916 GiB) |
| 모델 로딩 직후 CUDA allocated | 9,356,018,688 bytes (8.713 GiB) |
| 모델 parameter 수 | 4,143,404,816 |
| 모델 로딩 시간 | 58.575 s |
| episode 실행 시간 | 230.401 s |
| 전체 process wall time | 303.48 s |
| 평균 `select_action` latency | 1.5489 s (최초 compile 포함, 143회) |
| 최초 compile 이후 평균 `select_action` latency | 2.511 ms (142회) |
| action chunk 생성 평균 latency | 73.775 s (최초 compile 포함, 3회) |
| 최초 compile 이후 chunk 생성 평균 latency | 93.446 ms (2회) |
| 성공 여부 | 성공 (`reward=1.0`, `success=true`) |

최초 chunk 호출은 checkpoint에 저장된 `compile_model=true`,
`compile_mode=max-autotune` 때문에 221.139 s가 걸렸다. 이후 두 chunk는
각각 117.274 ms와 69.618 ms였다. 따라서 compile 포함 평균과 warm
inference 평균을 함께 기록했다.

## warning과 error

주요 warning:

- robosuite private macro 파일 없음. 기존 LIBERO smoke test에서는 환경
  생성/reset/step에 지장이 없었던 비치명적 안내다.
- `OpenGL_accelerate` 미설치 안내. EGL 환경 생성에는 치명적이지 않았다.
- `torch.jit.script_method` deprecation warning.
- PI05 vision embedding key 두 개가 handling 확인이 필요할 수 있다는
  warning. 직후 state dict는 모든 key가 성공적으로 로드됐다.
- checkpoint 설정의 `gradient_checkpointing=true`가 그대로 적용됐다.

초기 시도의 치명적 error:

```text
ValueError: Failed to instantiate processor step 'tokenizer_processor' ...
tokenizer_name: 'google/paligemma-3b-pt-224'
GatedRepoError: 401 Unauthorized
```

처음 Hub repo id와 offline mode를 함께 사용한 preflight에서는 parser가
revision override 전에 `main/config.json`을 조회해
`LocalEntryNotFoundError`가 났다. 해당 로그는
`outputs/pi05_task0_smoke/preflight_offline_repo_error/`에 보존했다.
또한 기본 sandbox에서 GPU가 보이지 않아 CPU로 자동 선택된 실행은 모델
추론 전에 직접 중단했으며 로그를
`outputs/pi05_task0_smoke/cpu_isolation_abort/`에 보존했다.

초기 tokenizer 누락 실행은
`outputs/pi05_task0_smoke/processor_tokenizer_missing/`에 보존했다. 현재
`outputs/pi05_task0_smoke/`의 `eval.log`, `instrumentation.json`,
`resource_samples.tsv`, `process_time.txt`, `ram_*.txt`, `gpu_*.csv`는
성공한 최종 실행의 자료다.

## 초기 시점의 진행 조건 (해결됨)

Hugging Face에서 `google/paligemma-3b-pt-224` 사용 조건에 동의하고,
해당 repository를 읽을 수 있는 토큰으로 인증해야 한다. 그 뒤 위에서
명시한 20.843 MiB tokenizer 파일만 지정된 workspace HF cache에 받은
후 동일한 단일 평가를 실행할 수 있다. 이 조건은 이후 인증 갱신으로
해결됐다. 토큰 문자열은 문서나 프로젝트 파일에 저장하지 않았다.

## 2026-07-16 인증 후 재개 시도

사용자가 Hugging Face 로그인과 PaliGemma 접근 동의를 완료했다고 알려와
기존 checkpoint를 재다운로드하지 않고 tokenizer 6개 파일만 다시
요청했다. `hf auth whoami`는 로그인 계정을 `jiyu99`로 정상 확인했다.
그러나 Hub의 실제 파일 요청은 다음 응답으로 거부됐다.

```text
403 Forbidden
Access to model google/paligemma-3b-pt-224 is restricted and you are not in
the authorized list.
```

따라서 이 재개 시도에서도 checkpoint 다운로드와 evaluator 실행은 하지
않았으며, 기존 실패 이후의 추가 episode 횟수는 0이다. Hugging Face의
`google/paligemma-3b-pt-224` 모델 페이지에서 계정 `jiyu99`의 접근 상태를
확인하고, fine-grained token이라면 사용 가능한 public gated repository의
내용을 읽을 수 있는 권한을 포함해야 한다. 권한이 반영된 뒤 tokenizer
6개 파일만 다시 요청하면 된다.

## 최종 재개 및 성공

계정 `jiyu99`의 gated repository 접근이 반영된 뒤 tokenizer 6개 파일을
다운로드했다. 첫 병렬 다운로드에서 Hub Xet 경로가
`Unable to parse string as hex hash value`를 반환해, Hugging Face Hub가
지원하는 `HF_HUB_DISABLE_XET=1`을 사용하여 일반 HTTP로 이어받았다.
최종 tokenizer 크기는 metadata와 같은 `21,855,253 bytes`이며, 기존
π0.5 checkpoint weight의 크기 `7,473,096,344 bytes`는 변하지 않았다.

최종 episode 결과:

- task: `libero_object`, id 0만 실행
- episode/batch: 1/1
- 실행 step: 143 (최대 280보다 먼저 성공 종료)
- reward: 1.0
- success: `true` (100%)
- 영상: `outputs/pi05_task0_smoke/eval/videos/libero_object_0/eval_episode_0.mp4`
- 영상 형식: H.264, 360×360, 80 fps, 1.7875 s
- eval result: `outputs/pi05_task0_smoke/eval/eval_info.json`
