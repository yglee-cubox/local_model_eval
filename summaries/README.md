# 결과 요약

`python3 summarize_results.py` 로 `results/` 와 `logs/` 에서 다시 만듭니다. 원본 결과는 저장소에 없습니다.
실행별 설정(vLLM 인자, 샘플링, 문제별 / 인스턴스별 결과)은 같은 폴더의 JSON 에 있습니다.

thinking: vLLM `--default-chat-template-kwargs '{"enable_thinking": ...}'` 로 지정한 값, 지정하지 않았으면 체크포인트 chat template 의 기본값
(Qwen3 계열은 켜짐, gemma-4 는 꺼짐). `n/a` 는 chat template 에 thinking 스위치가 없는 모델입니다.

## LiveCodeBench (codegeneration)

| 모델 | thinking | 실행 이름 | pass@1 | easy | medium | hard | 문제 수 | 기간 | n | temp | top_p | max_tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| google/gemma-4-31B-it | on | gemma-4-31B-it-think | **0.875** | 0.990 | 0.928 | 0.711 | 1055 | 2023-05-07 ~ 2025-04-06 | 10 | 1.0 | - | 32768 |
| google/gemma-4-31B-it | off | gemma-4-31B-it | **0.823** | 0.988 | 0.892 | 0.597 | 1055 | 2023-05-07 ~ 2025-04-06 | 10 | 0.2 | - | 8192 |
| models-deepseek-coder-1.3b-instruct | n/a | models-deepseek-coder-1.3b-instruct | **0.100** | 0.278 | 0.040 | 0.054 | 80 | 2025-03-01 ~ 2025-04-06 | 1 | 0.0 | - | - |
| Qwen/Qwen3.8-27B | on | Qwen3.8-27B | **0.862** | 0.987 | 0.912 | 0.692 | 1055 | 2023-05-07 ~ 2025-04-06 | 10 | 1.0 | - | 65536 |
| Qwen/Qwen3.8-27B | off | Qwen3.8-27B-nothink | **0.724** | 0.852 | 0.815 | 0.508 | 1055 | 2023-05-07 ~ 2025-04-06 | 10 | 0.7 | 0.8 | 32768 |
| Qwen3-1.7B | on | Qwen3-1.7B | **0.523** | 0.910 | 0.548 | 0.140 | 1055 | 2023-05-07 ~ 2025-04-06 | 1 | 0.6 | - | 32768 |

## SWE-bench Pro V2

공식 프로토콜 아님 (컨테이너에 네트워크와 /purestorage 접근 있음). 체크포인트끼리 비교용입니다.

| 모델 | thinking | 실행 이름 | resolved | rate | 인스턴스 | 평균 호출 수 | 에이전트 종료 상태 |
|---|---|---|---|---|---|---|---|
| google/gemma-4-31B-it | on | gemma-4-31B-it-think | 8 | **15.69%** | 51 (hard51_ids) | 23.7 | Submitted 47, RepeatedFormatError 4 |
| google/gemma-4-31B-it | off | gemma-4-31B-it | 5 | **9.80%** | 51 (hard51_ids) | 15.3 | Submitted 50, RepeatedFormatError 1 |
| Qwen/Qwen3.8-27B | on | Qwen3.8-27B | 27 | **52.94%** | 51 (hard51_ids) | 69.7 | Submitted 32, container_timeout_or_crash 9, TimeExceeded 7, RepeatedFormatError 3 |
| Qwen/Qwen3.8-27B | off | Qwen3.8-27B-nothink | 23 | **45.10%** | 51 (hard51_ids) | 93.8 | Submitted 41, LimitsExceeded 9, container_timeout_or_crash 1 |
| Qwen3-1.7B | on | Qwen3-1.7B | 0 | **0.00%** | 51 (hard51_ids) | 90.2 | Submitted 22, crash 17, RepeatedFormatError 7, LimitsExceeded 5 |
