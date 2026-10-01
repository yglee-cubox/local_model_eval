# 결과 요약

`python3 summarize_results.py` 로 `results/` 에서 다시 만듭니다. 원본 결과는 저장소에 없습니다.

## LiveCodeBench (codegeneration)

| 모델 | pass@1 | easy | medium | hard | 문제 수 | 기간 | n | temp |
|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | **0.523** | 0.910 | 0.548 | 0.140 | 1055 | 2023-05-07 ~ 2025-04-06 | 1 | 0.6 |
| models-deepseek-coder-1.3b-instruct | **0.100** | 0.278 | 0.040 | 0.054 | 80 | 2025-03-01 ~ 2025-04-06 | 1 | 0.0 |

## SWE-bench Pro V2

공식 프로토콜 아님 (컨테이너에 네트워크와 /purestorage 접근 있음). 체크포인트끼리 비교용입니다.

| 모델 | resolved | rate | 인스턴스 | 에이전트 종료 상태 |
|---|---|---|---|---|
| Qwen3-1.7B | 0 | **0.00%** | 51 | Submitted 22, crash 17, RepeatedFormatError 7, LimitsExceeded 5 |
