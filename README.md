# 직접 학습한 로컬 모델로 벤치마크 돌리기

공용 벤치마크(`/purestorage/ailab/datasets/share_data/benchmarks/`)를 **HuggingFace 형식 체크포인트**로 평가하는 스크립트입니다.
벤치마크 자체 설명은 공용 폴더의 `README.md` 를 보세요.

| 벤치마크 | 스크립트 | 하는 일 |
|---|---|---|
| LiveCodeBench | `lcb/lcb_local.sbatch`, `lcb/lcb_local_srun.sh` | vLLM 으로 코드 생성 → 공식 채점 (pass@1) |
| LiveCodeBench (새 아키텍처 모델) | `lcb/lcb_server.sbatch` | vLLM 0.30 서버 + lcb_runner 공식 OpenAI 러너로 생성 → 공식 채점 (1-1) |
| SWE-bench Pro V2 | `swepro/swepro_local.sbatch` | vLLM 서버 + mini-swe-agent(공식 설정)를 인스턴스 컨테이너(enroot)에서 실행 → 새 컨테이너에서 채점 (resolve rate) |
| Terminal-Bench 4.0 | 없음 | 클러스터에서 실행 불가 (공용 README 3장) |

## 설치 (clone 직후 한 번, mgmt 노드에서)

가상 환경과 `glibc-runtime/` 은 저장소에 없습니다. 다음으로 만듭니다. (Python 3.12)

```bash
uv venv -p 3.12 .venv-vllm  && uv pip install --python .venv-vllm/bin/python  -r requirements-vllm.txt
uv venv -p 3.12 .venv-agent && uv pip install --python .venv-agent/bin/python -r requirements-agent.txt

# glibc 로더 + 라이브러리: mgmt 노드(Ubuntu, glibc 2.43)의 시스템 라이브러리를 그대로 복사
mkdir -p glibc-runtime/lib && cd /lib/x86_64-linux-gnu && cp -L ld-linux-x86-64.so.2 libc.so.6 libcrypt.so.1 libdl.so.2 \
  libgcc_s.so.1 libm.so.6 libpthread.so.0 librt.so.1 libstdc++.so.6 libutil.so.1 libz.so.1 "$OLDPWD/glibc-runtime/lib/" && cd -
```

LiveCodeBench 는 `lcb_local_data.patch` 를 적용한 lcb_runner 저장소도 필요합니다 (공용 README 1-2, 경로는 `env.sh` 의 `LCB_REPO`).
스크립트 안의 `/purestorage/ailab/yglee/...` 경로는 clone 한 위치에 맞게 바꿔 주세요.

## 0. 체크포인트 준비

- vLLM 이 읽을 수 있는 **HF 형식 폴더**여야 합니다: `config.json`, 가중치(`*.safetensors`), 토크나이저 파일.
  - FSDP / DeepSpeed 체크포인트는 먼저 HF 형식으로 변환합니다. (예: `model.save_pretrained(dir, safe_serialization=True)` + `tokenizer.save_pretrained(dir)`)
  - LoRA 는 base 모델에 합쳐서(`merge_and_unload()`) 저장합니다.
- 채팅(instruct) 모델은 `tokenizer_config.json` 에 **`chat_template`** 이 있어야 합니다. 두 벤치마크 모두 이 템플릿으로 프롬프트를 만듭니다.
- 체크포인트는 /purestorage 에 둡니다. 로딩 속도는 노드마다 크게 다릅니다 (2026-10-01 측정, /purestorage 읽기):
  **h100 ~670 MB/s, a100 ~550 MB/s, a10 ~15 MB/s.** a10 에서는 7B 모델(15GB) 로딩만 15분 넘게 걸리니 a100/h100 을 쓰세요.

## 1. LiveCodeBench

`lcb/run_lcb_local.py` 가 체크포인트를 lcb_runner 모델 목록에 즉석에서 등록하고 공식 `lcb_runner` 로 생성과 채점을 합니다.
(lcb_runner 는 원래 `lm_styles.py` 에 등록된 모델만 받습니다.)

```bash
cd /purestorage/ailab/yglee/workspace/local_model_eval

# 전체 평가 (release_v6, n=10, temperature 0.2 = 공식 기본값)
sbatch --export=ALL,MODEL_PATH=/purestorage/ailab/<계정>/ckpt/step-1000 lcb/lcb_local.sbatch

# 큰 모델 / 추론 모델: GPU 수 = 텐서 병렬 수
sbatch -p h100 --gres=gpu:4 --export=ALL,MODEL_PATH=...,MODEL_NAME=my-32b-rl,MAX_TOKENS=32768 lcb/lcb_local.sbatch

# 빠른 확인 (터미널을 잡고 있음). 2025-03 이후 80문제, 샘플 1개
bash lcb/lcb_local_srun.sh /purestorage/ailab/<계정>/ckpt/step-1000 --start_date 2025-03-01 --n 1 --temperature 0
```

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MODEL_PATH` | (필수) | HF 형식 체크포인트 폴더 |
| `MODEL_NAME` | 경로 마지막 두 단계 | 결과 폴더 이름 |
| `PROMPT_STYLE` | `chat_template` | `chat_template`: 체크포인트의 chat template 사용 (instruct 모델) / `base`: few-shot 프롬프트 (base 모델) / `CodeQwenInstruct` 등 lcb_runner 의 `LMStyle` 이름 |
| `RELEASE` | `release_v6` | `release_v5`, `v6` (v6 에서 추가된 문제만) 등 |
| `N`, `TEMP`, `MAX_TOKENS` | 10, 0.2, 2000 | 샘플 수, 온도, 최대 생성 길이 |
| `OUT_DIR` | `results/lcb` | 결과 위치 |
| `EXTRA` | | 그 밖의 lcb_runner 옵션 (예: `"--start_date 2024-08-01"`) |

- 결과: `<OUT_DIR>/output/<MODEL_NAME>/Scenario.codegeneration_<N>_<TEMP>{,_eval,_eval_all}.json`. `_eval.json` 의 `pass@1` 이 점수입니다.
  파일별 형식과 읽는 코드는 공용 README **6-1** 에 있습니다.
- `chat_template` 스타일은 system 프롬프트 + 문제를 체크포인트의 템플릿으로 감싸고, 답의 마지막 ```` ```python ```` 블록을 코드로 씁니다.
  lcb_runner 기본 stop 문자열 `###` 이 마크다운 제목에서 답을 자르는 문제가 있어서, stop 을 토크나이저의 EOS 로 바꿉니다.
- `chat_template` 은 `codegeneration` 시나리오만 지원합니다.

### 1-1. lcb_runner 의 vLLM 이 못 읽는 모델: `lcb/lcb_server.sbatch`

`lcb_local.sbatch` 는 lcb_runner 저장소의 가상 환경 (vLLM 0.8.4, transformers 4.51) 으로 모델을 띄웁니다.
그래서 그보다 새로운 아키텍처는 읽지 못합니다. 예를 들어 Qwen3.5 계열 (`model_type: qwen3_5`, Qwen3.8-27B 등) 이 그렇습니다.
이런 모델은 `lcb_server.sbatch` 를 씁니다.

1. 잡의 GPU 에서 `.venv-vllm` (vLLM 0.30) 으로 OpenAI 호환 서버를 띄웁니다.
2. lcb_runner 의 공식 `OpenAIChat` 러너가 그 서버에 요청합니다.
   GPT-4o 같은 API 모델과 같은 경로이고, 프롬프트는 system + user 메시지입니다. chat template 은 서버가 적용합니다.
3. 생성이 끝나면 lcb_runner 가 그대로 채점합니다. 결과 파일 위치와 형식은 `lcb_local.sbatch` 와 같습니다.

```bash
cd /purestorage/ailab/yglee/workspace/local_model_eval

# GPU 8장, 텐서 병렬 2 × 데이터 병렬 4
sbatch -p h100 --gres=gpu:8 \
       --export=ALL,MODEL_PATH=<ckpt>,MODEL_NAME=my-27b,TP=2,TEMP=1.0,MAX_TOKENS=65536,PARALLEL=48 \
       lcb/lcb_server.sbatch
```

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MODEL_PATH`, `MODEL_NAME`, `RELEASE`, `N`, `TEMP`, `MAX_TOKENS`, `OUT_DIR`, `EXTRA` | 위 표와 같음 | |
| `TP` | 1 | 텐서 병렬 수. 데이터 병렬 수는 GPU 수 ÷ `TP` 입니다 |
| `PARALLEL` | 32 | 동시에 보내는 요청 수 (요청 하나가 문제 하나의 `N` 개 샘플) |
| `VLLM_ARGS` | | `vllm serve` 추가 옵션. thinking 끄기 등은 아래 "thinking 과 샘플링 설정" 참고 |
| `SAMPLING_KWARGS` | | 요청마다 더할 샘플링 값 (JSON). 아래 "thinking 과 샘플링 설정" 참고 |

- 요청 하나의 제한 시간은 4시간 (`--openai_timeout 14400`) 입니다. 긴 추론 모델도 시간 초과로 잘리지 않게 넉넉히 잡았습니다.
- lcb_runner 는 생성이 **전부 끝난 뒤에** 결과 파일을 씁니다. 중간에 잡이 죽으면 생성한 것이 남지 않습니다.
- vLLM 서버 로그: `<OUT_DIR>/vllm-<MODEL_NAME>-<잡번호>.log`
- top_p 는 lcb_runner 기본값 0.95 로 요청에 들어갑니다. 바꾸려면 `EXTRA="--top_p 0.8"` 처럼 줍니다.
  top_k 는 요청에 넣지 않으므로 서버 기본값 (체크포인트의 `generation_config.json`) 을 씁니다.
- **thinking 모델은 `--reasoning-parser` 를 주지 마세요.** 그러면 thinking 이 답에 같이 들어가고, 답의 마지막 ```` ```python ```` 블록을 코드로 씁니다.
  - parser 를 주면 thinking 이 `max_tokens` 에서 잘렸을 때 생성 전체가 thinking 으로 분류돼, 답 (content) 이 비고 코드를 못 꺼낼 수 있습니다.
  - parser 없이 thinking 이 잘리면 thinking 안의 미완성 코드가 답으로 쓰여 대개 런타임 에러가 납니다.
    잘린 답이 많으면 `MAX_TOKENS` 를 늘리세요 (예: Qwen3.8-27B 는 65536 에서 hard 문제 답의 약 18% 가 잘렸습니다. `summaries/analysis.md` 1-3).
- 확인한 예 (h100 8장, `TP=2`, Qwen3.8-27B, release_v6 전체, n=10):
  - thinking 켬: pass@1 0.862, 7시간 34분
  - thinking 끔: pass@1 0.724, 1시간 32분

## thinking 과 샘플링 설정 (LCB 서버 방식, SWE-bench Pro 공통)

`lcb/lcb_server.sbatch` 와 `swepro/swepro_local.sbatch` 는 둘 다 vLLM 서버를 띄웁니다. 그래서 같은 방법으로 thinking 과 샘플링을 정합니다.

| 하고 싶은 것 | 방법 |
|---|---|
| thinking 끄기 / 켜기 | `VLLM_ARGS='--default-chat-template-kwargs {"enable_thinking":false}'` (켜기는 `true`) |
| temperature, top_p, top_k 등의 서버 기본값 바꾸기 | `VLLM_ARGS='--override-generation-config {"temperature":0.7,"top_p":0.8,"top_k":20}'` |
| 서버 기본값으로 못 주는 값 (presence_penalty 등) | `SAMPLING_KWARGS='{"presence_penalty":1.5}'` |
| thinking 을 답에서 분리 (SWE-bench Pro) | `VLLM_ARGS='--reasoning-parser qwen3'` (gemma-4 는 `gemma4`) |

- **thinking 을 켜지도 끄지도 않으면 chat template 의 기본값을 따릅니다.** Qwen3 계열은 켜짐, gemma-4 는 꺼짐입니다.
- **`SAMPLING_KWARGS` 가 필요한 이유:**
  - vLLM 은 `generation_config` 의 temperature, top_p, top_k, min_p, repetition_penalty 만 서버 기본값으로 씁니다. presence_penalty 는 서버 기본값으로 줄 수 없습니다.
  - lcb_runner 의 OpenAI 러너는 요청마다 `presence_penalty=0` 을 직접 넣습니다.
  - `SAMPLING_KWARGS` 는 이 값들을 요청에 덮어씁니다 (`lcb/run_lcb_local.py`, `swepro/agent_in_container.py`).
- **SWE-bench Pro 에서는 thinking 모델에 `--reasoning-parser` 를 주세요.** LCB 와 반대입니다.
  - parser 가 없으면 thinking 이 답에 섞입니다.
  - thinking 안에 ```` ``` ```` 블록이 있으면 명령 형식 오류 (`FormatError`) 가 나고, 대화 기록도 길어집니다.
- **JSON 이 들어간 값은 `--export=ALL,...` 안에 쓰지 말고 미리 `export` 하세요.** `--export` 는 쉼표로 변수를 나눠서 JSON 이 깨집니다.

실제로 Qwen3.8-27B thinking 끔 실행에 쓴 명령입니다 (모델 README 의 non-thinking 권장값).

```bash
export MODEL_PATH=<ckpt> MODEL_NAME=Qwen3.8-27B-nothink TEMP=0.7
export SAMPLING_KWARGS='{"presence_penalty":1.5}'
NT='--default-chat-template-kwargs {"enable_thinking":false} --override-generation-config {"temperature":0.7,"top_p":0.8,"top_k":20}'

# LiveCodeBench
VLLM_ARGS="$NT" TP=2 N=10 MAX_TOKENS=32768 PARALLEL=48 EXTRA="--top_p 0.8" \
  sbatch -p h100 --gres=gpu:8 --export=ALL lcb/lcb_server.sbatch

# SWE-bench Pro HARD-51
VLLM_ARGS="$NT --reasoning-parser qwen3" CONC=16 MAX_TOKENS=32768 SAMPLING_KWARGS='{"top_p":0.8,"presence_penalty":1.5}' \
  sbatch -p h100 --gres=gpu:4 --cpus-per-task=64 --mem=600G --export=ALL swepro/swepro_local.sbatch
```

## 2. SWE-bench Pro V2

한 잡 안에서 다음을 모두 합니다.

1. 잡의 GPU 에서 체크포인트를 **vLLM OpenAI 호환 서버**로 띄웁니다. (`.venv-vllm`, vLLM 0.30)
2. 인스턴스마다 그 인스턴스의 공개 이미지(`ghcr.io/scaleapi/swe-bench_pro-v2:<id>`)로 **enroot 컨테이너**를 띄우고,
   그 안에서 **mini-swe-agent 2.4.6** 을 공식 설정(`v2-harbor/tooling/configs/mini_textbased.yaml`)으로 돌립니다.
   공식 `host_mini_swe.py` 와 같은 방식입니다: 명령 제한 60초, `git add -A && git diff --cached` 로 패치 저장.
3. 그 패치를 **새 컨테이너**에서 채점합니다. (공용 `scripts/grade_enroot.sh`)

```bash
cd /purestorage/ailab/yglee/workspace/local_model_eval

# HARD-51 (기본)
sbatch --export=ALL,MODEL_PATH=/purestorage/ailab/<계정>/ckpt/step-1000 swepro/swepro_local.sbatch

# 전체 642개, GPU 4장, 동시 16개
sbatch -p h100 --gres=gpu:4 --cpus-per-task=64 \
       --export=ALL,MODEL_PATH=...,IDS=$PWD/swepro/all_ids.txt,CONC=16 swepro/swepro_local.sbatch
```

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MODEL_PATH` | (필수) | HF 형식 체크포인트 폴더 (chat template 필요) |
| `MODEL_NAME` | 경로 마지막 두 단계 | 결과 폴더 이름. **같은 이름으로 다시 내면 끝난 인스턴스는 건너뜁니다** (이어하기) |
| `IDS` | HARD-51 | 인스턴스 id 목록 파일. 전체: `swepro/all_ids.txt` |
| `CONC` | 8 | 동시에 돌릴 인스턴스 수 |
| `STEP_LIMIT` | 250 | 에이전트 최대 스텝 (공식 Harbor 예시 값) |
| `AGENT_TIMEOUT` | 3000 | 인스턴스당 에이전트 시간 제한(초). `task.toml` 값 |
| `MAX_MODEL_LEN` | 모델 설정값 | vLLM 컨텍스트 길이. 메모리가 모자라면 줄입니다 |
| `TEMP`, `MAX_TOKENS` | 모델/vLLM 기본값 | 샘플링 |
| `GRADE` | 1 | 0 이면 패치만 만듭니다 |
| `VLLM_ARGS` | | `vllm serve` 추가 옵션 (예: `"--gpu-memory-utilization 0.85"`). thinking 끄기 등은 "thinking 과 샘플링 설정" 참고 |
| `SAMPLING_KWARGS` | | 요청마다 더할 샘플링 값 (JSON, 예: `'{"top_p":0.8,"presence_penalty":1.5}'`) |

결과 (`results/swepro/<MODEL_NAME>/`):

```
summary.txt                          resolved 12/51 = 23.53%
vllm-<잡번호>.log                     vLLM 서버 로그
agent/<instance_id>/model.patch      에이전트가 만든 패치
agent/<instance_id>/trajectory.json  대화 전체 (mini-swe-agent 형식)
agent/<instance_id>/result.json      exit_status (Submitted / LimitsExceeded / ...), 호출 수, 패치 크기, 시간
agent/<instance_id>/agent.log
grades/<instance_id>/verifier/reward.txt   1 = 해결 (output.json 에 테스트별 결과)
```

파일별 형식, `exit_status` 값, 읽는 코드는 공용 README **6-3** 에 있습니다.

주의:

- **공식 점수와 같은 조건이 아닙니다.** 공식 locked protocol 은 에이전트 단계에서 네트워크를 막는데, enroot 컨테이너는 호스트 네트워크와
  `/purestorage` 를 그대로 씁니다. 모델이 마음먹으면 인터넷이나 공용 폴더의 정답 패치를 볼 수 있다는 뜻입니다.
  체크포인트 간 비교용으로 쓰고, trajectory 에서 `/purestorage`, `curl`, `git log` 같은 수상한 명령이 있는지 확인하세요.
- 텍스트 형식 설정을 씁니다. 모델이 답마다 ```` ```mswea_bash_command ```` 블록 하나를 내야 합니다.
  형식을 못 지키면 `FormatError` 가 반복되고 결국 끝납니다. 작은 모델일수록 흔합니다.
- 컨텍스트가 모자라면 해당 인스턴스는 `ContextWindowExceededError` 로 끝나고, 그때까지의 패치로 채점합니다.
- 일부 이미지(vuls, teleport, protonmail 등)는 **Alpine(musl)** 이라 glibc 용 python 이 그냥은 실행되지 않습니다 (`execve(): ... No such file or directory`).
  `run_instance.sh` 는 `glibc-runtime/lib/ld-linux-x86-64.so.2 --library-path ...` 로 python 을 띄워서 이 문제를 피합니다. 에이전트가 실행하는 명령(bash, git, go ...)은 컨테이너 자체 libc 를 씁니다. (2026-10-01 수정 전 결과에서 `container_timeout_or_crash` 인 인스턴스는 이 문제입니다. `agent/<id>/` 를 지우고 다시 내세요)
- 이미지를 컨테이너로 풀 때(pyxis import) **잡의 `/tmp`(메모리, tmpfs)** 를 씁니다. `ENROOT_TEMP_PATH` 를 바꿔도 pyxis 는 무시합니다.
  teleport, protonmail 같은 큰 이미지 여러 개를 동시에 풀면 잡 메모리를 넘어서 `No space left on device` / `Out Of Memory` 로 컨테이너가 안 뜨고
  `container_timeout_or_crash` 로 끝납니다. 기본 `--mem=256G` 에 `CONC=8` 이면 생길 수 있으니 `sbatch --mem=600G ...` 처럼 늘리거나 `CONC` 를 줄이세요. (a100 노드 메모리 2TB)
- 이미지는 잡이 도는 노드에 받아집니다 (인스턴스당 수백 MB~수 GB, 7일 보관). 처음에는 받는 시간이 더 걸립니다.

## 공통

- 잡 로그: `logs/<잡이름>-<잡번호>.out`
- 다른 사람이 쓸 때: 이 폴더의 가상 환경과 스크립트는 읽기만 하면 되지만, 결과와 로그는 이 폴더에 쓰려고 합니다.
  `OUT_DIR=<내 폴더>` 를 넘기고 `sbatch -o <내 폴더>/%x-%j.out ...` 으로 로그 위치를 바꾸거나, 폴더째 복사해서 쓰세요.
- `env.sh` 가 잡 안에서 HF / vLLM / triton 캐시를 노드 로컬 `$HOME` 으로 돌립니다. (GPU 노드에서 /purestorage 파일 잠금이 안 되기 때문)
- GPU 노드에 nvcc 가 없어서, `env.sh` 는 실행 중에 커널을 컴파일하는 vLLM 의 FlashInfer 기능 두 가지를 끕니다.
  - 샘플러: `VLLM_USE_FLASHINFER_SAMPLER=0`
  - GPU 여러 장 (텐서 병렬) 에서 쓰는 all-reduce: `VLLM_ALLREDUCE_USE_FLASHINFER=0`.
    이게 켜져 있으면 GPU 2장 이상에서 vLLM 이 뜨다가 `Could not find nvcc` 로 죽습니다.
- **잡이 도는 동안 이 폴더의 bash 스크립트 (`run_instance.sh` 등) 를 고치지 마세요.**
  bash 는 스크립트를 실행하면서 읽기 때문에, 파일이 바뀌면 `Stale file handle` 로 남은 인스턴스가 채점 단계를 건너뜁니다.
  Python 파일은 시작할 때 한 번 읽으므로 비교적 안전합니다.

| 경로 | 내용 |
|---|---|
| `env.sh` | 공통 환경 변수 |
| `lcb/run_lcb_local.py` | LiveCodeBench 래퍼 (lcb_runner: `/purestorage/ailab/yglee/workspace/benchmarks/livecodebench`) |
| `lcb/lcb_server.sbatch` | LiveCodeBench, vLLM 0.30 서버 방식 (1-1) |
| `swepro/agent_in_container.py` | 컨테이너 안에서 도는 mini-swe-agent 실행기 |
| `swepro/run_instance.sh` | 인스턴스 하나: 에이전트 → 채점 |
| `swepro/all_ids.txt` | V2 642개 id |
| `summarize_results.py` | `results/` 와 `logs/` 에서 모델별·벤치마크별 요약을 `summaries/` 로 모읍니다 (끝난 실행만, `test-*` 제외). thinking on/off 는 vLLM 인자(`VLLM_ARGS` 의 `--default-chat-template-kwargs`)나 chat template 기본값에서 읽습니다. `results/`, `logs/` 는 git 에 올리지 않고 `summaries/` 만 올립니다 |
| `.venv-vllm/` | vLLM 0.30 (모델 서버) |
| `.venv-agent/` | mini-swe-agent 2.4.6 (컨테이너 안에서 /purestorage 경로로 실행) |
| `glibc-runtime/lib/` | glibc 로더와 라이브러리. Alpine(musl) 이미지에서도 `.venv-agent` 의 python 이 돌도록 이 로더로 실행합니다 (`run_instance.sh`) |
