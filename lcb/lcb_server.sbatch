#!/bin/bash
# LiveCodeBench (code generation) through a vLLM 0.30 OpenAI server, for checkpoints the LCB venv's vLLM 0.8.4
# cannot load (e.g. Qwen3.5-architecture models). lcb_runner talks to the server with its official OpenAIChat
# runner (system + user chat messages; the server applies the checkpoint's chat template).
#
#   sbatch -p h100 --gres=gpu:8 --export=ALL,MODEL_PATH=...,MODEL_NAME=qwen-27b,TP=2,TEMP=1.0,MAX_TOKENS=65536 \
#          lcb/lcb_server.sbatch
#
# env: MODEL_PATH (required)  MODEL_NAME (default: last 2 path parts)  TP (1; data parallel = GPUs / TP)
#      RELEASE (release_v6)  N (10)  TEMP (0.2)  MAX_TOKENS (2000)  PARALLEL (32 concurrent requests, each n samples)
#      OUT_DIR (local_model_eval/results/lcb)  VLLM_ARGS (extra `vllm serve` flags)  EXTRA (more lcb_runner flags)
#
#SBATCH --job-name=lcb-server
#SBATCH --partition=h100
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=/purestorage/ailab/yglee/workspace/local_model_eval/logs/%x-%j.out

set -uo pipefail
source /purestorage/ailab/yglee/workspace/local_model_eval/env.sh
: "${MODEL_PATH:?set MODEL_PATH}"
MODEL_NAME=${MODEL_NAME:-$(echo "${MODEL_PATH%/}" | awk -F/ '{print $(NF-1)"-"$NF}')}
OUT_DIR=${OUT_DIR:-$EVAL_ROOT/results/lcb}
NGPU=$(nvidia-smi -L | wc -l)
TP=${TP:-1}
PORT=$((20000 + SLURM_JOB_ID % 10000))
mkdir -p "$OUT_DIR"
VLLM_LOG=$OUT_DIR/vllm-$MODEL_NAME-$SLURM_JOB_ID.log
echo "model=$MODEL_PATH name=$MODEL_NAME gpus=$NGPU tp=$TP dp=$((NGPU / TP)) out=$OUT_DIR"

"$EVAL_ROOT/.venv-vllm/bin/vllm" serve "$MODEL_PATH" --served-model-name "$MODEL_NAME" \
  --host 127.0.0.1 --port "$PORT" --tensor-parallel-size "$TP" --data-parallel-size "$((NGPU / TP))" \
  ${VLLM_ARGS:-} > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

for i in $(seq 1 360); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
  if ! kill -0 $VLLM_PID 2>/dev/null || grep -q "Engine core initialization failed" "$VLLM_LOG"; then
    echo "vLLM failed to start, see $VLLM_LOG"; tail -5 "$VLLM_LOG"; exit 1
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "vLLM not ready after 30 min"; exit 1; }
echo "vLLM ready on port $PORT"

export OPENAI_BASE_URL=http://127.0.0.1:$PORT/v1 OPENAI_KEY=EMPTY
"$LCB_REPO/.venv/bin/python" "$EVAL_ROOT/lcb/run_lcb_local.py" \
  --model_path "$MODEL_PATH" --model_name "$MODEL_NAME" --prompt_style OpenAIChat --output_dir "$OUT_DIR" -- \
  --scenario codegeneration --release_version "${RELEASE:-release_v6}" \
  --n "${N:-10}" --temperature "${TEMP:-0.2}" --max_tokens "${MAX_TOKENS:-2000}" \
  --multiprocess "${PARALLEL:-32}" --openai_timeout 14400 \
  --evaluate --num_process_evaluate "$((SLURM_CPUS_PER_TASK - 4))" \
  ${EXTRA:-}
