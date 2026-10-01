#!/bin/bash
# SWE-bench Pro V2 with a local checkpoint, entirely on the cluster:
#   vLLM (OpenAI API) on this job's GPUs  +  mini-swe-agent in each instance's enroot container  +  enroot grading.
#
#   sbatch --export=ALL,MODEL_PATH=/purestorage/ailab/<me>/ckpt/step-1000 swepro/swepro_local.sbatch          # HARD-51
#   sbatch --export=ALL,MODEL_PATH=...,IDS=/path/ids.txt,CONC=16 -p h100 --gres=gpu:4 swepro/swepro_local.sbatch
#
# env: MODEL_PATH (required)   MODEL_NAME (default: last 2 path parts; results go to results/swepro/<MODEL_NAME>)
#      IDS        instance id list (default: HARD-51; all 642: $EVAL_ROOT/swepro/all_ids.txt)
#      CONC (8)   instances in parallel        STEP_LIMIT (250)    AGENT_TIMEOUT (3000 s, from task.toml)
#      MAX_MODEL_LEN (vLLM context; default = model config)    TEMP / MAX_TOKENS (sampling; default = model/vLLM)
#      GRADE (1)  set 0 to only produce patches    AGENT_CONFIG (official mini_textbased.yaml)
#      VLLM_ARGS  extra `vllm serve` flags
# Rerunning with the same MODEL_NAME resumes: finished instances are skipped.
#
#SBATCH --job-name=swepro-local
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=/purestorage/ailab/yglee/workspace/local_model_eval/logs/%x-%j.out

set -uo pipefail
source /purestorage/ailab/yglee/workspace/local_model_eval/env.sh
: "${MODEL_PATH:?set MODEL_PATH}"
MODEL_NAME=${MODEL_NAME:-$(echo "${MODEL_PATH%/}" | awk -F/ '{print $(NF-1)"-"$NF}')}
export OUT_DIR=${OUT_DIR:-$EVAL_ROOT/results/swepro/$MODEL_NAME}
export AGENT_CONFIG=${AGENT_CONFIG:-$BENCH_DATA/swe-bench-pro/v2-harbor/tooling/configs/mini_textbased.yaml}
export SERVED_MODEL=local
PORT=$((20000 + SLURM_JOB_ID % 10000))
export API_BASE=http://127.0.0.1:$PORT/v1
IDS=${IDS:-$BENCH_DATA/swe-bench-pro/v2-harbor/hard51_ids.txt}
NGPU=$(nvidia-smi -L | wc -l)
mkdir -p "$OUT_DIR"
echo "model=$MODEL_PATH name=$MODEL_NAME gpus=$NGPU ids=$IDS ($(wc -l < "$IDS")) conc=${CONC:-8} out=$OUT_DIR"

"$EVAL_ROOT/.venv-vllm/bin/vllm" serve "$MODEL_PATH" --served-model-name "$SERVED_MODEL" \
  --host 127.0.0.1 --port "$PORT" --tensor-parallel-size "$NGPU" \
  ${MAX_MODEL_LEN:+--max-model-len "$MAX_MODEL_LEN"} ${VLLM_ARGS:-} > "$OUT_DIR/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

for i in $(seq 1 360); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
  if ! kill -0 $VLLM_PID 2>/dev/null || grep -q "Engine core initialization failed" "$OUT_DIR/vllm-$SLURM_JOB_ID.log"; then
    echo "vLLM failed to start, see $OUT_DIR/vllm-$SLURM_JOB_ID.log"; tail -5 "$OUT_DIR/vllm-$SLURM_JOB_ID.log"; exit 1
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "vLLM not ready after 30 min"; exit 1; }
echo "vLLM ready on $API_BASE"

grep -v '^\s*$' "$IDS" | xargs -P "${CONC:-8}" -I{} bash "$EVAL_ROOT/swepro/run_instance.sh" {}

if [ "${GRADE:-1}" = 1 ]; then
  python3 "$BENCH_DATA/swe-bench-pro/scripts/summarize.py" "$OUT_DIR/grades" | tee "$OUT_DIR/summary.txt"
fi
