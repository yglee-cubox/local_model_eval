#!/bin/bash
# LiveCodeBench (code generation) for a local checkpoint.
#
#   sbatch --export=ALL,MODEL_PATH=/purestorage/ailab/<me>/ckpt/step-1000 lcb/lcb_local.sbatch
#   sbatch -p h100 --gres=gpu:4 --export=ALL,MODEL_PATH=...,MODEL_NAME=my-32b,MAX_TOKENS=32768 lcb/lcb_local.sbatch
#
# env: MODEL_PATH (required)  MODEL_NAME (default: last 2 path parts)  PROMPT_STYLE (chat_template|base|<LMStyle>)
#      RELEASE (release_v6)  N (10)  TEMP (0.2)  MAX_TOKENS (2000)  OUT_DIR (local_model_eval/results/lcb)
#      EXTRA (more lcb_runner flags, e.g. "--start_date 2024-08-01")
# Tensor parallel size = number of GPUs given with --gres.
#
#SBATCH --job-name=lcb-local
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/purestorage/ailab/yglee/workspace/local_model_eval/logs/%x-%j.out

set -euo pipefail
source /purestorage/ailab/yglee/workspace/local_model_eval/env.sh
: "${MODEL_PATH:?set MODEL_PATH}"

"$LCB_REPO/.venv/bin/python" "$EVAL_ROOT/lcb/run_lcb_local.py" \
  --model_path "$MODEL_PATH" ${MODEL_NAME:+--model_name "$MODEL_NAME"} \
  --prompt_style "${PROMPT_STYLE:-chat_template}" \
  --output_dir "${OUT_DIR:-$EVAL_ROOT/results/lcb}" -- \
  --scenario codegeneration --release_version "${RELEASE:-release_v6}" \
  --n "${N:-10}" --temperature "${TEMP:-0.2}" --max_tokens "${MAX_TOKENS:-2000}" \
  --evaluate --num_process_evaluate "$((SLURM_CPUS_PER_TASK - 2))" \
  ${EXTRA:-}
