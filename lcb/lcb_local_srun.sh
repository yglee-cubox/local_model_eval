#!/bin/bash
# Quick interactive LiveCodeBench run on a GPU node (blocks the terminal; use lcb_local.sbatch for full runs).
#
#   bash lcb/lcb_local_srun.sh <model_path> [extra lcb_runner flags...]
#   bash lcb/lcb_local_srun.sh /purestorage/ailab/<me>/ckpt/step-1000 --start_date 2025-03-01 --n 1
#
# env: PARTITION (a100; a10 nodes read /purestorage at only ~15 MB/s)  GPUS (1)  CPUS (16)  MEM (64G)  MODEL_NAME  PROMPT_STYLE  OUT_DIR
set -euo pipefail
MODEL_PATH=${1:?usage: lcb_local_srun.sh <model_path> [lcb_runner flags...]}; shift
EVAL_ROOT=/purestorage/ailab/yglee/workspace/local_model_eval
CPUS=${CPUS:-16}

srun -p "${PARTITION:-a100}" --gres=gpu:"${GPUS:-1}" --cpus-per-task="$CPUS" --mem="${MEM:-64G}" -J lcb-local-srun \
  bash -c 'source '"$EVAL_ROOT"'/env.sh
    "$LCB_REPO/.venv/bin/python" "$EVAL_ROOT/lcb/run_lcb_local.py" \
      --model_path "$0" ${MODEL_NAME:+--model_name "$MODEL_NAME"} --prompt_style "${PROMPT_STYLE:-chat_template}" \
      --output_dir "${OUT_DIR:-$EVAL_ROOT/results/lcb}" -- \
      --scenario codegeneration --evaluate --num_process_evaluate '"$((CPUS - 2))"' "$@"' \
  "$MODEL_PATH" "$@"
