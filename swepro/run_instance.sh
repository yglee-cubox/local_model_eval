#!/bin/bash
# One SWE-bench Pro V2 instance: agent in a fresh container -> model.patch -> grading in another fresh container.
# Called by swepro_local.sbatch inside its allocation (needs API_BASE, SERVED_MODEL, OUT_DIR from there).
#   bash run_instance.sh <instance_id>
set -uo pipefail
source /purestorage/ailab/yglee/workspace/local_model_eval/env.sh
IID=${1:?instance_id}
TASKS=$BENCH_DATA/swe-bench-pro/v2-harbor/tasks
T=$TASKS/$IID
A=$OUT_DIR/agent/$IID
mkdir -p "$A"

if [ -f "$A/result.json" ]; then
  echo "[$IID] agent already done, skipping (delete $A to rerun)"
else
  IMAGE=$(sed -n 's/^docker_image = "\(.*\)"/\1/p' "$T/task.toml"); IMAGE=${IMAGE/\//#}
  # /purestorage is mounted in every enroot container, so the agent venv and paths below are visible inside.
  # Some images are Alpine (musl, no glibc loader) and cannot exec the glibc venv python directly, so we start it
  # through our own glibc loader + libs (glibc-runtime/). --library-path only affects the python process itself;
  # commands the agent runs (bash, git, go, ...) still use the container's own libc.
  GLIBC=$EVAL_ROOT/glibc-runtime/lib
  timeout -k 30 "${AGENT_TIMEOUT:-3000}" \
  srun --ntasks=1 --overlap --container-image="$IMAGE" --container-remap-root \
    "$GLIBC/ld-linux-x86-64.so.2" --library-path "$GLIBC" --argv0 "$EVAL_ROOT/.venv-agent/bin/python" \
    "$(readlink -f "$EVAL_ROOT/.venv-agent/bin/python")" "$EVAL_ROOT/swepro/agent_in_container.py" \
      --task_dir "$T" --out_dir "$A" --config "$AGENT_CONFIG" \
      --model "hosted_vllm/$SERVED_MODEL" --api_base "$API_BASE" \
      --step_limit "${STEP_LIMIT:-250}" --wall_time "$(( ${AGENT_TIMEOUT:-3000} - 60 ))" \
      ${TEMP:+--temperature "$TEMP"} ${MAX_TOKENS:+--max_tokens "$MAX_TOKENS"} \
    > "$A/agent.log" 2>&1
  [ -f "$A/result.json" ] || echo '{"exit_status": "container_timeout_or_crash"}' > "$A/result.json"
  [ -f "$A/model.patch" ] || : > "$A/model.patch"
fi

if [ "${GRADE:-1}" = 1 ]; then
  bash "$BENCH_DATA/swe-bench-pro/scripts/grade_enroot.sh" "$IID" "$A/model.patch" "$OUT_DIR/grades" > "$A/grade.log" 2>&1
fi
echo "[$IID] $(tr -d '\n ' < "$A/result.json" | cut -c1-120) reward=$(cat "$OUT_DIR/grades/$IID/verifier/reward.txt" 2>/dev/null || echo -)"
