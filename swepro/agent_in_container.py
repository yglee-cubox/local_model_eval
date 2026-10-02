"""Run mini-swe-agent on one SWE-bench Pro V2 instance, *inside* that instance's container.

Mirrors v2/tooling/host_mini_swe.py of the official repo (same config file, same 60 s command timeout,
same `git add -A && git diff --cached` patch capture), but the sandbox is the enroot container we are
running in, so commands are executed locally (cwd=/app). The model is any OpenAI-compatible endpoint,
normally the vLLM server started by swepro_local.sbatch.

Writes to --out_dir: model.patch, trajectory.json, result.json
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model


def load_image_env():
    """pyxis starts the container with the host's environment, which hides the image's ENV (e.g. PATH with
    /usr/local/go/bin). enroot writes the image ENV to /etc/environment, so re-apply it for the agent's commands."""
    try:
        lines = Path("/etc/environment").read_text().splitlines()
    except OSError:
        return
    for line in lines:
        key, sep, value = line.partition("=")
        if sep and key.strip() and not key.startswith("#"):
            os.environ[key.strip()] = value.strip().strip('"')


def main():
    load_image_env()
    p = argparse.ArgumentParser()
    p.add_argument("--task_dir", required=True, help="v2-harbor/tasks/<instance_id>")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--config", required=True, help="mini-swe-agent yaml (official: mini_textbased.yaml)")
    p.add_argument("--model", required=True, help="litellm model name, e.g. hosted_vllm/local")
    p.add_argument("--api_base", required=True)
    p.add_argument("--step_limit", type=int, default=250)
    p.add_argument("--wall_time", type=int, default=2940, help="seconds; task.toml agent timeout is 3000")
    p.add_argument("--command_timeout", type=int, default=60)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    a = p.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    instruction = (Path(a.task_dir) / "instruction.md").read_text()
    config = yaml.safe_load(Path(a.config).read_text())

    agent_cfg = dict(config.get("agent", {}))
    agent_cfg.update(step_limit=a.step_limit, cost_limit=0, wall_time_limit_seconds=a.wall_time,
                     output_path=out / "trajectory.json")

    model_cfg = dict(config.get("model", {}))
    kwargs = dict(model_cfg.get("model_kwargs", {}))
    kwargs.pop("reasoning_effort", None)  # API-model setting; not meaningful for a local vLLM model
    kwargs.update(api_base=a.api_base, api_key="EMPTY")
    if a.temperature is not None:
        kwargs["temperature"] = a.temperature
    if a.max_tokens is not None:
        kwargs["max_tokens"] = a.max_tokens
    # Extra sampling params as JSON, e.g. SAMPLING_KWARGS='{"top_p":0.8,"presence_penalty":1.5}'
    kwargs.update(json.loads(os.environ.get("SAMPLING_KWARGS") or "{}"))
    model_cfg.update(model_kwargs=kwargs, cost_tracking="ignore_errors")

    cwd = "/app" if Path("/app").is_dir() else "/testbed"
    env_cfg = config.get("environment", {})
    env = LocalEnvironment(cwd=cwd, env=env_cfg.get("env", {}), timeout=a.command_timeout)

    agent = DefaultAgent(get_model(a.model, model_cfg), env, **agent_cfg)
    t0 = time.time()
    try:
        result = agent.run(instruction)
        exit_status = result.get("exit_status")
    except Exception as e:  # never lose the patch because the agent loop crashed
        exit_status = f"crash:{type(e).__name__}: {e}"

    diff = subprocess.run(f"cd {cwd} && git add -A && git diff --cached", shell=True,
                          capture_output=True, text=True, timeout=120).stdout
    subprocess.run(f"cd {cwd} && git reset -q", shell=True, timeout=60)
    (out / "model.patch").write_text(diff)
    (out / "result.json").write_text(json.dumps(
        {"exit_status": exit_status, "n_calls": agent.n_calls, "patch_bytes": len(diff),
         "seconds": round(time.time() - t0)}, indent=1))
    print(f"[agent] exit_status={exit_status} n_calls={agent.n_calls} patch_bytes={len(diff)}", flush=True)


if __name__ == "__main__":
    main()
