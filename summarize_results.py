"""Collect small per-model, per-benchmark summaries from results/ into summaries/ (committed to git).

  python3 summarize_results.py [--results results] [--logs logs] [--out summaries] [--include-test]

summaries/lcb/<run>__<scenario>.json   pass@1 overall / by difficulty / per problem, run settings
summaries/swepro/<run>.json            resolved count and rate, resolved ids, agent exit status counts
summaries/README.md                    one table per benchmark, grouped by base model with thinking on/off

<run> is the MODEL_NAME of the run (e.g. Qwen3.8-27B-nothink). Only finished runs are collected: LCB needs
`*_eval.json`, SWE-bench Pro needs `summary.txt`. Folders named `test-*` are skipped unless --include-test.

Thinking mode is read from the run's logs, in this order:
  1. vLLM `non-default args` -> default_chat_template_kwargs.enable_thinking (set with VLLM_ARGS)
  2. otherwise the checkpoint's chat template default (Qwen3: on unless enable_thinking=false; gemma-4: off)
  3. "n/a" if the template has no enable_thinking switch (or the checkpoint path is unknown)
"""
import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

DIFFS = ["easy", "medium", "hard"]


def job_id(path):
    m = re.search(r"-(\d+)\.(?:log|out)$", path.name)
    return int(m.group(1)) if m else 0


def vllm_args(log):
    """dict printed by `vllm serve` as `non-default args: {...}` (None if no log)."""
    if log is None:
        return None
    for line in open(log, errors="replace"):
        if "non-default args:" in line:
            return ast.literal_eval(line.split("non-default args:", 1)[1].strip())
    return None


def template_thinking_default(model_path):
    p = Path(model_path or "")
    t = p / "chat_template.jinja"
    if t.exists():
        tpl = t.read_text()
    elif (p / "tokenizer_config.json").exists():
        tpl = str(json.load(open(p / "tokenizer_config.json")).get("chat_template", ""))
    else:
        return None
    if "enable_thinking" not in tpl:
        return None
    if re.search(r"enable_thinking\s*\|\s*default\(\s*false", tpl, re.I):
        return False
    if re.search(r"enable_thinking is (?:defined and enable_thinking is false|undefined)", tpl):
        return True
    return None


def thinking(vargs, model_path):
    kw = (vargs or {}).get("default_chat_template_kwargs") or {}
    if "enable_thinking" in kw:
        return ("on" if kw["enable_thinking"] else "off"), "vllm default_chat_template_kwargs"
    d = template_thinking_default(model_path)
    return ("n/a", "no enable_thinking in chat template") if d is None else (("on" if d else "off"), "chat template default")


def base_model(model_path, run):
    if not model_path:
        return run
    m = re.search(r"models--([^/]+)--([^/]+)", model_path)  # HF hub cache: models--Qwen--Qwen3.8-27B/snapshots/<sha>
    return f"{m.group(1)}/{m.group(2)}" if m else Path(model_path).name


def server_args(vargs):
    drop = {"host", "port", "model", "model_tag", "served_model_name"}
    return {k: v for k, v in (vargs or {}).items() if k not in drop}


def lcb_run_info(run, results, logs):
    """lcb_runner args (+ extra request params) from the latest logs/lcb-*.out of this run, and its vLLM log."""
    cands = []
    for f in logs.glob("lcb-*.out"):
        text = open(f, errors="replace").read()
        m = re.search(rf"lcb_runner args: (--model {re.escape(run)} .*)", text)
        if m:
            cands.append((job_id(f), m.group(1), re.search(r"extra request params: (\{.*\})", text)))
    if not cands:
        return {}, None, None
    jid, argline, extra = max(cands)
    toks = argline.split()
    args = {toks[i][2:]: toks[i + 1] for i in range(len(toks) - 1)
            if toks[i].startswith("--") and not toks[i + 1].startswith("--")}
    if extra:
        args["request_params"] = ast.literal_eval(extra.group(1))
    vlog = results / "lcb" / f"vllm-{run}-{jid}.log"
    return args, args.get("local_model_path"), vlog if vlog.exists() else None


def lcb(model_dir, results, logs):
    run = model_dir.name
    args, model_path, vlog = lcb_run_info(run, results, logs)
    vargs = vllm_args(vlog)
    think, think_src = thinking(vargs, model_path)
    rows = []
    for ev in sorted(model_dir.glob("Scenario.*_eval.json")):
        scenario = ev.name[: -len("_eval.json")]
        m = re.match(r"Scenario\.(\w+?)_(\d+)_([\d.]+)$", scenario)
        per = json.load(open(ev.with_name(scenario + "_eval_all.json")))
        by_diff = {}
        for d in DIFFS:
            ps = [float(p["pass@1"]) for p in per if p["difficulty"] == d]
            if ps:
                by_diff[d] = {"pass@1": round(sum(ps) / len(ps), 4), "n_problems": len(ps)}
        dates = sorted(p["contest_date"][:10] for p in per)
        rows.append({
            "benchmark": "livecodebench",
            "run": run,
            "base_model": base_model(model_path, run),
            "thinking": think,
            "thinking_source": think_src,
            "scenario": m.group(1),
            "release": args.get("release_version"),
            "n": int(m.group(2)),
            "temperature": float(m.group(3)),
            "top_p": float(args["top_p"]) if "top_p" in args else None,
            "max_tokens": int(args["max_tokens"]) if "max_tokens" in args else None,
            "request_params": args.get("request_params", {}),
            "vllm_args": server_args(vargs),
            "n_problems": len(per),
            "contest_dates": [dates[0], dates[-1]],
            "pass@1": round(json.load(open(ev))[0]["pass@1"], 4),
            "by_difficulty": by_diff,
            "per_problem": {p["question_id"]: round(float(p["pass@1"]), 4) for p in per},
        })
    return rows


def swepro(model_dir, logs):
    run = model_dir.name
    summary = (model_dir / "summary.txt").read_text()
    ok, total = map(int, re.search(r"resolved (\d+)/(\d+)", summary).groups())
    rewards = {d.name: (d / "verifier/reward.txt").read_text().strip()
               for d in sorted((model_dir / "grades").iterdir()) if (d / "verifier/reward.txt").exists()}
    status, calls, secs = Counter(), [], []
    for f in sorted((model_dir / "agent").glob("*/result.json")):
        r = json.load(open(f))
        status[r.get("exit_status", "unknown").split(":")[0]] += 1  # "crash:ContextWindowExceededError: ..." -> "crash"
        calls.append(r.get("n_calls", 0))
        secs.append(r.get("seconds", 0))
    vlogs = sorted(model_dir.glob("vllm-*.log"), key=job_id)
    vargs = vllm_args(vlogs[-1]) if vlogs else None
    model_path = (vargs or {}).get("model")
    # first line: model=... name=... gpus=... ids=<file> (<count>) conc=... ; re-runs of a few failed ids reuse the
    # same name, so take the job that covered the most ids (latest on ties)
    headers = []
    for f in logs.glob("swepro-local-*.out"):
        first = open(f, errors="replace").readline()
        if f" name={run} " in first:
            n_ids = re.search(r"ids=\S+ \((\d+)\)", first)
            headers.append((int(n_ids.group(1)) if n_ids else 0, job_id(f), dict(re.findall(r"(\w+)=(\S+)", first))))
    header = max(headers, key=lambda h: h[:2])[2] if headers else {}
    think, think_src = thinking(vargs, model_path or header.get("model"))
    return {
        "benchmark": "swe-bench-pro-v2",
        "run": run,
        "base_model": base_model(model_path or header.get("model"), run),
        "thinking": think,
        "thinking_source": think_src,
        "instance_set": Path(header["ids"]).stem if "ids" in header else None,
        "gpus": int(header["gpus"]) if "gpus" in header else None,
        "concurrency": int(header["conc"]) if "conc" in header else None,
        "vllm_args": server_args(vargs),
        "n_instances": total,
        "resolved": ok,
        "resolve_rate": round(ok / max(total, 1), 4),
        "agent_exit_status": dict(status.most_common()),
        "mean_agent_calls": round(sum(calls) / max(len(calls), 1), 1),
        "mean_agent_seconds": round(sum(secs) / max(len(secs), 1)),
        "resolved_ids": sorted(k for k, v in rewards.items() if v == "1"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parent
    p.add_argument("--results", type=Path, default=root / "results")
    p.add_argument("--logs", type=Path, default=root / "logs")
    p.add_argument("--out", type=Path, default=root / "summaries")
    p.add_argument("--include-test", action="store_true")
    args = p.parse_args()

    def runs(d):
        return [m for m in sorted(d.iterdir()) if m.is_dir() and (args.include_test or not m.name.startswith("test-"))]

    lcb_rows, swe_rows = [], []
    for m in runs(args.results / "lcb/output"):
        lcb_rows += lcb(m, args.results, args.logs) or print(f"skip lcb/{m.name}: no *_eval.json") or []
    for m in runs(args.results / "swepro"):
        if (m / "summary.txt").exists():
            swe_rows.append(swepro(m, args.logs))
        else:
            print(f"skip swepro/{m.name}: no summary.txt (unfinished?)")

    for r in lcb_rows:
        f = args.out / "lcb" / f"{r['run']}__{r['scenario']}_{r['n']}_{r['temperature']}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(r, indent=1) + "\n")
    for r in swe_rows:
        f = args.out / "swepro" / f"{r['run']}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(r, indent=1) + "\n")

    order = {"on": 0, "off": 1, "n/a": 2}
    lcb_rows.sort(key=lambda r: (r["base_model"].lower(), order[r["thinking"]]))
    swe_rows.sort(key=lambda r: (r["base_model"].lower(), order[r["thinking"]]))
    md = ["# 결과 요약", "",
          "`python3 summarize_results.py` 로 `results/` 와 `logs/` 에서 다시 만듭니다. 원본 결과는 저장소에 없습니다.",
          "실행별 설정(vLLM 인자, 샘플링, 문제별 / 인스턴스별 결과)은 같은 폴더의 JSON 에 있습니다.", "",
          "thinking: vLLM `--default-chat-template-kwargs '{\"enable_thinking\": ...}'` 로 지정한 값, 지정하지 않았으면 체크포인트 chat template 의 기본값",
          "(Qwen3 계열은 켜짐, gemma-4 는 꺼짐). `n/a` 는 chat template 에 thinking 스위치가 없는 모델입니다.", "",
          "## LiveCodeBench (codegeneration)", "",
          "| 모델 | thinking | 실행 이름 | pass@1 | easy | medium | hard | 문제 수 | 기간 | n | temp | top_p | max_tokens |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in lcb_rows:
        d = [f"{r['by_difficulty'][k]['pass@1']:.3f}" if k in r["by_difficulty"] else "-" for k in DIFFS]
        md.append(f"| {r['base_model']} | {r['thinking']} | {r['run']} | **{r['pass@1']:.3f}** | {' | '.join(d)} | "
                  f"{r['n_problems']} | {r['contest_dates'][0]} ~ {r['contest_dates'][1]} | {r['n']} | {r['temperature']} | "
                  f"{r['top_p'] if r['top_p'] is not None else '-'} | {r['max_tokens'] or '-'} |")
    md += ["", "## SWE-bench Pro V2", "",
           "공식 프로토콜 아님 (컨테이너에 네트워크와 /purestorage 접근 있음). 체크포인트끼리 비교용입니다.", "",
           "| 모델 | thinking | 실행 이름 | resolved | rate | 인스턴스 | 평균 호출 수 | 에이전트 종료 상태 |",
           "|---|---|---|---|---|---|---|---|"]
    for r in swe_rows:
        st = ", ".join(f"{k} {v}" for k, v in r["agent_exit_status"].items())
        md.append(f"| {r['base_model']} | {r['thinking']} | {r['run']} | {r['resolved']} | **{r['resolve_rate']:.2%}** | "
                  f"{r['n_instances']} ({r['instance_set'] or '-'}) | {r['mean_agent_calls']} | {st} |")
    (args.out / "README.md").write_text("\n".join(md) + "\n")
    print(f"wrote {len(lcb_rows)} lcb, {len(swe_rows)} swepro summaries -> {args.out}")


if __name__ == "__main__":
    main()
