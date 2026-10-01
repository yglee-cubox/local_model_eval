"""Collect small per-model, per-benchmark summaries from results/ into summaries/ (committed to git).

  python3 summarize_results.py [--results results] [--out summaries] [--include-test]

summaries/lcb/<model>__<scenario>.json   pass@1 overall / by difficulty / per problem, run settings
summaries/swepro/<model>.json            resolved count and rate, resolved ids, agent exit status counts
summaries/README.md                      one table per benchmark

Only finished runs are collected: LCB needs `*_eval.json`, SWE-bench Pro needs `summary.txt`.
Folders named `test-*` are skipped unless --include-test.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

DIFFS = ["easy", "medium", "hard"]


def lcb(model_dir):
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
            "model": model_dir.name,
            "scenario": m.group(1),
            "n": int(m.group(2)),
            "temperature": float(m.group(3)),
            "n_problems": len(per),
            "contest_dates": [dates[0], dates[-1]],
            "pass@1": round(json.load(open(ev))[0]["pass@1"], 4),
            "by_difficulty": by_diff,
            "per_problem": {p["question_id"]: round(float(p["pass@1"]), 4) for p in per},
        })
    return rows


def swepro(model_dir):
    summary = (model_dir / "summary.txt").read_text()
    ok, total = map(int, re.search(r"resolved (\d+)/(\d+)", summary).groups())
    rewards = {d.name: (d / "verifier/reward.txt").read_text().strip()
               for d in sorted((model_dir / "grades").iterdir()) if (d / "verifier/reward.txt").exists()}
    status, calls = Counter(), []
    for f in sorted((model_dir / "agent").glob("*/result.json")):
        r = json.load(open(f))
        status[r.get("exit_status", "unknown").split(":")[0]] += 1  # "crash:ContextWindowExceededError: ..." -> "crash"
        calls.append(r.get("n_calls", 0))
    return {
        "benchmark": "swe-bench-pro-v2",
        "model": model_dir.name,
        "n_instances": total,
        "resolved": ok,
        "resolve_rate": round(ok / max(total, 1), 4),
        "agent_exit_status": dict(status.most_common()),
        "mean_agent_calls": round(sum(calls) / max(len(calls), 1), 1),
        "resolved_ids": sorted(k for k, v in rewards.items() if v == "1"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parent
    p.add_argument("--results", type=Path, default=root / "results")
    p.add_argument("--out", type=Path, default=root / "summaries")
    p.add_argument("--include-test", action="store_true")
    args = p.parse_args()

    def models(d):
        return [m for m in sorted(d.iterdir()) if m.is_dir() and (args.include_test or not m.name.startswith("test-"))]

    lcb_rows, swe_rows = [], []
    for m in models(args.results / "lcb/output"):
        lcb_rows += lcb(m) or print(f"skip lcb/{m.name}: no *_eval.json") or []
    for m in models(args.results / "swepro"):
        if (m / "summary.txt").exists():
            swe_rows.append(swepro(m))
        else:
            print(f"skip swepro/{m.name}: no summary.txt (unfinished?)")

    for r in lcb_rows:
        f = args.out / "lcb" / f"{r['model']}__{r['scenario']}_{r['n']}_{r['temperature']}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(r, indent=1) + "\n")
    for r in swe_rows:
        f = args.out / "swepro" / f"{r['model']}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(r, indent=1) + "\n")

    md = ["# 결과 요약", "", "`python3 summarize_results.py` 로 `results/` 에서 다시 만듭니다. 원본 결과는 저장소에 없습니다.", "",
          "## LiveCodeBench (codegeneration)", "",
          "| 모델 | pass@1 | easy | medium | hard | 문제 수 | 기간 | n | temp |", "|---|---|---|---|---|---|---|---|---|"]
    for r in lcb_rows:
        d = [f"{r['by_difficulty'][k]['pass@1']:.3f}" if k in r["by_difficulty"] else "-" for k in DIFFS]
        md.append(f"| {r['model']} | **{r['pass@1']:.3f}** | {' | '.join(d)} | {r['n_problems']} | "
                  f"{r['contest_dates'][0]} ~ {r['contest_dates'][1]} | {r['n']} | {r['temperature']} |")
    md += ["", "## SWE-bench Pro V2", "",
           "공식 프로토콜 아님 (컨테이너에 네트워크와 /purestorage 접근 있음). 체크포인트끼리 비교용입니다.", "",
           "| 모델 | resolved | rate | 인스턴스 | 에이전트 종료 상태 |", "|---|---|---|---|---|"]
    for r in swe_rows:
        st = ", ".join(f"{k} {v}" for k, v in r["agent_exit_status"].items())
        md.append(f"| {r['model']} | {r['resolved']} | **{r['resolve_rate']:.2%}** | {r['n_instances']} | {st} |")
    (args.out / "README.md").write_text("\n".join(md) + "\n")
    print(f"wrote {len(lcb_rows)} lcb, {len(swe_rows)} swepro summaries -> {args.out}")


if __name__ == "__main__":
    main()
