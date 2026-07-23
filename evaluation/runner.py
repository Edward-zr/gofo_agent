"""Independent benchmark question runner."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.agent_adapter import EvalAgent
from evaluation.dataset_loader import DATASETS_DIR, load_benchmark_suite
from evaluation.evaluator import aggregate_benchmark_results, evaluate_benchmark_item

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_benchmarks(
    *,
    mode: str = "mock",
    suites: list[str] | None = None,
    limit: int | None = None,
    agent: EvalAgent | None = None,
) -> dict[str, Any]:
    """Run all (or selected) independent benchmark questions."""
    items = load_benchmark_suite(suites=suites, limit=limit)
    eval_agent = agent or EvalAgent(mode=mode)
    results: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []

    for item in items:
        # Fresh session per independent question
        eval_agent.reset()
        # Optional setup / prior turn for memory & follow-up benchmarks
        setup = item.get("setup_question") or item.get("prior_question")
        if setup:
            eval_agent.ask(str(setup))
        question = str(item.get("question") or "")
        attachment_ids = item.get("attachment_ids")
        if isinstance(attachment_ids, str):
            attachment_ids = [attachment_ids]
        trace = eval_agent.ask(question, attachment_ids=attachment_ids)
        traces.append(trace)
        results.append(evaluate_benchmark_item(item, trace))

    summary = aggregate_benchmark_results(results)
    payload = {
        "type": "benchmark",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "dataset_dir": str(DATASETS_DIR),
        "count": len(results),
        "summary": summary,
        "results": results,
        "traces": traces,
    }
    return payload


def save_benchmark_results(payload: dict[str, Any], *, tag: str | None = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"benchmark_{tag}_{stamp}.json" if tag else f"benchmark_{stamp}.json"
    path = RESULTS_DIR / name
    latest = RESULTS_DIR / "latest_benchmark.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    with latest.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GOFO benchmark evaluation suite")
    parser.add_argument("--mode", choices=["mock", "live"], default="mock")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--suite", action="append", dest="suites")
    args = parser.parse_args()
    out = run_benchmarks(mode=args.mode, suites=args.suites, limit=args.limit)
    path = save_benchmark_results(out)
    print(f"Wrote {path}")
    print(json.dumps(out["summary"], indent=2))
