"""One-command entrypoint: python -m evaluation

Examples:
  python -m evaluation                 # mock mode, full suites
  python -m evaluation --mode live     # live GOFOAgent (needs API keys / DB)
  python -m evaluation --limit 20      # smoke sample
  python -m evaluation --benchmarks-only
  python -m evaluation --scenarios-only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is importable when launched as a module.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.dataset_loader import dataset_counts
from evaluation.evaluator import overall_agent_score
from evaluation.report import save_combined
from evaluation.runner import run_benchmarks, save_benchmark_results
from evaluation.scenario_runner import run_scenarios, save_scenario_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GOFO Operations Intelligence Agent — Evaluation Framework"
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default="mock",
        help="mock = offline heuristic agent; live = real GOFOAgent",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max items per runner")
    parser.add_argument(
        "--suite",
        action="append",
        dest="suites",
        help="Benchmark suite to include (repeatable): sop,sql,general,followup,memory,upload",
    )
    parser.add_argument("--benchmarks-only", action="store_true")
    parser.add_argument("--scenarios-only", action="store_true")
    parser.add_argument(
        "--generate-datasets",
        action="store_true",
        help="Regenerate evaluation/datasets/*.json then exit",
    )
    args = parser.parse_args(argv)

    if args.generate_datasets:
        from evaluation.generate_datasets import main as gen_main

        gen_main()
        return 0

    counts = dataset_counts()
    print("Dataset counts:", json.dumps(counts))
    if sum(v for k, v in counts.items() if k != "scenarios") == 0:
        print("No benchmark datasets found. Generating defaults...")
        from evaluation.generate_datasets import main as gen_main

        gen_main()

    benchmark_payload = {"summary": {}, "results": [], "count": 0}
    scenario_payload = {"summary": {}, "results": [], "count": 0}

    if not args.scenarios_only:
        print(f"Running benchmarks (mode={args.mode})...")
        benchmark_payload = run_benchmarks(
            mode=args.mode,
            suites=args.suites,
            limit=args.limit,
        )
        path = save_benchmark_results(benchmark_payload)
        print(f"  benchmark results → {path}")

    if not args.benchmarks_only:
        print(f"Running scenarios (mode={args.mode})...")
        scenario_payload = run_scenarios(mode=args.mode, limit=args.limit)
        path = save_scenario_results(scenario_payload)
        print(f"  scenario results → {path}")

    overall = overall_agent_score(
        benchmark_payload.get("summary") or {},
        scenario_payload.get("summary") or {},
    )
    combined = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "overall": overall,
        "benchmark": benchmark_payload,
        "scenarios": scenario_payload,
    }
    combined_path = save_combined(combined)
    report_path = Path(__file__).resolve().parent / "results" / "latest_report.md"

    print("")
    print("======== GOFO Evaluation Scorecard ========")
    print(f"Overall Score:           {overall.get('overall_score')}%")
    print(f"Benchmark Accuracy:      {overall.get('benchmark_accuracy')}%")
    print(f"Scenario Success Rate:   {overall.get('scenario_success_rate')}%")
    print(f"Planner Accuracy:        {overall.get('planner_accuracy')}%")
    print(f"Tool Orchestration:      {overall.get('tool_orchestration')}%")
    print(f"SQL Success:             {overall.get('sql_success')}%")
    print(f"Retrieval Confidence:    {overall.get('retrieval_confidence')}%")
    print(f"Latency:                 {overall.get('latency_sec')} sec")
    print(f"Average Tokens:          {overall.get('average_tokens')}")
    print(f"Estimated Cost:          ${overall.get('estimated_cost_per_query')}/query")
    print("===========================================")
    print(f"Combined JSON: {combined_path}")
    print(f"Markdown report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
