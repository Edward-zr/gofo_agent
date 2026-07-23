"""CLI for running Prompt Experiment Framework evaluations.

Examples:
  python -m evaluation.prompt_experiments
  python -m evaluation.prompt_experiments --experiment exp_planner_prompt
  python -m evaluation.prompt_experiments --activate planner.planner_prompt:v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.prompt_experiments import (
    load_experiments,
    run_prompt_experiment,
    write_experiment_report,
)
from core.prompt_manager import get_prompt_manager, reset_prompt_manager
from core.prompt_registry import reset_prompt_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GOFO Prompt Experiment Framework")
    parser.add_argument("--experiment", help="Run a single experiment id")
    parser.add_argument("--mode", choices=["mock", "live"], default=None)
    parser.add_argument(
        "--activate",
        help="Switch active prompt version without code changes (key:version)",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist --activate into registry.yaml",
    )
    parser.add_argument("--list", action="store_true", help="List registered prompts")
    args = parser.parse_args(argv)

    if args.list:
        for row in get_prompt_manager().list_prompts():
            print(json.dumps(row))
        return 0

    if args.activate:
        if ":" not in args.activate:
            print("--activate requires key:version", file=sys.stderr)
            return 2
        key, version = args.activate.split(":", 1)
        get_prompt_manager().set_active_version(key.strip(), version.strip(), persist=args.persist)
        print(f"Active version set: {key.strip()} → {version.strip()} (persist={args.persist})")
        return 0

    experiments = load_experiments()
    if args.experiment:
        experiments = [exp for exp in experiments if exp.id == args.experiment]
        if not experiments:
            print(f"Unknown experiment: {args.experiment}", file=sys.stderr)
            return 2

    for experiment in experiments:
        print(f"Running experiment {experiment.id} on {experiment.prompt_key}...")
        # Fresh registry/manager between experiments
        reset_prompt_registry()
        reset_prompt_manager()
        payload = run_prompt_experiment(experiment, mode=args.mode)
        path = write_experiment_report(payload)
        print(f"  recommended={payload.get('recommended_version')}")
        print(f"  report → {path}")
        print(json.dumps(payload.get("results"), indent=2, default=str)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
