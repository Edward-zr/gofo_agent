"""Multi-turn scenario evaluation runner."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.agent_adapter import EvalAgent
from evaluation.dataset_loader import DATASETS_DIR, load_scenarios
from evaluation.evaluator import (
    aggregate_scenario_results,
    evaluate_benchmark_item,
    evaluate_scenario,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _turn_as_item(scenario: dict[str, Any], turn: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": f"{scenario.get('scenario_id')}:turn_{index + 1}",
        "suite": "scenario",
        "question": turn.get("user") or turn.get("question") or "",
        "expected_behavior": turn.get("expected_behavior")
        or (scenario.get("expected_behaviors") or [""])[0],
        "expected_tools": turn.get("expected_tools") or scenario.get("expected_tools") or [],
        "expected_answer_keywords": turn.get("expected_answer_keywords") or [],
        "notes": turn.get("notes") or "",
    }


def run_scenarios(
    *,
    mode: str = "mock",
    limit: int | None = None,
    agent: EvalAgent | None = None,
) -> dict[str, Any]:
    """Run multi-turn conversational scenarios."""
    scenarios = load_scenarios(limit=limit)
    eval_agent = agent or EvalAgent(mode=mode)
    scenario_results: list[dict[str, Any]] = []

    for scenario in scenarios:
        eval_agent.reset()
        conversation = scenario.get("conversation") or []
        turn_traces: list[dict[str, Any]] = []
        turn_evals: list[dict[str, Any]] = []
        for index, turn in enumerate(conversation):
            question = str(turn.get("user") or turn.get("question") or "")
            attachment_ids = turn.get("attachment_ids")
            if isinstance(attachment_ids, str):
                attachment_ids = [attachment_ids]
            # Optional scripted clarification reply
            if turn.get("is_clarification_reply") and not question:
                question = str(turn.get("reply") or "1")
            trace = eval_agent.ask(question, attachment_ids=attachment_ids)
            turn_traces.append(trace)
            item = _turn_as_item(scenario, turn, index)
            turn_evals.append(evaluate_benchmark_item(item, trace))
        scenario_results.append(evaluate_scenario(scenario, turn_traces, turn_evals))

    summary = aggregate_scenario_results(scenario_results)
    return {
        "type": "scenarios",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "dataset_dir": str(DATASETS_DIR),
        "count": len(scenario_results),
        "summary": summary,
        "results": scenario_results,
    }


def save_scenario_results(payload: dict[str, Any], *, tag: str | None = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"scenarios_{tag}_{stamp}.json" if tag else f"scenarios_{stamp}.json"
    path = RESULTS_DIR / name
    latest = RESULTS_DIR / "latest_scenarios.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    with latest.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GOFO scenario evaluation suite")
    parser.add_argument("--mode", choices=["mock", "live"], default="mock")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    out = run_scenarios(mode=args.mode, limit=args.limit)
    path = save_scenario_results(out)
    print(f"Wrote {path}")
    print(json.dumps(out["summary"], indent=2))
