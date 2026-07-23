"""Unit tests for the evaluation framework (offline / mock mode)."""

from __future__ import annotations

from evaluation.dataset_loader import dataset_counts, load_benchmark_suite, load_scenarios
from evaluation.evaluator import (
    aggregate_benchmark_results,
    evaluate_benchmark_item,
    overall_agent_score,
)
from evaluation.generate_datasets import main as generate_datasets
from evaluation.metrics import keyword_hit_rate, tool_order_accuracy, tool_selection_accuracy
from evaluation.runner import run_benchmarks
from evaluation.scenario_runner import run_scenarios


def test_generate_and_load_datasets() -> None:
    generate_datasets()
    counts = dataset_counts()
    assert counts["sop"] == 200
    assert counts["sql"] == 100
    assert counts["general"] == 50
    assert counts["followup"] == 50
    assert counts["memory"] == 30
    assert counts["upload"] == 20
    assert counts["scenarios"] == 40
    assert len(load_benchmark_suite(limit=5)) == 5
    assert len(load_scenarios(limit=3)) == 3


def test_metric_helpers() -> None:
    assert keyword_hit_rate("Pickup rate improved in Chicago", ["pickup", "chicago"]) == 1.0
    assert tool_selection_accuracy(["SQL", "LLM"], ["SQL", "LLM", "MEMORY"]) > 0.5
    assert tool_order_accuracy(["SQL", "LLM"], ["MEMORY", "SQL", "LLM"]) == 1.0


def test_evaluate_benchmark_item_pass() -> None:
    item = {
        "id": "t1",
        "suite": "sql",
        "question": "Show today's pickup rate",
        "expected_behavior": "sql_analytics",
        "expected_tools": ["SQL", "LLM"],
        "expected_answer_keywords": ["pickup"],
    }
    trace = {
        "answer": "Pickup rate today is 92%.",
        "tools_used": ["SQL", "LLM"],
        "tool_order": ["SQL", "LLM"],
        "sql": "SELECT 1;",
        "sources": [],
        "charts": [],
        "recommendations": [],
        "capability": "sql",
        "latency_ms": 100,
        "total_tokens": 200,
        "estimated_cost_usd": 0.001,
        "error": None,
        "qa_retry_count": 0,
        "reflection_retry_count": 0,
        "requires_clarification": False,
        "memory_state": {},
        "execution_plan": {"steps": [{"tool": "SQL"}, {"tool": "LLM"}]},
    }
    result = evaluate_benchmark_item(item, trace)
    assert result["passed"] is True
    assert result["accuracy"] >= 0.6


def test_mock_benchmark_smoke() -> None:
    generate_datasets()
    payload = run_benchmarks(mode="mock", limit=15)
    assert payload["count"] == 15
    summary = aggregate_benchmark_results(payload["results"])
    assert summary["count"] == 15
    assert "overall_accuracy" in summary


def test_mock_scenario_smoke() -> None:
    generate_datasets()
    payload = run_scenarios(mode="mock", limit=3)
    assert payload["count"] == 3
    assert "multi_turn_success_rate" in payload["summary"]


def test_overall_scorecard() -> None:
    score = overall_agent_score(
        {
            "overall_accuracy": 90.0,
            "tool_selection_accuracy": 90.0,
            "sql_success_rate": 90.0,
            "retrieval_confidence": 90.0,
            "avg_latency_ms": 1000,
            "avg_tokens": 500,
            "avg_cost_usd": 0.001,
            "count": 10,
        },
        {
            "end_to_end_workflow_success_rate": 80.0,
            "planner_accuracy": 90.0,
            "tool_orchestration_accuracy": 90.0,
            "avg_latency_ms": 2000,
            "count": 5,
        },
    )
    assert 0 < score["overall_score"] <= 100
