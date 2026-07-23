"""Evaluate a single benchmark item or scenario turn/trace."""

from __future__ import annotations

from typing import Any

from evaluation import metrics


def _expects_tool(expected_tools: list[str], name: str) -> bool:
    return name.upper() in {t.upper() for t in expected_tools}


def evaluate_benchmark_item(item: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    """Score one independent benchmark question against its expectations."""
    expected_tools = list(item.get("expected_tools") or [])
    expected_behavior = item.get("expected_behavior") or ""
    keywords = item.get("expected_answer_keywords") or []

    keyword_score = metrics.keyword_hit_rate(trace.get("answer") or "", keywords)
    tool_sel = metrics.tool_selection_accuracy(expected_tools, trace.get("tools_used") or [])
    tool_ord = metrics.tool_order_accuracy(expected_tools, trace.get("tool_order") or [])
    behavior = metrics.behavior_match(expected_behavior, trace)
    sql_ok = metrics.sql_success(
        trace,
        expected_sql=_expects_tool(expected_tools, "SQL")
        or "sql" in expected_behavior.lower(),
    )
    retrieval = metrics.retrieval_confidence_score(
        trace,
        expected_rag=_expects_tool(expected_tools, "RAG")
        or "rag" in expected_behavior.lower()
        or "sop" in expected_behavior.lower(),
    )
    halluc = metrics.hallucination_risk(trace)
    has_error = 1.0 if trace.get("error") else 0.0
    retries = float(trace.get("qa_retry_count") or 0) + float(
        trace.get("reflection_retry_count") or 0
    )

    # Functional correctness blend (no exact answer match).
    accuracy = metrics.mean(
        [
            keyword_score,
            behavior,
            tool_sel,
            0.5 + 0.5 * tool_ord,  # order is softer
            sql_ok,
        ]
    )
    if has_error:
        accuracy *= 0.2

    passed = accuracy >= 0.6 and not trace.get("error")

    return {
        "id": item.get("id"),
        "suite": item.get("suite"),
        "question": item.get("question"),
        "passed": passed,
        "accuracy": round(accuracy, 4),
        "scores": {
            "answer_correctness": round(keyword_score, 4),
            "behavior": round(behavior, 4),
            "tool_selection": round(tool_sel, 4),
            "tool_order": round(tool_ord, 4),
            "sql_success": round(sql_ok, 4),
            "retrieval_confidence": round(retrieval, 4),
            "hallucination_risk": round(halluc, 4),
        },
        "expected_behavior": expected_behavior,
        "expected_tools": expected_tools,
        "actual_tools": trace.get("tools_used") or [],
        "tool_order": trace.get("tool_order") or [],
        "latency_ms": trace.get("latency_ms"),
        "total_tokens": trace.get("total_tokens"),
        "estimated_cost_usd": trace.get("estimated_cost_usd"),
        "error": trace.get("error"),
        "retries": retries,
        "requires_clarification": bool(trace.get("requires_clarification")),
        "answer_preview": (trace.get("answer") or "")[:400],
        "sql": trace.get("sql") or "",
        "confidence_level": trace.get("confidence_level"),
        "notes": item.get("notes") or "",
    }


def evaluate_scenario(
    scenario: dict[str, Any],
    turn_traces: list[dict[str, Any]],
    turn_evals: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score a multi-turn scenario from per-turn traces and evaluations."""
    expected_tools = list(scenario.get("expected_tools") or [])
    expected_behaviors = list(scenario.get("expected_behaviors") or [])

    turn_pass_rate = metrics.mean(
        [1.0 if item.get("passed") else 0.0 for item in turn_evals]
    ) if turn_evals else 0.0

    all_tools: list[str] = []
    for trace in turn_traces:
        for tool in trace.get("tool_order") or trace.get("tools_used") or []:
            all_tools.append(str(tool))
    tool_sel = metrics.tool_selection_accuracy(expected_tools, set(all_tools))
    tool_ord = metrics.tool_order_accuracy(expected_tools, all_tools)

    behavior_scores = []
    for behavior in expected_behaviors:
        # Behavior may be satisfied on any turn
        best = max(
            (metrics.behavior_match(behavior, trace) for trace in turn_traces),
            default=0.0,
        )
        behavior_scores.append(best)
    behavior_score = metrics.mean(behavior_scores) if behavior_scores else turn_pass_rate

    memory_ok = 0.0
    if turn_traces:
        # Multi-turn memory: later turns should resolve or keep entities/context
        later = turn_traces[1:] or turn_traces
        memory_signals = []
        for trace in later:
            mem = trace.get("memory_state") or {}
            resolved = trace.get("resolved_question") or ""
            memory_signals.append(
                1.0
                if (
                    mem.get("memory_turns")
                    or mem.get("current_entities")
                    or (resolved and resolved != trace.get("question"))
                )
                else 0.0
            )
        memory_ok = metrics.mean(memory_signals)

    clarification_ok = 1.0
    if any("clarif" in str(b).lower() for b in expected_behaviors):
        clarification_ok = 1.0 if any(t.get("requires_clarification") for t in turn_traces) else 0.0

    planner_ok = metrics.mean(
        [
            1.0 if (t.get("execution_plan") or {}).get("steps") or t.get("tools_used") else 0.0
            for t in turn_traces
        ]
    ) if turn_traces else 0.0

    errors = sum(1 for t in turn_traces if t.get("error"))
    e2e_success = turn_pass_rate >= 0.6 and errors == 0 and tool_sel >= 0.4

    overall = metrics.mean(
        [
            turn_pass_rate,
            behavior_score,
            tool_sel,
            0.5 + 0.5 * tool_ord,
            memory_ok,
            clarification_ok,
            planner_ok,
        ]
    )

    return {
        "scenario_id": scenario.get("scenario_id"),
        "description": scenario.get("description"),
        "passed": bool(e2e_success),
        "success_score": round(overall, 4),
        "multi_turn_success_rate": round(turn_pass_rate, 4),
        "planner_accuracy": round(planner_ok, 4),
        "tool_orchestration_accuracy": round(metrics.mean([tool_sel, tool_ord]), 4),
        "memory_accuracy": round(memory_ok, 4),
        "clarification_accuracy": round(clarification_ok, 4),
        "end_to_end_success": bool(e2e_success),
        "turn_count": len(turn_traces),
        "errors": errors,
        "total_latency_ms": round(sum(float(t.get("latency_ms") or 0) for t in turn_traces), 2),
        "total_tokens": int(sum(int(t.get("total_tokens") or 0) for t in turn_traces)),
        "estimated_cost_usd": round(
            sum(float(t.get("estimated_cost_usd") or 0) for t in turn_traces), 6
        ),
        "turns": turn_evals,
        "expected_tools": expected_tools,
        "actual_tools": list(dict.fromkeys(all_tools)),
        "expected_behaviors": expected_behaviors,
    }


def aggregate_benchmark_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll up benchmark item scores into suite + overall metrics."""
    if not results:
        return {
            "overall_accuracy": 0.0,
            "suite_accuracy": {},
            "tool_selection_accuracy": 0.0,
            "sql_success_rate": 0.0,
            "retrieval_confidence": 0.0,
            "avg_latency_ms": 0.0,
            "avg_tokens": 0.0,
            "avg_cost_usd": 0.0,
            "exception_rate": 0.0,
            "hallucination_rate": 0.0,
            "pass_rate": 0.0,
            "count": 0,
        }

    by_suite: dict[str, list[float]] = {}
    for item in results:
        by_suite.setdefault(str(item.get("suite") or "unknown"), []).append(
            float(item.get("accuracy") or 0)
        )

    suite_accuracy = {
        suite: metrics.pct(metrics.mean(scores)) for suite, scores in by_suite.items()
    }

    return {
        "overall_accuracy": metrics.pct(metrics.mean(r.get("accuracy") or 0 for r in results)),
        "suite_accuracy": suite_accuracy,
        "sop_accuracy": suite_accuracy.get("sop", 0.0),
        "sql_accuracy": suite_accuracy.get("sql", 0.0),
        "general_accuracy": suite_accuracy.get("general", 0.0),
        "followup_accuracy": suite_accuracy.get("followup", 0.0),
        "memory_accuracy": suite_accuracy.get("memory", 0.0),
        "upload_accuracy": suite_accuracy.get("upload", 0.0),
        "tool_selection_accuracy": metrics.pct(
            metrics.mean((r.get("scores") or {}).get("tool_selection") or 0 for r in results)
        ),
        "sql_success_rate": metrics.pct(
            metrics.mean((r.get("scores") or {}).get("sql_success") or 0 for r in results)
        ),
        "retrieval_confidence": metrics.pct(
            metrics.mean(
                (r.get("scores") or {}).get("retrieval_confidence") or 0 for r in results
            )
        ),
        "avg_latency_ms": round(metrics.mean(r.get("latency_ms") or 0 for r in results), 2),
        "avg_tokens": round(metrics.mean(r.get("total_tokens") or 0 for r in results), 1),
        "avg_cost_usd": round(metrics.mean(r.get("estimated_cost_usd") or 0 for r in results), 6),
        "exception_rate": metrics.pct(
            metrics.mean(1.0 if r.get("error") else 0.0 for r in results)
        ),
        "hallucination_rate": metrics.pct(
            metrics.mean((r.get("scores") or {}).get("hallucination_risk") or 0 for r in results)
        ),
        "pass_rate": metrics.pct(metrics.mean(1.0 if r.get("passed") else 0.0 for r in results)),
        "count": len(results),
        "failed_ids": [r.get("id") for r in results if not r.get("passed")],
    }


def aggregate_scenario_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "multi_turn_success_rate": 0.0,
            "planner_accuracy": 0.0,
            "tool_orchestration_accuracy": 0.0,
            "conversation_memory_accuracy": 0.0,
            "clarification_accuracy": 0.0,
            "end_to_end_workflow_success_rate": 0.0,
            "avg_latency_ms": 0.0,
            "avg_cost_usd": 0.0,
            "count": 0,
            "failed_ids": [],
        }
    return {
        "multi_turn_success_rate": metrics.pct(
            metrics.mean(r.get("multi_turn_success_rate") or 0 for r in results)
        ),
        "planner_accuracy": metrics.pct(
            metrics.mean(r.get("planner_accuracy") or 0 for r in results)
        ),
        "tool_orchestration_accuracy": metrics.pct(
            metrics.mean(r.get("tool_orchestration_accuracy") or 0 for r in results)
        ),
        "conversation_memory_accuracy": metrics.pct(
            metrics.mean(r.get("memory_accuracy") or 0 for r in results)
        ),
        "clarification_accuracy": metrics.pct(
            metrics.mean(r.get("clarification_accuracy") or 0 for r in results)
        ),
        "end_to_end_workflow_success_rate": metrics.pct(
            metrics.mean(1.0 if r.get("end_to_end_success") else 0.0 for r in results)
        ),
        "avg_latency_ms": round(
            metrics.mean(r.get("total_latency_ms") or 0 for r in results), 2
        ),
        "avg_cost_usd": round(
            metrics.mean(r.get("estimated_cost_usd") or 0 for r in results), 6
        ),
        "count": len(results),
        "failed_ids": [r.get("scenario_id") for r in results if not r.get("passed")],
    }


def overall_agent_score(benchmark_summary: dict[str, Any], scenario_summary: dict[str, Any]) -> dict[str, Any]:
    """Combine benchmark + scenario into one overall scorecard."""
    bench = (benchmark_summary.get("overall_accuracy") or 0) / 100.0
    scen = (scenario_summary.get("end_to_end_workflow_success_rate") or 0) / 100.0
    planner = (scenario_summary.get("planner_accuracy") or benchmark_summary.get("tool_selection_accuracy") or 0) / 100.0
    tools = (
        (
            (benchmark_summary.get("tool_selection_accuracy") or 0)
            + (scenario_summary.get("tool_orchestration_accuracy") or 0)
        )
        / 200.0
    )
    sql = (benchmark_summary.get("sql_success_rate") or 0) / 100.0
    retrieval = (benchmark_summary.get("retrieval_confidence") or 0) / 100.0

    overall = metrics.mean([bench, scen, planner, tools, sql, retrieval])
    return {
        "overall_score": metrics.pct(overall),
        "benchmark_accuracy": benchmark_summary.get("overall_accuracy", 0.0),
        "scenario_success_rate": scenario_summary.get("end_to_end_workflow_success_rate", 0.0),
        "planner_accuracy": metrics.pct(planner),
        "tool_orchestration": metrics.pct(tools),
        "sql_success": benchmark_summary.get("sql_success_rate", 0.0),
        "retrieval_confidence": benchmark_summary.get("retrieval_confidence", 0.0),
        "latency_sec": round(
            (
                (benchmark_summary.get("avg_latency_ms") or 0)
                + (scenario_summary.get("avg_latency_ms") or 0)
            )
            / 2000.0,
            3,
        )
        if (benchmark_summary.get("count") or scenario_summary.get("count"))
        else 0.0,
        "average_tokens": benchmark_summary.get("avg_tokens", 0.0),
        "estimated_cost_per_query": benchmark_summary.get("avg_cost_usd", 0.0),
    }
