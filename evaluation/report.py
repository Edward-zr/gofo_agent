"""Markdown report generator for evaluation runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.compare import compare_scorecards, find_previous_combined

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _num(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _failed_cases(benchmark: dict[str, Any], scenarios: dict[str, Any], *, limit: int = 25) -> list[str]:
    lines: list[str] = []
    for item in benchmark.get("results") or []:
        if item.get("passed"):
            continue
        lines.append(
            f"- `{item.get('id')}` ({item.get('suite')}): "
            f"acc={item.get('accuracy')} tools={item.get('actual_tools')} "
            f"err={item.get('error') or 'score below threshold'}"
        )
        if len(lines) >= limit:
            break
    for item in scenarios.get("results") or []:
        if item.get("passed"):
            continue
        lines.append(
            f"- scenario `{item.get('scenario_id')}`: "
            f"success={item.get('success_score')} tools={item.get('actual_tools')}"
        )
        if len(lines) >= limit * 2:
            break
    return lines or ["- None"]


def _bucket_failures(benchmark: dict[str, Any], scenarios: dict[str, Any]) -> dict[str, list[str]]:
    buckets = {
        "planner": [],
        "sql": [],
        "retrieval": [],
        "clarification": [],
        "memory": [],
        "tool_orchestration": [],
    }
    for item in benchmark.get("results") or []:
        if item.get("passed"):
            continue
        scores = item.get("scores") or {}
        suite = str(item.get("suite") or "")
        label = f"`{item.get('id')}` — {item.get('question')}"
        if scores.get("tool_selection", 1) < 0.5:
            buckets["tool_orchestration"].append(label)
            buckets["planner"].append(label)
        if suite == "sql" or scores.get("sql_success", 1) < 0.5:
            buckets["sql"].append(label)
        if suite == "sop" or scores.get("retrieval_confidence", 1) < 0.5:
            buckets["retrieval"].append(label)
        if "clarif" in str(item.get("expected_behavior") or "").lower():
            buckets["clarification"].append(label)
        if suite in {"memory", "followup"}:
            buckets["memory"].append(label)
    for item in scenarios.get("results") or []:
        if item.get("passed"):
            continue
        label = f"`{item.get('scenario_id')}` — {item.get('description')}"
        if (item.get("planner_accuracy") or 1) < 0.6:
            buckets["planner"].append(label)
        if (item.get("tool_orchestration_accuracy") or 1) < 0.6:
            buckets["tool_orchestration"].append(label)
        if (item.get("memory_accuracy") or 1) < 0.6:
            buckets["memory"].append(label)
        if (item.get("clarification_accuracy") or 1) < 0.6:
            buckets["clarification"].append(label)
    return buckets


def _slowest(benchmark: dict[str, Any], *, limit: int = 10) -> list[str]:
    rows = sorted(
        benchmark.get("results") or [],
        key=lambda r: float(r.get("latency_ms") or 0),
        reverse=True,
    )[:limit]
    return [
        f"- `{r.get('id')}`: {_num(r.get('latency_ms'), 1)} ms — {r.get('question')}"
        for r in rows
    ] or ["- None"]


def _expensive(benchmark: dict[str, Any], *, limit: int = 10) -> list[str]:
    rows = sorted(
        benchmark.get("results") or [],
        key=lambda r: float(r.get("estimated_cost_usd") or 0),
        reverse=True,
    )[:limit]
    return [
        f"- `{r.get('id')}`: ${_num(r.get('estimated_cost_usd'), 6)} "
        f"({r.get('total_tokens')} tokens) — {r.get('question')}"
        for r in rows
    ] or ["- None"]


def build_report_markdown(combined: dict[str, Any], comparison: dict[str, Any]) -> str:
    overall = combined.get("overall") or {}
    bench_sum = (combined.get("benchmark") or {}).get("summary") or {}
    scen_sum = (combined.get("scenarios") or {}).get("summary") or {}
    buckets = _bucket_failures(combined.get("benchmark") or {}, combined.get("scenarios") or {})

    lines = [
        "# GOFO Agent Evaluation Report",
        "",
        f"Generated: `{combined.get('generated_at')}`  ",
        f"Mode: `{combined.get('mode')}`  ",
        "",
        "## Executive Summary",
        "",
        f"**Overall Agent Score:** {_pct(overall.get('overall_score'))}",
        "",
        f"- Benchmark Accuracy: {_pct(overall.get('benchmark_accuracy'))}",
        f"- Scenario Success Rate: {_pct(overall.get('scenario_success_rate'))}",
        f"- Planner Accuracy: {_pct(overall.get('planner_accuracy'))}",
        f"- Tool Orchestration: {_pct(overall.get('tool_orchestration'))}",
        f"- SQL Success: {_pct(overall.get('sql_success'))}",
        f"- Retrieval Confidence: {_pct(overall.get('retrieval_confidence'))}",
        f"- Latency: {_num(overall.get('latency_sec'), 3)} sec",
        f"- Average Tokens: {_num(overall.get('average_tokens'), 1)}",
        f"- Estimated Cost: ${_num(overall.get('estimated_cost_per_query'), 6)}/query",
        "",
        "## Benchmark Results",
        "",
        f"- Questions evaluated: {bench_sum.get('count', 0)}",
        f"- Overall Accuracy: {_pct(bench_sum.get('overall_accuracy'))}",
        f"- SOP Accuracy: {_pct(bench_sum.get('sop_accuracy'))}",
        f"- SQL Accuracy: {_pct(bench_sum.get('sql_accuracy'))}",
        f"- General QA Accuracy: {_pct(bench_sum.get('general_accuracy'))}",
        f"- Follow-up Accuracy: {_pct(bench_sum.get('followup_accuracy'))}",
        f"- Memory Accuracy: {_pct(bench_sum.get('memory_accuracy'))}",
        f"- File Upload Accuracy: {_pct(bench_sum.get('upload_accuracy'))}",
        f"- Tool Selection Accuracy: {_pct(bench_sum.get('tool_selection_accuracy'))}",
        f"- SQL Success Rate: {_pct(bench_sum.get('sql_success_rate'))}",
        f"- Exception Rate: {_pct(bench_sum.get('exception_rate'))}",
        f"- Hallucination Rate: {_pct(bench_sum.get('hallucination_rate'))}",
        "",
        "## Scenario Results",
        "",
        f"- Scenarios evaluated: {scen_sum.get('count', 0)}",
        f"- Multi-turn Success Rate: {_pct(scen_sum.get('multi_turn_success_rate'))}",
        f"- Planner Accuracy: {_pct(scen_sum.get('planner_accuracy'))}",
        f"- Tool Orchestration Accuracy: {_pct(scen_sum.get('tool_orchestration_accuracy'))}",
        f"- Conversation Memory Accuracy: {_pct(scen_sum.get('conversation_memory_accuracy'))}",
        f"- Clarification Accuracy: {_pct(scen_sum.get('clarification_accuracy'))}",
        f"- End-to-End Workflow Success Rate: {_pct(scen_sum.get('end_to_end_workflow_success_rate'))}",
        "",
        "## Regression Analysis",
        "",
    ]

    if not comparison.get("has_previous"):
        lines.append("No previous combined evaluation found — baseline established.")
    else:
        lines.append("### Improvements")
        lines.append("")
        lines.extend([f"- {row}" for row in comparison.get("improvements") or ["- None"]])
        lines.append("")
        lines.append("### Regressions")
        lines.append("")
        lines.extend([f"- {row}" for row in comparison.get("regressions") or ["- None"]])
        lines.append("")
        lines.append("### Metric Deltas")
        lines.append("")
        for key, row in (comparison.get("metrics") or {}).items():
            lines.append(
                f"- **{key}**: {row.get('previous')} → {row.get('current')} ({row.get('arrow')})"
            )

    lines.extend(
        [
            "",
            "## Failed Cases",
            "",
            *_failed_cases(combined.get("benchmark") or {}, combined.get("scenarios") or {}),
            "",
            "## Planner Errors",
            "",
            *([f"- {x}" for x in buckets["planner"][:20]] or ["- None"]),
            "",
            "## SQL Errors",
            "",
            *([f"- {x}" for x in buckets["sql"][:20]] or ["- None"]),
            "",
            "## Retrieval Errors",
            "",
            *([f"- {x}" for x in buckets["retrieval"][:20]] or ["- None"]),
            "",
            "## Clarification Failures",
            "",
            *([f"- {x}" for x in buckets["clarification"][:20]] or ["- None"]),
            "",
            "## Memory Failures",
            "",
            *([f"- {x}" for x in buckets["memory"][:20]] or ["- None"]),
            "",
            "## Tool Orchestration Failures",
            "",
            *([f"- {x}" for x in buckets["tool_orchestration"][:20]] or ["- None"]),
            "",
            "## Slowest Queries",
            "",
            *_slowest(combined.get("benchmark") or {}),
            "",
            "## Most Expensive Queries",
            "",
            *_expensive(combined.get("benchmark") or {}),
            "",
            "## Recommendations",
            "",
        ]
    )

    recs: list[str] = []
    if (bench_sum.get("sql_success_rate") or 100) < 90:
        recs.append("- Investigate SQL failures and schema-retriever coverage for failing KPI questions.")
    if (bench_sum.get("sop_accuracy") or 100) < 90:
        recs.append("- Review RAG confidence thresholds and SOP corpus coverage for weak retrieval cases.")
    if (scen_sum.get("conversation_memory_accuracy") or 100) < 90:
        recs.append("- Strengthen multi-turn entity/memory resolution for follow-up turns.")
    if (scen_sum.get("clarification_accuracy") or 100) < 90:
        recs.append("- Expand Clarification Manager patterns for ambiguous ranking/compare prompts.")
    if (bench_sum.get("avg_latency_ms") or 0) > 3000:
        recs.append("- Profile planner + retrieval latency; consider caching and parallel tool waves.")
    if comparison.get("regressions"):
        recs.append("- Prioritize fixing listed regressions before merging further agent changes.")
    if not recs:
        recs.append("- No critical gaps detected. Expand scenarios for new tools as they ship.")
    lines.extend(recs)
    lines.append("")
    return "\n".join(lines)


def write_report(combined: dict[str, Any], comparison: dict[str, Any] | None = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison = comparison if comparison is not None else {"has_previous": False}
    markdown = build_report_markdown(combined, comparison)
    latest = RESULTS_DIR / "latest_report.md"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = RESULTS_DIR / f"report_{stamp}.md"
    latest.write_text(markdown, encoding="utf-8")
    archived.write_text(markdown, encoding="utf-8")
    return latest


def save_combined(combined: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"combined_{stamp}.json"
    latest = RESULTS_DIR / "latest_combined.json"
    # Avoid huge traces in combined archive for readability; keep summaries + results.
    slim = {
        "generated_at": combined.get("generated_at"),
        "mode": combined.get("mode"),
        "overall": combined.get("overall"),
        "benchmark": {
            "summary": (combined.get("benchmark") or {}).get("summary"),
            "results": (combined.get("benchmark") or {}).get("results"),
            "count": (combined.get("benchmark") or {}).get("count"),
        },
        "scenarios": {
            "summary": (combined.get("scenarios") or {}).get("summary"),
            "results": (combined.get("scenarios") or {}).get("results"),
            "count": (combined.get("scenarios") or {}).get("count"),
        },
        "comparison": combined.get("comparison"),
    }
    previous = find_previous_combined(path)
    # Write first so find_previous can see older files; use existing latest as previous.
    prev_payload = None
    latest_existing = RESULTS_DIR / "latest_combined.json"
    if latest_existing.exists():
        with latest_existing.open(encoding="utf-8") as handle:
            prev_payload = json.load(handle)
    comparison = compare_scorecards(slim, prev_payload)
    slim["comparison"] = comparison
    combined["comparison"] = comparison
    with path.open("w", encoding="utf-8") as handle:
        json.dump(slim, handle, indent=2, default=str)
    with latest.open("w", encoding="utf-8") as handle:
        json.dump(slim, handle, indent=2, default=str)
    write_report(combined, comparison)
    del previous  # reserved for future explicit path chaining
    return path
