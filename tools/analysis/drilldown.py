"""Automatic drilldown analysis for WHY questions."""

from __future__ import annotations

from typing import Any

from core.models import QueryResponse
from tools.analysis.root_cause import analyze_root_cause
from tools.analysis.recommender import recommend
from tools.sql.business_date import get_latest_business_date, resolve_relative_dates
from tools.sql.executor import execute


def run_drilldown(question: str) -> QueryResponse:
    """
    Run deterministic SQL drilldowns for operational why questions.

    Level 1: metric status
    Level 2: affected hub
    Level 3: driver/customer/status reason
    """
    latest_date = get_latest_business_date()
    rewritten_question = resolve_relative_dates(question, latest_date)
    date_filter = _date_filter(rewritten_question, latest_date.isoformat())
    sql_statements = _build_drilldown_sql(question, date_filter)
    rows_by_level: list[dict[str, Any]] = []
    for level, sql in enumerate(sql_statements, start=1):
        rows_by_level.append({"level": level, "sql": sql, "rows": execute(sql)})

    flattened_rows = [row for level in rows_by_level for row in level["rows"]]
    root_cause = analyze_root_cause(flattened_rows, question)
    recommendation = root_cause.get("recommendation") or recommend(question, flattened_rows)
    answer = _format_drilldown_answer(root_cause, recommendation)

    return QueryResponse(
        question=question,
        answer=answer,
        sources=[],
        capability="sql",
        rewritten_question=rewritten_question,
        latest_business_date=latest_date.isoformat(),
        generated_sql="\n\n".join(sql_statements),
        sql_rows=flattened_rows,
        execution_order=["sql", "root_cause", "recommendation"],
        planning_capability="sql",
        planning_intent="root_cause_drilldown",
        root_cause=root_cause,
        recommendation=recommendation,
    )


def _build_drilldown_sql(question: str, date_filter: str) -> list[str]:
    normalized = question.lower()
    status_filter = ""
    if "fail" in normalized:
        status_filter = "AND p.status = 'Failed'"
    elif "delay" in normalized:
        status_filter = "AND p.status = 'Delayed'"

    return [
        f"""
SELECT
    COUNT(*) AS pickup_count,
    SUM(CASE WHEN status='Completed' THEN 1 ELSE 0 END) AS completed_pickups,
    SUM(CASE WHEN status='Delayed' THEN 1 ELSE 0 END) AS delayed_pickups,
    SUM(CASE WHEN status='Failed' THEN 1 ELSE 0 END) AS failed_pickups,
    SUM(package_count) AS package_count
FROM pickups
WHERE {date_filter};
""".strip(),
        f"""
SELECT
    d.hub,
    COUNT(*) AS pickup_count,
    SUM(CASE WHEN p.status='Delayed' THEN 1 ELSE 0 END) AS delayed_pickups,
    SUM(CASE WHEN p.status='Failed' THEN 1 ELSE 0 END) AS failed_pickups,
    SUM(p.package_count) AS package_count
FROM pickups p
JOIN drivers d ON p.driver_id=d.driver_id
WHERE {date_filter.replace('pickup_date', 'p.pickup_date')}
{status_filter}
GROUP BY d.hub
ORDER BY failed_pickups DESC, delayed_pickups DESC, package_count DESC
LIMIT 5;
""".strip(),
        f"""
SELECT
    d.driver_name,
    d.hub,
    COUNT(*) AS pickup_count,
    SUM(CASE WHEN p.status='Delayed' THEN 1 ELSE 0 END) AS delayed_pickups,
    SUM(CASE WHEN p.status='Failed' THEN 1 ELSE 0 END) AS failed_pickups
FROM pickups p
JOIN drivers d ON p.driver_id=d.driver_id
WHERE {date_filter.replace('pickup_date', 'p.pickup_date')}
{status_filter}
GROUP BY d.driver_id, d.driver_name, d.hub
ORDER BY failed_pickups DESC, delayed_pickups DESC, pickup_count DESC
LIMIT 5;
""".strip(),
        f"""
SELECT
    e.reason,
    COUNT(*) AS failed_pickups
FROM pickups p
JOIN exceptions e ON p.pickup_id=e.pickup_id
WHERE {date_filter.replace('pickup_date', 'p.pickup_date')}
AND p.status='Failed'
GROUP BY e.reason
ORDER BY failed_pickups DESC
LIMIT 5;
""".strip(),
    ]


def _date_filter(rewritten_question: str, latest_date: str) -> str:
    import re

    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", rewritten_question)
    if len(dates) >= 2:
        return f"pickup_date BETWEEN '{dates[0]}' AND '{dates[-1]}'"
    if dates:
        return f"pickup_date = '{dates[0]}'"
    return f"pickup_date = '{latest_date}'"


def _format_drilldown_answer(root_cause: dict[str, Any], recommendation: str) -> str:
    causes = root_cause.get("main_causes") or []
    lines = [str(root_cause.get("issue") or "Operational issue identified.")]
    if causes:
        lines.append("")
        lines.append("Main reasons:")
        lines.extend(f"{index}. {cause}" for index, cause in enumerate(causes, start=1))
    lines.append("")
    lines.append(f"Recommendation: {recommendation}")
    return "\n".join(lines)
