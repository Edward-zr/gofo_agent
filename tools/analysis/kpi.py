"""KPI dictionary and operations overview analysis."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import config
from core.models import QueryResponse
from tools.analysis.anomaly import detect_anomalies
from tools.analysis.recommender import recommend
from tools.sql.business_date import get_latest_business_date
from tools.sql.executor import execute

KPI_DICTIONARY_PATH = config.PROJECT_ROOT / "config" / "kpi_dictionary.yaml"


@lru_cache(maxsize=1)
def load_kpi_dictionary(path: Path = KPI_DICTIONARY_PATH) -> dict[str, Any]:
    """Load the KPI dictionary from the project YAML file."""
    if not path.exists():
        return {}
    return _parse_simple_yaml(path.read_text(encoding="utf-8"))


def answer_operations_overview(question: str) -> QueryResponse:
    """Answer broad operations-health questions using KPI definitions."""
    latest_date = get_latest_business_date().isoformat()
    sql_statements = _overview_sql(latest_date)
    overview_rows = execute(sql_statements[0])
    hub_rows = execute(sql_statements[1])
    driver_rows = execute(sql_statements[2])
    trend_rows = execute(sql_statements[3])
    all_rows = overview_rows + hub_rows + driver_rows
    kpi_dictionary = load_kpi_dictionary()
    anomaly = detect_anomalies(trend_rows)
    recommendation = recommend("operations performance", all_rows)
    answer = _format_overview_answer(overview_rows, hub_rows, driver_rows, anomaly, recommendation)

    return QueryResponse(
        question=question,
        answer=answer,
        sources=[],
        capability="sql",
        latest_business_date=latest_date,
        generated_sql="\n\n".join(sql_statements),
        sql_rows=all_rows,
        needs_sql=True,
        needs_rag=False,
        execution_order=["sql", "kpi", "anomaly", "recommendation"],
        planning_capability="sql",
        planning_intent="operations_kpi_summary",
        business_metric="operations_health",
        analysis_dimension="hub",
        date_range=latest_date,
        kpi_summary={"dictionary": kpi_dictionary, "checked_metrics": list(kpi_dictionary.keys())},
        anomaly=anomaly,
        recommendation=recommendation,
    )


def _overview_sql(latest_date: str) -> list[str]:
    return [
        f"""
SELECT
    pickup_date,
    COUNT(*) AS pickup_count,
    SUM(package_count) AS package_volume,
    SUM(CASE WHEN status='Completed' THEN 1 ELSE 0 END) AS completed_pickups,
    SUM(CASE WHEN status='Failed' THEN 1 ELSE 0 END) AS failed_pickups,
    SUM(CASE WHEN status='Delayed' THEN 1 ELSE 0 END) AS delayed_pickups,
    ROUND(100.0 * SUM(CASE WHEN status='Completed' THEN 1 ELSE 0 END) / COUNT(*), 2) AS completion_rate,
    ROUND(100.0 * SUM(CASE WHEN status='Delayed' THEN 1 ELSE 0 END) / COUNT(*), 2) AS delay_rate,
    ROUND(100.0 * SUM(CASE WHEN status='Failed' THEN 1 ELSE 0 END) / COUNT(*), 2) AS failure_rate
FROM pickups
WHERE pickup_date='{latest_date}'
GROUP BY pickup_date;
""".strip(),
        f"""
SELECT
    d.hub,
    COUNT(*) AS pickup_count,
    SUM(p.package_count) AS package_volume,
    ROUND(100.0 * SUM(CASE WHEN p.status='Completed' THEN 1 ELSE 0 END) / COUNT(*), 2) AS completion_rate
FROM pickups p
JOIN drivers d ON p.driver_id=d.driver_id
WHERE p.pickup_date='{latest_date}'
GROUP BY d.hub
ORDER BY completion_rate DESC, pickup_count DESC;
""".strip(),
        f"""
SELECT
    d.driver_name,
    d.hub,
    COUNT(*) AS pickup_count,
    SUM(CASE WHEN p.status='Completed' THEN 1 ELSE 0 END) AS completed_pickups
FROM pickups p
JOIN drivers d ON p.driver_id=d.driver_id
WHERE p.pickup_date='{latest_date}'
GROUP BY d.driver_id, d.driver_name, d.hub
ORDER BY completed_pickups DESC, pickup_count DESC
LIMIT 10;
""".strip(),
        """
SELECT
    pickup_date,
    COUNT(*) AS pickup_count,
    ROUND(100.0 * SUM(CASE WHEN status='Completed' THEN 1 ELSE 0 END) / COUNT(*), 2) AS completion_rate,
    ROUND(100.0 * SUM(CASE WHEN status='Delayed' THEN 1 ELSE 0 END) / COUNT(*), 2) AS delay_rate,
    ROUND(100.0 * SUM(CASE WHEN status='Failed' THEN 1 ELSE 0 END) / COUNT(*), 2) AS failure_rate
FROM pickups
GROUP BY pickup_date
ORDER BY pickup_date;
""".strip(),
    ]


def _format_overview_answer(
    overview_rows: list[dict[str, Any]],
    hub_rows: list[dict[str, Any]],
    driver_rows: list[dict[str, Any]],
    anomaly: dict[str, Any],
    recommendation: str,
) -> str:
    if not overview_rows:
        return "No operational KPI records were found."
    overview = overview_rows[0]
    best_hub = hub_rows[0] if hub_rows else {}
    top_driver = driver_rows[0] if driver_rows else {}
    lines = [
        "Operational KPI summary:",
        f"- Pickup volume: {overview.get('pickup_count', 0)} pickups and {overview.get('package_volume', 0)} packages.",
        f"- Completion rate: {overview.get('completion_rate', 0)}%.",
        f"- Failed pickups: {overview.get('failed_pickups', 0)}.",
        f"- Delayed pickups: {overview.get('delayed_pickups', 0)}.",
    ]
    if best_hub:
        lines.append(f"- Best hub by current ranking: {best_hub.get('hub')}.")
    if top_driver:
        lines.append(f"- Top driver: {top_driver.get('driver_name')} from {top_driver.get('hub')}.")
    if anomaly.get("is_anomaly"):
        first = anomaly["anomalies"][0]
        lines.append(f"- Abnormal behavior detected: {first.get('type')} in {first.get('affected_dimension')}.")
    lines.append(f"Recommendation: {recommendation}")
    return "\n".join(lines)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small KPI dictionary YAML subset without adding dependencies."""
    result: dict[str, Any] = {}
    current_key: str | None = None
    current_list_key: str | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        if not raw_line.startswith(" "):
            current_key = raw_line.rstrip(":")
            result[current_key] = {}
            current_list_key = None
            continue
        if current_key is None:
            continue
        stripped = raw_line.strip()
        if stripped.endswith(":"):
            current_list_key = stripped.rstrip(":")
            result[current_key][current_list_key] = []
        elif stripped.startswith("- ") and current_list_key:
            result[current_key][current_list_key].append(stripped[2:])
        elif ":" in stripped:
            key, value = stripped.split(":", 1)
            result[current_key][key.strip()] = value.strip().strip('"')
            current_list_key = None
    return result
