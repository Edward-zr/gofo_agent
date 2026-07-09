"""Statistical anomaly detection for operational metrics."""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any

from core.models import QueryResponse
from tools.analysis.recommender import recommend
from tools.sql.business_date import get_latest_business_date
from tools.sql.executor import execute


def detect_anomalies(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Detect simple operational anomalies using moving average, percent change, and z-score.

    No ML models are used.
    """
    if not rows:
        return {"is_anomaly": False, "anomalies": [], "method": "statistics"}

    anomalies: list[dict[str, Any]] = []
    anomalies.extend(_detect_rate_spikes(rows, "delay_rate", "delay spike"))
    anomalies.extend(_detect_rate_spikes(rows, "failure_rate", "failure spike"))
    anomalies.extend(_detect_rate_drops(rows, "completion_rate", "completion drop"))
    anomalies.extend(_detect_volume_spikes(rows))

    return {
        "is_anomaly": bool(anomalies),
        "anomalies": anomalies,
        "method": "moving average, percentage change, z-score",
    }


def answer_anomaly_question(question: str) -> QueryResponse:
    """Run statistics-only anomaly detection for recent operational behavior."""
    latest_date = get_latest_business_date().isoformat()
    sql = """
SELECT
    pickup_date,
    COUNT(*) AS pickup_count,
    ROUND(100.0 * SUM(CASE WHEN status='Completed' THEN 1 ELSE 0 END) / COUNT(*), 2) AS completion_rate,
    ROUND(100.0 * SUM(CASE WHEN status='Delayed' THEN 1 ELSE 0 END) / COUNT(*), 2) AS delay_rate,
    ROUND(100.0 * SUM(CASE WHEN status='Failed' THEN 1 ELSE 0 END) / COUNT(*), 2) AS failure_rate
FROM pickups
GROUP BY pickup_date
ORDER BY pickup_date;
""".strip()
    rows = execute(sql)
    anomaly = detect_anomalies(rows)
    recommendation = recommend("abnormal operations", rows)
    answer = _format_anomaly_answer(anomaly, recommendation)
    return QueryResponse(
        question=question,
        answer=answer,
        sources=[],
        capability="sql",
        latest_business_date=latest_date,
        generated_sql=sql,
        sql_rows=rows,
        execution_order=["sql", "anomaly", "recommendation"],
        planning_capability="sql",
        planning_intent="anomaly_detection",
        business_metric="operations_anomaly",
        date_range=latest_date,
        anomaly=anomaly,
        recommendation=recommendation,
    )


def _format_anomaly_answer(anomaly: dict[str, Any], recommendation: str) -> str:
    if not anomaly.get("is_anomaly"):
        return f"No abnormal operational behavior was detected from the recent trend. Recommendation: {recommendation}"
    lines = ["Abnormal operational behavior detected."]
    for item in anomaly.get("anomalies", [])[:3]:
        lines.append(
            f"- {item.get('type')} for {item.get('affected_dimension')}: "
            f"{item.get('latest')} vs baseline {item.get('baseline')} "
            f"({item.get('percentage_change')}% change)."
        )
    lines.append(f"Recommendation: {recommendation}")
    return "\n".join(lines)


def _detect_rate_spikes(rows: list[dict[str, Any]], key: str, label: str) -> list[dict[str, Any]]:
    values = [_as_float(row.get(key)) for row in rows if row.get(key) is not None]
    if len(values) < 2:
        return []
    baseline = mean(values[:-1])
    latest = values[-1]
    if baseline <= 0:
        return []
    percentage_change = 100.0 * (latest - baseline) / baseline
    z_score = _z_score(latest, values[:-1])
    if latest >= baseline * 1.5 or percentage_change >= 50 or z_score >= 2:
        return [_anomaly(label, key, latest, baseline, percentage_change, z_score, rows[-1])]
    return []


def _detect_rate_drops(rows: list[dict[str, Any]], key: str, label: str) -> list[dict[str, Any]]:
    values = [_as_float(row.get(key)) for row in rows if row.get(key) is not None]
    if len(values) < 2:
        return []
    baseline = mean(values[:-1])
    latest = values[-1]
    percentage_change = 100.0 * (latest - baseline) / baseline if baseline else 0
    z_score = _z_score(latest, values[:-1])
    if latest <= baseline * 0.9 or percentage_change <= -10 or z_score <= -2:
        return [_anomaly(label, key, latest, baseline, percentage_change, z_score, rows[-1])]
    return []


def _detect_volume_spikes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key = "pickup_count"
    values = [_as_float(row.get(key)) for row in rows if row.get(key) is not None]
    if len(values) < 2:
        return []
    baseline = mean(values[:-1])
    latest = values[-1]
    if baseline <= 0:
        return []
    percentage_change = 100.0 * (latest - baseline) / baseline
    z_score = _z_score(latest, values[:-1])
    if latest >= baseline * 1.5 or percentage_change >= 50 or z_score >= 2:
        return [_anomaly("volume spike", key, latest, baseline, percentage_change, z_score, rows[-1])]
    return []


def _anomaly(
    label: str,
    metric: str,
    latest: float,
    baseline: float,
    percentage_change: float,
    z_score: float,
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": label,
        "metric": metric,
        "latest": round(latest, 2),
        "baseline": round(baseline, 2),
        "percentage_change": round(percentage_change, 2),
        "z_score": round(z_score, 2) if math.isfinite(z_score) else 0.0,
        "affected_dimension": row.get("hub") or row.get("driver_name") or row.get("pickup_date"),
    }


def _z_score(value: float, baseline_values: list[float]) -> float:
    deviation = pstdev(baseline_values)
    if deviation == 0:
        return 0.0
    return (value - mean(baseline_values)) / deviation


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
