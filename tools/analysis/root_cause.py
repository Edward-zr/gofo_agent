"""Root-cause analysis for SQL operational results."""

from __future__ import annotations

from typing import Any

from tools.analysis.recommender import recommend


def analyze_root_cause(rows: list[dict[str, Any]], question: str = "") -> dict[str, Any]:
    """Summarize likely causes from SQL result rows."""
    if not rows:
        return {
            "issue": "No operational rows were available for root-cause analysis.",
            "main_causes": [],
            "affected_dimensions": [],
            "recommendation": recommend(question, rows),
        }

    issue = _infer_issue(rows, question)
    causes = _main_causes(rows)
    affected = _affected_dimensions(rows)
    return {
        "issue": issue,
        "main_causes": causes,
        "affected_dimensions": affected,
        "recommendation": recommend(issue, rows),
    }


def _infer_issue(rows: list[dict[str, Any]], question: str) -> str:
    normalized = question.lower()
    if "delay" in normalized:
        return "Delay rate increased or delayed pickups are elevated."
    if "fail" in normalized:
        return "Failed pickups are concentrated in specific operational dimensions."
    if "drop" in normalized or "decrease" in normalized or "performance" in normalized:
        return "Operational performance decreased."
    if any("completion_rate" in row for row in rows):
        return "Completion performance needs review."
    return "Operational metric requires investigation."


def _main_causes(rows: list[dict[str, Any]]) -> list[str]:
    causes: list[str] = []
    highest_delay = _highest(rows, "delayed_pickups")
    highest_failed = _highest(rows, "failed_pickups")
    highest_volume = _highest(rows, "package_count") or _highest(rows, "packages")

    if highest_delay:
        causes.append(f"Delay volume is highest for {_dimension_label(highest_delay)}.")
    if highest_failed:
        causes.append(f"Failed pickups are highest for {_dimension_label(highest_failed)}.")
    if highest_volume:
        causes.append(f"Package volume is concentrated in {_dimension_label(highest_volume)}.")
    if not causes:
        causes.append("The available rows do not show a single dominant cause.")
    return causes


def _affected_dimensions(rows: list[dict[str, Any]]) -> list[str]:
    dimensions: list[str] = []
    for key in ("hub", "driver_name", "customer_name", "pickup_date", "status", "reason"):
        values = [str(row[key]) for row in rows if row.get(key)]
        if values:
            dimensions.append(f"{key}: {', '.join(values[:5])}")
    return dimensions


def _highest(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get(key) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: _as_float(row.get(key)))


def _dimension_label(row: dict[str, Any]) -> str:
    return str(
        row.get("hub")
        or row.get("driver_name")
        or row.get("customer_name")
        or row.get("pickup_date")
        or row.get("status")
        or "the returned result"
    )


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
