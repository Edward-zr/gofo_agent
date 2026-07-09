"""Generate operational recommendations from analytical findings."""

from __future__ import annotations

from typing import Any


def recommend(issue: str | None = None, rows: list[dict[str, Any]] | None = None) -> str:
    """Return a concise operational recommendation without executing actions."""
    normalized = (issue or "").lower()
    rows = rows or []

    if "delay" in normalized or _row_has_status(rows, "Delayed"):
        return (
            "Review driver assignment, check overloaded routes, and investigate "
            "the hubs contributing most to delayed pickups."
        )
    if "fail" in normalized or _row_has_status(rows, "Failed"):
        return (
            "Check failed pickup reasons, review capacity planning, and prioritize "
            "drivers or hubs with repeated failures."
        )
    if "completion" in normalized or "performance" in normalized:
        return (
            "Compare completion, delay, and failure rates by hub and driver, then "
            "rebalance capacity where completion is below target."
        )
    if "volume" in normalized or "package" in normalized:
        return (
            "Check whether package volume increased faster than driver capacity "
            "and review route allocation for overloaded hubs."
        )
    return "Monitor pickup volume, completion rate, delay rate, and failure reasons before changing operations."


def _row_has_status(rows: list[dict[str, Any]], status: str) -> bool:
    return any(str(row.get("status", "")).lower() == status.lower() for row in rows)
