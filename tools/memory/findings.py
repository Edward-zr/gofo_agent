"""Compatibility helpers for persistent operational findings."""

from __future__ import annotations

import os
from typing import Any

from tools.memory.long_memory import LongTermMemory


def remember_finding(question: str, finding: dict[str, Any]) -> None:
    """Persist a compact operational finding to SQLite long-term memory."""
    if not finding:
        return
    if os.getenv("PYTEST_CURRENT_TEST"):
        return
    root_cause = finding.get("root_cause") or {}
    LongTermMemory().save_finding(
        hub=_first_dimension_value(root_cause, "hub"),
        driver=_first_dimension_value(root_cause, "driver_name"),
        customer=_first_dimension_value(root_cause, "customer_name"),
        issue=str(root_cause.get("issue") or question),
        metric=str(finding.get("metric") or ""),
        root_cause="; ".join(root_cause.get("main_causes") or []),
        recommendation=str(finding.get("recommendation") or root_cause.get("recommendation") or ""),
        severity="high" if (finding.get("anomaly") or {}).get("is_anomaly") else None,
    )


def load_findings() -> list[dict[str, Any]]:
    """Load persisted operational findings from SQLite."""
    return LongTermMemory().search_similar_issue("", limit=100)


def _first_dimension_value(root_cause: dict[str, Any], key: str) -> str | None:
    prefix = f"{key}:"
    for item in root_cause.get("affected_dimensions") or []:
        text = str(item)
        if text.startswith(prefix):
            return text.split(":", 1)[1].split(",", 1)[0].strip()
    return None
