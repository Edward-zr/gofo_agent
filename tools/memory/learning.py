"""Learn recurring operational patterns from persistent findings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.memory.database import get_connection
from tools.memory.long_memory import LongTermMemory


def detect_repeated_patterns(
    memory: LongTermMemory | None = None,
    *,
    database_path: Path | None = None,
    threshold: int = 3,
) -> list[dict[str, Any]]:
    """Create learned patterns when the same operational problem recurs."""
    memory = memory or LongTermMemory(database_path=database_path)
    patterns: list[dict[str, Any]] = []
    patterns.extend(_hub_patterns(memory, database_path, threshold))
    patterns.extend(_driver_patterns(memory, database_path, threshold))
    return patterns


def _hub_patterns(
    memory: LongTermMemory,
    database_path: Path | None,
    threshold: int,
) -> list[dict[str, Any]]:
    with get_connection(database_path) as connection:
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT hub, metric, COUNT(*) AS issue_count, MAX(recommendation) AS previous_solution
                FROM operational_findings
                WHERE hub IS NOT NULL AND hub != ''
                GROUP BY hub, metric
                HAVING COUNT(*) >= ?;
                """,
                (threshold,),
            )
        ]
    created: list[dict[str, Any]] = []
    for row in rows:
        metric = row.get("metric") or "operational issues"
        pattern_name = f"{row['hub']} recurring {metric}"
        if memory.save_pattern(
            pattern_name=pattern_name,
            description=f"{row['hub']} has recurring {metric} findings.",
            trigger_condition=f"{row['hub']} appears in {row['issue_count']} findings for {metric}.",
            previous_solution=row.get("previous_solution"),
        ):
            created.append({"pattern_name": pattern_name, **row})
    return created


def _driver_patterns(
    memory: LongTermMemory,
    database_path: Path | None,
    threshold: int,
) -> list[dict[str, Any]]:
    with get_connection(database_path) as connection:
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT driver, metric, COUNT(*) AS issue_count, MAX(recommendation) AS previous_solution
                FROM operational_findings
                WHERE driver IS NOT NULL AND driver != ''
                GROUP BY driver, metric
                HAVING COUNT(*) >= ?;
                """,
                (threshold,),
            )
        ]
    created: list[dict[str, Any]] = []
    for row in rows:
        metric = row.get("metric") or "operational issues"
        pattern_name = f"{row['driver']} recurring {metric}"
        if memory.save_pattern(
            pattern_name=pattern_name,
            description=f"{row['driver']} has repeated {metric} findings.",
            trigger_condition=f"{row['driver']} appears in {row['issue_count']} findings for {metric}.",
            previous_solution=row.get("previous_solution"),
        ):
            created.append({"pattern_name": pattern_name, **row})
    return created
