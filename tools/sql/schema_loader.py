"""Load SQLite schema metadata for SQL planning.

Prefer SchemaRegistry for structured metadata. This module keeps a text helper
for backward-compatible callers/tests.
"""

from __future__ import annotations

from functools import lru_cache

from tools.sql.schema import SCHEMA
from tools.sql.schema_registry import get_schema_registry


@lru_cache(maxsize=1)
def get_database_schema() -> str:
    """
    Return live SQLite table and column metadata as text.

    Prefer SchemaRegistry + SchemaRetriever for planner prompts. This helper
    remains for debug / legacy callers.
    """
    try:
        snapshot = get_schema_registry().get_snapshot()
        if not snapshot.tables:
            return SCHEMA
        lines = ["Live SQLite schema", "------------------"]
        for table_name in snapshot.table_names():
            table = snapshot.tables[table_name]
            lines.append("")
            lines.append(f"TABLE {table_name}:")
            for column in table.columns:
                lines.append(f"- {column.name} {column.data_type}")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return SCHEMA


def refresh_database_schema_cache() -> None:
    """Clear text-schema cache after registry refresh."""
    get_database_schema.cache_clear()
    get_schema_registry().refresh()
