"""Load SQLite schema metadata for SQL planning."""

from __future__ import annotations

import sqlite3
from functools import lru_cache

import config
from tools.sql.schema import SCHEMA


@lru_cache(maxsize=1)
def get_database_schema() -> str:
    """
    Return live SQLite table and column metadata for planner prompts.

    If the configured database is unavailable, return the static documented
    schema so tests and prompt construction remain usable.
    """
    database_path = config.SQLITE_DATABASE
    if not database_path.exists():
        return SCHEMA

    connection = sqlite3.connect(str(database_path))
    try:
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        ).fetchall()
        table_names = [str(row[0]) for row in table_rows]
        lines = ["Live SQLite schema", "------------------"]
        for table_name in table_names:
            columns = connection.execute(f"PRAGMA table_info({table_name});").fetchall()
            lines.append("")
            lines.append(f"TABLE {table_name}:")
            for column in columns:
                column_name = column[1]
                column_type = column[2] or "TEXT"
                lines.append(f"- {column_name} {column_type}")
        return "\n".join(lines)
    finally:
        connection.close()
