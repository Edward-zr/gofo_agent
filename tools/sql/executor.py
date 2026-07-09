"""Execute SQL queries against the GOFO SQLite analytics database."""

from __future__ import annotations

import sqlite3
from typing import Any

import config


def execute(sql: str) -> list[dict[str, Any]]:
    """
    Execute a SQL statement and return all result rows.

    Raises FileNotFoundError when the configured database file does not exist.
    """
    database_path = config.SQLITE_DATABASE
    if not database_path.exists():
        raise FileNotFoundError(
            f"SQLite database not found at {database_path}. "
            "Ensure the analytics database is set up before running SQL queries."
        )

    connection = sqlite3.connect(str(database_path))
    connection.row_factory = sqlite3.Row
    try:
        cursor = connection.execute(sql)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()
