"""Resolve operational dates from analytics data instead of the system clock."""

from __future__ import annotations

import re
import sqlite3
from datetime import date, timedelta
from functools import lru_cache

import config


@lru_cache(maxsize=1)
def get_latest_business_date() -> date:
    """
    Return the latest pickup_date available in the analytics database.

    Treat this date as operational "today" for relative date resolution.
    """
    database_path = config.SQLITE_DATABASE
    if not database_path.exists():
        raise FileNotFoundError(
            f"SQLite database not found at {database_path}. "
            "Ensure the analytics database is set up before running SQL queries."
        )

    connection = sqlite3.connect(str(database_path))
    try:
        row = connection.execute("SELECT MAX(pickup_date) FROM pickups").fetchone()
        if not row or row[0] is None:
            raise ValueError("No pickup dates found in the database.")
        return date.fromisoformat(str(row[0]))
    finally:
        connection.close()


def resolve_relative_dates(question: str, latest_date: date) -> str:
    """
    Rewrite common temporal expressions using the latest business date.

    Only rewrites known relative date phrases; all other text is unchanged.
    """
    this_week_start = latest_date - timedelta(days=latest_date.weekday())
    last_week_end = this_week_start - timedelta(days=1)
    last_week_start = last_week_end - timedelta(days=6)
    this_month = latest_date.strftime("%B %Y")
    last_month = (latest_date.replace(day=1) - timedelta(days=1)).strftime("%B %Y")

    replacements: list[tuple[str, str]] = [
        (
            r"\blast week\b",
            f"week of {last_week_start.isoformat()} to {last_week_end.isoformat()}",
        ),
        (
            r"\bthis week\b",
            f"week of {this_week_start.isoformat()} to {latest_date.isoformat()}",
        ),
        (r"\blast month\b", last_month),
        (r"\bthis month\b", this_month),
        (r"\byesterday\b", (latest_date - timedelta(days=1)).isoformat()),
        (r"\btoday\b", latest_date.isoformat()),
    ]

    rewritten = question
    for pattern, replacement in replacements:
        rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
    return rewritten
