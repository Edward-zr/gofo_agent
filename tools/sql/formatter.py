"""Format SQL query results for display."""

from __future__ import annotations

from typing import Any


def format_result(rows: list[dict[str, Any]]) -> str:
    """Pretty-print SQL rows as a Markdown table."""
    if not rows:
        return "No records found."

    headers = list(rows[0].keys())
    header_line = "| " + " | ".join(str(header) for header in headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    data_lines = [
        "| " + " | ".join(str(row.get(header, "")) for header in headers) + " |"
        for row in rows
    ]

    return "\n".join([header_line, separator_line, *data_lines])
