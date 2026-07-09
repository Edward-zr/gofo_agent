"""Run sample analytics queries against the GOFO demo SQLite database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "data" / "gofo_demo.db"


def _print_table(title: str, rows: list[sqlite3.Row]) -> None:
    """Print query results as a readable fixed-width table."""
    print()
    print(title)
    print("-" * len(title))

    if not rows:
        print("No records found.")
        return

    columns = rows[0].keys()
    headers = [str(column) for column in columns]
    values = [[str(row[column]) for column in columns] for row in rows]
    widths = [
        max(len(header), *(len(value[index]) for value in values))
        for index, header in enumerate(headers)
    ]

    header_line = " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    divider = "-+-".join("-" * width for width in widths)
    print(header_line)
    print(divider)
    for value_row in values:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(value_row)))


def _run_query(connection: sqlite3.Connection, title: str, sql: str) -> None:
    """Execute one SQL query and print its results."""
    cursor = connection.execute(sql)
    _print_table(title, cursor.fetchall())


def main() -> None:
    """Connect to the demo database and run representative analytics queries."""
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Demo database not found at {DATABASE_PATH}. "
            "Run scripts/create_demo_db.py first."
        )

    queries: list[tuple[str, str]] = [
        (
            "Total pickups yesterday",
            """
            SELECT COUNT(*) AS total_pickups
            FROM pickups
            WHERE pickup_date = DATE('now', '-1 day');
            """,
        ),
        (
            "Total packages yesterday",
            """
            SELECT COALESCE(SUM(package_count), 0) AS total_packages
            FROM pickups
            WHERE pickup_date = DATE('now', '-1 day');
            """,
        ),
        (
            "Top 10 customers by package count",
            """
            SELECT c.customer_name,
                   SUM(p.package_count) AS total_packages
            FROM pickups p
            JOIN customers c ON c.customer_id = p.customer_id
            GROUP BY c.customer_name
            ORDER BY total_packages DESC
            LIMIT 10;
            """,
        ),
        (
            "Top 5 drivers by completed pickups",
            """
            SELECT d.driver_name,
                   COUNT(*) AS completed_pickups
            FROM pickups p
            JOIN drivers d ON d.driver_id = p.driver_id
            WHERE p.status = 'Completed'
            GROUP BY d.driver_name
            ORDER BY completed_pickups DESC
            LIMIT 5;
            """,
        ),
        (
            "Failed pickups",
            """
            SELECT pickup_id,
                   pickup_date,
                   package_count,
                   status
            FROM pickups
            WHERE status = 'Failed'
            ORDER BY pickup_date DESC, pickup_id DESC
            LIMIT 20;
            """,
        ),
        (
            "Delayed pickups",
            """
            SELECT pickup_id,
                   pickup_date,
                   package_count,
                   status
            FROM pickups
            WHERE status = 'Delayed'
            ORDER BY pickup_date DESC, pickup_id DESC
            LIMIT 20;
            """,
        ),
        (
            "Average packages per pickup",
            """
            SELECT ROUND(AVG(package_count), 2) AS avg_packages_per_pickup
            FROM pickups;
            """,
        ),
        (
            "Pickup count by hub",
            """
            SELECT d.hub,
                   COUNT(*) AS pickup_count
            FROM pickups p
            JOIN drivers d ON d.driver_id = p.driver_id
            GROUP BY d.hub
            ORDER BY pickup_count DESC;
            """,
        ),
        (
            "Daily pickup trend for the last 30 days",
            """
            SELECT pickup_date,
                   COUNT(*) AS pickup_count,
                   SUM(package_count) AS total_packages
            FROM pickups
            WHERE pickup_date >= DATE('now', '-29 days')
            GROUP BY pickup_date
            ORDER BY pickup_date;
            """,
        ),
    ]

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        print(f"Connected to {DATABASE_PATH.relative_to(PROJECT_ROOT)}")
        for title, sql in queries:
            _run_query(connection, title, sql)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
