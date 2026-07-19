"""SQL planning for GOFO operational analytics questions."""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from tools.llm.client import get_llm
from tools.sql.schema import SCHEMA
from tools.sql.schema_loader import get_database_schema

_ALLOWED_TABLES = {"pickups", "drivers", "customers", "addresses", "exceptions"}
_ALLOWED_COLUMNS = {
    "pickups": {"pickup_id", "customer_id", "driver_id", "address_id", "pickup_date", "package_count", "status"},
    "drivers": {"driver_id", "driver_name", "hub"},
    "customers": {"customer_id", "customer_name"},
    "addresses": {"address_id", "city", "state"},
    "exceptions": {"exception_id", "pickup_id", "reason", "created_at"},
}
_TABLE_REFERENCE_RE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
_TABLE_ALIAS_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
    re.IGNORECASE,
)
_QUALIFIED_COLUMN_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")

_DATABASE_SCHEMA = """
Concrete database schema
------------------------

TABLE pickups:

columns:
pickup_id
customer_id
driver_id
address_id
pickup_date
package_count
status

Meaning:
Each row = one pickup task.

status values:
Completed
Failed
Delayed

TABLE drivers:
driver_id
driver_name
hub

TABLE customers:
customer_id
customer_name

TABLE addresses:
address_id
city
state

TABLE exceptions:
exception_id
pickup_id
reason
created_at

Relationships:
pickups.driver_id = drivers.driver_id
pickups.customer_id = customers.customer_id
pickups.address_id = addresses.address_id
exceptions.pickup_id = pickups.pickup_id
"""

_FEW_SHOT_EXAMPLES = """
Example 1

Question
How many pickups 2026-06-29?

SQL
SELECT COUNT(*)
FROM pickups
WHERE pickup_date = '2026-06-29';

Example 2

Question
Pickup details 2026-06-29

SQL
SELECT p.pickup_id,
       p.pickup_date,
       p.status,
       p.package_count,
       d.driver_name,
       d.hub,
       c.customer_name
FROM pickups p
LEFT JOIN drivers d ON p.driver_id = d.driver_id
LEFT JOIN customers c ON p.customer_id = c.customer_id
WHERE p.pickup_date = '2026-06-29';

Example 3

Question
Rank hubs

SQL
SELECT d.hub,
       COUNT(*) AS pickups,
       SUM(p.package_count) AS packages
FROM pickups p
JOIN drivers d ON p.driver_id = d.driver_id
GROUP BY d.hub
ORDER BY packages DESC;

Example 4

Question
Top 5 drivers for week of 2026-06-23 to 2026-06-29

SQL
SELECT d.driver_name,
       COUNT(*) AS pickup_count
FROM pickups p
JOIN drivers d ON p.driver_id = d.driver_id
WHERE p.pickup_date BETWEEN '2026-06-23' AND '2026-06-29'
GROUP BY d.driver_id, d.driver_name
ORDER BY pickup_count DESC
LIMIT 5;

Example 5

Question
Average packages per pickup

SQL
SELECT AVG(package_count) AS avg_packages_per_pickup
FROM pickups;

Example 6

Question
Top 10 customers by pickup volume

SQL
SELECT c.customer_name,
       COUNT(*) AS pickup_count,
       SUM(p.package_count) AS package_count
FROM pickups p
JOIN customers c ON p.customer_id = c.customer_id
GROUP BY c.customer_name
ORDER BY pickup_count DESC
LIMIT 10;

Example 7

Question
Compare this week and last week.

SQL
SELECT period,
       COUNT(*) AS pickup_count,
       SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) AS completed_pickups,
       SUM(CASE WHEN status = 'Failed' THEN 1 ELSE 0 END) AS failed_pickups,
       SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END) AS delayed_pickups,
       SUM(package_count) AS package_count
FROM (
    SELECT CASE
               WHEN pickup_date >= DATE('now', '-6 days') THEN 'this_week'
               WHEN pickup_date >= DATE('now', '-13 days')
                    AND pickup_date < DATE('now', '-6 days') THEN 'last_week'
           END AS period,
           status,
           package_count
    FROM pickups
    WHERE pickup_date >= DATE('now', '-13 days')
)
WHERE period IS NOT NULL
GROUP BY period
ORDER BY period;

Example 8

Question
Monthly KPI

SQL
SELECT SUBSTR(pickup_date, 1, 7) AS month,
       COUNT(*) AS pickup_count,
       SUM(package_count) AS package_count,
       SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) AS completed_pickups,
       ROUND(
           100.0 * SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) / COUNT(*),
           2
       ) AS completion_rate
FROM pickups
GROUP BY month
ORDER BY month;

Example 9

Question
Warehouse performance

SQL
SELECT d.hub AS warehouse,
       COUNT(*) AS pickup_count,
       SUM(package_count) AS package_count,
       SUM(CASE WHEN p.status = 'Completed' THEN 1 ELSE 0 END) AS completed_pickups,
       ROUND(
           100.0 * SUM(CASE WHEN p.status = 'Completed' THEN 1 ELSE 0 END) / COUNT(*),
           2
       ) AS completion_rate
FROM pickups p
JOIN drivers d ON p.driver_id = d.driver_id
GROUP BY d.hub
ORDER BY completion_rate DESC;

Example 10

Question
Highest performing driver

SQL
SELECT d.driver_name,
       COUNT(*) AS completed_pickups
FROM pickups p
JOIN drivers d ON p.driver_id = d.driver_id
WHERE p.status = 'Completed'
GROUP BY d.driver_id, d.driver_name
ORDER BY completed_pickups DESC
LIMIT 1;

Example 11

Question
Pickup trends

SQL
SELECT pickup_date,
       COUNT(*) AS pickup_count,
       SUM(package_count) AS package_count
FROM pickups
GROUP BY pickup_date
ORDER BY pickup_date;

Example 12

Question
Average pickup per driver

SQL
SELECT d.driver_name,
       AVG(p.package_count) AS average_packages_per_pickup,
       COUNT(*) AS pickup_count
FROM pickups p
JOIN drivers d ON p.driver_id = d.driver_id
GROUP BY d.driver_id, d.driver_name
ORDER BY average_packages_per_pickup DESC;

Example 13

Question
Failed pickups

SQL
SELECT p.pickup_id,
       p.pickup_date,
       p.status,
       p.package_count,
       e.reason
FROM pickups p
LEFT JOIN exceptions e ON p.pickup_id = e.pickup_id
WHERE p.status = 'Failed'
ORDER BY p.pickup_date;

Example 14

Question
Delayed pickups

SQL
SELECT pickup_date,
       COUNT(*) AS delayed_pickups,
       SUM(package_count) AS delayed_packages
FROM pickups
WHERE status = 'Delayed'
GROUP BY pickup_date
ORDER BY pickup_date;
"""

_BUSINESS_CONTEXT = """
Business vocabulary
-------------------
highest performing driver, best driver, top driver:
    rank drivers by completed pickups in descending order.

pickup trend, pickup trends:
    group pickups by pickup_date and order by pickup_date.

warehouse performance:
    warehouse and hub mean operational warehouse/station.
    warehouse means drivers.hub in this demo database; group by drivers.hub.
    ALWAYS use drivers.hub. NEVER use customers.customer_name for hub.

completion rate:
    completed pickups divided by total pickups, usually as a percentage.

failed pickups:
    filter pickups.status = 'Failed'.

delayed pickups:
    filter pickups.status = 'Delayed'.

pickup volume, pickup count, average pickups:
    aggregate rows in pickups and/or package_count.

monthly KPI:
    group by SUBSTR(pickup_date, 1, 7).

weekly KPI:
    group by STRFTIME('%Y-%W', pickup_date).

daily KPI:
    group by pickup_date.

comparison, growth, increase, decrease:
    return comparable period-level metrics; include pickup_count, completed,
    failed, delayed, and package_count when relevant.

customers:
    join pickups.customer_id to customers.customer_id.

drivers:
    join pickups.driver_id to drivers.driver_id.

addresses:
    join pickups.address_id to addresses.address_id.

exceptions:
    join exceptions.pickup_id to pickups.pickup_id.

details:
    return actual pickup rows, not only COUNT. Include pickup_id, pickup_date,
    status, package_count, and relevant joined names when useful.

dates:
    always use pickups.pickup_date for operational dates. If the question
    contains a concrete ISO date such as 2026-06-29, use that literal date in
    the WHERE clause instead of DATE('now').

Known pickup fields
-------------------
pickup_id, pickup_date, status, package_count, customer_id, driver_id, address_id.

Known statuses
--------------
Completed, Delayed, Failed.
"""


def _build_system_prompt() -> str:
    """Return schema-aware instructions for SQLite analytics SQL generation."""
    return (
        "You are an expert SQLite analytics assistant.\n"
        "You generate SQL ONLY.\n\n"
        "Database engine:\n"
        "SQLite\n\n"
        f"Database schema:\n{get_database_schema()}\n\n"
        f"Documented schema:\n{SCHEMA}\n"
        f"{_DATABASE_SCHEMA}\n"
        f"{_BUSINESS_CONTEXT}\n"
        "Rules\n"
        "1. Output SQL only.\n"
        "2. Never explain.\n"
        "3. Never use markdown.\n"
        "4. Never invent tables.\n"
        "5. Never invent columns.\n"
        "6. Use SQLite syntax only.\n"
        "7. Always use pickup_date for operational dates.\n"
        "8. If the user question contains a concrete ISO date, filter with "
        "pickup_date = 'YYYY-MM-DD' or an explicit BETWEEN range.\n"
        "9. For driver questions, JOIN drivers.\n"
        "10. For hub or warehouse questions, JOIN drivers and use drivers.hub. "
        "Hub means operational warehouse/station. For hub or warehouse "
        "questions, ALWAYS JOIN drivers and use drivers.hub. NEVER use "
        "customers.customer_name as hub.\n"
        "11. For customer questions, JOIN customers.\n"
        "12. For city or location questions, JOIN addresses.\n"
        "13. For failure reason questions, JOIN exceptions.\n"
        "14. For details questions, return actual rows, not only COUNT.\n"
        "15. Never use MySQL syntax.\n"
        "16. Never use PostgreSQL syntax.\n"
        "17. Never use SQL Server syntax.\n"
        "18. Use DATE('now') instead of CURDATE() only if no concrete date was provided.\n"
        "19. Use DATE('now', '-1 day') instead of CURDATE() - INTERVAL 1 DAY only if no concrete date was provided.\n"
        "20. Use LIMIT instead of TOP.\n"
        "21. Attempt a best-effort analytics query whenever the question maps to "
        "pickups, drivers, customers, addresses, hubs/warehouses, packages, "
        "statuses, exceptions, dates, KPIs, trends, rankings, comparisons, "
        "growth, increases, or decreases.\n"
        "22. Never return SELECT 'UNKNOWN'; for valid operational analytics "
        "questions that can be approximated from the schema.\n"
        "23. Return SELECT 'UNKNOWN'; only when the question truly cannot be "
        "mapped to the available analytics schema.\n"
        "24. Conversation context is NOT a SQL filter. Only add WHERE filters "
        "when the current question explicitly names a hub, driver, customer, "
        "city, status, or date.\n"
        "25. Global driver ranking must GROUP BY driver and must NOT inherit a "
        "previous hub filter unless the question explicitly asks for that hub.\n"
        "26. Global hub ranking must GROUP BY drivers.hub without a single-hub "
        "WHERE clause unless the question explicitly names one hub.\n"
        "27. Highest or lowest driver questions rank all drivers globally by "
        "default. Do not add hub='...' unless the user explicitly specifies a hub.\n"
        "28. Worst hub or best hub follow-up questions may filter to a named hub "
        "only when the question names that hub directly.\n\n"
        "SELECT 'UNKNOWN';\n\n"
        "Few-shot examples\n"
        f"{_FEW_SHOT_EXAMPLES}"
    )


def _build_user_prompt(question: str) -> str:
    """Return the user message containing the analytics question."""
    return f"Question:\n{question}\n\nSQL:"


def _references_only_known_tables(sql: str) -> bool:
    """Return whether FROM/JOIN table references are in the analytics schema."""
    table_names = {match.group(1).lower() for match in _TABLE_REFERENCE_RE.finditer(sql)}
    return table_names.issubset(_ALLOWED_TABLES)


def _references_only_known_columns(sql: str) -> bool:
    """Return whether qualified alias.column references exist in the schema."""
    aliases: dict[str, str] = {}
    for match in _TABLE_ALIAS_RE.finditer(sql):
        table = match.group(1).lower()
        alias = (match.group(2) or table).lower()
        if table in _ALLOWED_TABLES:
            aliases[alias] = table
            aliases[table] = table

    for alias, column in _QUALIFIED_COLUMN_RE.findall(sql):
        table = aliases.get(alias.lower())
        if table is None:
            continue
        if column.lower() not in _ALLOWED_COLUMNS[table]:
            return False
    return True


def plan(question: str) -> str:
    """
    Convert an English analytics question into SQL.

    Does not execute SQL or connect to a database.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    messages = [
        SystemMessage(content=_build_system_prompt()),
        HumanMessage(content=_build_user_prompt(question)),
    ]
    response = get_llm().invoke(messages)
    content = response.content

    if isinstance(content, str):
        sql = content.strip()
    else:
        sql = str(content).strip()

    if not _references_only_known_tables(sql) or not _references_only_known_columns(sql):
        return "SELECT 'UNKNOWN';"

    return sql
