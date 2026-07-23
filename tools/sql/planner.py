"""SQL planning for GOFO operational analytics questions."""

from __future__ import annotations

from typing import Any

from tools.llm.client import get_llm
from tools.sql.schema_registry import get_schema_registry
from tools.sql.schema_retriever import RetrievedSchema, SchemaRetriever
from tools.sql.validator import validate_sql

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

Example 5b

Question
Average pickup per driver

SQL
SELECT d.driver_name,
       AVG(p.package_count) AS avg_packages
FROM pickups p
JOIN drivers d ON p.driver_id = d.driver_id
GROUP BY d.driver_id, d.driver_name
ORDER BY avg_packages DESC;

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
WITH bounds AS (
  SELECT DATE('now', 'weekday 0', '-6 days') AS this_week_start,
         DATE('now') AS this_week_end,
         DATE('now', 'weekday 0', '-13 days') AS last_week_start,
         DATE('now', 'weekday 0', '-7 days') AS last_week_end
)
SELECT 'this_week' AS period,
       COUNT(*) AS pickup_count,
       SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) AS completed,
       SUM(CASE WHEN status = 'Failed' THEN 1 ELSE 0 END) AS failed,
       SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END) AS delayed,
       SUM(package_count) AS package_count
FROM pickups, bounds
WHERE pickup_date BETWEEN bounds.this_week_start AND bounds.this_week_end
UNION ALL
SELECT 'last_week' AS period,
       COUNT(*) AS pickup_count,
       SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) AS completed,
       SUM(CASE WHEN status = 'Failed' THEN 1 ELSE 0 END) AS failed,
       SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END) AS delayed,
       SUM(package_count) AS package_count
FROM pickups, bounds
WHERE pickup_date BETWEEN bounds.last_week_start AND bounds.last_week_end;

Example 8

Question
Monthly KPI

SQL
SELECT SUBSTR(pickup_date, 1, 7) AS month,
       COUNT(*) AS pickup_count,
       ROUND(100.0 * SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) / COUNT(*), 2)
         AS completion_rate
FROM pickups
GROUP BY SUBSTR(pickup_date, 1, 7)
ORDER BY month;

Example 9

Question
Weekly KPI

SQL
SELECT STRFTIME('%Y-%W', pickup_date) AS week,
       COUNT(*) AS pickup_count,
       SUM(package_count) AS package_count
FROM pickups
GROUP BY STRFTIME('%Y-%W', pickup_date)
ORDER BY week;

Example 10

Question
Pickup trends

SQL
SELECT pickup_date,
       COUNT(*) AS pickup_count,
       SUM(package_count) AS package_count
FROM pickups
GROUP BY pickup_date
ORDER BY pickup_date;

Example 11

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

Example 12

Question
Warehouse performance

SQL
SELECT d.hub,
       COUNT(*) AS pickup_count,
       ROUND(AVG(p.package_count), 2) AS average_packages_per_pickup
FROM pickups p
JOIN drivers d ON p.driver_id = d.driver_id
GROUP BY d.hub
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


def _build_system_prompt(retrieved_schema: RetrievedSchema | None = None) -> str:
    """Render SQL generator system prompt from the Prompt Registry."""
    from core.prompt_manager import get_prompt_manager

    snapshot = get_schema_registry().get_snapshot()
    if retrieved_schema is None:
        retrieved_schema = SchemaRetriever(registry=get_schema_registry()).retrieve(
            "show operational pickup metrics"
        )
    schema_block = retrieved_schema.to_prompt_text(snapshot)
    _selection, rendered = get_prompt_manager().render(
        "sql.generator_prompt",
        {
            "schema": schema_block,
            "business_context": _BUSINESS_CONTEXT,
            "few_shot_examples": _FEW_SHOT_EXAMPLES,
            "question": "",
        },
    )
    return rendered


def _build_user_prompt(question: str) -> str:
    """Return the user message containing the analytics question."""
    return f"Question:\n{question}\n\nSQL:"


def plan(
    question: str,
    *,
    retrieved_schema: RetrievedSchema | None = None,
    business_intent: str | None = None,
    semantic_context: dict[str, Any] | None = None,
) -> str:
    """
    Convert an English analytics question into SQL using retrieved schema only.

    Does not execute SQL. Validation runs after generation.
    Prompt text comes from the Prompt Registry; sampling config from prompt metadata.
    """
    from core.prompt_manager import get_prompt_manager

    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    if retrieved_schema is None:
        retrieved_schema = SchemaRetriever().retrieve(
            question,
            business_intent=business_intent,
            semantic_context=semantic_context,
        )

    manager = get_prompt_manager()
    selection, messages = manager.build_messages(
        "sql.generator_prompt",
        {
            "question": question,
            "schema": retrieved_schema.to_prompt_text(get_schema_registry().get_snapshot()),
            "business_context": _BUSINESS_CONTEXT,
            "few_shot_examples": _FEW_SHOT_EXAMPLES,
        },
        user_content=_build_user_prompt(question),
    )
    meta = selection.metadata
    response = get_llm(
        temperature=meta.temperature,
        max_tokens=meta.max_tokens,
        model=meta.model,
    ).invoke(messages)
    content = response.content
    sql = content.strip() if isinstance(content, str) else str(content).strip()
    return validate_sql(sql)
