"""SQL validator — ensure generated SQL only references known schema objects."""

from __future__ import annotations

import re

from tools.sql.schema_registry import SchemaRegistry, SchemaSnapshot, get_schema_registry

_TABLE_REFERENCE_RE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
_TABLE_ALIAS_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
    re.IGNORECASE,
)
_QUALIFIED_COLUMN_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")


def references_only_known_tables(sql: str, snapshot: SchemaSnapshot) -> bool:
    allowed = snapshot.allowed_tables()
    table_names = {match.group(1).lower() for match in _TABLE_REFERENCE_RE.finditer(sql)}
    return table_names.issubset(allowed)


def references_only_known_columns(sql: str, snapshot: SchemaSnapshot) -> bool:
    allowed_columns = snapshot.allowed_columns()
    aliases: dict[str, str] = {}
    for match in _TABLE_ALIAS_RE.finditer(sql):
        table = match.group(1).lower()
        alias = (match.group(2) or table).lower()
        if table in allowed_columns:
            aliases[alias] = table
            aliases[table] = table

    for alias, column in _QUALIFIED_COLUMN_RE.findall(sql):
        table = aliases.get(alias.lower())
        if table is None:
            continue
        if column.lower() not in allowed_columns.get(table, set()):
            return False
    return True


def validate_sql(
    sql: str,
    *,
    registry: SchemaRegistry | None = None,
    snapshot: SchemaSnapshot | None = None,
) -> str:
    """Return SQL unchanged if valid; otherwise SELECT 'UNKNOWN';."""
    snap = snapshot or (registry or get_schema_registry()).get_snapshot()
    cleaned = (sql or "").strip()
    if not cleaned:
        return "SELECT 'UNKNOWN';"
    if not references_only_known_tables(cleaned, snap):
        return "SELECT 'UNKNOWN';"
    if not references_only_known_columns(cleaned, snap):
        return "SELECT 'UNKNOWN';"
    return cleaned
