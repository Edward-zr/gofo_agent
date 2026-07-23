"""Schema Retriever — select only relevant tables/columns/joins for SQL generation.

Never returns the full schema unless the user explicitly asks for it.
Retrieval uses business intent + keyword matching + descriptions, with an
optional embedding search hook for future vector-based schema retrieval.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from pydantic import BaseModel, Field

import config
from core.logger import get_logger
from tools.sql.schema_registry import (
    SchemaRegistry,
    SchemaSnapshot,
    get_schema_registry,
)

logger = get_logger("schema_retriever")

_FULL_SCHEMA_PHRASES = (
    "full schema",
    "entire schema",
    "complete schema",
    "all tables",
    "whole database schema",
    "show me the schema",
    "list all tables",
    "database schema",
)

# Intent → tables that should almost always be considered.
_INTENT_TABLE_HINTS: dict[str, list[str]] = {
    "sql_query": ["pickups"],
    "sql_analysis": ["pickups", "drivers"],
    "ranking": ["pickups", "drivers"],
    "trend": ["pickups"],
    "comparison": ["pickups", "drivers"],
    "root_cause": ["pickups", "exceptions"],
    "failure": ["pickups", "exceptions"],
    "kpi": ["pickups"],
    "hub": ["pickups", "drivers"],
    "driver": ["pickups", "drivers"],
    "customer": ["pickups", "customers"],
    "city": ["pickups", "addresses"],
    "location": ["pickups", "addresses"],
}


class RetrievedSchema(BaseModel):
    """Structured subset of schema sent to the SQL generator."""

    tables: list[str] = Field(default_factory=list)
    columns: dict[str, list[str]] = Field(default_factory=dict)
    relationships: list[str] = Field(default_factory=list)
    business_intent: str | None = None
    candidate_tables: list[str] = Field(default_factory=list)
    candidate_columns: dict[str, list[str]] = Field(default_factory=dict)
    scores: dict[str, float] = Field(default_factory=dict)
    full_schema_requested: bool = False
    reasoning: str = ""

    def to_prompt_text(self, snapshot: SchemaSnapshot) -> str:
        """Render a compact schema prompt using ONLY retrieved objects."""
        lines = [
            "Retrieved schema (use ONLY these tables/columns — never invent others)",
            "---------------------------------------------------------------------",
        ]
        for table_name in self.tables:
            table = snapshot.tables.get(table_name)
            if table is None:
                continue
            lines.append("")
            lines.append(f"TABLE {table_name}:")
            if table.description:
                lines.append(f"  description: {table.description}")
            selected_cols = set(self.columns.get(table_name) or [])
            for column in table.columns:
                if selected_cols and column.name not in selected_cols:
                    # Always keep PKs/FKs needed for joins even if not scored.
                    if not column.primary_key and column.name not in {
                        fk.from_column for fk in table.foreign_keys
                    }:
                        continue
                desc = f" — {column.description}" if column.description else ""
                pk = " PK" if column.primary_key else ""
                lines.append(f"  - {column.name} {column.data_type}{pk}{desc}")
            if table.primary_keys:
                lines.append(f"  primary_keys: {', '.join(table.primary_keys)}")
        if self.relationships:
            lines.append("")
            lines.append("Relationships:")
            for rel in self.relationships:
                lines.append(f"  - {rel}")
        lines.append("")
        lines.append(
            "Rules: Never invent tables or columns. Use only the objects listed above."
        )
        return "\n".join(lines)


class SchemaEmbedder(Protocol):
    """Optional vector search over schema metadata (future-ready)."""

    def score_tables(self, question: str, snapshot: SchemaSnapshot) -> dict[str, float]: ...


class NullSchemaEmbedder:
    """Default no-op embedder — keeps retrieval keyword-first."""

    def score_tables(self, question: str, snapshot: SchemaSnapshot) -> dict[str, float]:
        return {}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _wants_full_schema(text: str) -> bool:
    return any(phrase in text for phrase in _FULL_SCHEMA_PHRASES)


def _infer_business_intent(
    question: str,
    *,
    business_intent: str | None,
    semantic_context: dict[str, Any] | None,
) -> str:
    if business_intent:
        return str(business_intent).strip().lower()
    semantic_context = semantic_context or {}
    for key in ("sub_intent", "intent", "business_intent", "metric"):
        value = semantic_context.get(key)
        if value:
            return str(value).strip().lower()
    text = _normalize(question)
    if any(token in text for token in ("root cause", "why failed", "failure reason", "exception")):
        return "root_cause"
    if any(token in text for token in ("rank", "top", "worst", "best", "highest", "lowest")):
        return "ranking"
    if any(token in text for token in ("compare", "vs", "versus", "difference")):
        return "comparison"
    if any(token in text for token in ("trend", "over time", "week over week")):
        return "trend"
    if "hub" in text or "warehouse" in text:
        return "hub"
    if "driver" in text:
        return "driver"
    if "customer" in text:
        return "customer"
    if "city" in text or "location" in text:
        return "city"
    if any(token in text for token in ("kpi", "rate", "pickup")):
        return "kpi"
    return "sql_query"


def _keyword_score(text: str, keywords: list[str]) -> float:
    score = 0.0
    for keyword in keywords:
        key = keyword.lower()
        if not key:
            continue
        if " " in key:
            if key in text:
                score += 2.0
        elif re.search(rf"\b{re.escape(key)}\b", text):
            score += 1.0
        elif key in text:
            score += 0.35
    return score


def _score_tables(text: str, intent: str, snapshot: SchemaSnapshot) -> dict[str, float]:
    scores: dict[str, float] = {name: 0.0 for name in snapshot.tables}
    for hint_key, tables in _INTENT_TABLE_HINTS.items():
        if hint_key in intent or intent in hint_key:
            for table in tables:
                if table in scores:
                    scores[table] += 1.5

    for name, table in snapshot.tables.items():
        blob = " ".join(
            [
                name,
                table.description,
                " ".join(table.keywords),
                " ".join(f"{c.name} {c.description}" for c in table.columns),
            ]
        ).lower()
        scores[name] += _keyword_score(text, table.keywords)
        scores[name] += _keyword_score(text, [name])
        # Soft boost when description terms appear in the question.
        for token in re.findall(r"[a-z_]{4,}", table.description.lower()):
            if re.search(rf"\b{re.escape(token)}\b", text):
                scores[name] += 0.15
        # Direct column-name hits
        for column in table.columns:
            if re.search(rf"\b{re.escape(column.name.lower())}\b", text):
                scores[name] += 1.25
            scores[name] += 0.25 * _keyword_score(text, column.keywords)
        # Tiny presence of ops vocabulary keeps pickups alive.
        if name == "pickups" and any(
            token in text
            for token in ("pickup", "package", "status", "today", "rate", "count", "kpi", "failed")
        ):
            scores[name] += 1.0
        _ = blob  # reserved for future embedder context
    return scores


def _select_columns(
    text: str,
    table_name: str,
    snapshot: SchemaSnapshot,
    *,
    include_join_keys: bool = True,
) -> list[str]:
    table = snapshot.tables[table_name]
    selected: list[str] = []
    scored: list[tuple[float, str]] = []
    join_keys = {fk.from_column for fk in table.foreign_keys} | set(table.primary_keys)
    # Also destination keys when this table is referenced as parent.
    for fk in snapshot.relationships:
        if fk.to_table == table_name:
            join_keys.add(fk.to_column)

    for column in table.columns:
        score = 0.0
        if re.search(rf"\b{re.escape(column.name.lower())}\b", text):
            score += 3.0
        score += _keyword_score(text, column.keywords)
        if column.description:
            for token in re.findall(r"[a-z_]{4,}", column.description.lower()):
                if re.search(rf"\b{re.escape(token)}\b", text):
                    score += 0.2
        # Always keep structural keys for joins.
        if include_join_keys and column.name in join_keys:
            score = max(score, 0.5)
        # Keep commonly needed measure columns lightly for pickups.
        if table_name == "pickups" and column.name in {"pickup_date", "status", "package_count"}:
            score = max(score, 0.75)
        if table_name == "drivers" and column.name in {"driver_name", "hub"}:
            score = max(score, 0.75)
        if score > 0:
            scored.append((score, column.name))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [name for _, name in scored]
    if not selected:
        # Fallback: PKs + a few descriptive columns.
        selected = list(table.primary_keys)
        for column in table.columns:
            if column.name not in selected:
                selected.append(column.name)
            if len(selected) >= min(4, len(table.columns)):
                break
    # Preserve table column order for stable prompts.
    order = {column.name: index for index, column in enumerate(table.columns)}
    return sorted(set(selected), key=lambda name: order.get(name, 999))


def _expand_join_closure(
    tables: list[str],
    snapshot: SchemaSnapshot,
) -> tuple[list[str], list[str]]:
    """Ensure join partners needed to connect selected satellite tables exist.

    Does NOT fan out from pickups to every FK parent — only adds pickups when a
    related dimension/fact table was selected, then emits joins among the final set.
    """
    selected = set(tables)
    relationships: list[str] = []

    satellites = {"drivers", "customers", "addresses", "exceptions"}
    if selected & satellites and "pickups" in snapshot.tables:
        selected.add("pickups")

    # Relationships only among currently selected tables (no new table discovery).
    for fk in snapshot.relationships:
        if fk.from_table in selected and fk.to_table in selected:
            join = fk.as_join()
            if join not in relationships:
                relationships.append(join)

    ordered = sorted(selected, key=lambda name: (0 if name == "pickups" else 1, name))
    return ordered, relationships


class SchemaRetriever:
    """Retrieve a minimal relevant schema subset for SQL generation."""

    def __init__(
        self,
        registry: SchemaRegistry | None = None,
        *,
        embedder: SchemaEmbedder | None = None,
        min_table_score: float = 0.75,
        max_tables: int = 4,
    ) -> None:
        self._registry = registry or get_schema_registry()
        self._embedder = embedder or NullSchemaEmbedder()
        self._min_table_score = min_table_score
        self._max_tables = max_tables

    def retrieve(
        self,
        question: str,
        *,
        business_intent: str | None = None,
        semantic_context: dict[str, Any] | None = None,
    ) -> RetrievedSchema:
        question = (question or "").strip()
        snapshot = self._registry.get_snapshot()
        text = _normalize(question)
        intent = _infer_business_intent(
            question,
            business_intent=business_intent,
            semantic_context=semantic_context,
        )

        if _wants_full_schema(text):
            columns = {
                name: table.column_names() for name, table in snapshot.tables.items()
            }
            relationships = [fk.as_join() for fk in snapshot.relationships]
            result = RetrievedSchema(
                tables=snapshot.table_names(),
                columns=columns,
                relationships=relationships,
                business_intent=intent,
                candidate_tables=snapshot.table_names(),
                candidate_columns=columns,
                scores={name: 1.0 for name in snapshot.tables},
                full_schema_requested=True,
                reasoning="User explicitly requested the full schema.",
            )
            self._debug(result)
            return result

        scores = _score_tables(text, intent, snapshot)
        if getattr(config, "SCHEMA_EMBEDDING_RETRIEVAL_ENABLED", False):
            for name, embed_score in self._embedder.score_tables(question, snapshot).items():
                scores[name] = scores.get(name, 0.0) + float(embed_score)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        candidate_tables = [name for name, score in ranked if score > 0]
        selected = [name for name, score in ranked if score >= self._min_table_score]

        if not selected:
            # Always fall back to the core fact table rather than dumping everything.
            selected = ["pickups"] if "pickups" in snapshot.tables else snapshot.table_names()[:1]

        selected = selected[: self._max_tables]
        selected, relationships = _expand_join_closure(selected, snapshot)

        columns: dict[str, list[str]] = {}
        for table_name in selected:
            columns[table_name] = _select_columns(text, table_name, snapshot)

        result = RetrievedSchema(
            tables=selected,
            columns=columns,
            relationships=relationships,
            business_intent=intent,
            candidate_tables=candidate_tables or selected,
            candidate_columns=columns,
            scores={name: float(scores.get(name, 0.0)) for name in selected},
            full_schema_requested=False,
            reasoning=(
                f"intent={intent}; ranked={ranked[:5]}; "
                f"selected={selected}; relationships={len(relationships)}"
            ),
        )
        self._debug(result)
        return result

    def _debug(self, result: RetrievedSchema) -> None:
        message = (
            "Schema retrieval\n"
            f"  intent: {result.business_intent}\n"
            f"  tables: {result.tables}\n"
            f"  columns: {result.columns}\n"
            f"  relationships: {result.relationships}\n"
            f"  candidates: {result.candidate_tables}\n"
            f"  scores: {result.scores}\n"
            f"  reasoning: {result.reasoning}"
        )
        if config.DEBUG or getattr(config, "SCHEMA_RETRIEVER_DEBUG", False):
            print("----------------------------------")
            print(message)
            print("----------------------------------")
        logger.info(message)


def retrieve_schema(
    question: str,
    *,
    business_intent: str | None = None,
    semantic_context: dict[str, Any] | None = None,
) -> RetrievedSchema:
    """Module-level helper."""
    return SchemaRetriever().retrieve(
        question,
        business_intent=business_intent,
        semantic_context=semantic_context,
    )
