"""Scoring primitives for benchmark and scenario evaluation."""

from __future__ import annotations

import re
from typing import Any, Iterable


def _norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def keyword_hit_rate(answer: str, keywords: Iterable[str] | None) -> float:
    """Fraction of expected keywords found in the answer (semantic proxy)."""
    keys = [k for k in (keywords or []) if str(k).strip()]
    if not keys:
        return 1.0  # nothing to check → do not penalize
    text = _norm(answer)
    words = text.split()
    hits = 0
    for key in keys:
        token = _norm(str(key))
        if not token:
            continue
        if token in text:
            hits += 1
            continue
        # Soft stem/prefix match: "return" ↔ "returns", "pickup" ↔ "pickups"
        stem = token[: max(4, len(token) - 1)]
        if any(stem and (stem in word or word.startswith(stem[:4])) for word in words):
            hits += 1
            continue
        if any(part and part in text for part in token.split() if len(part) > 2):
            hits += 1
    return hits / len(keys)


def tool_selection_accuracy(expected: Iterable[str] | None, actual: Iterable[str] | None) -> float:
    """Jaccard overlap between expected and actual tool sets."""
    exp = {str(t).upper() for t in (expected or []) if t}
    act = {str(t).upper() for t in (actual or []) if t}
    if not exp:
        return 1.0
    if not act:
        return 0.0
    return len(exp & act) / len(exp | act)


def tool_order_accuracy(expected: Iterable[str] | None, actual: Iterable[str] | None) -> float:
    """Score whether expected tools appear in order within the actual sequence."""
    exp = [str(t).upper() for t in (expected or []) if t]
    act = [str(t).upper() for t in (actual or []) if t]
    if not exp:
        return 1.0
    if not act:
        return 0.0
    idx = 0
    matched = 0
    for tool in act:
        if idx < len(exp) and tool == exp[idx]:
            matched += 1
            idx += 1
    return matched / len(exp)


def behavior_match(expected_behavior: str | None, trace: dict[str, Any]) -> float:
    """Match coarse expected behavior labels to trace signals."""
    behavior = _norm(expected_behavior)
    if not behavior:
        return 1.0

    answer = _norm(trace.get("answer"))
    capability = _norm(str(trace.get("capability")))
    tools = {str(t).upper() for t in (trace.get("tools_used") or [])}
    checks: list[bool] = []

    def has(*words: str) -> bool:
        return any(w in behavior for w in words)

    if has("clarif"):
        checks.append(bool(trace.get("requires_clarification")) or capability == "clarification")
    if has("sql", "query", "analytics", "metric", "kpi"):
        checks.append("SQL" in tools or capability == "sql" or bool(trace.get("sql")))
    if has("rag", "sop", "policy", "document"):
        checks.append("RAG" in tools or capability == "rag" or bool(trace.get("sources")))
    if has("chart", "visual", "plot", "graph"):
        checks.append(bool(trace.get("charts")) or "VISUALIZATION" in tools)
    if has("recommend"):
        checks.append(bool(trace.get("recommendations")) or "RECOMMENDATION" in tools)
    if has("memory", "follow", "context", "previous"):
        mem = trace.get("memory_state") or {}
        checks.append(bool(mem.get("memory_turns")) or bool(trace.get("resolved_question")))
    if has("upload", "attachment", "file"):
        checks.append("ATTACHMENT" in tools or "upload" in answer or "attach" in answer)
    if has("greet", "chitchat", "general", "conversation"):
        checks.append(capability in {"conversation", "llm"} or "LLM" in tools)
    if has("refuse", "unknown", "insufficient"):
        checks.append(
            "don't know" in answer
            or "not enough" in answer
            or "couldn't find" in answer
            or bool(trace.get("requires_clarification"))
        )
    if has("cautious", "uncertain"):
        checks.append(
            "based on the available" in answer
            or "suggests" in answer
            or str(trace.get("fallback_strategy") or "").upper() == "GENERATE_CAUTIOUS"
        )

    if not checks:
        # Soft fallback: non-empty answer without exception
        return 1.0 if answer and not trace.get("error") else 0.0
    return sum(1 for ok in checks if ok) / len(checks)


def sql_success(trace: dict[str, Any], *, expected_sql: bool = False) -> float:
    if trace.get("error"):
        return 0.0
    sql = (trace.get("sql") or "").strip()
    if expected_sql:
        if not sql:
            return 0.0
        if "UNKNOWN" in sql.upper() and "SELECT 'UNKNOWN'" in sql.upper().replace(" ", ""):
            return 0.0
        return 1.0
    # If SQL not expected, success is vacuously 1 unless a failed SQL appears.
    if sql and "UNKNOWN" in sql.upper():
        return 0.5
    return 1.0


def retrieval_confidence_score(trace: dict[str, Any], *, expected_rag: bool = False) -> float:
    level = str(trace.get("confidence_level") or "").upper()
    score = trace.get("confidence_score")
    if expected_rag:
        if score is None and not trace.get("sources"):
            return 0.0
        if level == "LOW" and not trace.get("requires_clarification"):
            # Low confidence without fallback is a reliability miss
            return 0.4
        if isinstance(score, (int, float)):
            return max(0.0, min(1.0, float(score)))
        return 0.7 if trace.get("sources") else 0.0
    if isinstance(score, (int, float)):
        return max(0.0, min(1.0, float(score)))
    return 1.0


def hallucination_risk(trace: dict[str, Any]) -> float:
    """Higher is worse. Used for reliability aggregation (inverted later)."""
    if trace.get("error"):
        return 1.0
    level = str(trace.get("confidence_level") or "").upper()
    if level == "LOW" and not trace.get("requires_clarification") and not trace.get("fallback_strategy"):
        return 0.8
    answer = _norm(trace.get("answer"))
    if not answer:
        return 0.6
    return 0.0


def mean(values: Iterable[float]) -> float:
    data = [float(v) for v in values]
    if not data:
        return 0.0
    return sum(data) / len(data)


def pct(value: float) -> float:
    return round(100.0 * max(0.0, min(1.0, value)), 2)
