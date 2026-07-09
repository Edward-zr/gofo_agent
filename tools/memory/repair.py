"""Detect and resolve user corrections in short-term conversations."""

from __future__ import annotations

import re
from typing import Any

from tools.memory.conversation import ConversationMemory

_REPAIR_PHRASES = (
    "no",
    "not that",
    "i mean",
    "i meant",
    "actually",
    "wrong",
    "that's not what i asked",
    "thats not what i asked",
    "i'm talking about",
    "im talking about",
    "i asked for",
    "instead",
)

_DIMENSION_WORDS = {
    "customer": ("customer", "customers"),
    "hub": ("hub", "hubs", "warehouse", "warehouses", "station", "stations"),
    "driver": ("driver", "drivers"),
}

_DIMENSION_REPLACEMENTS = {
    "customer": "hub",
    "hub": "driver",
    "driver": "customer",
}

_DATE_REPLACEMENTS = {
    "today": ("yesterday", "last month"),
    "yesterday": ("today", "last month"),
    "last month": ("today", "yesterday"),
}


def detect_repair(question: str, memory: ConversationMemory) -> dict[str, Any]:
    """
    Detect whether the user is correcting the previous turn.

    Returns a plain dict so callers can attach the metadata directly to
    QueryResponse debug fields without introducing a new model dependency.
    """
    question = question.strip()
    previous_question = _previous_question(memory)
    default = {
        "is_repair": False,
        "repair_type": None,
        "corrected_question": question,
        "changed_dimension": None,
    }
    if not question or not previous_question:
        return default

    normalized = question.lower()
    if not _contains_repair_phrase(normalized):
        return default

    corrected = previous_question
    changed_dimension = None
    repair_type = "correction"

    dimension = _mentioned_dimension(normalized)
    if dimension:
        previous_dimension = _previous_dimension(previous_question, memory)
        corrected = _replace_dimension(corrected, previous_dimension, dimension)
        changed_dimension = dimension
        repair_type = "dimension_replacement"

    if _mentions_all_hubs(normalized):
        corrected = _replace_single_hub_with_all_hubs(corrected)
        changed_dimension = "hub"
        repair_type = "scope_replacement"

    if "lowest" in normalized or "worst" in normalized:
        corrected = _replace_words(corrected, {"highest": "lowest", "best": "worst", "top": "bottom"})
        changed_dimension = changed_dimension or "ranking_direction"
        repair_type = "ranking_direction"
    elif "highest" in normalized or "best" in normalized:
        corrected = _replace_words(corrected, {"lowest": "highest", "worst": "best", "bottom": "top"})
        changed_dimension = changed_dimension or "ranking_direction"
        repair_type = "ranking_direction"

    date_phrase = _mentioned_date(normalized)
    if date_phrase:
        corrected = _replace_date(corrected, date_phrase)
        changed_dimension = changed_dimension or "date_range"
        repair_type = "date_replacement"

    if corrected == previous_question and dimension:
        corrected = f"{previous_question} by {dimension}"

    return {
        "is_repair": True,
        "repair_type": repair_type,
        "corrected_question": _normalize_spaces(corrected),
        "changed_dimension": changed_dimension,
    }


def _contains_repair_phrase(normalized_question: str) -> bool:
    return any(phrase in normalized_question for phrase in _REPAIR_PHRASES)


def _previous_question(memory: ConversationMemory) -> str | None:
    history = memory.get_recent_history(1)
    if not history:
        return None
    latest = history[-1]
    return latest.get("resolved_question") or latest.get("user_question")


def _previous_dimension(previous_question: str, memory: ConversationMemory) -> str | None:
    state = memory.get_current_state()
    dimension = state.get("analysis_dimension")
    if dimension:
        return str(dimension)
    return _mentioned_dimension(previous_question.lower())


def _mentioned_dimension(normalized_question: str) -> str | None:
    for dimension, words in _DIMENSION_WORDS.items():
        if any(re.search(rf"\b{re.escape(word)}\b", normalized_question) for word in words):
            return dimension
    return None


def _replace_dimension(question: str, previous_dimension: str | None, new_dimension: str) -> str:
    target_dimension = previous_dimension or _DIMENSION_REPLACEMENTS.get(new_dimension)
    if target_dimension:
        old_words = _DIMENSION_WORDS.get(target_dimension, (target_dimension,))
        for old_word in sorted(old_words, key=len, reverse=True):
            pattern = re.compile(rf"\b{re.escape(old_word)}\b", re.IGNORECASE)
            if pattern.search(question):
                replacement = "hubs" if old_word.endswith("s") and new_dimension == "hub" else new_dimension
                if old_word.endswith("s") and not replacement.endswith("s"):
                    replacement = f"{replacement}s"
                return pattern.sub(replacement, question)
    return f"{question} by {new_dimension}"


def _mentions_all_hubs(normalized_question: str) -> bool:
    return "all hubs" in normalized_question or "all warehouses" in normalized_question


def _replace_single_hub_with_all_hubs(question: str) -> str:
    without_specific_hub = re.sub(
        r"\b(?:for|in|at|only)\s+[A-Za-z][A-Za-z\s-]*\s+(?:hub|warehouse)\b",
        " for all hubs",
        question,
        flags=re.IGNORECASE,
    )
    without_named_hub = re.sub(
        r"\b[A-Z][A-Za-z]+\s+hub\b",
        "all hubs",
        without_specific_hub,
        flags=re.IGNORECASE,
    )
    if re.search(r"\ball hubs\b", without_named_hub, flags=re.IGNORECASE):
        return without_named_hub
    return f"{question} for all hubs"


def _replace_words(question: str, replacements: dict[str, str]) -> str:
    corrected = question
    for old, new in replacements.items():
        corrected = re.sub(rf"\b{old}\b", new, corrected, flags=re.IGNORECASE)
    return corrected


def _mentioned_date(normalized_question: str) -> str | None:
    for new_date, old_dates in _DATE_REPLACEMENTS.items():
        if re.search(rf"\b{re.escape(new_date)}\b", normalized_question):
            return new_date
        for old_date in old_dates:
            if re.search(rf"\b{re.escape(old_date)}\b", normalized_question):
                return old_date
    return None


def _replace_date(question: str, new_date: str) -> str:
    corrected = question
    for old_date, possible_new_dates in _DATE_REPLACEMENTS.items():
        if old_date == new_date:
            continue
        pattern = re.compile(rf"\b{re.escape(old_date)}\b", re.IGNORECASE)
        if pattern.search(corrected):
            return pattern.sub(new_date, corrected)
        for possible_new_date in possible_new_dates:
            if possible_new_date == new_date:
                pattern = re.compile(rf"\b{re.escape(old_date)}\b", re.IGNORECASE)
                if pattern.search(corrected):
                    return pattern.sub(new_date, corrected)
    return f"{question} for {new_date}"


def _normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
