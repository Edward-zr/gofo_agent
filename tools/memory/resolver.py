"""Resolve follow-up questions using short-term conversation memory."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from tools.llm.client import get_llm
from tools.memory.conversation import ConversationMemory
from tools.memory.repair import detect_repair

_FOLLOW_UP_PREFIXES = (
    "what about",
    "how about",
    "compare",
    "why",
    "only",
    "which",
    "give me",
    "show",
    "and ",
    "also",
    "then",
)

_CONTEXTUAL_PHRASES = (
    " he ",
    " she ",
    "him",
    "her",
    "this driver",
    "that driver",
    "this warehouse",
    "that warehouse",
    "this customer",
    "that customer",
    "failed ones",
    "delayed ones",
    "same period",
    "last one",
)

_RESULT_REFERENCE_PHRASES = (
    "them",
    "that",
    "it",
    "this",
    "these",
    "those",
    "same ones",
    "above",
    "previous result",
    "previous results",
    "rank them",
    "sort them",
    "compare these",
    "highest one",
    "lowest one",
    "first one",
    "top 3",
    "top 5",
    "top five",
    "top",
    "bottom",
    "average",
    "difference",
    "percentage",
)


def _looks_like_follow_up(question: str) -> bool:
    """Return True when a question is likely contextual."""
    normalized = question.lower().strip()
    if not normalized:
        return False
    if normalized in {"why?", "why"}:
        return True
    if any(phrase in normalized for phrase in _CONTEXTUAL_PHRASES):
        return True
    if references_previous_result(question):
        return True
    if any(normalized.startswith(prefix) for prefix in _FOLLOW_UP_PREFIXES):
        return True
    if _mentions_active_entity(normalized):
        return True
    return len(normalized.split()) <= 3 and not normalized.startswith(("what is", "explain"))


def _build_system_prompt() -> str:
    """Return the resolver system prompt."""
    return (
        "You rewrite user follow-up questions.\n\n"
        "You receive:\n"
        "Conversation history\n"
        "Current state\n"
        "Active entities\n"
        "Last result context\n"
        "New question\n\n"
        "Return a standalone question.\n"
        "Do not answer.\n"
        "Preserve the user's meaning and combine relevant context from memory.\n"
        "If the current question corrects the prior turn, replace the incorrect "
        "dimension, date range, ranking direction, or filter instead of adding "
        "another condition.\n"
        "Memory fields can include previous_question, previous_sql, "
        "previous_answer, previous_rows, current_metric, analysis_dimension, "
        "date_range, and filters.\n"
        "Resolve pronouns and references like him, her, this driver, that warehouse, "
        "failed ones, or only delayed ones using active entities when possible.\n"
        "Resolve references like them, these, those, above, previous result, "
        "rank them, sort them, highest one, lowest one, first one, it, this, "
        "that, and same ones using last result context when available.\n"
        "For example, if previous results list Drew and John and the user asks "
        "'Which hub are they from?', rewrite to find hubs for Drew and John.\n"
        "Return only the rewritten standalone question."
    )


def _build_user_prompt(
    question: str,
    memory: ConversationMemory,
) -> str:
    """Return the resolver user prompt."""
    state = memory.get_current_state()
    return (
        f"Conversation history:\n{memory.get_recent_history()}\n\n"
        f"Current state:\n{state}\n\n"
        f"Active entities:\n{state.get('active_entities', {})}\n\n"
        f"Active metrics:\n{state.get('active_metrics', {})}\n\n"
        f"Current metric:\n{state.get('current_metric')}\n\n"
        f"Analysis dimension:\n{state.get('analysis_dimension')}\n\n"
        f"Date range:\n{state.get('date_range')}\n\n"
        f"Filters:\n{state.get('filters', {})}\n\n"
        f"Previous question:\n{state.get('previous_question')}\n\n"
        f"Previous answer:\n{state.get('previous_answer')}\n\n"
        f"Previous rows:\n{state.get('previous_rows')}\n\n"
        f"Last SQL context:\n{state.get('last_sql_context')}\n\n"
        f"Last result context:\n{state.get('last_result_context')}\n\n"
        f"New question:\n{question}\n\n"
        "Standalone question:"
    )


def references_previous_result(question: str) -> bool:
    """Return True when the question refers to the previous result table."""
    normalized = question.lower().strip()
    return any(phrase in normalized for phrase in _RESULT_REFERENCE_PHRASES)


def _mentions_active_entity(normalized_question: str) -> bool:
    """Detect common detail requests that often refer to active entities."""
    return "details" in normalized_question or "detail" in normalized_question


def resolve(question: str, memory: ConversationMemory) -> str:
    """
    Resolve a user question into a standalone question using memory.

    Non-follow-up questions, empty memory, and resolver failures return the
    original question unchanged.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    if not memory.get_recent_history():
        return question

    repair = detect_repair(question, memory)
    if repair.get("is_repair"):
        return str(repair.get("corrected_question") or question)

    if not _looks_like_follow_up(question):
        return question

    try:
        messages = [
            SystemMessage(content=_build_system_prompt()),
            HumanMessage(content=_build_user_prompt(question, memory)),
        ]
        response = get_llm().invoke(messages)
        content = response.content
        resolved = content.strip() if isinstance(content, str) else str(content).strip()
        return resolved or question
    except Exception:
        return question
