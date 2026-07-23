"""Production intent classification for the GOFO Operations Intelligence Agent.

This module is the first stage of the agent pipeline. It decides *what* the user
wants; it does not execute tools or answer the question.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

import config
from core.logger import get_logger
from tools.llm.client import get_llm

logger = get_logger("intent_classifier")

CONFIDENCE_THRESHOLD = 0.5


class IntentType(StrEnum):
    """Primary user intents supported by the classifier."""

    Greeting = "Greeting"
    ChitChat = "ChitChat"
    General_Knowledge = "General_Knowledge"
    SOP_QA = "SOP_QA"
    SOP_Summary = "SOP_Summary"
    SOP_Compare = "SOP_Compare"
    SQL_Query = "SQL_Query"
    SQL_Analysis = "SQL_Analysis"
    Dashboard = "Dashboard"
    Explain_Result = "Explain_Result"
    Follow_Up = "Follow_Up"
    Upload_File = "Upload_File"
    Coding = "Coding"
    Unknown = "Unknown"


class IntentClassification(BaseModel):
    """Structured output of intent classification."""

    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    requires_sql: bool = False
    requires_rag: bool = False
    requires_memory: bool = False
    requires_planner: bool = False
    requires_clarification: bool = False
    reasoning: str = Field(default="", description="Debug-only explanation.")

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, score))


_GREETING_EXACT = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "thx",
    }
)

_CHITCHAT_PHRASES = (
    "how are you",
    "how's it going",
    "tell me a joke",
    "who are you",
    "what can you do",
)

_FOLLOWUP_PHRASES = (
    "what about",
    "how about",
    "and ",
    "compare with",
    "explain more",
    "tell me more",
    "show top",
    "show only",
    "instead",
    "same for",
    "drill down",
    "break it down",
    "why is that",
    "go on",
    "continue",
    "more detail",
)

_UPLOAD_PHRASES = (
    "analyze this",
    "analyse this",
    "summarize this",
    "summarise this",
    "this csv",
    "this excel",
    "this pdf",
    "this file",
    "uploaded file",
    "upload a file",
    "i will upload",
    "i'll upload",
)

_DASHBOARD_PHRASES = (
    "dashboard",
    "kpi visualization",
    "kpi visualisation",
    "plot pickup",
    "plot trend",
    "create a chart",
    "create charts",
    "visualize",
    "visualise",
)

_CODING_PHRASES = (
    "write python",
    "write a python",
    "debug sql",
    "debug this sql",
    "explain langgraph",
    "write code",
    "implement a function",
)

_SOP_PHRASES = (
    "sop",
    "procedure",
    "policy",
    "workflow",
    "check in",
    "check-in",
    "exception handling",
    "cbt",
)

_SQL_ANALYSIS_PHRASES = (
    "why did",
    "root cause",
    "worst",
    "best",
    "lowest",
    "highest",
    "compare",
    "vs ",
    "versus",
    "week over week",
    "wow",
    "decrease",
    "increase",
    "performance",
    "success rate",
    "completion rate",
    "failure rate",
)

_SQL_QUERY_PHRASES = (
    "show today's",
    "show todays",
    "list all",
    "list failed",
    "how many",
    "count of",
    "pickups today",
    "today's pickups",
    "todays pickups",
)


def _history_from_memory(conversation_memory: Any) -> list[dict[str, Any]]:
    if conversation_memory is None:
        return []
    if hasattr(conversation_memory, "get_recent_history"):
        return list(conversation_memory.get_recent_history() or [])
    if isinstance(conversation_memory, list):
        return conversation_memory
    if isinstance(conversation_memory, dict):
        return list(conversation_memory.get("turns") or conversation_memory.get("history") or [])
    return []


def _format_history(history: list[dict[str, Any]], limit: int = 6) -> str:
    if not history:
        return "(no prior turns)"
    lines: list[str] = []
    for turn in history[-limit:]:
        user_q = turn.get("user_question") or turn.get("question") or ""
        answer = turn.get("assistant_answer") or turn.get("answer") or ""
        lines.append(f"User: {user_q}")
        if answer:
            lines.append(f"Assistant: {str(answer)[:240]}")
    return "\n".join(lines) if lines else "(no prior turns)"


def _defaults_for_intent(intent: IntentType) -> dict[str, bool]:
    """Return default tool/memory flags for a primary intent."""
    if intent in {
        IntentType.Greeting,
        IntentType.ChitChat,
        IntentType.General_Knowledge,
        IntentType.Coding,
    }:
        return {
            "requires_sql": False,
            "requires_rag": False,
            "requires_memory": False,
            "requires_planner": False,
            "requires_clarification": False,
        }
    if intent in {IntentType.SOP_QA, IntentType.SOP_Summary}:
        return {
            "requires_sql": False,
            "requires_rag": True,
            "requires_memory": False,
            "requires_planner": False,
            "requires_clarification": False,
        }
    if intent == IntentType.SOP_Compare:
        return {
            "requires_sql": False,
            "requires_rag": True,
            "requires_memory": False,
            "requires_planner": True,
            "requires_clarification": False,
        }
    if intent == IntentType.SQL_Query:
        return {
            "requires_sql": True,
            "requires_rag": False,
            "requires_memory": False,
            "requires_planner": False,
            "requires_clarification": False,
        }
    if intent == IntentType.SQL_Analysis:
        return {
            "requires_sql": True,
            "requires_rag": False,
            "requires_memory": True,
            "requires_planner": True,
            "requires_clarification": False,
        }
    if intent == IntentType.Dashboard:
        return {
            "requires_sql": True,
            "requires_rag": False,
            "requires_memory": False,
            "requires_planner": True,
            "requires_clarification": False,
        }
    if intent in {IntentType.Explain_Result, IntentType.Follow_Up}:
        return {
            "requires_sql": False,
            "requires_rag": False,
            "requires_memory": True,
            "requires_planner": True,
            "requires_clarification": False,
        }
    if intent == IntentType.Upload_File:
        return {
            "requires_sql": False,
            "requires_rag": False,
            "requires_memory": True,
            "requires_planner": True,
            "requires_clarification": False,
        }
    return {
        "requires_sql": False,
        "requires_rag": False,
        "requires_memory": False,
        "requires_planner": False,
        "requires_clarification": True,
    }


def _apply_defaults(classification: IntentClassification) -> IntentClassification:
    defaults = _defaults_for_intent(classification.intent)
    # Preserve LLM overrides when they strengthen requirements; fill gaps from defaults.
    data = classification.model_dump()
    for key, default_value in defaults.items():
        if key == "requires_clarification":
            data[key] = bool(data.get(key)) or default_value
        elif default_value:
            data[key] = True if data.get(key) is None else bool(data.get(key)) or default_value
        elif data.get(key) is None:
            data[key] = default_value
    return IntentClassification.model_validate(data)


def _enforce_confidence(classification: IntentClassification) -> IntentClassification:
    if classification.confidence < CONFIDENCE_THRESHOLD:
        return IntentClassification(
            intent=IntentType.Unknown,
            confidence=classification.confidence,
            requires_sql=False,
            requires_rag=False,
            requires_memory=classification.requires_memory,
            requires_planner=False,
            requires_clarification=True,
            reasoning=(
                classification.reasoning
                or f"Confidence {classification.confidence:.2f} below threshold {CONFIDENCE_THRESHOLD}."
            ),
        )
    return classification


def _looks_like_followup(normalized: str, has_history: bool) -> bool:
    if not has_history:
        return False
    if any(phrase in normalized for phrase in _FOLLOWUP_PHRASES):
        return True
    tokens = normalized.split()
    if len(tokens) <= 6 and any(
        token in {"it", "that", "them", "this", "those", "same", "instead"} for token in tokens
    ):
        return True
    if normalized in {"why", "why?", "continue", "details", "more", "explain"}:
        return True
    return False


def _heuristic_classify(
    question: str,
    conversation_memory: Any,
) -> IntentClassification | None:
    """Fast path for obvious intents; returns None when GPT should decide."""
    normalized = question.lower().strip()
    if not normalized:
        return IntentClassification(
            intent=IntentType.Unknown,
            confidence=0.99,
            requires_clarification=True,
            reasoning="Empty question.",
        )

    history = _history_from_memory(conversation_memory)
    has_history = bool(history)

    if normalized in _GREETING_EXACT:
        return _apply_defaults(
            IntentClassification(
                intent=IntentType.Greeting,
                confidence=0.99,
                reasoning="Exact greeting match.",
            )
        )

    if any(phrase in normalized for phrase in _CHITCHAT_PHRASES):
        return _apply_defaults(
            IntentClassification(
                intent=IntentType.ChitChat,
                confidence=0.96,
                reasoning="Chit-chat phrase match.",
            )
        )

    if _looks_like_followup(normalized, has_history):
        explain = any(
            phrase in normalized
            for phrase in ("explain", "why is this", "why are these", "these numbers", "this chart")
        )
        intent = IntentType.Explain_Result if explain and has_history else IntentType.Follow_Up
        return _apply_defaults(
            IntentClassification(
                intent=intent,
                confidence=0.93,
                requires_memory=True,
                requires_planner=True,
                reasoning="Short conversational follow-up with prior turns.",
            )
        )

    if any(phrase in normalized for phrase in _UPLOAD_PHRASES) or re.search(
        r"\b(csv|xlsx|xls|pdf|docx)\b", normalized
    ):
        return _apply_defaults(
            IntentClassification(
                intent=IntentType.Upload_File,
                confidence=0.94,
                reasoning="Upload / document analysis phrasing.",
            )
        )

    if any(phrase in normalized for phrase in _DASHBOARD_PHRASES):
        return _apply_defaults(
            IntentClassification(
                intent=IntentType.Dashboard,
                confidence=0.94,
                reasoning="Dashboard / visualization request.",
            )
        )

    if any(phrase in normalized for phrase in _CODING_PHRASES):
        return _apply_defaults(
            IntentClassification(
                intent=IntentType.Coding,
                confidence=0.93,
                reasoning="Coding / debugging request.",
            )
        )

    sop_hit = any(phrase in normalized for phrase in _SOP_PHRASES)
    if sop_hit:
        if "compare" in normalized or " vs " in normalized or "versus" in normalized:
            intent = IntentType.SOP_Compare
        elif "summarize" in normalized or "summarise" in normalized or "key points" in normalized:
            intent = IntentType.SOP_Summary
        else:
            intent = IntentType.SOP_QA
        return _apply_defaults(
            IntentClassification(
                intent=intent,
                confidence=0.92,
                reasoning="SOP / policy phrasing.",
            )
        )

    ops_entity = any(
        term in normalized
        for term in (
            "hub",
            "hubs",
            "driver",
            "drivers",
            "pickup",
            "pickups",
            "customer",
            "customers",
            "warehouse",
            "chicago",
            "performance",
            "rate",
            "delay",
            "delayed",
            "failed",
            "operations",
            "kpi",
        )
    )

    if any(phrase in normalized for phrase in _SQL_ANALYSIS_PHRASES) and ops_entity:
        return _apply_defaults(
            IntentClassification(
                intent=IntentType.SQL_Analysis,
                confidence=0.91,
                reasoning="Analytical SQL / ops performance phrasing.",
            )
        )

    if re.search(r"\brank\b", normalized) and ops_entity:
        return _apply_defaults(
            IntentClassification(
                intent=IntentType.SQL_Analysis,
                confidence=0.9,
                reasoning="Ranking request over operational entities.",
            )
        )

    if any(phrase in normalized for phrase in _SQL_QUERY_PHRASES) or (
        ("show" in normalized or "list" in normalized or "how many" in normalized)
        and ops_entity
    ):
        return _apply_defaults(
            IntentClassification(
                intent=IntentType.SQL_Query,
                confidence=0.9,
                reasoning="Direct SQL listing / count phrasing.",
            )
        )

    if ops_entity and any(
        term in normalized
        for term in ("today", "yesterday", "last week", "this week", "highest", "lowest")
    ):
        return _apply_defaults(
            IntentClassification(
                intent=IntentType.SQL_Query,
                confidence=0.86,
                reasoning="Operational entity + time/rank phrasing.",
            )
        )

    return None


def _parse_json(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
    return json.loads(content)


def _system_prompt(*, question: str = "", history: str = "") -> str:
    """Render intent classifier system prompt from the Prompt Registry."""
    from core.prompt_manager import get_prompt_manager

    intents = ", ".join(item.value for item in IntentType)
    _selection, rendered = get_prompt_manager().render(
        "router.intent_classifier_prompt",
        {
            "question": question or "(see user message)",
            "allowed_intents": intents,
            "history": history or "(none)",
        },
    )
    return rendered


def _debug_print(classification: IntentClassification) -> None:
    required_tools: list[str] = []
    if classification.requires_sql:
        required_tools.append("SQL")
    if classification.requires_rag:
        required_tools.append("RAG")
    if classification.intent == IntentType.Dashboard:
        required_tools.append("VISUALIZATION")
    if classification.intent == IntentType.Upload_File:
        required_tools.append("ATTACHMENT")
    if not required_tools and classification.intent in {
        IntentType.Greeting,
        IntentType.ChitChat,
        IntentType.General_Knowledge,
        IntentType.Coding,
    }:
        required_tools.append("LLM")

    lines = [
        "----------------------------------",
        f"Detected Intent: {classification.intent.value}",
        f"Confidence: {classification.confidence}",
        f"Required Tools: {', '.join(required_tools) or 'none'}",
        f"Memory Required: {classification.requires_memory}",
        f"Planner Required: {classification.requires_planner}",
        f"Clarification Required: {classification.requires_clarification}",
        f"Reasoning: {classification.reasoning}",
        "----------------------------------",
    ]
    message = "\n".join(lines)
    if config.DEBUG:
        print(message)
    logger.info(message)


class IntentClassifier:
    """Classify user questions into a single primary intent."""

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    def classify(
        self,
        question: str,
        conversation_memory: Any = None,
    ) -> IntentClassification:
        """Classify a question using heuristics first, then GPT when needed."""
        question = (question or "").strip()
        heuristic = _heuristic_classify(question, conversation_memory)
        if heuristic is not None:
            result = _enforce_confidence(heuristic)
            _debug_print(result)
            return result

        history = _history_from_memory(conversation_memory)
        history_text = _format_history(history)
        from core.prompt_manager import get_prompt_manager

        manager = get_prompt_manager()
        selection, messages = manager.build_messages(
            "router.intent_classifier_prompt",
            {
                "question": question,
                "allowed_intents": ", ".join(item.value for item in IntentType),
                "history": history_text,
            },
            user_content=(
                f"Conversation history:\n{history_text}\n\n"
                f"Latest user message:\n{question}"
            ),
        )
        try:
            llm = self._llm or manager.llm_for(selection)
            response = llm.invoke(messages)
            content = response.content if hasattr(response, "content") else response
            raw = content if isinstance(content, str) else str(content)
            parsed = _parse_json(raw)
            classification = IntentClassification.model_validate(parsed)
        except Exception as exc:  # noqa: BLE001 - fall back safely
            logger.warning("Intent classifier LLM failed: %s", exc)
            classification = IntentClassification(
                intent=IntentType.Unknown,
                confidence=0.2,
                requires_clarification=True,
                reasoning=f"Classifier LLM unavailable or invalid output: {exc}",
            )

        classification = _apply_defaults(classification)
        classification = _enforce_confidence(classification)
        _debug_print(classification)
        return classification


def classify_intent(
    question: str,
    conversation_memory: Any = None,
    *,
    llm: Any | None = None,
) -> IntentClassification:
    """Module-level helper matching the planner/classifier API style."""
    return IntentClassifier(llm=llm).classify(question, conversation_memory)
