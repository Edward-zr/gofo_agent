"""LLM-based semantic request analysis for GOFO routing."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from core.logger import get_logger
from tools.llm.client import get_llm
from tools.orchestration.models import SemanticAnalysis

logger = get_logger("orchestration.semantic")


def analyze_request(
    question: str,
    *,
    resolved_question: str | None = None,
    conversation_state: dict[str, Any] | None = None,
    repair_detected: bool = False,
    has_attachments: bool = False,
) -> SemanticAnalysis:
    """Understand user intent, domain, and execution plan using the configured LLM."""
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    resolved = resolved_question or question
    if has_attachments:
        return _attachment_fallback(question, resolved)

    state = conversation_state or {}
    context_summary = _summarize_conversation_state(state)
    try:
        messages = [
            SystemMessage(content=_system_prompt()),
            HumanMessage(
                content=_user_prompt(
                    question=question,
                    resolved_question=resolved,
                    context_summary=context_summary,
                    repair_detected=repair_detected,
                    has_attachments=has_attachments,
                )
            ),
        ]
        response = get_llm().invoke(messages)
        content = response.content
        raw = content if isinstance(content, str) else str(content)
        payload = _parse_json(raw)
        analysis = SemanticAnalysis.model_validate(payload)
        if not analysis.resolved_question:
            analysis.resolved_question = resolved
        return _normalize_analysis(analysis, question=question, resolved_question=resolved)
    except Exception as exc:
        logger.warning("Semantic analysis fallback after LLM failure: %s", exc)
        return _heuristic_fallback(question, resolved_question=resolved, has_attachments=has_attachments)


def _system_prompt() -> str:
    return (
        "You are the semantic request analyzer for the GOFO Operations Intelligence Agent.\n"
        "Your job is to UNDERSTAND the user's request and decide how the operations analyst should respond.\n"
        "Do NOT answer operational data questions with fabricated metrics.\n"
        "For general conversation, greetings, help, or capability questions, provide a helpful direct_reply.\n\n"
        "Classify into:\n"
        "1. domain:\n"
        "   - general_conversation: greetings, thanks, help, what can you do, small talk\n"
        "   - sop_knowledge: SOP, policy, procedure, CBT, workflow, how-to process questions\n"
        "   - data_analytics: pickups, hubs, drivers, customers, KPIs, rankings, trends, performance\n"
        "   - attachment: uploaded file/image analysis (only when attachments are present)\n"
        "   - memory: follow-up using prior analysis without new data retrieval\n\n"
        "2. sub_intent (pick the best fit):\n"
        "   ranking, detail, analysis, lookup, comparison, trend, summary, recommendation,\n"
        "   root_cause, explanation, overview, anomaly, greeting, help, clarification, unknown\n\n"
        "3. capability:\n"
        "   - sql: operational database analytics\n"
        "   - rag: SOP / policy knowledge only\n"
        "   - multi: needs both live operational data and SOP guidance\n"
        "   - conversation: greetings, help, or general analyst replies without SQL/RAG\n"
        "   - unknown: only if truly unrelated and no helpful direct_reply is possible\n\n"
        "4. response_mode: direct, analytical, executive, investigative, conversational\n\n"
        "Rules:\n"
        "- Prefer sql for operational metrics, rankings, lookups, comparisons, and trends.\n"
        "- Prefer rag for SOP definitions and process/policy questions.\n"
        "- Prefer multi when the user needs both current metrics and operational guidance.\n"
        "- Prefer conversation for hello/thanks/help instead of unknown.\n"
        "- If conversation context exists, set use_previous_result or use_previous_analysis when appropriate.\n"
        "- Never return unknown for greetings; use conversation with direct_reply.\n"
        "- resolved_question should be a standalone, explicit operations question.\n\n"
        "Return ONLY JSON:\n"
        "{\n"
        '  "domain": "general_conversation|sop_knowledge|data_analytics|attachment|memory",\n'
        '  "sub_intent": "ranking|detail|analysis|lookup|comparison|trend|summary|recommendation|'
        'root_cause|explanation|overview|anomaly|greeting|help|clarification|unknown",\n'
        '  "capability": "sql|rag|multi|conversation|unknown",\n'
        '  "response_mode": "direct|analytical|executive|investigative|conversational",\n'
        '  "confidence": 0.0,\n'
        '  "resolved_question": "standalone question",\n'
        '  "reasoning": "brief reason",\n'
        '  "entities": {},\n'
        '  "metric": null,\n'
        '  "dimension": null,\n'
        '  "use_previous_result": false,\n'
        '  "use_previous_analysis": false,\n'
        '  "requires_sql": false,\n'
        '  "requires_rag": false,\n'
        '  "direct_reply": null,\n'
        '  "planner_intent": "short_snake_case_intent"\n'
        "}"
    )


def _user_prompt(
    *,
    question: str,
    resolved_question: str,
    context_summary: str,
    repair_detected: bool,
    has_attachments: bool,
) -> str:
    return (
        f"User question:\n{question}\n\n"
        f"Resolved question:\n{resolved_question}\n\n"
        f"Conversation context:\n{context_summary or 'None'}\n\n"
        f"Repair detected: {repair_detected}\n"
        f"Attachments present: {has_attachments}\n"
    )


def _summarize_conversation_state(state: dict[str, Any]) -> str:
    if not state:
        return ""
    parts = []
    if state.get("last_question"):
        parts.append(f"Last question: {state['last_question']}")
    if state.get("last_resolved_question"):
        parts.append(f"Last resolved question: {state['last_resolved_question']}")
    if state.get("last_intent"):
        parts.append(f"Last intent: {state['last_intent']}")
    if state.get("last_metric"):
        parts.append(f"Last metric: {state['last_metric']}")
    if state.get("last_dimension"):
        parts.append(f"Last dimension: {state['last_dimension']}")
    if state.get("last_entities"):
        parts.append(f"Last entities: {state['last_entities']}")
    if state.get("summary"):
        parts.append(f"Last summary: {str(state['summary'])[:400]}")
    return "\n".join(parts)


def _parse_json(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
    return json.loads(content)


def _normalize_analysis(
    analysis: SemanticAnalysis,
    *,
    question: str,
    resolved_question: str,
) -> SemanticAnalysis:
    """Apply deterministic guardrails after LLM classification."""
    normalized = question.lower().strip()
    if normalized in {"hello", "hi", "hey", "good morning", "good afternoon"}:
        return analysis.model_copy(
            update={
                "domain": "general_conversation",
                "sub_intent": "greeting",
                "capability": "conversation",
                "response_mode": "conversational",
                "requires_sql": False,
                "requires_rag": False,
                "direct_reply": analysis.direct_reply
                or (
                    "Hello. I am the GOFO Operations Intelligence Analyst. "
                    "I can analyze pickup performance, investigate root causes, "
                    "compare hubs and drivers, and answer SOP questions."
                ),
                "planner_intent": "greeting",
                "confidence": max(analysis.confidence, 0.9),
            }
        )
    if any(phrase in normalized for phrase in ("what can you do", "help me", "how can you help")):
        return analysis.model_copy(
            update={
                "domain": "general_conversation",
                "sub_intent": "help",
                "capability": "conversation",
                "response_mode": "conversational",
                "requires_sql": False,
                "requires_rag": False,
                "direct_reply": analysis.direct_reply
                or (
                    "I can help with GOFO operational analytics and SOP knowledge. "
                    "Ask about pickup volume, hub or driver performance, delays, root causes, "
                    "recommendations, or process questions like pickup SOP and CBT."
                ),
                "planner_intent": "help",
                "confidence": max(analysis.confidence, 0.9),
            }
        )
    if analysis.capability == "unknown" and analysis.domain == "data_analytics":
        return analysis.model_copy(
            update={
                "capability": "sql",
                "requires_sql": True,
                "requires_rag": False,
                "confidence": max(analysis.confidence, 0.55),
                "reasoning": analysis.reasoning or "Operational phrasing suggests database analytics.",
            }
        )
    if not analysis.resolved_question:
        analysis.resolved_question = resolved_question
    return analysis


def _attachment_fallback(question: str, resolved_question: str) -> SemanticAnalysis:
    normalized = question.lower()
    sub_intent = "summary"
    if any(phrase in normalized for phrase in ("analyze", "analysis", "discover", "insight", "problem", "risk")):
        sub_intent = "analysis"
    elif any(phrase in normalized for phrase in ("compare", "versus", "vs")):
        sub_intent = "comparison"
    return SemanticAnalysis(
        domain="attachment",
        sub_intent=sub_intent,
        capability="conversation",
        response_mode="analytical",
        confidence=0.9,
        resolved_question=resolved_question,
        reasoning="Attachment present; route to file analysis without semantic LLM.",
        planner_intent="file_analysis",
    )


def _heuristic_fallback(
    question: str,
    *,
    resolved_question: str,
    has_attachments: bool,
) -> SemanticAnalysis:
    if has_attachments:
        return _attachment_fallback(question, resolved_question)
    normalized = question.lower().strip()
    if normalized in {"hello", "hi", "hey", "good morning", "good afternoon"}:
        return _normalize_analysis(
            SemanticAnalysis(
                domain="general_conversation",
                sub_intent="greeting",
                capability="conversation",
                response_mode="conversational",
                confidence=0.75,
                resolved_question=resolved_question,
                reasoning="Heuristic greeting fallback.",
                planner_intent="greeting",
            ),
            question=question,
            resolved_question=resolved_question,
        )
    return SemanticAnalysis(
        domain="data_analytics",
        sub_intent="analysis",
        capability="sql",
        response_mode="analytical",
        confidence=0.5,
        resolved_question=resolved_question,
        reasoning="Heuristic fallback after semantic LLM failure.",
        requires_sql=True,
        planner_intent="operational_analysis",
    )
