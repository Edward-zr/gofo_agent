"""Adapter that runs GOFOAgent and captures an evaluation trace."""

from __future__ import annotations

import time
import traceback
from typing import Any, Callable

from evaluation.cost import estimate_usage


def _tools_from_response(payload: dict[str, Any]) -> list[str]:
    analysis = payload.get("analysis") or {}
    plan = analysis.get("execution_plan") or {}
    tools: list[str] = []
    for step in plan.get("steps") or []:
        tool = step.get("tool")
        if tool and tool not in tools:
            tools.append(str(tool))
    for item in analysis.get("step_results_summary") or []:
        tool = item.get("tool")
        if tool and str(tool) not in tools:
            tools.append(str(tool))
    if not tools:
        capability = str(analysis.get("capability") or "").lower()
        if capability in {"sql", "rag", "clarification", "conversation"}:
            tools.append(capability.upper() if capability != "conversation" else "LLM")
    return tools


def _tool_order_from_response(payload: dict[str, Any]) -> list[str]:
    analysis = payload.get("analysis") or {}
    plan = analysis.get("execution_plan") or {}
    order: list[str] = []
    for step in sorted(plan.get("steps") or [], key=lambda s: s.get("step_number") or 0):
        tool = step.get("tool")
        if tool:
            order.append(str(tool))
    if order:
        return order
    summary = analysis.get("step_results_summary") or []
    return [str(item.get("tool")) for item in summary if item.get("tool")]


def extract_trace_fields(payload: dict[str, Any], *, question: str, latency_ms: float) -> dict[str, Any]:
    """Normalize a GOFOAgent.ask() dict into a stable evaluation trace."""
    analysis = payload.get("analysis") or {}
    raw = payload.get("raw") or {}
    answer = payload.get("answer") or ""
    usage = estimate_usage(
        prompt_text=question,
        completion_text=answer,
        usage=(raw.get("token_usage") or analysis.get("token_usage")),
    )
    sources = payload.get("sources") or []
    charts = payload.get("charts") or analysis.get("charts") or []
    return {
        "question": question,
        "answer": answer,
        "sql": payload.get("sql") or raw.get("generated_sql") or "",
        "sql_rows": payload.get("data") or raw.get("sql_rows") or [],
        "sources": sources,
        "retrieved_documents": sources,
        "charts": charts,
        "recommendations": payload.get("recommendations") or [],
        "capability": analysis.get("capability") or raw.get("capability"),
        "intent": analysis.get("primary_intent") or analysis.get("intent"),
        "intent_classification": analysis.get("intent_classification") or {},
        "execution_plan": analysis.get("execution_plan") or {},
        "step_results_summary": analysis.get("step_results_summary") or [],
        "tools_used": _tools_from_response(payload),
        "tool_order": _tool_order_from_response(payload),
        "requires_clarification": bool(analysis.get("requires_clarification")),
        "clarification_question": analysis.get("clarification_question"),
        "clarification_options": analysis.get("clarification_options") or [],
        "missing_fields": analysis.get("missing_fields") or [],
        "resolved_question": analysis.get("resolved_question"),
        "memory_state": {
            "memory_turns": analysis.get("memory_turns"),
            "current_entities": analysis.get("current_entities") or {},
            "previous_question": analysis.get("previous_question"),
            "new_state": analysis.get("new_state") or {},
            "pending_clarification": (analysis.get("agent_state") or {}).get("pending_clarification"),
        },
        "agent_state": analysis.get("agent_state") or raw.get("agent_state") or {},
        "confidence_score": raw.get("confidence_score")
        or (analysis.get("agent_state") or {}).get("confidence_score"),
        "confidence_level": raw.get("confidence_level")
        or (analysis.get("agent_state") or {}).get("confidence_level"),
        "fallback_strategy": raw.get("fallback_strategy")
        or (analysis.get("agent_state") or {}).get("fallback_strategy"),
        "quality_report": analysis.get("quality_report") or {},
        "qa_retry_count": analysis.get("qa_retry_count") or 0,
        "reflection_retry_count": analysis.get("reflection_retry_count") or 0,
        "latency_ms": latency_ms,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "estimated_cost_usd": usage["estimated_cost_usd"],
        "error": None,
    }


def run_live_question(
    agent: Any,
    question: str,
    *,
    attachment_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Execute one live agent turn and return a trace dict."""
    started = time.perf_counter()
    try:
        payload = agent.ask(question, attachment_ids=attachment_ids)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if not isinstance(payload, dict):
            payload = {"answer": str(payload), "analysis": {}, "raw": {}}
        return extract_trace_fields(payload, question=question, latency_ms=latency_ms)
    except Exception as exc:  # noqa: BLE001 — evaluation must capture failures
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = estimate_usage(prompt_text=question, completion_text="")
        return {
            "question": question,
            "answer": "",
            "sql": "",
            "sql_rows": [],
            "sources": [],
            "retrieved_documents": [],
            "charts": [],
            "recommendations": [],
            "capability": None,
            "intent": None,
            "intent_classification": {},
            "execution_plan": {},
            "step_results_summary": [],
            "tools_used": [],
            "tool_order": [],
            "requires_clarification": False,
            "clarification_question": None,
            "clarification_options": [],
            "missing_fields": [],
            "resolved_question": question,
            "memory_state": {},
            "agent_state": {},
            "confidence_score": None,
            "confidence_level": None,
            "fallback_strategy": None,
            "quality_report": {},
            "qa_retry_count": 0,
            "reflection_retry_count": 0,
            "latency_ms": latency_ms,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "estimated_cost_usd": usage["estimated_cost_usd"],
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def make_mock_responder(fixtures: dict[str, dict[str, Any]] | None = None) -> Callable[[str], dict[str, Any]]:
    """Build a deterministic responder for offline framework tests."""

    fixtures = fixtures or {}

    def _respond(question: str, attachment_ids: list[str] | None = None) -> dict[str, Any]:
        del attachment_ids
        key = question.strip().lower()
        for pattern, fixture in fixtures.items():
            if pattern.lower() in key:
                started = time.perf_counter()
                time.sleep(0)  # keep signature similar
                latency_ms = (time.perf_counter() - started) * 1000.0
                payload = {
                    "answer": fixture.get("answer", "Mock answer."),
                    "sql": fixture.get("sql", ""),
                    "data": fixture.get("sql_rows", []),
                    "sources": fixture.get("sources", []),
                    "charts": fixture.get("charts", []),
                    "recommendations": fixture.get("recommendations", []),
                    "analysis": {
                        "capability": fixture.get("capability", "sql"),
                        "primary_intent": fixture.get("intent"),
                        "execution_plan": fixture.get("execution_plan")
                        or {
                            "steps": [
                                {"step_number": i + 1, "tool": tool}
                                for i, tool in enumerate(fixture.get("tools") or [])
                            ]
                        },
                        "step_results_summary": [
                            {"tool": tool} for tool in (fixture.get("tools") or [])
                        ],
                        "requires_clarification": fixture.get("requires_clarification", False),
                        "clarification_question": fixture.get("clarification_question"),
                        "memory_turns": fixture.get("memory_turns", 1),
                        "current_entities": fixture.get("entities") or {},
                        "resolved_question": fixture.get("resolved_question", question),
                        "agent_state": fixture.get("agent_state") or {},
                        "qa_retry_count": fixture.get("qa_retry_count", 0),
                    },
                    "raw": {
                        "confidence_score": fixture.get("confidence_score"),
                        "confidence_level": fixture.get("confidence_level"),
                        "fallback_strategy": fixture.get("fallback_strategy"),
                        "generated_sql": fixture.get("sql", ""),
                    },
                }
                return extract_trace_fields(payload, question=question, latency_ms=max(latency_ms, 5.0))
        # Default heuristic mock
        started = time.perf_counter()
        q = key
        tools: list[str] = []
        capability = "llm"
        answer = "I can help with GOFO operations."
        sql = ""
        requires_clarification = False
        clarification_question = None
        charts: list[dict[str, Any]] = []
        recommendations: list[str] = []
        sources: list[dict[str, Any]] = []
        if any(w in q for w in ("sop", "return", "policy", "procedure", "how do i")):
            tools = ["RAG", "LLM"]
            capability = "rag"
            answer = "Based on the available SOP documentation, follow the documented procedure."
            sources = [{"id": "sop::1", "text": "SOP procedure", "score": 0.9, "metadata": {}}]
        elif any(w in q for w in ("pickup", "hub", "driver", "rate", "sql", "today", "chicago", "rank")):
            tools = ["SQL", "LLM"]
            capability = "sql"
            answer = "Pickup performance summary for the requested scope."
            sql = "SELECT 1;"
        elif "chart" in q or "bar" in q or "plot" in q:
            tools = ["SQL", "VISUALIZATION", "LLM"]
            capability = "sql"
            answer = "Here is a bar chart of the requested metric."
            charts = [{"type": "bar", "title": "Performance"}]
        elif "recommend" in q or "improvement" in q:
            tools = ["SQL", "STATISTICS", "RECOMMENDATION", "LLM"]
            capability = "sql"
            answer = "Recommend focusing coaching on low-rate hubs."
            recommendations = ["Coach low-rate drivers", "Rebalance capacity"]
        elif "upload" in q or "attach" in q or "file" in q:
            tools = ["ATTACHMENT", "LLM"]
            capability = "attachment"
            answer = "Please upload the file using the attachment control."
        elif any(w in q for w in ("best driver", "show performance", "compare pickup")):
            tools = []
            capability = "clarification"
            requires_clarification = True
            clarification_question = "I need a bit more detail before I run analytics."
            answer = clarification_question
        elif any(w in q for w in ("hello", "hi ", "what can you", "who are you")):
            tools = ["LLM"]
            capability = "conversation"
            answer = "Hello. I am the GOFO Operations Intelligence Analyst."
        latency_ms = (time.perf_counter() - started) * 1000.0 + 12.0
        payload = {
            "answer": answer,
            "sql": sql,
            "data": [{"ok": 1}] if sql else [],
            "sources": sources,
            "charts": charts,
            "recommendations": recommendations,
            "analysis": {
                "capability": capability,
                "execution_plan": {
                    "steps": [{"step_number": i + 1, "tool": t} for i, t in enumerate(tools)]
                },
                "step_results_summary": [{"tool": t} for t in tools],
                "requires_clarification": requires_clarification,
                "clarification_question": clarification_question,
                "memory_turns": 1,
                "resolved_question": question,
                "agent_state": {},
            },
            "raw": {
                "confidence_score": 0.85 if sources else None,
                "confidence_level": "HIGH" if sources else None,
                "generated_sql": sql,
            },
        }
        return extract_trace_fields(payload, question=question, latency_ms=latency_ms)

    return _respond


class EvalAgent:
    """Thin wrapper: live GOFOAgent or mock responder."""

    def __init__(self, *, mode: str = "live", mock_responder: Callable | None = None) -> None:
        self.mode = mode
        self._base_mock = mock_responder or make_mock_responder()
        self._turn = 0
        self._history: list[str] = []
        self._entities: dict[str, str] = {}
        self._agent = None
        if mode == "live":
            from core.agent import GOFOAgent

            self._agent = GOFOAgent()

    def ask(self, question: str, attachment_ids: list[str] | None = None) -> dict[str, Any]:
        if self.mode == "live" and self._agent is not None:
            return run_live_question(self._agent, question, attachment_ids=attachment_ids)

        self._turn += 1
        q = question.strip().lower()
        for hub in ("chicago", "los angeles", "atlanta", "dallas", "seattle"):
            if hub in q:
                self._entities["hub"] = hub.title()
        if "driver" in q:
            self._entities["dimension"] = "driver"
        if "pickup rate" in q:
            self._entities["metric"] = "pickup rate"

        trace = self._base_mock(question, attachment_ids=attachment_ids)
        # Enrich mock answers with question keywords for offline scoring stability.
        tokens = [t for t in q.replace("?", " ").split() if len(t) > 3][:6]
        answer = trace.get("answer") or ""
        for token in tokens:
            if token not in answer.lower():
                answer = f"{answer} {token}".strip()
        if self._entities.get("hub") and self._entities["hub"].lower() not in answer.lower():
            answer = f"{answer} Focus: {self._entities['hub']}."
        if self._entities.get("metric") and "rate" not in answer.lower():
            answer = f"{answer} Metric: {self._entities['metric']}."
        trace["answer"] = answer

        tools = list(trace.get("tools_used") or [])
        order = list(trace.get("tool_order") or [])
        if self._turn > 1 and "MEMORY" not in {t.upper() for t in tools}:
            # Follow-up turns in mock mode include memory.
            tools = ["MEMORY", *tools] if tools else ["MEMORY", "LLM"]
            order = ["MEMORY", *order] if order else ["MEMORY", "LLM"]
            plan_steps = [{"step_number": i + 1, "tool": t} for i, t in enumerate(order)]
            trace["tools_used"] = tools
            trace["tool_order"] = order
            trace["execution_plan"] = {"steps": plan_steps}
            trace["step_results_summary"] = [{"tool": t} for t in order]
        mem = dict(trace.get("memory_state") or {})
        mem["memory_turns"] = self._turn
        mem["current_entities"] = dict(self._entities)
        if self._history:
            mem["previous_question"] = self._history[-1]
            if self._entities.get("hub") and self._entities["hub"].lower() not in q:
                trace["resolved_question"] = f"{question} ({self._entities['hub']})"
        trace["memory_state"] = mem
        self._history.append(question)
        return trace

    def reset(self) -> None:
        """Start a fresh conversation (new agent session)."""
        self._turn = 0
        self._history = []
        self._entities = {}
        if self.mode == "live":
            from core.agent import GOFOAgent

            self._agent = GOFOAgent()
