"""Multi-step execution planning for the GOFO Operations Intelligence Agent.

The Planner never answers the user and never executes tools. It only returns a
structured ExecutionPlan for the PlanExecutor.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

import config
from core.intent_classifier import IntentClassification, IntentType
from core.logger import get_logger
from tools.llm.client import get_llm

logger = get_logger("planner")

PLANNER_CONFIDENCE_THRESHOLD = 0.5


class ToolName(StrEnum):
    """Known tools. Values are strings so future tools can be added safely."""

    SQL = "SQL"
    RAG = "RAG"
    MEMORY = "MEMORY"
    PYTHON = "PYTHON"
    VISUALIZATION = "VISUALIZATION"
    ATTACHMENT = "ATTACHMENT"
    LLM = "LLM"


class ExecutionStep(BaseModel):
    """A single planned tool action."""

    step_number: int = Field(ge=1)
    tool: str
    action: str
    description: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[int] = Field(default_factory=list)
    status: Literal["pending", "running", "done", "skipped", "failed"] = "pending"


class ExecutionPlan(BaseModel):
    """Structured multi-step plan produced by the Planner."""

    goal: str
    steps: list[ExecutionStep] = Field(default_factory=list)
    estimated_tool_calls: int = 0
    requires_clarification: bool = False
    clarification_question: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    reasoning: str = Field(default="", description="Debug-only explanation.")

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, score))


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


def _memory_state(conversation_memory: Any) -> dict[str, Any]:
    if conversation_memory is None:
        return {}
    if hasattr(conversation_memory, "get_current_state"):
        return dict(conversation_memory.get_current_state() or {})
    if isinstance(conversation_memory, dict):
        return dict(conversation_memory)
    return {}


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


def _finalize_plan(plan: ExecutionPlan) -> ExecutionPlan:
    """Normalize step numbering, estimated calls, and low-confidence clarification."""
    steps = sorted(plan.steps, key=lambda step: step.step_number)
    renumbered: list[ExecutionStep] = []
    for index, step in enumerate(steps, start=1):
        renumbered.append(
            step.model_copy(
                update={
                    "step_number": index,
                    "status": "pending",
                    "depends_on": [dep for dep in step.depends_on if isinstance(dep, int) and dep < index],
                }
            )
        )
    estimated = plan.estimated_tool_calls or len(renumbered)
    requires_clarification = plan.requires_clarification or plan.confidence < PLANNER_CONFIDENCE_THRESHOLD
    clarification = plan.clarification_question
    if requires_clarification and not clarification:
        clarification = (
            "I need a bit more detail to plan this correctly.\n\n"
            "Do you want:\n"
            "• today's data\n"
            "• this week's data\n"
            "• all historical data?"
        )
    if requires_clarification:
        renumbered = []
        estimated = 0
    return plan.model_copy(
        update={
            "steps": renumbered,
            "estimated_tool_calls": estimated,
            "requires_clarification": requires_clarification,
            "clarification_question": clarification,
        }
    )


def _step(
    number: int,
    tool: str,
    action: str,
    description: str,
    *,
    depends_on: list[int] | None = None,
    inputs: dict[str, Any] | None = None,
) -> ExecutionStep:
    return ExecutionStep(
        step_number=number,
        tool=tool,
        action=action,
        description=description,
        depends_on=depends_on or [],
        inputs=inputs or {},
    )


def _context_inputs(question: str, conversation_memory: Any) -> dict[str, Any]:
    state = _memory_state(conversation_memory)
    return {
        "question": question,
        "date_range": state.get("date_range"),
        "filters": state.get("filters") or state.get("active_filters") or {},
        "active_entities": state.get("active_entities") or {},
        "previous_question": state.get("previous_question"),
        "current_metric": state.get("current_metric"),
    }


def _template_plan(
    question: str,
    classification: IntentClassification,
    conversation_memory: Any,
    attachment_context: dict[str, Any] | None,
) -> ExecutionPlan | None:
    """Return a deterministic plan for high-confidence simple intents."""
    intent = classification.intent
    ctx = _context_inputs(question, conversation_memory)
    attachment_context = attachment_context or {}

    if classification.requires_clarification or intent == IntentType.Unknown:
        return _finalize_plan(
            ExecutionPlan(
                goal="Clarify the user request before executing tools",
                steps=[],
                estimated_tool_calls=0,
                requires_clarification=True,
                clarification_question=(
                    "I think you want operational analysis, but the time range is unclear.\n\n"
                    "Do you want:\n"
                    "• today's data\n"
                    "• this week's data\n"
                    "• all historical data?"
                ),
                confidence=classification.confidence,
                reasoning="Classifier requested clarification or intent is Unknown.",
            )
        )

    if intent == IntentType.Greeting:
        return _finalize_plan(
            ExecutionPlan(
                goal="Greet the user and introduce GOFO capabilities",
                steps=[
                    _step(1, ToolName.LLM, "greet", "Respond with a brief GOFO greeting", inputs=ctx),
                ],
                estimated_tool_calls=1,
                confidence=0.99,
                reasoning="Greeting template.",
            )
        )

    if intent in {IntentType.ChitChat, IntentType.General_Knowledge, IntentType.Coding}:
        return _finalize_plan(
            ExecutionPlan(
                goal=f"Answer via general LLM ({intent.value})",
                steps=[
                    _step(1, ToolName.LLM, "general_reply", "Answer with the general LLM", inputs=ctx),
                ],
                estimated_tool_calls=1,
                confidence=0.95,
                reasoning=f"{intent.value} template.",
            )
        )

    if intent == IntentType.SOP_QA:
        return _finalize_plan(
            ExecutionPlan(
                goal="Answer SOP / policy question from RAG",
                steps=[
                    _step(1, ToolName.RAG, "retrieve_sop", "Retrieve relevant SOP chunks", inputs=ctx),
                    _step(
                        2,
                        ToolName.LLM,
                        "summarize_sop",
                        "Grounded SOP answer from retrieved chunks",
                        depends_on=[1],
                        inputs=ctx,
                    ),
                ],
                estimated_tool_calls=2,
                confidence=0.94,
                reasoning="SOP_QA template.",
            )
        )

    if intent == IntentType.SOP_Summary:
        return _finalize_plan(
            ExecutionPlan(
                goal="Summarize SOP document key points",
                steps=[
                    _step(1, ToolName.RAG, "retrieve_sop", "Retrieve SOP chunks for summary", inputs=ctx),
                    _step(
                        2,
                        ToolName.LLM,
                        "summarize_key_points",
                        "Produce key-point summary",
                        depends_on=[1],
                        inputs=ctx,
                    ),
                ],
                estimated_tool_calls=2,
                confidence=0.93,
                reasoning="SOP_Summary template.",
            )
        )

    if intent == IntentType.SOP_Compare:
        return _finalize_plan(
            ExecutionPlan(
                goal="Compare SOP documents",
                steps=[
                    _step(1, ToolName.RAG, "retrieve_sop_a", "Retrieve first SOP corpus chunks", inputs=ctx),
                    _step(
                        2,
                        ToolName.RAG,
                        "retrieve_sop_b",
                        "Retrieve second SOP corpus chunks",
                        depends_on=[1],
                        inputs=ctx,
                    ),
                    _step(
                        3,
                        ToolName.LLM,
                        "compare_sops",
                        "Compare retrieved SOP guidance",
                        depends_on=[1, 2],
                        inputs=ctx,
                    ),
                ],
                estimated_tool_calls=3,
                confidence=0.9,
                reasoning="SOP_Compare template.",
            )
        )

    if intent == IntentType.SQL_Query:
        return _finalize_plan(
            ExecutionPlan(
                goal="Query operational SQL data and summarize",
                steps=[
                    _step(1, ToolName.SQL, "query_records", "Query operational records", inputs=ctx),
                    _step(
                        2,
                        ToolName.LLM,
                        "summarize_results",
                        "Summarize SQL results for operations",
                        depends_on=[1],
                        inputs=ctx,
                    ),
                ],
                estimated_tool_calls=2,
                confidence=0.93,
                reasoning="SQL_Query template.",
            )
        )

    if intent == IntentType.Dashboard:
        return _finalize_plan(
            ExecutionPlan(
                goal="Build pickup KPI dashboard with charts",
                steps=[
                    _step(1, ToolName.SQL, "query_trends", "Query pickup trend metrics", inputs=ctx),
                    _step(
                        2,
                        ToolName.PYTHON,
                        "build_dataframe",
                        "Prepare dataframe for visualization",
                        depends_on=[1],
                        inputs=ctx,
                    ),
                    _step(
                        3,
                        ToolName.VISUALIZATION,
                        "create_charts",
                        "Generate matplotlib KPI charts",
                        depends_on=[2],
                        inputs=ctx,
                    ),
                    _step(
                        4,
                        ToolName.LLM,
                        "summarize_dashboard",
                        "Summarize dashboard insights",
                        depends_on=[1, 3],
                        inputs=ctx,
                    ),
                ],
                estimated_tool_calls=4,
                confidence=0.92,
                reasoning="Dashboard template.",
            )
        )

    if intent == IntentType.Upload_File:
        return _finalize_plan(
            ExecutionPlan(
                goal="Analyze uploaded document / spreadsheet",
                steps=[
                    _step(
                        1,
                        ToolName.ATTACHMENT,
                        "analyze_attachment",
                        "Run attachment / ADA analysis",
                        inputs={**ctx, **attachment_context},
                    ),
                    _step(
                        2,
                        ToolName.LLM,
                        "summarize_attachment",
                        "Summarize attachment findings",
                        depends_on=[1],
                        inputs=ctx,
                    ),
                ],
                estimated_tool_calls=2,
                confidence=0.92,
                reasoning="Upload_File template.",
            )
        )

    if intent == IntentType.Explain_Result:
        return _finalize_plan(
            ExecutionPlan(
                goal="Explain prior analytical results using memory",
                steps=[
                    _step(1, ToolName.MEMORY, "load_prior_result", "Load last result context", inputs=ctx),
                    _step(
                        2,
                        ToolName.LLM,
                        "explain_result",
                        "Explain numbers / charts from memory",
                        depends_on=[1],
                        inputs=ctx,
                    ),
                ],
                estimated_tool_calls=2,
                confidence=0.91,
                reasoning="Explain_Result template.",
            )
        )

    if intent == IntentType.Follow_Up:
        return _finalize_plan(
            ExecutionPlan(
                goal="Resolve conversational follow-up using memory and prior route tools",
                steps=[
                    _step(1, ToolName.MEMORY, "load_prior_context", "Load prior conversation context", inputs=ctx),
                    _step(
                        2,
                        ToolName.SQL,
                        "query_with_inherited_filters",
                        "Query operational data using inherited date/entity filters",
                        depends_on=[1],
                        inputs=ctx,
                    ),
                    _step(
                        3,
                        ToolName.LLM,
                        "summarize_follow_up",
                        "Summarize follow-up answer",
                        depends_on=[2],
                        inputs=ctx,
                    ),
                ],
                estimated_tool_calls=3,
                confidence=0.9,
                reasoning="Follow_Up template with inherited memory filters.",
            )
        )

    if intent == IntentType.SQL_Analysis:
        if _looks_complex_analysis(question):
            # Deterministic multi-step analytics plan (no planner LLM required).
            return _finalize_plan(
                ExecutionPlan(
                    goal="Multi-step operational analysis",
                    steps=[
                        _step(1, ToolName.SQL, "query_primary_metrics", "Retrieve primary operational metrics", inputs=ctx),
                        _step(
                            2,
                            ToolName.SQL,
                            "query_comparison_or_failures",
                            "Retrieve comparison period or failure breakdown",
                            depends_on=[1],
                            inputs=ctx,
                        ),
                        _step(
                            3,
                            ToolName.PYTHON,
                            "calculate_changes",
                            "Calculate rates / week-over-week changes",
                            depends_on=[1, 2],
                            inputs=ctx,
                        ),
                        _step(
                            4,
                            ToolName.LLM,
                            "summarize_findings",
                            "Generate business summary from analysis steps",
                            depends_on=[3],
                            inputs=ctx,
                        ),
                    ],
                    estimated_tool_calls=4,
                    confidence=0.9,
                    reasoning="SQL_Analysis multi-step template.",
                )
            )
        return _finalize_plan(
            ExecutionPlan(
                goal="Analytical SQL query with summary",
                steps=[
                    _step(1, ToolName.SQL, "query_metrics", "Retrieve analytical metrics", inputs=ctx),
                    _step(
                        2,
                        ToolName.PYTHON,
                        "derive_kpis",
                        "Derive ranking / rate KPIs",
                        depends_on=[1],
                        inputs=ctx,
                    ),
                    _step(
                        3,
                        ToolName.LLM,
                        "summarize_findings",
                        "Summarize analytical findings",
                        depends_on=[2],
                        inputs=ctx,
                    ),
                ],
                estimated_tool_calls=3,
                confidence=0.91,
                reasoning="SQL_Analysis standard template.",
            )
        )

    # Ambiguous / unsupported intents may use LLM planner.
    return None


def _parse_json(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
    return json.loads(content)


def _system_prompt() -> str:
    tools = ", ".join(item.value for item in ToolName)
    return (
        "You are an operations planning manager for GOFO logistics.\n"
        "You NEVER answer the user's question.\n"
        "You ONLY produce a structured multi-step execution plan.\n\n"
        f"Available tools: {tools}\n"
        "Future tools may appear as plain strings (Snowflake, BigQuery, WebSearch, etc.).\n\n"
        "Rules:\n"
        "- Break complex questions into small reliable steps.\n"
        "- Prefer multiple smaller steps over one uncertain step.\n"
        "- Minimize total tool calls while maximizing answer quality.\n"
        "- Use depends_on for ordering (step numbers).\n"
        "- For follow-ups, inherit date/entity context from conversation history via step inputs.\n"
        "- If the request is ambiguous, set requires_clarification=true and provide clarification_question.\n"
        "- Always end analytical plans with an LLM summarize step.\n"
        "- Do not invent SQL or final answers.\n\n"
        "Return ONLY JSON with this shape:\n"
        "{\n"
        '  "goal": "short goal",\n'
        '  "steps": [\n'
        "    {\n"
        '      "step_number": 1,\n'
        '      "tool": "SQL",\n'
        '      "action": "query_pickups",\n'
        '      "description": "Retrieve pickup metrics",\n'
        '      "inputs": {"question": "..."},\n'
        '      "depends_on": []\n'
        "    }\n"
        "  ],\n"
        '  "estimated_tool_calls": 2,\n'
        '  "requires_clarification": false,\n'
        '  "clarification_question": null,\n'
        '  "confidence": 0.9,\n'
        '  "reasoning": "debug only"\n'
        "}"
    )


def _debug_print(plan: ExecutionPlan) -> None:
    tools = [step.tool for step in plan.steps]
    deps = {step.step_number: step.depends_on for step in plan.steps}
    lines = [
        "----------------------------------",
        f"Goal: {plan.goal}",
        "Execution Plan:",
    ]
    for step in plan.steps:
        lines.append(
            f"  {step.step_number}. [{step.tool}] {step.action} — {step.description} "
            f"(depends_on={step.depends_on})"
        )
    if not plan.steps and plan.requires_clarification:
        lines.append(f"  Clarification: {plan.clarification_question}")
    lines.extend(
        [
            f"Chosen Tools: {', '.join(tools) or 'none'}",
            f"Dependencies: {deps}",
            f"Estimated Tool Calls: {plan.estimated_tool_calls}",
            f"Confidence: {plan.confidence}",
            f"Reasoning: {plan.reasoning}",
            "----------------------------------",
        ]
    )
    message = "\n".join(lines)
    if config.DEBUG:
        print(message)
    logger.info(message)


def _looks_complex_analysis(question: str) -> bool:
    normalized = question.lower()
    return bool(
        re.search(r"\bwhy\b", normalized)
        or "compare" in normalized
        or "root cause" in normalized
        or "week" in normalized
        or "vs " in normalized
        or "versus" in normalized
        or "decrease" in normalized
        or "increase" in normalized
    )


class Planner:
    """Create structured multi-step execution plans without running tools."""

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    def plan(
        self,
        question: str,
        classification: IntentClassification,
        conversation_memory: Any = None,
        *,
        attachment_context: dict[str, Any] | None = None,
    ) -> ExecutionPlan:
        """Return an ExecutionPlan for the given classified question."""
        question = (question or "").strip()
        template = _template_plan(question, classification, conversation_memory, attachment_context)
        # Prefer deterministic templates; call the LLM planner only when needed.
        if template is not None:
            _debug_print(template)
            return template

        history = _history_from_memory(conversation_memory)
        state = _memory_state(conversation_memory)
        messages = [
            SystemMessage(content=_system_prompt()),
            HumanMessage(
                content=(
                    f"Intent classification:\n{classification.model_dump_json(indent=2)}\n\n"
                    f"Conversation history:\n{_format_history(history)}\n\n"
                    f"Memory state filters/entities:\n"
                    f"{json.dumps({'date_range': state.get('date_range'), 'filters': state.get('filters') or state.get('active_filters'), 'active_entities': state.get('active_entities')}, default=str)}\n\n"
                    f"Attachment context:\n{json.dumps(attachment_context or {}, default=str)}\n\n"
                    f"User question:\n{question}"
                )
            ),
        ]
        llm = self._llm or get_llm()
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else response
        raw = content if isinstance(content, str) else str(content)
        try:
            parsed = _parse_json(raw)
            plan = ExecutionPlan.model_validate(parsed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Planner LLM parse failed: %s", exc)
            if template is not None:
                _debug_print(template)
                return template
            plan = ExecutionPlan(
                goal="Clarify ambiguous request",
                steps=[],
                estimated_tool_calls=0,
                requires_clarification=True,
                clarification_question=(
                    "I could not build a reliable execution plan.\n\n"
                    "Do you want:\n"
                    "• today's data\n"
                    "• this week's data\n"
                    "• all historical data?"
                ),
                confidence=0.2,
                reasoning=f"Failed to parse planner output: {exc}",
            )

        plan = _finalize_plan(plan)
        # Prefer a richer multi-step LLM plan for SQL_Analysis; fall back to SQL_Query template shape.
        if not plan.steps and not plan.requires_clarification and template is not None:
            plan = template
        _debug_print(plan)
        return plan

    def plan_retry(
        self,
        question: str,
        classification: IntentClassification,
        quality_report: Any,
        *,
        previous_plan: ExecutionPlan | dict[str, Any] | None = None,
        previous_answer: str | None = None,
        conversation_memory: Any = None,
    ) -> ExecutionPlan:
        """Build an additional execution plan from a QA QualityReport.

        The planner still does not execute tools or answer the user.
        """
        ctx = _context_inputs(question, conversation_memory)
        ctx["qa_feedback"] = getattr(quality_report, "feedback", None) or []
        ctx["missing_information"] = getattr(quality_report, "missing_information", None) or []
        ctx["previous_answer"] = previous_answer or ""
        ctx["increased_top_k"] = True

        should_retry_retrieval = bool(getattr(quality_report, "should_retry_retrieval", False))
        should_retry_sql = bool(getattr(quality_report, "should_retry_sql", False))
        should_retry_python = bool(getattr(quality_report, "should_retry_python", False))
        should_ask = bool(getattr(quality_report, "should_ask_user", False))
        action = str(getattr(getattr(quality_report, "action", None), "value", getattr(quality_report, "action", "")))

        if should_ask or action == "ask_user":
            return _finalize_plan(
                ExecutionPlan(
                    goal="Ask user for clarification based on QA",
                    steps=[],
                    estimated_tool_calls=0,
                    requires_clarification=True,
                    clarification_question=(
                        "I need a bit more detail to improve the analysis.\n\n"
                        "Do you want:\n"
                        "• today's data\n"
                        "• this week's data\n"
                        "• all historical data?"
                    ),
                    confidence=0.8,
                    reasoning="QA requested user clarification.",
                )
            )

        steps: list[ExecutionStep] = []
        step_no = 1
        if should_retry_retrieval or action == "retry_retrieval":
            steps.append(
                _step(
                    step_no,
                    ToolName.RAG,
                    "retrieve_sop_expanded",
                    "Retry retrieval with broader top_k and rerank",
                    inputs={**ctx, "top_k": 12, "merge_chunks": True, "dedupe": True},
                )
            )
            step_no += 1
        if should_retry_sql or action in {"retry_sql", "retry_plan"}:
            depends = [step_no - 1] if steps else []
            steps.append(
                _step(
                    step_no,
                    ToolName.SQL,
                    "query_additional_metrics",
                    "Additional SQL for failures / comparison / root cause",
                    depends_on=depends,
                    inputs={
                        **ctx,
                        "include_failures": True,
                        "include_previous_period": True,
                        "include_exception_reasons": True,
                    },
                )
            )
            step_no += 1
        if should_retry_python or action in {"retry_python", "retry_plan"}:
            depends = [step_no - 1] if steps else []
            steps.append(
                _step(
                    step_no,
                    ToolName.PYTHON,
                    "recalculate_kpis",
                    "Recalculate rates / comparisons from refreshed evidence",
                    depends_on=depends,
                    inputs=ctx,
                )
            )
            step_no += 1

        if not steps:
            # Generic improvement path.
            steps = [
                _step(1, ToolName.SQL, "query_metrics", "Refresh operational metrics", inputs=ctx),
                _step(2, ToolName.LLM, "summarize_findings", "Regenerate answer with QA feedback", depends_on=[1], inputs=ctx),
            ]
        else:
            depends = [step.step_number for step in steps]
            steps.append(
                _step(
                    step_no,
                    ToolName.LLM,
                    "summarize_findings",
                    "Regenerate answer using previous answer + QA feedback + new evidence",
                    depends_on=depends,
                    inputs=ctx,
                )
            )

        prior_goal = None
        if isinstance(previous_plan, ExecutionPlan):
            prior_goal = previous_plan.goal
        elif isinstance(previous_plan, dict):
            prior_goal = previous_plan.get("goal")

        plan = _finalize_plan(
            ExecutionPlan(
                goal=f"QA retry: {prior_goal or 'improve answer quality'}",
                steps=steps,
                estimated_tool_calls=len(steps),
                confidence=0.82,
                reasoning=f"Built from QA action={action}; missing={ctx['missing_information']}",
            )
        )
        _debug_print(plan)
        return plan


def create_plan(
    question: str,
    classification: IntentClassification,
    conversation_memory: Any = None,
    *,
    attachment_context: dict[str, Any] | None = None,
    llm: Any | None = None,
) -> ExecutionPlan:
    """Module-level helper for planning."""
    return Planner(llm=llm).plan(
        question,
        classification,
        conversation_memory,
        attachment_context=attachment_context,
    )
