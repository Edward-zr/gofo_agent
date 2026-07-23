"""Multi-step execution planning for the GOFO Operations Intelligence Agent.

The Planner never answers the user and never executes tools. It only returns a
structured ExecutionPlan for the PlanExecutor.

Flow:
  DataSourceSelector → ExecutionPlan steps → ToolOrchestrator
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

import config
from core.data_source_selector import (
    DataSource,
    DataSourceSelection,
    DataSourceSelector,
)
from core.intent_classifier import IntentClassification, IntentType
from core.logger import get_logger
from tools.llm.client import get_llm

logger = get_logger("planner")

PLANNER_CONFIDENCE_THRESHOLD = 0.5


class ToolName(StrEnum):
    """Known tools. Values are strings so future tools can be added safely."""

    SQL = "SQL"
    KNOWLEDGE_GRAPH = "KNOWLEDGE_GRAPH"
    RAG = "RAG"
    MEMORY = "MEMORY"
    TRANSFORM = "TRANSFORM"
    STATISTICS = "STATISTICS"
    PYTHON = "PYTHON"  # alias → StatisticsTool (backward compatible)
    VISUALIZATION = "VISUALIZATION"
    RECOMMENDATION = "RECOMMENDATION"
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
    selected_data_sources: list[str] = Field(
        default_factory=list,
        description="Minimum data sources chosen before orchestration.",
    )
    source_reasons: dict[str, str] = Field(
        default_factory=dict,
        description="Why each selected data source was chosen.",
    )
    expected_outputs: list[str] = Field(
        default_factory=list,
        description="Expected output per selected source / step.",
    )
    data_source_selection: dict[str, Any] | None = Field(
        default=None,
        description="Full DataSourceSelection payload for debugging.",
    )
    retrieval_policy: dict[str, Any] | None = Field(
        default=None,
        description=(
            "RAG retrieval policy for the Confidence Engine: "
            "allow_general_knowledge, minimum_confidence, allow_clarification, "
            "allow_document_request."
        ),
    )
    required_capabilities: list[str] = Field(
        default_factory=list,
        description=(
            "Capability tokens for Supervisor routing (never agent names). "
            "Example: database_query, document_retrieval, statistics."
        ),
    )

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


def _default_rag_retrieval_policy() -> dict[str, Any]:
    """Strict SOP policy: no confident answers from weak retrieval."""
    return {
        "allow_general_knowledge": False,
        "minimum_confidence": "MEDIUM",
        "allow_clarification": True,
        "allow_document_request": True,
    }


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

    selected = plan.selected_data_sources or [
        step.tool for step in renumbered if step.tool != ToolName.LLM.value
    ]
    # Keep LLM in selected_data_sources when it is the only tool.
    if not selected and any(step.tool == ToolName.LLM.value for step in renumbered):
        selected = [ToolName.LLM.value]

    retrieval_policy = plan.retrieval_policy
    uses_rag = any(step.tool == ToolName.RAG.value for step in renumbered)
    if retrieval_policy is None and uses_rag:
        retrieval_policy = _default_rag_retrieval_policy()

    if retrieval_policy and renumbered:
        with_policy: list[ExecutionStep] = []
        for step in renumbered:
            inputs = dict(step.inputs or {})
            if step.tool in {ToolName.RAG.value, ToolName.LLM.value}:
                inputs.setdefault("retrieval_policy", retrieval_policy)
            with_policy.append(step.model_copy(update={"inputs": inputs}))
        renumbered = with_policy

    from agents.base import TOOL_TO_CAPABILITY

    required_capabilities = list(plan.required_capabilities or [])
    if not required_capabilities:
        for step in renumbered:
            cap = TOOL_TO_CAPABILITY.get(step.tool)
            if cap and cap not in required_capabilities:
                required_capabilities.append(cap)

    return plan.model_copy(
        update={
            "steps": renumbered,
            "estimated_tool_calls": estimated,
            "requires_clarification": requires_clarification,
            "clarification_question": clarification,
            "selected_data_sources": list(dict.fromkeys(selected)),
            "retrieval_policy": retrieval_policy,
            "required_capabilities": required_capabilities,
        }
    )


def _attach_selection(plan: ExecutionPlan, selection: DataSourceSelection) -> ExecutionPlan:
    """Copy data-source selection metadata onto the execution plan."""
    return plan.model_copy(
        update={
            "selected_data_sources": list(selection.selected_sources),
            "source_reasons": selection.reasons_map(),
            "expected_outputs": list(selection.expected_outputs),
            "data_source_selection": selection.model_dump(),
            "confidence": max(plan.confidence, selection.confidence) if plan.confidence else selection.confidence,
            "reasoning": (
                f"{plan.reasoning} | sources: {selection.reasoning} | {selection.combine_reason}"
                if plan.reasoning
                else f"{selection.reasoning} | {selection.combine_reason}"
            ),
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


def _action_for_source(source: str, question: str) -> tuple[str, str]:
    """Return (action, description) for a selected data source."""
    normalized = question.lower()
    mapping: dict[str, tuple[str, str]] = {
        DataSource.MEMORY.value: (
            "load_prior_context",
            "Load prior conversation filters and last result context",
        ),
        DataSource.SQL.value: (
            "query_metrics",
            "Query operational metrics / counts / rankings from SQLite",
        ),
        DataSource.KNOWLEDGE_GRAPH.value: (
            "resolve_relationships",
            "Resolve Driver→Hub→Region / Manager→Hub / SOP ownership",
        ),
        DataSource.RAG.value: (
            "retrieve_sop",
            "Retrieve SOP / policy / unstructured knowledge chunks",
        ),
        DataSource.TRANSFORM.value: (
            "prepare_dataframe",
            "Transform / filter / group DataFrame for analytics",
        ),
        DataSource.STATISTICS.value: (
            "compute_statistics",
            "Compute summary stats / ratios / rankings (no LLM math)",
        ),
        DataSource.PYTHON.value: (
            "compute_statistics",
            "Legacy PYTHON alias → StatisticsTool",
        ),
        DataSource.VISUALIZATION.value: (
            "create_charts",
            "Generate matplotlib charts from prepared data",
        ),
        DataSource.RECOMMENDATION.value: (
            "generate_recommendations",
            "Produce business recommendations from computed results",
        ),
        DataSource.ATTACHMENT.value: (
            "analyze_attachment",
            "Analyze uploaded spreadsheet / document",
        ),
        DataSource.LLM.value: (
            "summarize_findings",
            "Synthesize final answer from selected evidence",
        ),
    }
    if source == DataSource.SQL.value:
        if "compare" in normalized or "yesterday" in normalized or "vs " in normalized:
            return ("query_comparison_metrics", "Query comparison-period operational metrics")
        if "lowest" in normalized or "highest" in normalized or "rank" in normalized:
            return ("query_ranking_metrics", "Query ranking / extreme KPI metrics")
        if "today" in normalized and "rate" in normalized:
            return ("query_today_kpi", "Query today's KPI / pickup rate")
    if source == DataSource.LLM.value and re.search(r"\b(hi|hello|hey)\b", normalized):
        return ("greet", "Respond with a brief GOFO greeting")
    return mapping.get(source, (f"run_{source.lower()}", f"Execute {source}"))


def _build_plan_from_selection(
    question: str,
    classification: IntentClassification,
    selection: DataSourceSelection,
    conversation_memory: Any,
    attachment_context: dict[str, Any] | None,
) -> ExecutionPlan:
    """Build the minimum ExecutionPlan from a DataSourceSelection."""
    ctx = _context_inputs(question, conversation_memory)
    attachment_context = attachment_context or {}

    if selection.requires_clarification:
        return _finalize_plan(
            _attach_selection(
                ExecutionPlan(
                    goal="Clarify the user request before executing tools",
                    steps=[],
                    estimated_tool_calls=0,
                    requires_clarification=True,
                    clarification_question=selection.clarification_question,
                    confidence=selection.confidence,
                    reasoning="Data source selector requested clarification.",
                ),
                selection,
            )
        )

    sources = [source for source in selection.execution_order if source in selection.selected_sources]
    if not sources:
        sources = list(selection.selected_sources)

    evidence_sources = [s for s in sources if s != DataSource.LLM.value]
    include_llm = DataSource.LLM.value in sources or bool(evidence_sources)

    steps: list[ExecutionStep] = []
    step_ids: dict[str, int] = {}
    number = 1

    # Dependency policy:
    # MEMORY → SQL → KG → RAG → TRANSFORM → STATISTICS → VISUALIZATION → RECOMMENDATION → LLM
    for source in evidence_sources:
        action, description = _action_for_source(source, question)
        depends_on: list[int] = []
        inputs = {**ctx}
        if source == DataSource.ATTACHMENT.value:
            inputs = {**ctx, **attachment_context}
        if source == DataSource.KNOWLEDGE_GRAPH.value and DataSource.SQL.value in step_ids:
            depends_on = [step_ids[DataSource.SQL.value]]
        elif source == DataSource.TRANSFORM.value and DataSource.SQL.value in step_ids:
            depends_on = [step_ids[DataSource.SQL.value]]
        elif source == DataSource.STATISTICS.value:
            if DataSource.TRANSFORM.value in step_ids:
                depends_on = [step_ids[DataSource.TRANSFORM.value]]
            elif DataSource.SQL.value in step_ids:
                depends_on = [step_ids[DataSource.SQL.value]]
        elif source == DataSource.PYTHON.value:
            # Legacy alias behaves like STATISTICS
            if DataSource.TRANSFORM.value in step_ids:
                depends_on = [step_ids[DataSource.TRANSFORM.value]]
            elif DataSource.SQL.value in step_ids:
                depends_on = [step_ids[DataSource.SQL.value]]
        elif source == DataSource.VISUALIZATION.value:
            if DataSource.STATISTICS.value in step_ids:
                depends_on = [step_ids[DataSource.STATISTICS.value]]
            elif DataSource.TRANSFORM.value in step_ids:
                depends_on = [step_ids[DataSource.TRANSFORM.value]]
            elif DataSource.PYTHON.value in step_ids:
                depends_on = [step_ids[DataSource.PYTHON.value]]
            elif DataSource.SQL.value in step_ids:
                depends_on = [step_ids[DataSource.SQL.value]]
        elif source == DataSource.RECOMMENDATION.value:
            if DataSource.STATISTICS.value in step_ids:
                depends_on = [step_ids[DataSource.STATISTICS.value]]
            elif DataSource.TRANSFORM.value in step_ids:
                depends_on = [step_ids[DataSource.TRANSFORM.value]]
            elif DataSource.SQL.value in step_ids:
                depends_on = [step_ids[DataSource.SQL.value]]
        elif source == DataSource.SQL.value and DataSource.MEMORY.value in step_ids:
            depends_on = [step_ids[DataSource.MEMORY.value]]
        elif source == DataSource.RAG.value and DataSource.MEMORY.value in step_ids:
            depends_on = [step_ids[DataSource.MEMORY.value]]
        # SOP compare keeps two RAG retrievals only when intent says so.
        if (
            source == DataSource.RAG.value
            and classification.intent == IntentType.SOP_Compare
            and DataSource.RAG.value not in step_ids
        ):
            steps.append(
                _step(
                    number,
                    ToolName.RAG,
                    "retrieve_sop_a",
                    "Retrieve first SOP corpus chunks",
                    depends_on=depends_on,
                    inputs=inputs,
                )
            )
            step_ids[DataSource.RAG.value] = number
            number += 1
            steps.append(
                _step(
                    number,
                    ToolName.RAG,
                    "retrieve_sop_b",
                    "Retrieve second SOP corpus chunks",
                    depends_on=[number - 1],
                    inputs=inputs,
                )
            )
            step_ids["RAG_B"] = number
            number += 1
            continue

        steps.append(
            _step(
                number,
                source,
                action,
                description,
                depends_on=depends_on,
                inputs=inputs,
            )
        )
        step_ids[source] = number
        number += 1

    if include_llm:
        llm_depends = [step.step_number for step in steps]
        # Greeting / single-LLM plans have no deps.
        if not evidence_sources:
            llm_depends = []
        action, description = _action_for_source(DataSource.LLM.value, question)
        if classification.intent == IntentType.Greeting:
            action, description = "greet", "Respond with a brief GOFO greeting"
        elif classification.intent in {
            IntentType.ChitChat,
            IntentType.General_Knowledge,
            IntentType.Coding,
        }:
            action, description = "general_reply", "Answer with the general LLM"
        elif classification.intent == IntentType.SOP_Compare:
            action, description = "compare_sops", "Compare retrieved SOP guidance"
        steps.append(
            _step(
                number,
                ToolName.LLM,
                action,
                description,
                depends_on=llm_depends,
                inputs=ctx,
            )
        )

    goal = _goal_from_selection(question, selection, classification)
    plan = ExecutionPlan(
        goal=goal,
        steps=steps,
        estimated_tool_calls=len(steps),
        confidence=selection.confidence,
        reasoning="Built from DataSourceSelection (minimum necessary tools).",
    )
    return _finalize_plan(_attach_selection(plan, selection))


def _goal_from_selection(
    question: str,
    selection: DataSourceSelection,
    classification: IntentClassification,
) -> str:
    primary = [s for s in selection.selected_sources if s != DataSource.LLM.value]
    if not primary:
        return f"Answer via LLM ({classification.intent.value})"
    if primary == [DataSource.SQL.value]:
        return "Answer operational KPI / metric question from SQL only"
    if primary == [DataSource.KNOWLEDGE_GRAPH.value]:
        return "Resolve entity relationship from Knowledge Graph only"
    if primary == [DataSource.RAG.value]:
        return "Answer unstructured SOP / policy question from RAG only"
    if set(primary) == {DataSource.SQL.value, DataSource.RAG.value}:
        return "Compare/query SQL metrics and explain with RAG knowledge"
    if set(primary) == {DataSource.SQL.value, DataSource.KNOWLEDGE_GRAPH.value}:
        return "Find operational extreme via SQL then resolve relationships via Knowledge Graph"
    if DataSource.VISUALIZATION.value in primary or DataSource.STATISTICS.value in primary or DataSource.TRANSFORM.value in primary:
        return "Query SQL metrics and run specialized Python analytics / charts"
    if DataSource.PYTHON.value in primary:
        return "Query SQL metrics and generate analytical / chart outputs"
    return f"Answer using selected sources: {', '.join(primary)}"


def _template_plan(
    question: str,
    classification: IntentClassification,
    conversation_memory: Any,
    attachment_context: dict[str, Any] | None,
) -> ExecutionPlan | None:
    """Legacy intent templates — used only when data-source selection is disabled."""
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
                selected_data_sources=[ToolName.LLM.value],
                source_reasons={ToolName.LLM.value: "Greeting only."},
                expected_outputs=["LLM: Greeting reply"],
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
                selected_data_sources=[ToolName.LLM.value],
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
                selected_data_sources=[ToolName.RAG.value, ToolName.LLM.value],
                retrieval_policy=_default_rag_retrieval_policy(),
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
                selected_data_sources=[ToolName.RAG.value, ToolName.LLM.value],
                retrieval_policy=_default_rag_retrieval_policy(),
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
                selected_data_sources=[ToolName.RAG.value, ToolName.LLM.value],
                retrieval_policy=_default_rag_retrieval_policy(),
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
                selected_data_sources=[ToolName.SQL.value, ToolName.LLM.value],
            )
        )

    if intent == IntentType.Dashboard:
        return _finalize_plan(
            ExecutionPlan(
                goal="Build pickup KPI dashboard with specialized Python tools",
                steps=[
                    _step(1, ToolName.SQL, "query_trends", "Query pickup trend metrics", inputs=ctx),
                    _step(
                        2,
                        ToolName.TRANSFORM,
                        "prepare_dataframe",
                        "Prepare dataframe for analytics",
                        depends_on=[1],
                        inputs=ctx,
                    ),
                    _step(
                        3,
                        ToolName.STATISTICS,
                        "compute_statistics",
                        "Compute KPI statistics",
                        depends_on=[2],
                        inputs=ctx,
                    ),
                    _step(
                        4,
                        ToolName.VISUALIZATION,
                        "create_charts",
                        "Generate matplotlib KPI charts",
                        depends_on=[2],
                        inputs=ctx,
                    ),
                    _step(
                        5,
                        ToolName.RECOMMENDATION,
                        "generate_recommendations",
                        "Produce operational recommendations",
                        depends_on=[3],
                        inputs=ctx,
                    ),
                    _step(
                        6,
                        ToolName.LLM,
                        "summarize_dashboard",
                        "Summarize dashboard insights",
                        depends_on=[3, 4, 5],
                        inputs=ctx,
                    ),
                ],
                estimated_tool_calls=6,
                confidence=0.92,
                reasoning="Dashboard template with specialized Python tools.",
                selected_data_sources=[
                    ToolName.SQL.value,
                    ToolName.TRANSFORM.value,
                    ToolName.STATISTICS.value,
                    ToolName.VISUALIZATION.value,
                    ToolName.RECOMMENDATION.value,
                    ToolName.LLM.value,
                ],
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
                selected_data_sources=[ToolName.ATTACHMENT.value, ToolName.LLM.value],
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
                selected_data_sources=[ToolName.MEMORY.value, ToolName.LLM.value],
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
                selected_data_sources=[ToolName.MEMORY.value, ToolName.SQL.value, ToolName.LLM.value],
            )
        )

    if intent == IntentType.SQL_Analysis:
        # Prefer SQL-only unless the question explicitly needs python/charts/compare calc.
        needs_python = bool(
            re.search(
                r"\b(calculate|forecast|statistics|chart|plot|visualize|dashboard)\b",
                question.lower(),
            )
        )
        steps = [
            _step(1, ToolName.SQL, "query_metrics", "Retrieve analytical metrics", inputs=ctx),
        ]
        if needs_python:
            steps.append(
                _step(
                    2,
                    ToolName.PYTHON,
                    "derive_kpis",
                    "Derive ranking / rate KPIs",
                    depends_on=[1],
                    inputs=ctx,
                )
            )
            steps.append(
                _step(
                    3,
                    ToolName.LLM,
                    "summarize_findings",
                    "Summarize analytical findings",
                    depends_on=[2],
                    inputs=ctx,
                )
            )
        else:
            steps.append(
                _step(
                    2,
                    ToolName.LLM,
                    "summarize_findings",
                    "Summarize analytical findings",
                    depends_on=[1],
                    inputs=ctx,
                )
            )
        return _finalize_plan(
            ExecutionPlan(
                goal="Analytical SQL query with summary",
                steps=steps,
                estimated_tool_calls=len(steps),
                confidence=0.91,
                reasoning="SQL_Analysis minimum template.",
                selected_data_sources=[step.tool for step in steps],
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


def _system_prompt(
    *,
    question: str = "",
    memory: str = "",
    intent_classification: str = "",
) -> str:
    """Render planner system prompt from the Prompt Registry."""
    from core.prompt_manager import get_prompt_manager

    tools = ", ".join(item.value for item in ToolName)
    _selection, rendered = get_prompt_manager().render(
        "planner.planner_prompt",
        {
            "question": question or "(see user message)",
            "available_tools": tools,
            "memory": memory or "(none)",
            "intent_classification": intent_classification or "{}",
        },
    )
    return rendered


def _debug_print(plan: ExecutionPlan) -> None:
    tools = [step.tool for step in plan.steps]
    deps = {step.step_number: step.depends_on for step in plan.steps}
    lines = [
        "----------------------------------",
        f"Goal: {plan.goal}",
        f"Selected Data Sources: {', '.join(plan.selected_data_sources) or 'none'}",
    ]
    if plan.source_reasons:
        lines.append("Source Reasons:")
        for source, reason in plan.source_reasons.items():
            lines.append(f"  - {source}: {reason}")
    lines.append("Execution Plan:")
    for step in plan.steps:
        lines.append(
            f"  {step.step_number}. [{step.tool}] {step.action} — {step.description} "
            f"(depends_on={step.depends_on})"
        )
    if not plan.steps and plan.requires_clarification:
        lines.append(f"  Clarification: {plan.clarification_question}")
    if plan.expected_outputs:
        lines.append("Expected Outputs:")
        for item in plan.expected_outputs:
            lines.append(f"  - {item}")
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


class Planner:
    """Create structured multi-step execution plans without running tools."""

    def __init__(
        self,
        llm: Any | None = None,
        *,
        data_source_selector: DataSourceSelector | None = None,
    ) -> None:
        self._llm = llm
        self._selector = data_source_selector or DataSourceSelector()

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

        # Primary path: Data Source Selection → minimum plan.
        if getattr(config, "DATA_SOURCE_SELECTION_ENABLED", True):
            selection = self._selector.select(
                question,
                classification,
                conversation_memory,
                attachment_context=attachment_context,
            )
            plan = _build_plan_from_selection(
                question,
                classification,
                selection,
                conversation_memory,
                attachment_context,
            )
            _debug_print(plan)
            return plan

        template = _template_plan(question, classification, conversation_memory, attachment_context)
        # Prefer deterministic templates; call the LLM planner only when needed.
        if template is not None:
            _debug_print(template)
            return template

        history = _history_from_memory(conversation_memory)
        state = _memory_state(conversation_memory)
        memory_blob = json.dumps(
            {
                "history": _format_history(history),
                "date_range": state.get("date_range"),
                "filters": state.get("filters") or state.get("active_filters"),
                "active_entities": state.get("active_entities"),
                "attachment_context": attachment_context or {},
            },
            default=str,
        )
        from core.prompt_manager import get_prompt_manager

        manager = get_prompt_manager()
        selection, messages = manager.build_messages(
            "planner.planner_prompt",
            {
                "question": question,
                "available_tools": ", ".join(item.value for item in ToolName),
                "memory": memory_blob,
                "intent_classification": classification.model_dump_json(indent=2),
            },
            user_content=(
                f"Intent classification:\n{classification.model_dump_json(indent=2)}\n\n"
                f"Conversation history:\n{_format_history(history)}\n\n"
                f"Memory state filters/entities:\n"
                f"{json.dumps({'date_range': state.get('date_range'), 'filters': state.get('filters') or state.get('active_filters'), 'active_entities': state.get('active_entities')}, default=str)}\n\n"
                f"Attachment context:\n{json.dumps(attachment_context or {}, default=str)}\n\n"
                f"User question:\n{question}"
            ),
        )
        llm = self._llm or manager.llm_for(selection)
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
        prior_sources: list[str] = []
        if isinstance(previous_plan, ExecutionPlan):
            prior_goal = previous_plan.goal
            prior_sources = list(previous_plan.selected_data_sources)
        elif isinstance(previous_plan, dict):
            prior_goal = previous_plan.get("goal")
            prior_sources = list(previous_plan.get("selected_data_sources") or [])

        plan = _finalize_plan(
            ExecutionPlan(
                goal=f"QA retry: {prior_goal or 'improve answer quality'}",
                steps=steps,
                estimated_tool_calls=len(steps),
                confidence=0.82,
                reasoning=f"Built from QA action={action}; missing={ctx['missing_information']}",
                selected_data_sources=prior_sources or [step.tool for step in steps],
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
