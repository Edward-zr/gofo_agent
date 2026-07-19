"""Request and response models shared across agent capabilities."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Input for a knowledge or ops query."""

    question: str = Field(..., min_length=1, description="User question.")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to retrieve.")
    collection: Optional[str] = Field(
        default=None,
        description="Optional Chroma collection override.",
    )
    filters: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters for vector search.",
    )
    result_context: Optional[dict[str, Any]] = Field(
        default=None,
        description="Previous analytical result context for memory analysis.",
    )
    attachments: Optional[list[str]] = Field(
        default=None,
        description="Attachment IDs referenced for this question.",
    )
    file_context: Optional[dict[str, Any]] = Field(
        default=None,
        description="Structured processed attachment context for reasoning.",
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Client conversation/session identifier.",
    )
    semantic_context: Optional[dict[str, Any]] = Field(
        default=None,
        description="Semantic request analysis used for routing and response orchestration.",
    )


class SourceChunk(BaseModel):
    """A retrieved document chunk with provenance."""

    id: str
    text: str
    score: float = Field(description="Relevance score; higher values are more relevant.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Structured output from a query capability."""

    question: str
    answer: Optional[str] = Field(
        default=None,
        description="Generated answer when an LLM synthesis step is used.",
    )
    sources: list[SourceChunk] = Field(default_factory=list)
    capability: str = Field(default="rag", description="Capability that produced this response.")
    rewritten_question: Optional[str] = Field(
        default=None,
        description="Rewritten question used for retrieval or SQL date resolution.",
    )
    latest_business_date: Optional[str] = Field(
        default=None,
        description="Latest operational pickup date used for SQL relative date resolution.",
    )
    generated_sql: Optional[str] = Field(
        default=None,
        description="Generated SQL used for analytics queries.",
    )
    sql_rows: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Raw SQL result rows for debug display.",
    )
    needs_sql: Optional[bool] = Field(
        default=None,
        description="Whether SQL analytics was planned for this question.",
    )
    needs_rag: Optional[bool] = Field(
        default=None,
        description="Whether SOP retrieval was planned for this question.",
    )
    plan_reason: Optional[str] = Field(
        default=None,
        description="Explanation of the selected tool plan.",
    )
    sql_output: Optional[str] = Field(
        default=None,
        description="SQL tool answer before final synthesis.",
    )
    rag_output: Optional[str] = Field(
        default=None,
        description="RAG tool answer before final synthesis.",
    )
    execution_order: Optional[list[str]] = Field(
        default=None,
        description="Order in which tools were executed.",
    )
    planning_capability: Optional[str] = Field(
        default=None,
        description="Capability selected by the natural language planner.",
    )
    planning_intent: Optional[str] = Field(
        default=None,
        description="Business intent extracted by the natural language planner.",
    )
    planning_confidence: Optional[float] = Field(
        default=None,
        description="Planner confidence score between 0 and 1.",
    )
    planning_reasoning: Optional[str] = Field(
        default=None,
        description="Planner reasoning for the selected capability.",
    )
    planning_entities: Optional[dict[str, Any]] = Field(
        default=None,
        description="Business entities extracted by the natural language planner.",
    )
    original_question: Optional[str] = Field(
        default=None,
        description="Original user question before session memory resolution.",
    )
    resolved_question: Optional[str] = Field(
        default=None,
        description="Standalone question after session memory resolution.",
    )
    memory_history_count: Optional[int] = Field(
        default=None,
        description="Number of turns currently retained in short-term memory.",
    )
    memory_current_state: Optional[dict[str, Any]] = Field(
        default=None,
        description="Current short-term conversation state after this turn.",
    )
    last_result_context: Optional[dict[str, Any]] = Field(
        default=None,
        description="Latest analytical result context available to memory.",
    )
    repair_detected: Optional[bool] = Field(
        default=None,
        description="Whether the latest user turn was detected as a correction.",
    )
    repair_type: Optional[str] = Field(
        default=None,
        description="Type of detected conversation repair.",
    )
    changed_dimension: Optional[str] = Field(
        default=None,
        description="Dimension or concept changed by a correction.",
    )
    business_metric: Optional[str] = Field(
        default=None,
        description="Primary operational metric inferred for the question.",
    )
    analysis_dimension: Optional[str] = Field(
        default=None,
        description="Primary business dimension for analysis.",
    )
    date_range: Optional[str] = Field(
        default=None,
        description="Operational date or period inferred for analysis.",
    )
    analysis_filters: Optional[dict[str, Any]] = Field(
        default=None,
        description="Structured operational filters inferred for analysis.",
    )
    root_cause: Optional[dict[str, Any]] = Field(
        default=None,
        description="Root-cause analysis derived from operational rows.",
    )
    anomaly: Optional[dict[str, Any]] = Field(
        default=None,
        description="Statistical anomaly detection result.",
    )
    recommendation: Optional[str] = Field(
        default=None,
        description="Operational recommendation generated from analysis.",
    )
    kpi_summary: Optional[dict[str, Any]] = Field(
        default=None,
        description="KPI dictionary-backed summary for broad operations questions.",
    )
    long_memory_matches: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Relevant historical conversations or findings retrieved before analysis.",
    )
    previous_issues_found: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Historical operational issues similar to the current question.",
    )
    saved_memory: Optional[bool] = Field(
        default=None,
        description="Whether this response was persisted to long-term memory.",
    )
    learned_patterns: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Recurring operational patterns learned from persisted findings.",
    )
    classifier_intent: Optional[str] = Field(
        default=None,
        description="Intent selected by the conversation-aware intent classifier.",
    )
    inherited_context: Optional[dict[str, Any]] = Field(
        default=None,
        description="Conversation context inherited by the resolver/context builder.",
    )
    business_findings: Optional[list[str]] = Field(
        default=None,
        description="Evidence-backed business findings generated from SQL rows.",
    )
    memory_updated: Optional[bool] = Field(
        default=None,
        description="Whether semantic conversation memory was updated for this response.",
    )
    sql_cache_hit: Optional[bool] = Field(
        default=None,
        description="Whether a previous identical SQL response was reused.",
    )
    attachment_ids: Optional[list[str]] = Field(
        default=None,
        description="Attachment IDs used for this response.",
    )
    attachment_filenames: Optional[list[str]] = Field(
        default=None,
        description="Human-readable attachment filenames used in the answer.",
    )
    data_sources: Optional[list[str]] = Field(
        default=None,
        description="Selected evidence sources for this response.",
    )
    file_context_summary: Optional[dict[str, Any]] = Field(
        default=None,
        description="Compact summary of attachment context used in reasoning.",
    )
    semantic_domain: Optional[str] = Field(
        default=None,
        description="High-level semantic domain for the user request.",
    )
    semantic_sub_intent: Optional[str] = Field(
        default=None,
        description="Semantic sub-intent such as ranking, detail, or analysis.",
    )
    response_mode: Optional[str] = Field(
        default=None,
        description="Selected response presentation mode.",
    )
    semantic_reasoning: Optional[str] = Field(
        default=None,
        description="Semantic analyzer reasoning for routing and response mode.",
    )
    suggested_next_steps: Optional[list[str]] = Field(
        default=None,
        description="Recommended follow-up questions or actions for the user.",
    )
    charts: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Renderable chart specifications generated from attachment analysis.",
    )
