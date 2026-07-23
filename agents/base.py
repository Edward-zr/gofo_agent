"""Shared agent interface and standardized response models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from time import perf_counter
from typing import Any, ClassVar

from pydantic import BaseModel, Field


class AgentStatus(StrEnum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    UNHEALTHY = "UNHEALTHY"


class AgentCapability(StrEnum):
    """Capability tokens used by Planner → Supervisor → Registry routing."""

    # RAG
    SOP_QA = "sop_qa"
    DOCUMENT_RETRIEVAL = "document_retrieval"
    DOCUMENT_SUMMARIZATION = "document_summarization"
    # SQL
    SQL_PLANNING = "sql_planning"
    SQL_GENERATION = "sql_generation"
    DATABASE_QUERY = "database_query"
    # Analytics
    STATISTICS = "statistics"
    CHARTS = "charts"
    DATA_ANALYSIS = "data_analysis"
    # General
    GENERAL_KNOWLEDGE = "general_knowledge"
    BRAINSTORMING = "brainstorming"
    CONVERSATION = "conversation"
    # Memory
    CONVERSATION_MEMORY = "conversation_memory"
    CONTEXT_TRACKING = "context_tracking"
    SLOT_REUSE = "slot_reuse"
    # Reflection
    ANSWER_REVIEW = "answer_review"
    CONSISTENCY_CHECK = "consistency_check"
    HALLUCINATION_DETECTION = "hallucination_detection"
    # Recommendation
    SUGGESTED_ANALYSES = "suggested_analyses"
    OPERATIONAL_RECOMMENDATIONS = "operational_recommendations"
    NEXT_STEPS = "next_steps"
    # Future-ready
    RELATIONSHIP_LOOKUP = "relationship_lookup"
    FORECAST = "forecast"
    SEARCH = "search"
    VISION = "vision"
    API_CALL = "api_call"


# Planner tool name → primary capability (capability routing, not agent names).
TOOL_TO_CAPABILITY: dict[str, str] = {
    "SQL": AgentCapability.DATABASE_QUERY.value,
    "RAG": AgentCapability.DOCUMENT_RETRIEVAL.value,
    "TRANSFORM": AgentCapability.DATA_ANALYSIS.value,
    "STATISTICS": AgentCapability.STATISTICS.value,
    "PYTHON": AgentCapability.STATISTICS.value,
    "VISUALIZATION": AgentCapability.CHARTS.value,
    "RECOMMENDATION": AgentCapability.OPERATIONAL_RECOMMENDATIONS.value,
    "MEMORY": AgentCapability.CONVERSATION_MEMORY.value,
    "LLM": AgentCapability.CONVERSATION.value,
    "ATTACHMENT": AgentCapability.DATA_ANALYSIS.value,
    "KNOWLEDGE_GRAPH": AgentCapability.RELATIONSHIP_LOOKUP.value,
}


class AgentTask(BaseModel):
    """Work unit dispatched by the Supervisor."""

    task_id: str
    capability: str
    tool: str | None = None
    action: str = ""
    description: str = ""
    question: str = ""
    resolved_question: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """Standardized agent output."""

    agent: str
    status: AgentStatus = AgentStatus.SUCCESS
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    result: Any = None
    summary: str = ""
    error: str | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class AgentHealth(BaseModel):
    """Runtime health snapshot for an agent."""

    name: str
    status: str = "healthy"
    version: str = "1.0"
    capabilities: list[str] = Field(default_factory=list)
    average_latency_ms: float = 0.0
    success_rate: float = 1.0
    total_calls: int = 0
    success_calls: int = 0
    last_error: str | None = None
    healthy: bool = True


class BaseAgent(ABC):
    """Contract every specialized / support agent must implement."""

    name: ClassVar[str] = "base"
    version: ClassVar[str] = "1.0"
    capabilities: ClassVar[list[str]] = []

    def __init__(self) -> None:
        self._initialized = False
        self._total_calls = 0
        self._success_calls = 0
        self._latency_sum_ms = 0.0
        self._last_error: str | None = None
        self._healthy = True

    def initialize(self) -> None:
        """One-time setup. Override for heavy resources."""
        self._initialized = True

    def can_handle(self, task: AgentTask) -> bool:
        """Return True when this agent can execute the task capability."""
        if not self._healthy:
            return False
        caps = {c.value if isinstance(c, AgentCapability) else str(c) for c in self.capabilities}
        return task.capability in caps

    @abstractmethod
    def execute(self, task: AgentTask) -> AgentResponse:
        """Run the task and return a standardized response."""

    def validate(self, response: AgentResponse) -> bool:
        """Validate agent output shape / basic success."""
        if response.status == AgentStatus.FAILED:
            return False
        return response.result is not None or bool(response.summary)

    def summarize(self, response: AgentResponse) -> str:
        """Short human-readable summary of the result."""
        if response.summary:
            return response.summary
        if isinstance(response.result, dict):
            answer = response.result.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
        if isinstance(response.result, str):
            return response.result
        return f"{response.agent}: {response.status.value}"

    def health_check(self) -> AgentHealth:
        avg = (
            self._latency_sum_ms / self._total_calls if self._total_calls else 0.0
        )
        rate = (
            self._success_calls / self._total_calls if self._total_calls else 1.0
        )
        return AgentHealth(
            name=self.name,
            status="healthy" if self._healthy else "unhealthy",
            version=self.version,
            capabilities=[
                c.value if isinstance(c, AgentCapability) else str(c)
                for c in self.capabilities
            ],
            average_latency_ms=round(avg, 2),
            success_rate=round(rate, 4),
            total_calls=self._total_calls,
            success_calls=self._success_calls,
            last_error=self._last_error,
            healthy=self._healthy and rate >= 0.5,
        )

    def run(self, task: AgentTask) -> AgentResponse:
        """Execute with latency / success tracking (used by Supervisor)."""
        started = perf_counter()
        self._total_calls += 1
        try:
            if not self._initialized:
                self.initialize()
            response = self.execute(task)
            if not isinstance(response, AgentResponse):
                response = AgentResponse(
                    agent=self.name,
                    status=AgentStatus.FAILED,
                    error="Agent.execute did not return AgentResponse",
                )
            response.latency_ms = (perf_counter() - started) * 1000.0
            response.agent = self.name
            if response.status in {AgentStatus.SUCCESS, AgentStatus.PARTIAL}:
                self._success_calls += 1
                self._last_error = None
            else:
                self._last_error = response.error or response.status.value
            self._latency_sum_ms += response.latency_ms
            if not response.summary:
                response.summary = self.summarize(response)
            return response
        except Exception as exc:  # noqa: BLE001 — agent boundary
            latency = (perf_counter() - started) * 1000.0
            self._latency_sum_ms += latency
            self._last_error = str(exc)
            return AgentResponse(
                agent=self.name,
                status=AgentStatus.FAILED,
                confidence=0.0,
                error=str(exc),
                latency_ms=latency,
                summary=f"{self.name} failed: {exc}",
            )

    def mark_unhealthy(self, reason: str | None = None) -> None:
        self._healthy = False
        if reason:
            self._last_error = reason

    def mark_healthy(self) -> None:
        self._healthy = True
