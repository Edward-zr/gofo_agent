"""RAG Agent — SOP QA, document retrieval, summarization."""

from __future__ import annotations

from agents._bridge import run_tool_handler
from agents.base import AgentCapability, AgentResponse, AgentStatus, AgentTask, BaseAgent


class RAGAgent(BaseAgent):
    name = "RAG"
    version = "1.0"
    capabilities = [
        AgentCapability.SOP_QA,
        AgentCapability.DOCUMENT_RETRIEVAL,
        AgentCapability.DOCUMENT_SUMMARIZATION,
    ]

    def execute(self, task: AgentTask) -> AgentResponse:
        from core.plan_executor import _handle_rag

        result = run_tool_handler(_handle_rag, task)
        answer = result.get("answer") or ""
        confidence = float(result.get("confidence_score") or result.get("confidence") or 0.8)
        status = AgentStatus.SUCCESS
        if result.get("fallback_strategy") in {"clarify", "request_documents"}:
            status = AgentStatus.PARTIAL
            confidence = min(confidence, 0.55)
        return AgentResponse(
            agent=self.name,
            status=status,
            confidence=confidence,
            result=result,
            summary=answer if isinstance(answer, str) else str(answer),
            metadata={"tool": "RAG", "action": task.action},
        )
