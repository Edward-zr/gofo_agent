"""Reflection Agent — answer review, consistency, hallucination detection."""

from __future__ import annotations

from typing import Any

from agents.base import AgentCapability, AgentResponse, AgentStatus, AgentTask, BaseAgent


class ReflectionAgent(BaseAgent):
    name = "Reflection"
    version = "1.0"
    capabilities = [
        AgentCapability.ANSWER_REVIEW,
        AgentCapability.CONSISTENCY_CHECK,
        AgentCapability.HALLUCINATION_DETECTION,
    ]

    def execute(self, task: AgentTask) -> AgentResponse:
        from core.models import QueryResponse
        from core.reflection import ReflectionAgent as CoreReflectionAgent

        ctx = task.context or {}
        draft = ctx.get("draft_response") or task.inputs.get("draft_response")
        question = task.resolved_question or task.question

        if draft is None:
            return AgentResponse(
                agent=self.name,
                status=AgentStatus.SKIPPED,
                confidence=0.0,
                summary="No draft response to critique",
                metadata={"action": task.action},
            )

        if isinstance(draft, QueryResponse):
            response = draft
        elif isinstance(draft, dict):
            response = QueryResponse.model_validate(draft)
        else:
            response = QueryResponse(question=question, answer=str(draft), capability="unknown")

        critic = ctx.get("reflection_agent") or CoreReflectionAgent(
            use_llm=bool(ctx.get("use_llm", False)),
        )
        result = critic.critique_response(
            question,
            response,
            conversation_memory=ctx.get("conversation_memory"),
            retry_count=int(ctx.get("retry_count") or 0),
        )
        payload: dict[str, Any] = (
            result.model_dump() if hasattr(result, "model_dump") else dict(result)
        )
        approved = bool(payload.get("approved") or payload.get("action") == "approve")
        confidence = float(payload.get("confidence") or (0.85 if approved else 0.45))
        return AgentResponse(
            agent=self.name,
            status=AgentStatus.SUCCESS if approved else AgentStatus.PARTIAL,
            confidence=confidence,
            result=payload,
            summary=payload.get("feedback") or payload.get("reason") or "Reflection complete",
            metadata={"action": task.action, "approved": approved},
        )
