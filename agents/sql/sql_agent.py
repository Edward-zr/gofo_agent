"""SQL Agent — SQL planning, generation, and database querying."""

from __future__ import annotations

from agents._bridge import run_tool_handler
from agents.base import AgentCapability, AgentResponse, AgentStatus, AgentTask, BaseAgent


class SQLAgent(BaseAgent):
    name = "SQL"
    version = "1.0"
    capabilities = [
        AgentCapability.SQL_PLANNING,
        AgentCapability.SQL_GENERATION,
        AgentCapability.DATABASE_QUERY,
        AgentCapability.RELATIONSHIP_LOOKUP,  # temporary until dedicated KG agent ships
    ]

    def execute(self, task: AgentTask) -> AgentResponse:
        if task.capability == AgentCapability.RELATIONSHIP_LOOKUP.value or task.tool == "KNOWLEDGE_GRAPH":
            from core.plan_executor import _handle_knowledge_graph

            result = run_tool_handler(_handle_knowledge_graph, task)
        else:
            from core.plan_executor import _handle_sql

            result = run_tool_handler(_handle_sql, task)

        answer = result.get("answer") or ""
        sql = result.get("generated_sql") or result.get("sql")
        ok = bool(sql or answer or result.get("sql_rows") or result.get("facts"))
        unknown = isinstance(sql, str) and "UNKNOWN" in sql.upper()
        confidence = 0.5 if unknown else (0.92 if ok else 0.3)
        return AgentResponse(
            agent=self.name,
            status=AgentStatus.SUCCESS if ok else AgentStatus.PARTIAL,
            confidence=confidence,
            result=result,
            summary=answer if isinstance(answer, str) and answer else (sql or "SQL agent completed"),
            metadata={"tool": task.tool or "SQL", "action": task.action},
        )
