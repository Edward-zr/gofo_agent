"""Request verification — Stage 6 of the planner-driven execution loop.

Compares tool output against TaskSpec. Never requests SQL retry for SOP/RAG
answers (prevents SELECT UNKNOWN overwriting glossary answers).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from core.intent_router import RouteIntent
from core.logger import get_logger
from core.task_decomposition import TaskSpec

logger = get_logger("request_verifier")


class VerificationReport(BaseModel):
    """Structured verification outcome for planner re-plan feedback."""

    passed: bool = False
    capability_ok: bool = True
    tool_ok: bool = True
    chart_type_ok: bool = True
    limit_ok: bool = True
    aggregation_ok: bool = True
    filters_ok: bool = True
    attachment_ok: bool = True
    knowledge_source_ok: bool = True
    feedback: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    suggested_tool_calls: list[str] = Field(default_factory=list)
    should_retry_retrieval: bool = False
    should_retry_sql: bool = False
    should_retry_python: bool = False
    should_ask_user: bool = False
    action: str = "approve"
    reasoning: str = ""

    # Compatibility aliases used by Planner.plan_retry
    @property
    def approved(self) -> bool:
        return self.passed


def verify(
    *,
    task_spec: TaskSpec | dict[str, Any],
    response: Any,
    route_decision: dict[str, Any] | None = None,
    execution_plan: dict[str, Any] | None = None,
) -> VerificationReport:
    """Verify response against TaskSpec constraints."""
    spec = task_spec if isinstance(task_spec, TaskSpec) else TaskSpec.model_validate(task_spec)
    route = dict(route_decision or {})
    plan = dict(execution_plan or {})
    report = VerificationReport(passed=True, action="approve")

    answer = _answer(response)
    capability = (_capability(response) or "").lower()
    charts = _charts(response)
    sql = _sql(response)
    sources = _sources(response)
    rows = _rows(response)
    attachment_ids = _attachment_ids(response)
    route_intent = str(route.get("intent") or spec.route_intent or "")

    # --- Capability / domain ---
    expected_caps = _expected_capabilities(spec)
    if expected_caps and capability and capability not in expected_caps and capability not in {
        "llm",
        "planner",
        "orchestrator",
        "multi_agent",
        "clarification",
    }:
        # Soft check: domain SOP allows llm/rag; SQL allows sql/llm/business_analysis
        if not _capability_matches_domain(spec.domain, capability):
            report.capability_ok = False
            report.feedback.append(
                f"Capability '{capability}' does not match domain '{spec.domain}'."
            )

    # --- Knowledge source (SOP must not become SQL) ---
    if spec.domain == "SOP" or route_intent == RouteIntent.SOP_QA:
        if sql and "UNKNOWN" in str(sql).upper():
            report.knowledge_source_ok = False
            report.feedback.append("SOP/RAG answer was overwritten by SQL UNKNOWN.")
        if capability in {"sql"} and not sources:
            report.knowledge_source_ok = False
            report.feedback.append("SOP question returned SQL capability without RAG sources.")
        # Never suggest SQL retry for SOP
        report.should_retry_sql = False

    if spec.domain == "SQL" and not answer and not rows:
        report.tool_ok = False
        report.feedback.append("SQL analytics returned no answer and no rows.")
        report.should_retry_sql = True
        report.suggested_tool_calls.append("SQL")

    if spec.domain == "SOP" and not answer and not sources:
        report.tool_ok = False
        report.feedback.append("SOP/RAG returned empty answer and no sources.")
        report.should_retry_retrieval = True
        report.suggested_tool_calls.append("RAG")

    # --- Chart type ---
    if spec.chart_type:
        chart_types = {str(c.get("type") or "").lower() for c in charts if isinstance(c, dict)}
        wanted = spec.chart_type.lower()
        aliases = {wanted}
        if wanted == "horizontal_bar":
            aliases.add("bar")
        if wanted == "bar":
            aliases.add("horizontal_bar")
        if charts and not (chart_types & aliases):
            report.chart_type_ok = False
            report.feedback.append(
                f"Requested chart_type '{spec.chart_type}' but got {sorted(chart_types) or 'none'}."
            )
            if spec.attachment_relevant or spec.domain == "ADA":
                report.suggested_tool_calls.append("ATTACHMENT")
            elif "VISUALIZATION" not in report.suggested_tool_calls:
                report.suggested_tool_calls.append("VISUALIZATION")
        elif not charts and wanted:
            report.chart_type_ok = False
            report.feedback.append(f"Requested chart_type '{spec.chart_type}' but no charts were produced.")
            report.suggested_tool_calls.append(
                "ATTACHMENT" if spec.attachment_relevant or spec.domain == "ADA" else "VISUALIZATION"
            )

    # --- Limit (best-effort: ranking answers / row counts) ---
    if spec.limit and rows and len(rows) > spec.limit * 3:
        # Large overshoot suggests wrong limit handling
        report.limit_ok = False
        report.feedback.append(
            f"Requested limit≈{spec.limit} but received {len(rows)} rows."
        )

    # --- Attachment usage ---
    if spec.attachment_relevant and not attachment_ids and capability not in {
        "file_analysis",
        "file_comparison",
        "attachment",
        "image_analysis",
    }:
        # Wait / clarify paths exempt
        if capability not in {"clarification", "conversation", "wait_for_upload"}:
            report.attachment_ok = False
            report.feedback.append("Attachment-relevant request did not use attachment analysis.")
            report.suggested_tool_calls.append("ATTACHMENT")
    if not spec.attachment_relevant and capability in {
        "file_analysis",
        "file_comparison",
        "attachment",
    } and spec.domain in {"SOP", "CHAT", "SQL"}:
        report.attachment_ok = False
        report.feedback.append(
            f"Domain '{spec.domain}' should not bind attachment session for this ask."
        )
        if spec.domain == "SOP":
            report.should_retry_retrieval = True
            report.suggested_tool_calls.append("RAG")

    # --- Empty answer ---
    if not (answer or "").strip() and capability != "clarification":
        report.tool_ok = False
        report.feedback.append("Empty answer.")
        if spec.domain == "SOP":
            report.should_retry_retrieval = True
            report.suggested_tool_calls.append("RAG")
        elif spec.domain == "SQL":
            report.should_retry_sql = True
            report.suggested_tool_calls.append("SQL")

    # SOP safety: never enable SQL retry
    if spec.domain == "SOP" or route_intent == RouteIntent.SOP_QA:
        report.should_retry_sql = False
        report.suggested_tool_calls = [
            t for t in report.suggested_tool_calls if t.upper() != "SQL"
        ]

    report.passed = all(
        [
            report.capability_ok,
            report.tool_ok,
            report.chart_type_ok,
            report.limit_ok,
            report.aggregation_ok,
            report.filters_ok,
            report.attachment_ok,
            report.knowledge_source_ok,
        ]
    )
    if report.passed:
        report.action = "approve"
        report.reasoning = "Output matches TaskSpec constraints."
    else:
        if report.should_retry_retrieval:
            report.action = "retry_retrieval"
        elif report.should_retry_sql:
            report.action = "retry_sql"
        elif report.suggested_tool_calls:
            report.action = "retry_plan"
        else:
            report.action = "retry_plan"
        report.reasoning = "; ".join(report.feedback) or "Verification failed."

    # Deduplicate suggested tools
    report.suggested_tool_calls = list(dict.fromkeys(report.suggested_tool_calls))

    logger.info(
        "[Verify] passed=%s action=%s domain=%s feedback=%s",
        report.passed,
        report.action,
        spec.domain,
        report.feedback[:3],
    )
    return report


def _expected_capabilities(spec: TaskSpec) -> set[str]:
    mapping = {
        "SOP": {"rag", "llm"},
        "SQL": {"sql", "llm", "business_analysis", "planner", "orchestrator"},
        "ADA": {
            "file_analysis",
            "file_comparison",
            "attachment",
            "image_analysis",
            "llm",
        },
        "CHAT": {"conversation", "llm", "general"},
        "WAIT_UPLOAD": {"wait_for_upload", "conversation"},
        "HYBRID": {"rag", "sql", "file_analysis", "llm", "attachment"},
    }
    return set(mapping.get(spec.domain, set()))


def _capability_matches_domain(domain: str, capability: str) -> bool:
    return capability in _expected_capabilities(
        TaskSpec(domain=domain)  # type: ignore[arg-type]
    ) or capability in {"llm", "planner", "orchestrator", "multi_agent"}


def _answer(response: Any) -> str:
    if isinstance(response, dict):
        return str(response.get("answer") or "")
    return str(getattr(response, "answer", None) or "")


def _capability(response: Any) -> str | None:
    if isinstance(response, dict):
        analysis = response.get("analysis") or {}
        return analysis.get("capability") or response.get("capability")
    return getattr(response, "capability", None)


def _charts(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, dict):
        charts = response.get("charts") or (response.get("analysis") or {}).get("charts") or []
        return list(charts)
    return list(getattr(response, "charts", None) or [])


def _sql(response: Any) -> str | None:
    if isinstance(response, dict):
        return response.get("sql") or (response.get("raw") or {}).get("generated_sql")
    return getattr(response, "generated_sql", None)


def _sources(response: Any) -> list[Any]:
    if isinstance(response, dict):
        return list(response.get("sources") or [])
    return list(getattr(response, "sources", None) or [])


def _rows(response: Any) -> list[Any]:
    if isinstance(response, dict):
        return list(response.get("data") or [])
    return list(getattr(response, "sql_rows", None) or [])


def _attachment_ids(response: Any) -> list[str]:
    if isinstance(response, dict):
        analysis = response.get("analysis") or {}
        return list(analysis.get("attachment_ids") or response.get("attachment_ids") or [])
    return list(getattr(response, "attachment_ids", None) or [])
