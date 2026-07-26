"""Unit tests for the multi-stage Quality Assurance pipeline."""

from __future__ import annotations

from core.intent_classifier import IntentClassification, IntentType
from core.planner import Planner
from core.quality_assurance import (
    AnswerContext,
    CompletenessValidator,
    DecisionEngine,
    EvidenceValidator,
    QAAction,
    QualityAssurancePipeline,
    ReasoningValidator,
    build_answer_context_from_response,
)


def test_complete_answer_is_approved() -> None:
    context = AnswerContext(
        question="How many pickups today?",
        answer="There were 120 pickups today with a 92% completion rate.",
        sql="SELECT COUNT(*) AS pickup_count FROM pickups",
        sql_rows=[{"pickup_count": 120, "completion_rate": 0.92}],
        capability="sql",
    )
    report = QualityAssurancePipeline().evaluate(context)
    assert report.evidence_score >= 0.75
    assert report.reasoning_score >= 0.75
    assert report.completeness_score >= 0.75
    assert report.approved is True
    assert report.action == QAAction.APPROVE


def test_weak_retrieval_triggers_retry_retrieval() -> None:
    context = AnswerContext(
        question="What is the pickup SOP?",
        answer="Drivers should always scan packages before departure according to policy.",
        sources=[],
        capability="rag",
    )
    evidence = EvidenceValidator().validate(context)
    assert evidence.score < 0.75
    assert evidence.should_retry_retrieval is True

    report = QualityAssurancePipeline().evaluate(context)
    assert report.should_retry_retrieval is True
    assert report.action in {QAAction.RETRY_RETRIEVAL, QAAction.RETRY_PLAN}
    assert report.approved is False


def test_hallucinated_answer_lowers_evidence_score() -> None:
    context = AnswerContext(
        question="What is the CVG SOP check-in process?",
        answer="I believe drivers probably skip scanning because the hub might be understaffed.",
        sources=[{"id": "1", "text": "Drivers must scan every package at check-in.", "score": 0.8}],
        capability="rag",
    )
    evidence = EvidenceValidator().validate(context)
    assert evidence.score < 0.75
    assert "possible_hallucination_language" in evidence.problems


def test_incomplete_sql_analysis_missing_root_cause() -> None:
    context = AnswerContext(
        question="Why did Chicago performance decrease last week?",
        answer="Chicago had 80 pickups last week.",
        sql="SELECT COUNT(*) FROM pickups",
        sql_rows=[{"pickup_count": 80}],
        capability="sql",
    )
    reasoning = ReasoningValidator().validate(context)
    completeness = CompletenessValidator().validate(context)
    assert reasoning.score < 0.75
    assert "Root cause analysis" in completeness.missing_information or completeness.score < 0.75

    report = QualityAssurancePipeline().evaluate(context)
    assert report.approved is False
    assert report.action in {
        QAAction.RETRY_PLAN,
        QAAction.RETRY_SQL,
        QAAction.RETRY_PYTHON,
        QAAction.RETRY_RETRIEVAL,
    }
    assert report.should_retry_sql or report.should_retry_python or report.completeness_score < 0.75


def test_missing_comparison_detected() -> None:
    context = AnswerContext(
        question="Compare Chicago and New York performance this month",
        answer="Chicago completed 100 pickups.",
        sql_rows=[{"hub": "Chicago", "pickups": 100}],
        capability="sql",
    )
    completeness = CompletenessValidator().validate(context)
    assert completeness.score < 0.75
    assert any("comparison" in item.lower() for item in completeness.missing_information)


def test_ambiguous_request_asks_user() -> None:
    context = AnswerContext(
        question="check performance",
        answer="Performance varies by hub.",
        sql_rows=[{"hub": "ORD", "rate": 0.9}],
        capability="sql",
        ambiguous=True,
    )
    report = QualityAssurancePipeline().evaluate(context)
    assert report.should_ask_user is True
    assert report.action == QAAction.ASK_USER
    assert report.approved is False


def test_decision_engine_approve_when_all_high() -> None:
    engine = DecisionEngine()
    report = engine.decide(
        [
            EvidenceValidator().validate(
                AnswerContext(
                    question="Show today's pickups",
                    answer="120 pickups today.",
                    sql_rows=[{"pickup_count": 120}],
                    capability="sql",
                )
            ),
            ReasoningValidator().validate(
                AnswerContext(
                    question="Show today's pickups",
                    answer="120 pickups today.",
                    sql_rows=[{"pickup_count": 120}],
                    capability="sql",
                )
            ),
            CompletenessValidator().validate(
                AnswerContext(
                    question="Show today's pickups",
                    answer="120 pickups today.",
                    sql_rows=[{"pickup_count": 120}],
                    capability="sql",
                )
            ),
        ]
    )
    assert report.action == QAAction.APPROVE
    assert report.approved is True


def test_retry_limit_accepts_best_effort() -> None:
    context = AnswerContext(
        question="Why did Chicago performance decrease last week?",
        answer="Chicago declined.",
        sql_rows=[{"hub": "Chicago", "pickups": 10}],
        capability="sql",
    )
    report = QualityAssurancePipeline().evaluate(context, retry_count=2)
    assert report.action == QAAction.ACCEPT_WITH_LIMITS
    assert report.approved is True
    assert "Retry limit" in (report.retry_reason or "")


def test_successful_retry_plan_from_qa() -> None:
    report = QualityAssurancePipeline().evaluate(
        AnswerContext(
            question="Why did Chicago performance decrease last week?",
            answer="Chicago had fewer pickups.",
            sql_rows=[{"hub": "Chicago", "pickups": 10}],
            capability="sql",
        )
    )
    assert report.approved is False
    classification = IntentClassification(
        intent=IntentType.SQL_Analysis,
        confidence=0.9,
        requires_sql=True,
        requires_planner=True,
        reasoning="test",
    )
    plan = Planner().plan_retry(
        "Why did Chicago performance decrease last week?",
        classification,
        report,
        previous_answer="Chicago had fewer pickups.",
    )
    tools = [step.tool for step in plan.steps]
    assert "SQL" in tools or "PYTHON" in tools
    assert tools[-1] == "LLM"
    assert plan.requires_clarification is False


def test_follow_up_conversation_context_in_completeness() -> None:
    context = AnswerContext(
        question="What about Chicago?",
        answer="It looks fine.",
        capability="sql",
        sql_rows=[],
        conversation_memory={
            "previous_question": "Show today's pickups by hub",
            "filters": {},
        },
    )
    completeness = CompletenessValidator().validate(context)
    assert completeness.score < 0.85
    assert "follow_up_context_ignored" in completeness.problems or completeness.missing_information


def test_build_answer_context_from_response_dict() -> None:
    context = build_answer_context_from_response(
        question="Show pickups",
        response={
            "answer": "10 pickups",
            "sources": [],
            "generated_sql": "SELECT 1",
            "sql_rows": [{"n": 10}],
            "capability": "sql",
        },
    )
    assert context.answer == "10 pickups"
    assert context.sql_rows[0]["n"] == 10


def test_pipeline_supports_extra_validator() -> None:
    class DummyValidator:
        name = "cost"

        def validate(self, context: AnswerContext):
            from core.quality_assurance import ValidatorResult

            return ValidatorResult(name=self.name, score=0.9, feedback=["cost ok"])

    pipeline = QualityAssurancePipeline()
    pipeline.register_validator(DummyValidator())  # type: ignore[arg-type]
    report = pipeline.evaluate(
        AnswerContext(
            question="Show today's pickups",
            answer="120 pickups today.",
            sql_rows=[{"pickup_count": 120}],
            capability="sql",
        )
    )
    assert any(result.name == "cost" for result in report.validator_results)


def test_qa_never_executes_tools(monkeypatch) -> None:
    called = {"sql": False}

    def boom(*args, **kwargs):
        called["sql"] = True
        raise AssertionError("QA must not execute tools")

    monkeypatch.setattr("tools.sql.service.answer", boom)
    report = QualityAssurancePipeline().evaluate(
        AnswerContext(
            question="Show today's pickups",
            answer="120 pickups",
            sql_rows=[{"pickup_count": 120}],
            capability="sql",
        )
    )
    assert called["sql"] is False
    assert report.evidence_score > 0


def test_file_analysis_does_not_trigger_sql_retry() -> None:
    """Attachment ADA answers must not be overwritten by warehouse SQL retries."""
    context = AnswerContext(
        question="inspect the file",
        answer=(
            "Executive Summary — order.xlsx\n\n"
            "Dataset overview:\n- 100 rows across hubs ORD and ATL\n\n"
            "Important metrics:\n- package_count average is 12"
        ),
        sql_rows=[
            {"hub": "ORD", "package_count": 12},
            {"hub": "ATL", "package_count": 8},
        ],
        capability="file_analysis",
    )
    evidence = EvidenceValidator().validate(context)
    assert evidence.should_retry_sql is False
    assert evidence.details.get("file_mode") is True

    report = QualityAssurancePipeline().evaluate(context)
    assert report.should_retry_sql is False
    assert report.action != QAAction.RETRY_SQL
    assert report.approved is True
