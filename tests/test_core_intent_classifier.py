"""Unit tests for core IntentClassifier."""

from __future__ import annotations

from types import SimpleNamespace

from core.intent_classifier import IntentClassifier, IntentType


class _FakeMemory:
    def __init__(self, turns: list[dict] | None = None) -> None:
        self._turns = turns or []

    def get_recent_history(self, limit: int = 10) -> list[dict]:
        return self._turns[-limit:]


class _FakeLLM:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0

    def invoke(self, messages):  # noqa: ANN001
        self.calls += 1
        return SimpleNamespace(content=self.payload)


def test_greeting_heuristic() -> None:
    result = IntentClassifier().classify("hello")
    assert result.intent == IntentType.Greeting
    assert result.confidence >= 0.9
    assert result.requires_clarification is False


def test_chitchat_heuristic() -> None:
    result = IntentClassifier().classify("tell me a joke")
    assert result.intent == IntentType.ChitChat


def test_sop_qa_heuristic() -> None:
    result = IntentClassifier().classify("How do drivers check in?")
    assert result.intent == IntentType.SOP_QA
    assert result.requires_rag is True


def test_sop_summary_heuristic() -> None:
    result = IntentClassifier().classify("Summarize the CVG SOP key points")
    assert result.intent == IntentType.SOP_Summary
    assert result.requires_rag is True


def test_sop_compare_heuristic() -> None:
    result = IntentClassifier().classify("Compare CVG SOP with ORD SOP")
    assert result.intent == IntentType.SOP_Compare
    assert result.requires_planner is True


def test_sql_query_heuristic() -> None:
    result = IntentClassifier().classify("Show today's pickups")
    assert result.intent == IntentType.SQL_Query
    assert result.requires_sql is True


def test_sql_analysis_heuristic() -> None:
    result = IntentClassifier().classify("Which hub has the worst pickup rate?")
    assert result.intent == IntentType.SQL_Analysis
    assert result.requires_planner is True
    assert result.requires_sql is True


def test_dashboard_heuristic() -> None:
    result = IntentClassifier().classify("Create a dashboard of pickup trends")
    assert result.intent == IntentType.Dashboard
    assert result.requires_planner is True


def test_upload_file_heuristic() -> None:
    result = IntentClassifier().classify("Analyze this CSV file")
    assert result.intent == IntentType.Upload_File


def test_coding_heuristic() -> None:
    result = IntentClassifier().classify("Write Python to parse pickups")
    assert result.intent == IntentType.Coding


def test_follow_up_uses_memory() -> None:
    memory = _FakeMemory(
        [
            {
                "user_question": "Show today's pickups",
                "assistant_answer": "There were 120 pickups today.",
            }
        ]
    )
    result = IntentClassifier().classify("What about Chicago?", memory)
    assert result.intent == IntentType.Follow_Up
    assert result.requires_memory is True


def test_follow_up_not_triggered_without_history() -> None:
    llm = _FakeLLM(
        '{"intent":"SQL_Query","confidence":0.8,"requires_sql":true,'
        '"requires_rag":false,"requires_memory":false,"requires_planner":false,'
        '"requires_clarification":false,"reasoning":"standalone city query"}'
    )
    result = IntentClassifier(llm=llm).classify("What about Chicago?")
    assert result.intent != IntentType.Follow_Up
    assert llm.calls == 1


def test_low_confidence_becomes_unknown() -> None:
    llm = _FakeLLM(
        '{"intent":"SQL_Analysis","confidence":0.2,"requires_sql":true,'
        '"requires_rag":false,"requires_memory":false,"requires_planner":true,'
        '"requires_clarification":false,"reasoning":"guess"}'
    )
    # Use a question that skips heuristics so GPT path runs.
    result = IntentClassifier(llm=llm).classify("Please help with the purple widgets situation")
    assert result.intent == IntentType.Unknown
    assert result.requires_clarification is True
    assert llm.calls == 1


def test_gpt_path_parses_structured_output() -> None:
    llm = _FakeLLM(
        '{"intent":"General_Knowledge","confidence":0.88,"requires_sql":false,'
        '"requires_rag":false,"requires_memory":false,"requires_planner":false,'
        '"requires_clarification":false,"reasoning":"general concept"}'
    )
    result = IntentClassifier(llm=llm).classify("What is machine learning?")
    assert result.intent == IntentType.General_Knowledge
    assert result.confidence == 0.88
    assert llm.calls == 1
