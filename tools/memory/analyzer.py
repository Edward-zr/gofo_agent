"""Backward-compatible import shim for result analysis."""

from tools.llm.client import get_llm
from tools.memory import result_analyzer as _result_analyzer


def analyze_previous_result(question: str, last_result_context: dict) -> str:
    """Analyze previous results using the rebuilt result analyzer."""
    original_get_llm = _result_analyzer.get_llm
    _result_analyzer.get_llm = get_llm
    try:
        return _result_analyzer.analyze_previous_result(question, last_result_context)
    finally:
        _result_analyzer.get_llm = original_get_llm

__all__ = ["analyze_previous_result"]
