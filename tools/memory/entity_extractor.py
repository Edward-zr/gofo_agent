"""Backward-compatible import shim for memory entity extraction."""

from tools.llm.client import get_llm
from tools.memory import summarizer as _summarizer


def extract_entities(question: str, answer: str) -> dict:
    """Extract entities using the rebuilt memory summarizer."""
    original_get_llm = _summarizer.get_llm
    _summarizer.get_llm = get_llm
    try:
        return _summarizer.extract_entities(question, answer)
    finally:
        _summarizer.get_llm = original_get_llm

__all__ = ["extract_entities"]
