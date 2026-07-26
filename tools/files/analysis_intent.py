"""Detect per-prompt analysis intent for uploaded attachments."""

from __future__ import annotations

import re
from enum import StrEnum


class AnalysisIntent(StrEnum):
    WAIT_FOR_UPLOAD = "WAIT_FOR_UPLOAD"
    EXECUTIVE_SUMMARY = "EXECUTIVE_SUMMARY"
    BUSINESS_REPORT = "BUSINESS_REPORT"
    VISUALIZE = "VISUALIZE"
    RANKING = "RANKING"
    COMPARISON = "COMPARISON"
    ANOMALY = "ANOMALY"
    FORECAST = "FORECAST"
    FILTER = "FILTER"
    AGGREGATION = "AGGREGATION"
    NARRATIVE = "NARRATIVE"
    LOOKUP = "LOOKUP"
    RECOMMENDATION = "RECOMMENDATION"
    ROOT_CAUSE = "ROOT_CAUSE"
    RECORDS = "RECORDS"
    GENERAL = "GENERAL"


_WAIT_PHRASES = (
    "i will upload",
    "i'll upload",
    "i am going to upload",
    "i'm going to upload",
    "going to upload",
    "about to upload",
    "want you to analyze an excel",
    "want you to analyze a csv",
    "want you to analyze a file",
    "i want to upload",
    "let me upload",
    "i will send a file",
    "i'll send a file",
)


def detect_analysis_intent(question: str) -> AnalysisIntent:
    """Detect the analysis intent for the current user prompt."""
    normalized = question.lower().strip()
    if not normalized:
        return AnalysisIntent.GENERAL

    if any(phrase in normalized for phrase in _WAIT_PHRASES):
        return AnalysisIntent.WAIT_FOR_UPLOAD

    if any(phrase in normalized for phrase in ("visualize", "visualise", "show chart", "show charts", "plot", "graph", "dashboard", "heatmap", "histogram", "scatter", "pie chart", "bar chart", "line chart")):
        return AnalysisIntent.VISUALIZE

    if any(phrase in normalized for phrase in ("find anomal", "detect anomal", "outlier", "unusual", "spike")):
        return AnalysisIntent.ANOMALY

    if any(phrase in normalized for phrase in ("predict", "forecast", "projection", "next week", "next month")):
        return AnalysisIntent.FORECAST

    if any(phrase in normalized for phrase in ("compare", "versus", "vs ", "difference between", "changed the most")):
        return AnalysisIntent.COMPARISON

    if any(phrase in normalized for phrase in ("what should operations do", "recommend", "recommendation", "next action")):
        return AnalysisIntent.RECOMMENDATION

    if re.search(r"\bwhy\b", normalized) or "root cause" in normalized or "what caused" in normalized:
        return AnalysisIntent.ROOT_CAUSE

    if any(phrase in normalized for phrase in ("filter", "only ", "where ", "for chicago", "for ord", "in chicago", "in ord")):
        return AnalysisIntent.FILTER

    if any(
        phrase in normalized
        for phrase in (
            "top 10",
            "top ten",
            "bottom 10",
            "show top",
            "aggregate",
            "group by",
            "grouped by",
            "total by",
            "sum by",
            "average by",
            "count by",
            "breakdown by",
            "for each",
            "based on each",
            "how many packages for each",
            "how many for each",
            "distribution of",
            "distribution based",
            "packages for each",
            "volume by",
            "count per",
            "number of packages",
        )
    ):
        return AnalysisIntent.AGGREGATION

    if re.search(r"\bcolumns?\b", normalized) and any(
        phrase in normalized for phrase in ("how many", "count", "distribution", "packages", "for each", "based on")
    ):
        return AnalysisIntent.AGGREGATION

    if any(phrase in normalized for phrase in ("rank", "highest", "lowest", "best", "worst", "top performer", "bottom performer")):
        return AnalysisIntent.RANKING

    if any(phrase in normalized for phrase in ("show his records", "show her records", "show their records", "show records", "list records")):
        return AnalysisIntent.RECORDS

    if any(phrase in normalized for phrase in ("belong to", "which hub", "which driver", "which customer", "who is")):
        return AnalysisIntent.LOOKUP

    if any(phrase in normalized for phrase in ("what happened", "explain", "narrative", "tell me about", "walk me through")):
        return AnalysisIntent.NARRATIVE

    if any(phrase in normalized for phrase in ("report", "business report", "full report", "ops report")):
        return AnalysisIntent.BUSINESS_REPORT

    if any(phrase in normalized for phrase in ("summarize", "summarise", "summary", "what is in", "what's in", "overview", "analyze this", "analyse this", "analyze the file", "analyse the file", "inspect the file", "inspect this file", "inspect file", "look at the file", "look at this file", "review the file", "review this file")):
        return AnalysisIntent.EXECUTIVE_SUMMARY

    return AnalysisIntent.GENERAL


def is_wait_for_upload(question: str) -> bool:
    return detect_analysis_intent(question) == AnalysisIntent.WAIT_FOR_UPLOAD


WAIT_FOR_UPLOAD_REPLY = (
    "I understand.\n\n"
    "Upload the file and I will:\n\n"
    "• inspect the file\n"
    "• summarize it\n"
    "• detect trends\n"
    "• identify anomalies\n"
    "• recommend actions\n"
    "• generate charts\n"
    "• answer follow-up questions."
)
