"""Format and print query results for the CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import config
from core.models import QueryResponse, SourceChunk
from cli.colors import Colors


@dataclass(frozen=True)
class RagDebugStats:
    """Retrieval diagnostics derived from a QueryResponse."""

    question: str
    rewritten_question: Optional[str]
    retrieved_chunks: int
    best_similarity: float | None
    average_similarity: float | None
    decision: str


def _extract_source_pdf(metadata: dict) -> str:
    """Resolve the source PDF filename from chunk metadata."""
    return str(metadata.get("filename") or metadata.get("source") or "unknown")


def _extract_page_number(metadata: dict) -> str:
    """Resolve a human-readable page number from chunk metadata."""
    page_label = metadata.get("page_label")
    if page_label is not None and str(page_label).strip():
        return str(page_label)

    page = metadata.get("page")
    if page is not None:
        try:
            return str(int(page) + 1)
        except (TypeError, ValueError):
            return str(page)

    return "unknown"


def _best_similarity_score(sources: list[SourceChunk]) -> float | None:
    """Return the highest relevance score among retrieved chunks."""
    if not sources:
        return None
    return max(chunk.score for chunk in sources)


def _average_similarity_score(sources: list[SourceChunk]) -> float | None:
    """Return the mean relevance score across retrieved chunks."""
    if not sources:
        return None
    return sum(chunk.score for chunk in sources) / len(sources)


def _rag_decision(sources: list[SourceChunk]) -> str:
    """Mirror service.answer() routing without calling retrieval or generation."""
    if not sources:
        return "Returning UNKNOWN_ANSWER (no retrieved chunks)"

    best_score = _best_similarity_score(sources)
    if best_score is not None and best_score < config.SIMILARITY_THRESHOLD:
        return "Returning UNKNOWN_ANSWER (low confidence)"

    return "Calling GPT"


def compute_rag_debug_stats(response: QueryResponse) -> RagDebugStats:
    """Compute retrieval diagnostics for debug display."""
    return RagDebugStats(
        question=response.question,
        rewritten_question=response.rewritten_question,
        retrieved_chunks=len(response.sources),
        best_similarity=_best_similarity_score(response.sources),
        average_similarity=_average_similarity_score(response.sources),
        decision=_rag_decision(response.sources),
    )


def _format_similarity(score: float | None) -> str:
    """Format a similarity score for debug output."""
    if score is None:
        return "N/A"
    return f"{score:.4f}"


def _print_rag_debug_panel(stats: RagDebugStats, colors: Colors) -> None:
    """Print retrieval diagnostics before the answer in debug mode."""
    divider = colors.divider("-", 50)

    print(divider)
    print(colors.header("RAG Debug"))
    print(divider)
    print(colors.label("Original Question:"))
    print(stats.question)
    print()
    print(colors.label("Rewritten Question:"))
    print(stats.rewritten_question or stats.question)
    print()
    print(colors.label("Retrieved chunks:"))
    print(str(stats.retrieved_chunks))
    print()
    print(colors.label("Best similarity:"))
    print(_format_similarity(stats.best_similarity))
    print()
    print(colors.label("Average similarity:"))
    print(_format_similarity(stats.average_similarity))
    print()
    print(colors.label("Decision:"))
    print(stats.decision)
    print(divider)
    print()


def _print_sql_debug_panel(response: QueryResponse, colors: Colors) -> None:
    """Print SQL diagnostics before the answer in debug mode."""
    from tools.sql.formatter import format_result

    divider = colors.divider("-", 50)

    print(divider)
    print(colors.header("SQL Debug"))
    print(divider)
    print(colors.label("Original Question:"))
    print(response.question)
    print()
    print(colors.label("Latest Business Date:"))
    print(response.latest_business_date or "N/A")
    print()
    print(colors.label("Rewritten Question:"))
    print(response.rewritten_question or response.question)
    print()
    print(colors.label("Generated SQL:"))
    print(response.generated_sql or "")
    print()
    print(colors.label("Raw rows:"))
    print(format_result(response.sql_rows or []))
    print(divider)
    print()


def _print_planner_debug_panel(response: QueryResponse, colors: Colors) -> None:
    """Print natural-language planner diagnostics."""
    divider = colors.divider("=", 26)
    entities = response.planning_entities or {}

    print(divider, colors.header("Planner"), divider)
    print(colors.label("Capability:"), response.planning_capability or response.capability)
    print(colors.label("Intent:"), response.planning_intent or "unknown")
    confidence = response.planning_confidence
    print(colors.label("Confidence:"), "N/A" if confidence is None else f"{confidence:.2f}")
    print(colors.label("Reasoning:"), response.planning_reasoning or response.plan_reason or "")
    print(colors.label("Entities:"), str(entities))
    print()


def _print_memory_debug_panel(response: QueryResponse, colors: Colors) -> None:
    """Print short-term memory diagnostics."""
    divider = colors.divider("=", 34)
    current_state = response.memory_current_state or {}
    last_result = current_state.get("last_result_context") or response.last_result_context or {}
    previous_sql = (current_state.get("last_sql_context") or {}).get("sql")

    print(divider)
    print(colors.header("MEMORY DEBUG"))
    print(divider)
    print(colors.label("Original Question:"))
    print(response.original_question or response.question)
    print()
    print(colors.label("Resolved Question:"))
    print(response.resolved_question or response.question)
    print()
    print(colors.label("Repair detected:"))
    print(str(bool(response.repair_detected)))
    print()
    print(colors.label("Repair type:"))
    print(response.repair_type or "N/A")
    print()
    print(colors.label("Changed Dimension:"))
    print(response.changed_dimension or "N/A")
    print()
    print(colors.label("Intent:"))
    print(response.classifier_intent or response.planning_intent or "N/A")
    print()
    print(colors.label("Metric:"))
    print(response.business_metric or str(current_state.get("current_metric")) or "N/A")
    print()
    print(colors.label("Dimension:"))
    print(response.analysis_dimension or str(current_state.get("analysis_dimension")) or "N/A")
    print()
    print(colors.label("Date range:"))
    print(response.date_range or str(current_state.get("date_range")) or "N/A")
    print()
    print(colors.label("Filters:"))
    print(str(response.analysis_filters or current_state.get("filters", {})))
    print()
    print(colors.label("Inherited Context:"))
    print(str(response.inherited_context or {}))
    print()
    print(colors.label("Business Findings:"))
    print(str(response.business_findings or []))
    print()
    print(colors.label("SQL Cache Hit:"))
    print(str(bool(response.sql_cache_hit)))
    print()
    print(colors.label("Memory Updated:"))
    print(str(bool(response.memory_updated)))
    print()
    print(colors.label("Attachment IDs:"))
    print(str(response.attachment_ids or []))
    print()
    print(colors.label("Attachment Filenames:"))
    print(str(response.attachment_filenames or []))
    print()
    print(colors.label("Data Sources:"))
    print(str(response.data_sources or []))
    print()
    print(colors.label("File Context Summary:"))
    print(str(response.file_context_summary or {}))
    print()
    print(colors.label("Last 10 turns:"))
    print(colors.label("History count:"), str(response.memory_history_count or 0))
    print()
    print(colors.label("Current Topic:"))
    print(str(current_state.get("current_topic")))
    print()
    print(colors.label("Active Entities:"))
    print(str(current_state.get("active_entities", {})))
    print()
    print(colors.label("Previous SQL:"))
    print(previous_sql or "N/A")
    print()
    print(colors.label("Previous Result Rows:"))
    print(str(len(last_result.get("rows", []))))
    print()
    print(colors.label("Long Memory Matches:"))
    print(str(response.long_memory_matches or []))
    print()
    print(colors.label("Previous Issues Found:"))
    print(str(response.previous_issues_found or []))
    print()
    print(colors.label("Saved Memory:"))
    print(str(bool(response.saved_memory)))
    print()
    print(colors.label("Learned Patterns:"))
    print(str(response.learned_patterns or []))
    print()
    print(colors.label("Root cause:"))
    print(str(response.root_cause or "N/A"))
    print()
    print(colors.label("Anomaly:"))
    print(str(response.anomaly or "N/A"))
    print()
    print(colors.label("Recommendation:"))
    print(response.recommendation or "N/A")
    print(divider)
    print()


def _print_multi_debug_panel(response: QueryResponse, colors: Colors) -> None:
    """Print multi-tool diagnostics before the answer in debug mode."""
    from tools.sql.formatter import format_result

    divider = colors.divider("-", 50)
    execution_order = ", ".join(response.execution_order or [])

    print(divider)
    print(colors.header("Multi-Tool Execution"))
    print(divider)
    print(colors.label("Needs SQL:"), str(response.needs_sql))
    print(colors.label("Needs RAG:"), str(response.needs_rag))
    print(colors.label("Reason:"), response.plan_reason or "")
    print(colors.label("Execution Order:"), execution_order or "N/A")
    print()
    print(colors.label("SQL Output:"))
    print(response.sql_output or "")
    if response.generated_sql:
        print()
        print(colors.label("Generated SQL:"))
        print(response.generated_sql)
    if response.sql_rows is not None:
        print()
        print(colors.label("Raw rows:"))
        print(format_result(response.sql_rows))
    print()
    print(colors.label("RAG Output:"))
    print(response.rag_output or "")
    print()
    print(colors.label("RAG Sources:"))
    if not response.sources:
        print(colors.wrap("No sources retrieved.", colors.YELLOW))
    else:
        for chunk in response.sources:
            filename = _extract_source_pdf(chunk.metadata)
            page = _extract_page_number(chunk.metadata)
            print(f"- {colors.value(filename)} (page {colors.value(page)})")
    print(divider)
    print()


def _print_sources(
    sources: list[SourceChunk],
    colors: Colors,
    *,
    debug: bool,
) -> None:
    """Print source citations; debug mode includes retrieval diagnostics."""
    print(colors.label("Sources:"))

    if not sources:
        print(colors.wrap("No sources retrieved.", colors.YELLOW))
        return

    for chunk in sources:
        filename = _extract_source_pdf(chunk.metadata)
        page = _extract_page_number(chunk.metadata)

        if debug:
            print(f"- {colors.value(filename)}")
            print(f"  page: {colors.value(page)}")
            print(f"  similarity score: {colors.value(f'{chunk.score:.4f}')}")
            print(f"  chunk ID: {colors.value(chunk.id)}")
        else:
            print(f"- {colors.value(filename)} (page {colors.value(page)})")


def print_query_response(
    response: QueryResponse,
    colors: Optional[Colors] = None,
    *,
    debug: bool = False,
) -> None:
    """Print a RAG answer with source citations; debug adds retrieval details."""
    colors = colors or Colors()

    print()
    if debug:
        _print_memory_debug_panel(response, colors)
        _print_planner_debug_panel(response, colors)
        if response.capability == "multi":
            _print_multi_debug_panel(response, colors)
        elif response.capability == "sql":
            _print_sql_debug_panel(response, colors)
        elif response.capability == "rag":
            _print_rag_debug_panel(compute_rag_debug_stats(response), colors)

    answer_label = "Synthesized Answer:" if debug and response.capability == "multi" else "Answer:"
    print(colors.label(answer_label))
    print(response.answer or "")
    print()
    _print_sources(response.sources, colors, debug=debug)


def print_retrieval_results(
    question: str,
    chunks: list[SourceChunk],
    colors: Optional[Colors] = None,
) -> None:
    """Print retrieved chunks in a readable debug format."""
    colors = colors or Colors()

    print()
    print(colors.header("GOFO Retrieval Debug"))
    print(colors.label("Question:"), colors.value(question))
    print(colors.label("Results:"), colors.value(str(len(chunks))))
    print()

    if not chunks:
        print(colors.wrap("No chunks retrieved. Run ingest.py and try again.", colors.RED))
        return

    for rank, chunk in enumerate(chunks, start=1):
        print(colors.divider("=", 40))
        print(colors.label("Rank:"), colors.value(str(rank)))
        print(colors.label("Similarity Score:"), colors.value(f"{chunk.score:.4f}"))
        print(colors.label("Source PDF:"), colors.value(_extract_source_pdf(chunk.metadata)))
        print(colors.label("Page Number:"), colors.value(_extract_page_number(chunk.metadata)))
        print(colors.label("Chunk ID:"), colors.value(chunk.id))
        print(colors.label("Chunk Text:"))
        print(chunk.text)
        print(colors.divider("=", 40))
        print()
