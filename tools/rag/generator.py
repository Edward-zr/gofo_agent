"""GPT answer generation from retrieved GOFO SOP context (no retrieval)."""

from __future__ import annotations

from typing import Any

from core.models import SourceChunk
from core.prompt_manager import get_prompt_manager

UNKNOWN_ANSWER = "I don't know based on the available SOP documents."

GENERAL_KNOWLEDGE_DISCLAIMER = (
    "This answer is based on general logistics knowledge rather than your "
    "internal SOP documentation."
)


def _extract_source_pdf(metadata: dict) -> str:
    return str(metadata.get("filename") or metadata.get("source") or "unknown")


def _extract_page_number(metadata: dict) -> str:
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


def _format_context_block(chunks: list[SourceChunk]) -> str:
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        pdf_name = _extract_source_pdf(chunk.metadata)
        page_number = _extract_page_number(chunk.metadata)
        blocks.append(
            f"[Source {index} | PDF: {pdf_name} | Page: {page_number}]\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def generate(
    question: str,
    chunks: list[SourceChunk],
    *,
    confidence_context: dict[str, Any] | None = None,
) -> str:
    """
    Generate an answer from retrieved SOP chunks via the Prompt Registry.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    ctx = confidence_context or {}
    if not chunks and not ctx.get("use_general_knowledge"):
        return UNKNOWN_ANSWER

    # Prefer cautious candidate prompt when confidence is MEDIUM/LOW.
    version = None
    level = str(ctx.get("confidence_level") or "HIGH").upper()
    if level in {"MEDIUM", "LOW"} or ctx.get("fallback_strategy") == "GENERATE_CAUTIOUS":
        version = "v2"

    sop_context = _format_context_block(chunks) if chunks else "(no SOP context)"
    if ctx.get("use_general_knowledge") or ctx.get("fallback_strategy") == "GENERAL_KNOWLEDGE":
        sop_context = (
            "(insufficient SOP context)\n"
            f"Use general logistics knowledge and begin with: {GENERAL_KNOWLEDGE_DISCLAIMER}"
        )

    try:
        answer = get_prompt_manager().invoke(
            "rag.generator_prompt",
            {
                "question": question,
                "sop_context": sop_context,
                "confidence_level": ctx.get("confidence_level") or "HIGH",
                "confidence_score": ctx.get("confidence_score") or "",
                "confidence_reason": ctx.get("confidence_reason") or "",
                "fallback_strategy": ctx.get("fallback_strategy") or "GENERATE_CONFIDENT",
            },
            version=version,
        )
        return answer or UNKNOWN_ANSWER
    except Exception:
        return UNKNOWN_ANSWER
