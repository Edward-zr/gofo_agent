"""GPT answer generation from retrieved GOFO SOP context (no retrieval)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from core.models import SourceChunk
from tools.llm.client import get_llm

UNKNOWN_ANSWER = "I don't know based on the available SOP documents."


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


def _format_context_block(chunks: list[SourceChunk]) -> str:
    """Serialize retrieved chunks into a single prompt context block."""
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        pdf_name = _extract_source_pdf(chunk.metadata)
        page_number = _extract_page_number(chunk.metadata)
        blocks.append(
            f"[Source {index} | PDF: {pdf_name} | Page: {page_number}]\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def _build_system_prompt() -> str:
    """Return static system instructions for grounded SOP answers."""
    return (
        "You are the GOFO Operations Intelligence Assistant.\n"
        "Answer ONLY using the SOP context provided in the user message.\n"
        "Never invent steps, policies, numbers, or names that are not in the context.\n"
        f"If the context does not contain enough information to answer the question, "
        f'respond exactly with: "{UNKNOWN_ANSWER}"\n'
        "Be concise and operational, focusing on procedures, requirements, and exceptions.\n"
        "Do not mention context, chunks, or retrieved documents in your answer."
    )


def _build_user_prompt(question: str, context_block: str) -> str:
    """Return the user message containing the question and SOP context."""
    return (
        f"Question:\n{question}\n\n"
        f"SOP Context:\n{context_block}\n\n"
        "Answer the question using only the SOP Context above."
    )


def _build_messages(question: str, chunks: list[SourceChunk]) -> list[SystemMessage | HumanMessage]:
    """Build LangChain chat messages for the generation request."""
    context_block = _format_context_block(chunks)
    return [
        SystemMessage(content=_build_system_prompt()),
        HumanMessage(content=_build_user_prompt(question, context_block)),
    ]


def generate(question: str, chunks: list[SourceChunk]) -> str:
    """
    Generate an answer from retrieved SOP chunks.

    Does not retrieve documents or access ChromaDB.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    if not chunks:
        return UNKNOWN_ANSWER

    messages = _build_messages(question, chunks)
    response = get_llm().invoke(messages)
    content = response.content

    if isinstance(content, str):
        return content.strip()

    return str(content).strip()
