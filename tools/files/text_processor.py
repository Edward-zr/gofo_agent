"""Plain text and markdown attachment processing."""

from __future__ import annotations

from tools.files.models import AttachmentMetadata, ProcessedFileContext, ProcessingStatus


def process_text(metadata: AttachmentMetadata, content: bytes) -> ProcessedFileContext:
    """Extract text content from TXT or MD files."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return _failed(metadata, "The text file could not be decoded as UTF-8.")

    if not text.strip():
        return _failed(metadata, "The text file is empty.")

    sections = [section.strip() for section in text.split("\n\n") if section.strip()]
    return ProcessedFileContext(
        attachment_id=metadata.attachment_id,
        filename=metadata.original_filename,
        file_type="text",
        processing_status=ProcessingStatus.READY,
        summary=f"Text document with {len(sections)} section(s).",
        paragraphs=sections,
        full_data={"text": text},
        source_references=[metadata.original_filename],
    )


def _failed(metadata: AttachmentMetadata, message: str) -> ProcessedFileContext:
    return ProcessedFileContext(
        attachment_id=metadata.attachment_id,
        filename=metadata.original_filename,
        file_type="text",
        processing_status=ProcessingStatus.FAILED,
        error_message=message,
        source_references=[metadata.original_filename],
    )
