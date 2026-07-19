"""Dispatch uploaded files to the correct processor."""

from __future__ import annotations

from tools.files.csv_processor import process_csv
from tools.files.detector import detect_file_type
from tools.files.document_processor import process_docx
from tools.files.excel_processor import process_excel
from tools.files.image_processor import process_image
from tools.files.models import AttachmentMetadata, ProcessedFileContext, ProcessingStatus
from tools.files.pdf_processor import process_pdf
from tools.files.text_processor import process_text

_PROCESSORS = {
    "csv": process_csv,
    "excel": process_excel,
    "pdf": process_pdf,
    "docx": process_docx,
    "text": process_text,
}


def process_attachment(
    metadata: AttachmentMetadata,
    content: bytes,
    *,
    question: str | None = None,
) -> ProcessedFileContext:
    """Process an attachment into structured file context."""
    file_type = detect_file_type(metadata.original_filename, metadata.file_type)
    if file_type == "image":
        return process_image(metadata, content, question=question)
    processor = _PROCESSORS.get(file_type)
    if processor is None:
        return ProcessedFileContext(
            attachment_id=metadata.attachment_id,
            filename=metadata.original_filename,
            file_type=file_type,
            processing_status=ProcessingStatus.FAILED,
            error_message="This file type is not supported.",
            source_references=[metadata.original_filename],
        )
    return processor(metadata, content)
