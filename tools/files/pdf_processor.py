"""PDF attachment processing."""

from __future__ import annotations

import io

from pypdf import PdfReader

from tools.files.models import AttachmentMetadata, ProcessedFileContext, ProcessingStatus


def process_pdf(metadata: AttachmentMetadata, content: bytes) -> ProcessedFileContext:
    """Extract page-level text from a PDF attachment."""
    try:
        reader = PdfReader(io.BytesIO(content))
    except Exception:
        return _failed(metadata, "The PDF could not be read.")

    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append({"page": index, "text": text})

    extracted_text = "\n\n".join(page["text"] for page in pages if page["text"])
    if not extracted_text.strip():
        return ProcessedFileContext(
            attachment_id=metadata.attachment_id,
            filename=metadata.original_filename,
            file_type="pdf",
            processing_status=ProcessingStatus.READY,
            summary="PDF uploaded, but text extraction found little or no readable text. The document may be scanned or image-based.",
            pages=pages,
            image_analysis={"needs_ocr": True},
            source_references=[metadata.original_filename],
        )

    return ProcessedFileContext(
        attachment_id=metadata.attachment_id,
        filename=metadata.original_filename,
        file_type="pdf",
        processing_status=ProcessingStatus.READY,
        summary=f"PDF with {len(pages)} page(s) and extracted text content.",
        pages=pages,
        paragraphs=[paragraph.strip() for paragraph in extracted_text.split("\n\n") if paragraph.strip()],
        full_data={"text": extracted_text},
        source_references=[f"{metadata.original_filename}, page {page['page']}" for page in pages[:3]],
    )


def _failed(metadata: AttachmentMetadata, message: str) -> ProcessedFileContext:
    return ProcessedFileContext(
        attachment_id=metadata.attachment_id,
        filename=metadata.original_filename,
        file_type="pdf",
        processing_status=ProcessingStatus.FAILED,
        error_message=message,
        source_references=[metadata.original_filename],
    )
