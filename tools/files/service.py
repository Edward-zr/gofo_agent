"""Upload, process, and retrieve attachment intelligence."""

from __future__ import annotations

from typing import Any

from core.errors import FileError
from tools.files.models import AttachmentMetadata, ProcessedFileContext, ProcessingStatus
from tools.files.processor import process_attachment
from tools.files.storage import AttachmentStore
from tools.files.validator import validate_upload


class AttachmentService:
    """Manage attachment uploads and processing."""

    def __init__(self, store: AttachmentStore | None = None) -> None:
        self.store = store or AttachmentStore()
        self._processed_cache: dict[str, ProcessedFileContext] = {}

    def upload(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str | None,
        conversation_id: str,
    ) -> AttachmentMetadata:
        """Validate and store an uploaded file."""
        extension = validate_upload(
            filename=filename,
            content_type=content_type,
            file_size=len(content),
            content=content,
        )
        metadata = self.store.save_upload(
            original_filename=filename,
            extension=extension,
            content=content,
            content_type=content_type,
            conversation_id=conversation_id,
        )
        return metadata

    def process(
        self,
        attachment_id: str,
        *,
        question: str | None = None,
    ) -> ProcessedFileContext:
        """Process an uploaded attachment into structured context."""
        metadata = self._require_metadata(attachment_id)
        if attachment_id in self._processed_cache and not question:
            return self._processed_cache[attachment_id]

        metadata.processing_status = ProcessingStatus.PROCESSING
        self.store.update(metadata)
        content = self.store.read_bytes(metadata)
        processed = process_attachment(metadata, content, question=question)
        metadata.processing_status = processed.processing_status
        metadata.error_message = processed.error_message
        metadata.metadata = {
            "summary": processed.summary,
            "file_type": processed.file_type,
            "file_schema": processed.file_schema,
            "statistics": processed.statistics,
        }
        self.store.update(metadata)
        if processed.processing_status == ProcessingStatus.READY:
            self._processed_cache[attachment_id] = processed
        return processed

    def get_metadata(self, attachment_id: str) -> AttachmentMetadata | None:
        return self.store.get(attachment_id)

    def get_processed(self, attachment_id: str, *, question: str | None = None) -> ProcessedFileContext:
        if attachment_id in self._processed_cache and not question:
            return self._processed_cache[attachment_id]
        return self.process(attachment_id, question=question)

    def get_many_processed(
        self,
        attachment_ids: list[str],
        *,
        question: str | None = None,
    ) -> list[ProcessedFileContext]:
        contexts = []
        for attachment_id in attachment_ids:
            contexts.append(self.get_processed(attachment_id, question=question))
        return contexts

    def list_for_conversation(self, conversation_id: str) -> list[AttachmentMetadata]:
        return self.store.list_for_conversation(conversation_id)

    def _require_metadata(self, attachment_id: str) -> AttachmentMetadata:
        self.store.reload_index()
        metadata = self.store.get(attachment_id)
        if metadata is None:
            raise FileError("The requested attachment could not be found.")
        return metadata
