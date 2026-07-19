"""Persistent attachment storage and metadata index."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import config
from tools.files.models import AttachmentMetadata, ProcessingStatus


class AttachmentStore:
    """Store uploaded files and metadata on disk."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or config.UPLOAD_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "attachments_index.json"
        self._index = self._load_index()

    def save_upload(
        self,
        *,
        original_filename: str,
        extension: str,
        content: bytes,
        content_type: str | None,
        conversation_id: str,
    ) -> AttachmentMetadata:
        """Persist a validated upload and return metadata."""
        attachment_id = str(uuid4())
        safe_name = _safe_storage_name(attachment_id, extension)
        storage_path = self.base_dir / safe_name
        storage_path.write_bytes(content)

        metadata = AttachmentMetadata(
            attachment_id=attachment_id,
            filename=safe_name,
            original_filename=original_filename,
            content_type=content_type,
            file_type=extension,
            file_size=len(content),
            storage_path=str(storage_path),
            conversation_id=conversation_id,
            processing_status=ProcessingStatus.UPLOADED,
        )
        self._index[attachment_id] = metadata.model_dump()
        self._save_index()
        return metadata

    def get(self, attachment_id: str) -> AttachmentMetadata | None:
        """Return attachment metadata by ID."""
        payload = self._index.get(attachment_id)
        if not payload:
            self.reload_index()
            payload = self._index.get(attachment_id)
        if not payload:
            return None
        return AttachmentMetadata.model_validate(payload)

    def reload_index(self) -> None:
        """Reload attachment metadata from disk."""
        self._index = self._load_index()

    def update(self, metadata: AttachmentMetadata) -> AttachmentMetadata:
        """Persist updated attachment metadata."""
        self._index[metadata.attachment_id] = metadata.model_dump()
        self._save_index()
        return metadata

    def list_for_conversation(self, conversation_id: str) -> list[AttachmentMetadata]:
        """Return attachments uploaded for a conversation."""
        return [
            AttachmentMetadata.model_validate(payload)
            for payload in self._index.values()
            if payload.get("conversation_id") == conversation_id
        ]

    def read_bytes(self, metadata: AttachmentMetadata) -> bytes:
        """Read stored file bytes."""
        path = Path(metadata.storage_path)
        if not path.exists():
            raise FileNotFoundError(f"Attachment file not found for {metadata.attachment_id}.")
        return path.read_bytes()

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {}
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save_index(self) -> None:
        self.index_path.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _safe_storage_name(attachment_id: str, extension: str) -> str:
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", attachment_id)
    return f"{safe_id}.{extension.lower()}"
