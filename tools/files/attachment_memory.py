"""Session-level attachment memory for conversational file analysis."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from tools.context.file_context_builder import build_file_context, resolve_file_reference
from tools.files.dataframe_store import StoredAttachment, context_to_stored_attachment
from tools.files.models import ProcessedFileContext


class AttachmentMemory:
    """Track parsed attachments as reusable conversation context."""

    def __init__(self) -> None:
        self.active_attachment_ids: list[str] = []
        self.processed_contexts: dict[str, ProcessedFileContext] = {}
        self.stored_attachments: dict[str, StoredAttachment] = {}
        self.active_filters: dict[str, Any] = {}
        self.last_analysis: dict[str, Any] | None = None

    def register_contexts(
        self,
        contexts: list[ProcessedFileContext],
        *,
        replace_active: bool = True,
    ) -> None:
        """Register newly processed attachments and keep parsed objects in memory."""
        if replace_active and contexts:
            # Uploading a new attachment replaces the active attachment set.
            self.active_attachment_ids = []
            self.active_filters = {}

        for context in contexts:
            self.processed_contexts[context.attachment_id] = context
            self.stored_attachments[context.attachment_id] = context_to_stored_attachment(context)
            if context.attachment_id not in self.active_attachment_ids:
                self.active_attachment_ids.append(context.attachment_id)

    def deactivate(self) -> None:
        """Stop auto-binding prior uploads to unrelated questions."""
        self.active_attachment_ids = []

    def activate(self, attachment_ids: list[str] | None = None) -> None:
        """Re-enable attachment binding for routed attachment turns.

        - Explicit IDs replace the active set (do not silently re-append old files).
        - With no IDs, keep the current focus if still valid; otherwise bind only
          the newest processed upload so follow-ups do not fall back to file #1.
        """
        if attachment_ids:
            self.active_attachment_ids = [
                attachment_id
                for attachment_id in attachment_ids
                if attachment_id in self.processed_contexts
            ]
            return

        self.active_attachment_ids = [
            attachment_id
            for attachment_id in self.active_attachment_ids
            if attachment_id in self.processed_contexts
        ]
        if self.active_attachment_ids:
            return

        if self.processed_contexts:
            newest_id = list(self.processed_contexts.keys())[-1]
            self.active_attachment_ids = [newest_id]

    def get_stored(self, attachment_id: str) -> StoredAttachment | None:
        return self.stored_attachments.get(attachment_id)

    def get_active_stored(self) -> list[StoredAttachment]:
        return [
            self.stored_attachments[attachment_id]
            for attachment_id in self.active_attachment_ids
            if attachment_id in self.stored_attachments
        ]

    def resolve_for_question(
        self,
        question: str,
        *,
        attachment_ids: list[str] | None = None,
        use_attachments: bool = True,
    ) -> tuple[list[ProcessedFileContext], dict[str, Any]]:
        """Resolve active attachments and build file context for a question."""
        if not use_attachments and not attachment_ids:
            return [], {}
        selected_ids = attachment_ids or (self.active_attachment_ids if use_attachments else [])
        contexts = [
            self.processed_contexts[attachment_id]
            for attachment_id in selected_ids
            if attachment_id in self.processed_contexts
        ]
        if not contexts and self.processed_contexts:
            contexts = resolve_file_reference(
                question, list(self.processed_contexts.values())
            )
        elif len(contexts) > 1:
            # Narrow multi-file active sets using the question (newest / "new file").
            contexts = resolve_file_reference(question, contexts)
        file_context = build_file_context(
            question=question,
            contexts=contexts,
            active_attachment_ids=[context.attachment_id for context in contexts],
        )
        return contexts, file_context

    def update_analysis(self, analysis: dict[str, Any]) -> None:
        self.last_analysis = deepcopy(analysis)
        if "active_filter" in analysis:
            self.active_filters = dict(analysis.get("active_filter") or {})

    def snapshot(self) -> dict[str, Any]:
        return {
            "active_attachment_ids": list(self.active_attachment_ids),
            "active_filters": dict(self.active_filters),
            "attachments": [
                {
                    "attachment_id": context.attachment_id,
                    "filename": context.filename,
                    "file_type": context.file_type,
                    "processing_status": context.processing_status,
                    "summary": context.summary,
                    "stored": self.stored_attachments[context.attachment_id].snapshot()
                    if context.attachment_id in self.stored_attachments
                    else None,
                }
                for context in self.processed_contexts.values()
            ],
            "last_analysis": self.last_analysis,
        }
