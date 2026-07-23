"""Render prompt templates with ``{{variable}}`` placeholders."""

from __future__ import annotations

import re
from typing import Any

from core.prompt_validator import PromptMetadata, PromptValidationError, validate_required_variables

_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class PromptRenderer:
    """Substitute template variables and enforce required variable contracts."""

    def render(
        self,
        template: str,
        variables: dict[str, Any] | None = None,
        *,
        metadata: PromptMetadata | None = None,
        strict: bool = True,
    ) -> str:
        variables = dict(variables or {})
        if metadata is not None:
            missing = validate_required_variables(metadata, variables)
            if missing:
                raise PromptValidationError(
                    f"Missing required prompt variables for '{metadata.name}': {', '.join(missing)}"
                )

        def _replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in variables:
                if strict and metadata and key in (metadata.required_variables or []):
                    raise PromptValidationError(f"Missing required variable: {key}")
                if strict and metadata is None:
                    # When no metadata, leave unresolved required-looking tokens as error
                    raise PromptValidationError(f"Unresolved template variable: {key}")
                return ""
            value = variables[key]
            if value is None:
                return ""
            if isinstance(value, (list, tuple, set)):
                return "\n".join(str(item) for item in value)
            if isinstance(value, dict):
                import json

                return json.dumps(value, indent=2, default=str)
            return str(value)

        rendered = _VAR_RE.sub(_replace, template)
        return rendered

    def find_placeholders(self, template: str) -> list[str]:
        return sorted(set(_VAR_RE.findall(template)))

    def estimate_tokens(self, text: str) -> int:
        return max(1, int(len(text) / 4)) if text else 0
