"""Parse and validate prompt frontmatter metadata."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

PromptStatus = Literal["active", "candidate", "experimental", "deprecated", "draft"]


class PromptMetadata(BaseModel):
    """Validated metadata for a versioned prompt asset."""

    name: str
    version: str = "1.0"
    owner: str = "unknown"
    description: str = ""
    temperature: float = 0.0
    max_tokens: int | None = 1200
    output_format: str = "text"
    required_variables: list[str] = Field(default_factory=list)
    optional_variables: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    status: PromptStatus = "active"
    includes: list[str] = Field(
        default_factory=list,
        description="Shared prompt names to prepend (e.g. shared.system_rules).",
    )
    role: str = "system"
    model: str | None = None

    @field_validator("version", mode="before")
    @classmethod
    def _version_str(cls, value: Any) -> str:
        if value is None:
            return "1.0"
        return str(value)

    @field_validator("temperature", mode="before")
    @classmethod
    def _clamp_temperature(cls, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(2.0, score))

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _parse_max_tokens(cls, value: Any) -> int | None:
        if value is None or value == "" or str(value).lower() in {"none", "null"}:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return 1200

    @field_validator("required_variables", "optional_variables", "tags", "includes", mode="before")
    @classmethod
    def _ensure_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


class PromptValidationError(ValueError):
    """Raised when prompt metadata or body is invalid."""


def parse_prompt_document(text: str, *, source: str = "<memory>") -> tuple[PromptMetadata, str]:
    """Split YAML frontmatter from the markdown body and validate metadata."""
    text = text.lstrip("\ufeff")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise PromptValidationError(
            f"Prompt {source} must begin with YAML frontmatter between --- delimiters."
        )
    raw_meta, body = match.group(1), match.group(2)
    if yaml is None:
        raise PromptValidationError(
            "PyYAML is required for the Prompt Registry. Install with: pip install pyyaml"
        )
    try:
        data = yaml.safe_load(raw_meta) or {}
    except Exception as exc:  # noqa: BLE001
        raise PromptValidationError(f"Invalid YAML frontmatter in {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise PromptValidationError(f"Frontmatter in {source} must be a YAML mapping.")
    try:
        metadata = PromptMetadata.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        raise PromptValidationError(f"Invalid prompt metadata in {source}: {exc}") from exc
    if not metadata.name.strip():
        raise PromptValidationError(f"Prompt {source} is missing required metadata field 'name'.")
    return metadata, body.strip() + "\n"


def validate_required_variables(metadata: PromptMetadata, variables: dict[str, Any]) -> list[str]:
    """Return names of required variables that are missing or empty."""
    missing: list[str] = []
    for name in metadata.required_variables:
        if name not in variables:
            missing.append(name)
            continue
        value = variables[name]
        if value is None:
            missing.append(name)
        elif isinstance(value, str) and not value.strip():
            missing.append(name)
    return missing
