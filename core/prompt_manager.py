"""High-level Prompt Manager — the only API components should use for prompts."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

import config
from core.logger import get_logger
from core.prompt_registry import PromptRegistry, PromptSelection, get_prompt_registry
from core.prompt_renderer import PromptRenderer
from core.prompt_validator import PromptValidationError

logger = get_logger("prompt_manager")


class PromptManager:
    """Resolve, render, and invoke prompts via the Prompt Registry."""

    def __init__(self, registry: PromptRegistry | None = None) -> None:
        self.registry = registry or get_prompt_registry()
        self.renderer = self.registry.renderer or PromptRenderer()

    def get(
        self,
        key: str,
        *,
        version: str | None = None,
    ) -> PromptSelection:
        """Fetch a prompt selection (does not hardcode filesystem paths)."""
        return self.registry.get(key, version=version)

    def render(
        self,
        key: str,
        variables: dict[str, Any] | None = None,
        *,
        version: str | None = None,
    ) -> tuple[PromptSelection, str]:
        """Render a prompt; returns (selection, rendered_text)."""
        selection = self.get(key, version=version)
        rendered = self.renderer.render(
            selection.full_template,
            variables or {},
            metadata=selection.metadata,
            strict=True,
        )
        self._debug_render(selection, variables or {}, rendered)
        return selection, rendered

    def build_messages(
        self,
        key: str,
        variables: dict[str, Any] | None = None,
        *,
        version: str | None = None,
        user_content: str | None = None,
        user_prompt_key: str | None = None,
    ) -> tuple[PromptSelection, list[SystemMessage | HumanMessage]]:
        """Build LangChain messages from a system prompt (+ optional user prompt)."""
        selection, system_text = self.render(key, variables, version=version)
        messages: list[SystemMessage | HumanMessage] = [SystemMessage(content=system_text)]

        if user_prompt_key:
            _user_sel, user_text = self.render(user_prompt_key, variables, version=version)
            messages.append(HumanMessage(content=user_text))
        elif user_content is not None:
            messages.append(HumanMessage(content=user_content))
        elif "user_message" in (variables or {}):
            messages.append(HumanMessage(content=str(variables["user_message"])))

        return selection, messages

    def invoke(
        self,
        key: str,
        variables: dict[str, Any] | None = None,
        *,
        version: str | None = None,
        user_content: str | None = None,
        user_prompt_key: str | None = None,
        llm: Any | None = None,
    ) -> str:
        """Render prompt, apply metadata LLM config, invoke model, return text."""
        selection, messages = self.build_messages(
            key,
            variables,
            version=version,
            user_content=user_content,
            user_prompt_key=user_prompt_key,
        )
        client = llm or self.llm_for(selection)
        response = client.invoke(messages)
        content = response.content if hasattr(response, "content") else response
        return content.strip() if isinstance(content, str) else str(content).strip()

    def llm_for(self, selection: PromptSelection) -> Any:
        """Build an LLM client configured from prompt metadata (no hardcoded params)."""
        from tools.llm.client import get_llm

        meta = selection.metadata
        return get_llm(
            temperature=meta.temperature,
            max_tokens=meta.max_tokens,
            model=meta.model,
        )

    def set_active_version(self, key: str, version: str, *, persist: bool = False) -> None:
        self.registry.set_active_version(key, version, persist=persist)

    def list_prompts(self) -> list[dict[str, Any]]:
        return self.registry.list_prompts()

    def _debug_render(
        self,
        selection: PromptSelection,
        variables: dict[str, Any],
        rendered: str,
    ) -> None:
        if not (
            getattr(config, "DEBUG", False)
            or getattr(config, "PROMPT_DEBUG", False)
        ):
            return
        provided = sorted(variables.keys())
        required = selection.metadata.required_variables
        missing = [name for name in required if name not in variables]
        tokens = self.renderer.estimate_tokens(rendered)
        print("----------------------------------")
        print("PromptManager.render")
        print(f"  prompt={selection.key}@{selection.version_key}")
        print(f"  metadata={selection.metadata.model_dump()}")
        print(f"  variables={provided}")
        print(f"  missing_required={missing}")
        print(f"  rendered_chars={len(rendered)} estimated_tokens={tokens}")
        print(f"  experiment={selection.experiment}")
        print("----------------------------------")


_MANAGER: PromptManager | None = None


def get_prompt_manager() -> PromptManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = PromptManager()
    return _MANAGER


def reset_prompt_manager() -> None:
    global _MANAGER
    _MANAGER = None
