"""Central Prompt Registry — single source of truth for versioned prompt assets."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

import config
from core.logger import get_logger
from core.prompt_renderer import PromptRenderer
from core.prompt_validator import PromptMetadata, PromptValidationError, parse_prompt_document

logger = get_logger("prompt_registry")

VersionRole = Literal["active", "candidate", "experimental"]


@dataclass
class PromptVersion:
    """One versioned prompt file."""

    family: str
    name: str
    version_key: str
    file_path: Path
    metadata: PromptMetadata
    body: str
    mtime: float
    role: VersionRole = "active"


@dataclass
class PromptSelection:
    """Resolved prompt ready for rendering / LLM invocation."""

    key: str
    family: str
    name: str
    version_key: str
    metadata: PromptMetadata
    body: str
    file_path: Path
    experiment: str | None = None
    includes_text: str = ""

    @property
    def full_template(self) -> str:
        if self.includes_text:
            return f"{self.includes_text.rstrip()}\n\n{self.body}"
        return self.body


@dataclass
class RegistryState:
    versions: dict[str, PromptVersion] = field(default_factory=dict)
    # family.name -> active version_key
    active_map: dict[str, str] = field(default_factory=dict)
    # family.name -> {candidate: [...], experimental: [...]}
    roles: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    loaded_at: float = 0.0


class PromptRegistry:
    """Discover, version, cache, and select prompts under ``prompts/``."""

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        hot_reload: bool | None = None,
        renderer: PromptRenderer | None = None,
    ) -> None:
        self.root = Path(root or getattr(config, "PROMPT_REGISTRY_DIR", None) or Path("prompts"))
        if not self.root.is_absolute():
            self.root = Path(config.PROJECT_ROOT) / self.root
        self.hot_reload = (
            bool(getattr(config, "PROMPT_HOT_RELOAD", False))
            if hot_reload is None
            else hot_reload
        )
        self.renderer = renderer or PromptRenderer()
        self._lock = threading.RLock()
        self._state = RegistryState()
        self._overrides: dict[str, str] = {}  # key -> version_key
        self._experiment: str | None = None
        self.reload(force=True)

    # ------------------------------------------------------------------
    # Discovery / loading
    # ------------------------------------------------------------------

    def reload(self, *, force: bool = False) -> None:
        with self._lock:
            if not force and not self.hot_reload and self._state.loaded_at:
                return
            versions: dict[str, PromptVersion] = {}
            active_map: dict[str, str] = {}
            roles: dict[str, dict[str, list[str]]] = {}

            if not self.root.exists():
                logger.warning("Prompt registry root missing: %s", self.root)
                self._state = RegistryState(loaded_at=time.time())
                return

            for family_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
                family = family_dir.name
                registry_path = family_dir / "registry.yaml"
                family_cfg = self._load_yaml(registry_path) if registry_path.exists() else {}
                prompts_cfg = family_cfg.get("prompts") or {}

                # Shared family has loose .md files without registry versions.
                if family == "shared":
                    for md in sorted(family_dir.glob("*.md")):
                        version = self._load_markdown(family, md.stem, "v1", md, role="active")
                        key = f"{family}.{version.metadata.name}"
                        versions[f"{key}@v1"] = version
                        active_map[key] = "v1"
                        roles[key] = {"candidate": [], "experimental": []}
                    continue

                for prompt_name, prompt_cfg in prompts_cfg.items():
                    prompt_cfg = prompt_cfg or {}
                    active = str(prompt_cfg.get("active") or "v1")
                    candidates = [str(v) for v in (prompt_cfg.get("candidates") or [])]
                    experimental = [str(v) for v in (prompt_cfg.get("experimental") or [])]
                    version_files = prompt_cfg.get("versions") or {}
                    key = f"{family}.{prompt_name}"
                    active_map[key] = active
                    roles[key] = {"candidate": candidates, "experimental": experimental}

                    if version_files:
                        for version_key, meta in version_files.items():
                            filename = (
                                meta.get("file")
                                if isinstance(meta, dict)
                                else str(meta)
                            )
                            path = family_dir / str(filename)
                            role: VersionRole = "active"
                            if str(version_key) == active:
                                role = "active"
                            elif str(version_key) in candidates:
                                role = "candidate"
                            elif str(version_key) in experimental:
                                role = "experimental"
                            version = self._load_markdown(
                                family, prompt_name, str(version_key), path, role=role
                            )
                            versions[f"{key}@{version_key}"] = version
                    else:
                        # Auto-discover prompt_name_v*.md
                        for path in sorted(family_dir.glob(f"{prompt_name}_v*.md")):
                            version_key = path.stem.split("_")[-1]
                            role = (
                                "active"
                                if version_key == active
                                else (
                                    "candidate"
                                    if version_key in candidates
                                    else (
                                        "experimental"
                                        if version_key in experimental
                                        else "active"
                                    )
                                )
                            )
                            version = self._load_markdown(
                                family, prompt_name, version_key, path, role=role
                            )
                            versions[f"{key}@{version_key}"] = version

            self._state = RegistryState(
                versions=versions,
                active_map=active_map,
                roles=roles,
                loaded_at=time.time(),
            )
            logger.info(
                "Prompt registry loaded: %s versions from %s",
                len(versions),
                self.root,
            )

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        if yaml is None:
            raise PromptValidationError("PyYAML is required for Prompt Registry.")
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data if isinstance(data, dict) else {}

    def _load_markdown(
        self,
        family: str,
        name: str,
        version_key: str,
        path: Path,
        *,
        role: VersionRole,
    ) -> PromptVersion:
        if not path.exists():
            raise PromptValidationError(f"Prompt file not found: {path}")
        text = path.read_text(encoding="utf-8")
        metadata, body = parse_prompt_document(text, source=str(path))
        # Prefer registry name if frontmatter differs
        if metadata.name != name and family != "shared":
            metadata = metadata.model_copy(update={"name": name})
        return PromptVersion(
            family=family,
            name=name,
            version_key=version_key,
            file_path=path,
            metadata=metadata,
            body=body,
            mtime=path.stat().st_mtime,
            role=role,
        )

    def _maybe_reload(self) -> None:
        if self.hot_reload:
            self.reload(force=True)

    # ------------------------------------------------------------------
    # Selection / overrides / experiments
    # ------------------------------------------------------------------

    def set_experiment(self, experiment_id: str | None) -> None:
        self._experiment = experiment_id

    def clear_overrides(self) -> None:
        self._overrides.clear()

    def set_active_version(self, key: str, version_key: str, *, persist: bool = False) -> None:
        """Switch active version in memory (and optionally rewrite registry.yaml)."""
        key = self._normalize_key(key)
        with self._lock:
            self._overrides[key] = version_key
            self._state.active_map[key] = version_key
            if persist:
                self._persist_active(key, version_key)

    def _persist_active(self, key: str, version_key: str) -> None:
        family, name = key.split(".", 1)
        registry_path = self.root / family / "registry.yaml"
        data = self._load_yaml(registry_path) if registry_path.exists() else {"family": family, "prompts": {}}
        prompts = data.setdefault("prompts", {})
        entry = prompts.setdefault(name, {})
        entry["active"] = version_key
        if yaml is None:
            raise PromptValidationError("PyYAML is required to persist active versions.")
        with registry_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)

    def _normalize_key(self, key: str) -> str:
        key = key.strip()
        if "." not in key:
            # Allow short names when unique
            matches = [k for k in self._state.active_map if k.endswith(f".{key}") or k == key]
            if len(matches) == 1:
                return matches[0]
            raise KeyError(f"Ambiguous or unknown prompt key: {key}")
        return key

    def resolve_version_key(self, key: str, *, version: str | None = None) -> str:
        key = self._normalize_key(key)
        if version:
            return version
        if key in self._overrides:
            return self._overrides[key]
        # Experiment hook: PROMPT_EXPERIMENT=planner.planner_prompt:v2
        experiment = self._experiment or getattr(config, "PROMPT_EXPERIMENT", "") or ""
        if experiment:
            for part in str(experiment).split(","):
                part = part.strip()
                if ":" in part:
                    exp_key, exp_ver = part.split(":", 1)
                    if self._normalize_key(exp_key) == key:
                        return exp_ver.strip()
        return self._state.active_map.get(key, "v1")

    def get(
        self,
        key: str,
        *,
        version: str | None = None,
        include_shared: bool = True,
    ) -> PromptSelection:
        """Load the selected prompt version (with optional shared includes)."""
        self._maybe_reload()
        key = self._normalize_key(key)
        version_key = self.resolve_version_key(key, version=version)
        version_id = f"{key}@{version_key}"
        with self._lock:
            prompt = self._state.versions.get(version_id)
            if prompt is None:
                # Fallback to any available version
                available = [
                    vid for vid in self._state.versions if vid.startswith(f"{key}@")
                ]
                if not available:
                    raise KeyError(f"Unknown prompt: {key}")
                version_id = sorted(available)[-1]
                prompt = self._state.versions[version_id]
                version_key = prompt.version_key

        includes_text = ""
        if include_shared:
            includes_text = self._render_includes(prompt.metadata.includes)

        selection = PromptSelection(
            key=key,
            family=prompt.family,
            name=prompt.name,
            version_key=version_key,
            metadata=prompt.metadata,
            body=prompt.body,
            file_path=prompt.file_path,
            experiment=self._experiment,
            includes_text=includes_text,
        )
        self._debug_selection(selection)
        return selection

    def _render_includes(self, includes: list[str]) -> str:
        blocks: list[str] = []
        for include_key in includes:
            try:
                # Shared includes do not recursively include others by default
                selection = self.get(include_key, include_shared=False)
                blocks.append(selection.body.strip())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to include prompt %s: %s", include_key, exc)
        return "\n\n".join(blocks)

    def list_prompts(self) -> list[dict[str, Any]]:
        self._maybe_reload()
        rows: list[dict[str, Any]] = []
        for key, active in sorted(self._state.active_map.items()):
            role_info = self._state.roles.get(key) or {}
            versions = sorted(
                {
                    vid.split("@", 1)[1]
                    for vid in self._state.versions
                    if vid.startswith(f"{key}@")
                }
            )
            rows.append(
                {
                    "key": key,
                    "active": self.resolve_version_key(key),
                    "configured_active": active,
                    "versions": versions,
                    "candidates": role_info.get("candidate") or [],
                    "experimental": role_info.get("experimental") or [],
                }
            )
        return rows

    def list_versions(self, key: str) -> list[str]:
        key = self._normalize_key(key)
        return sorted(
            {
                vid.split("@", 1)[1]
                for vid in self._state.versions
                if vid.startswith(f"{key}@")
            }
        )

    def _debug_selection(self, selection: PromptSelection) -> None:
        if not (
            getattr(config, "DEBUG", False)
            or getattr(config, "PROMPT_DEBUG", False)
        ):
            return
        print("----------------------------------")
        print("PromptRegistry")
        print(f"  name={selection.key}")
        print(f"  version={selection.version_key}")
        print(f"  owner={selection.metadata.owner}")
        print(f"  temperature={selection.metadata.temperature}")
        print(f"  max_tokens={selection.metadata.max_tokens}")
        print(f"  output_format={selection.metadata.output_format}")
        print(f"  required={selection.metadata.required_variables}")
        print(f"  status={selection.metadata.status}")
        print(f"  experiment={selection.experiment}")
        print(f"  file={selection.file_path}")
        print("----------------------------------")


_REGISTRY: PromptRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_prompt_registry() -> PromptRegistry:
    """Process-wide Prompt Registry singleton."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            _REGISTRY = PromptRegistry()
        return _REGISTRY


def reset_prompt_registry() -> None:
    """Clear singleton (tests / hot swap)."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        _REGISTRY = None
