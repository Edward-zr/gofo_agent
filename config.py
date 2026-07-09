"""Shared configuration for the GOFO Operations Intelligence Agent."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from core.errors import ConfigurationError

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "docs"
_chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
_chroma_path = Path(_chroma_dir)
CHROMA_PERSIST_DIR = (
    _chroma_path if _chroma_path.is_absolute() else PROJECT_ROOT / _chroma_path
)
CHROMA_PATH = CHROMA_PERSIST_DIR

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "gofo_sop")

# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "100"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "5"))
SIMILARITY_THRESHOLD = 0.40

# ---------------------------------------------------------------------------
# SQL analytics
# ---------------------------------------------------------------------------
_sqlite_db = os.getenv("SQLITE_DATABASE", "data/gofo_demo.db")
_sqlite_path = Path(_sqlite_db)
SQLITE_DATABASE = (
    _sqlite_path if _sqlite_path.is_absolute() else PROJECT_ROOT / _sqlite_path
)

_memory_db = os.getenv("MEMORY_DATABASE", "data/memory.db")
_memory_path = Path(_memory_db)
MEMORY_DATABASE = (
    _memory_path if _memory_path.is_absolute() else PROJECT_ROOT / _memory_path
)

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
_log_file = os.getenv("LOG_FILE", "logs/agent.log")
_log_path = Path(_log_file)
LOG_FILE = _log_path if _log_path.is_absolute() else PROJECT_ROOT / _log_path


def require_openai_api_key() -> str:
    """Return a validated OpenAI API key or raise ValueError."""
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        raise ConfigurationError(
            "OPENAI_API_KEY is missing or unset. Add a valid key to .env."
        )
    return OPENAI_API_KEY


def require_existing_sqlite_database() -> Path:
    """Return the configured SQLite analytics path or raise a clear error."""
    if not SQLITE_DATABASE.exists():
        raise ConfigurationError(
            f"SQLITE_DATABASE does not exist at {SQLITE_DATABASE}."
        )
    return SQLITE_DATABASE
