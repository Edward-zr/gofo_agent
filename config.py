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
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.40"))

HYBRID_RETRIEVAL_ENABLED = os.getenv("HYBRID_RETRIEVAL_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
HYBRID_DENSE_TOP_K = int(os.getenv("HYBRID_DENSE_TOP_K", "20"))
HYBRID_BM25_TOP_K = int(os.getenv("HYBRID_BM25_TOP_K", "20"))
HYBRID_RRF_K = int(os.getenv("HYBRID_RRF_K", "60"))
HYBRID_RERANK_TOP_N = int(os.getenv("HYBRID_RERANK_TOP_N", "20"))
HYBRID_RERANK_ENABLED = os.getenv("HYBRID_RERANK_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
HYBRID_RERANKER_MODEL = os.getenv("HYBRID_RERANKER_MODEL", "BAAI/bge-reranker-base")
HYBRID_QUERY_EXPANSION = os.getenv("HYBRID_QUERY_EXPANSION", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
HYBRID_EXPANSION_USE_LLM = os.getenv("HYBRID_EXPANSION_USE_LLM", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
HYBRID_EXPANSION_MIN_WORDS = int(os.getenv("HYBRID_EXPANSION_MIN_WORDS", "7"))
HYBRID_EXPANSION_MAX_QUERIES = int(os.getenv("HYBRID_EXPANSION_MAX_QUERIES", "4"))
HYBRID_RETRIEVAL_DEBUG = os.getenv("HYBRID_RETRIEVAL_DEBUG", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

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

# ---------------------------------------------------------------------------
# File uploads
# ---------------------------------------------------------------------------
_upload_dir = os.getenv("UPLOAD_DIR", "data/uploads")
_upload_path = Path(_upload_dir)
UPLOAD_DIR = _upload_path if _upload_path.is_absolute() else PROJECT_ROOT / _upload_path

MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

ALLOWED_UPLOAD_EXTENSIONS = frozenset(
    ext.strip().lower()
    for ext in os.getenv(
        "ALLOWED_UPLOAD_TYPES",
        "pdf,csv,xlsx,xls,txt,md,docx,png,jpg,jpeg,webp",
    ).split(",")
    if ext.strip()
)

BLOCKED_UPLOAD_EXTENSIONS = frozenset(
    {
        "exe",
        "dmg",
        "pkg",
        "sh",
        "bat",
        "dll",
        "bin",
        "app",
        "msi",
        "deb",
        "rpm",
        "js",
        "jar",
        "zip",
        "rar",
        "7z",
        "tar",
        "gz",
    }
)


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
