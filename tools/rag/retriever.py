"""ChromaDB retrieval for GOFO SOP knowledge (no LLM generation)."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import chromadb
from chromadb.api.models.Collection import Collection
from langchain_openai import OpenAIEmbeddings

import config
from core.models import QueryRequest, SourceChunk


@lru_cache(maxsize=1)
def _get_embeddings() -> OpenAIEmbeddings:
    """Return a cached OpenAI embeddings client aligned with ingest settings."""
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.require_openai_api_key(),
    )


@lru_cache(maxsize=1)
def _get_chroma_client() -> chromadb.ClientAPI:
    """Return a cached persistent Chroma client."""
    config.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))


def get_collection(collection_name: Optional[str] = None) -> Collection:
    """Open an existing Chroma collection."""
    name = collection_name or config.COLLECTION_NAME
    client = _get_chroma_client()

    try:
        return client.get_collection(name=name)
    except Exception as exc:
        raise FileNotFoundError(
            f"Collection '{name}' not found in {config.CHROMA_PERSIST_DIR}. "
            "Run ingest.py after adding PDFs to docs/."
        ) from exc


def _distance_to_score(distance: float) -> float:
    """Convert Chroma distance to a higher-is-better relevance score."""
    return 1.0 / (1.0 + distance)


def retrieve(request: QueryRequest) -> list[SourceChunk]:
    """
    Retrieve the top-k most similar chunks for a question.

    Embeds the query with OpenAI embeddings and searches ChromaDB.
    Does not call a chat/completion model.
    """
    question = request.question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    collection = get_collection(request.collection)
    query_embedding = _get_embeddings().embed_query(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=request.top_k,
        where=request.filters,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents") or [[]]
    ids = results.get("ids") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    if not documents[0]:
        return []

    chunks: list[SourceChunk] = []
    for chunk_id, text, metadata, distance in zip(
        ids[0],
        documents[0],
        metadatas[0],
        distances[0],
    ):
        chunks.append(
            SourceChunk(
                id=chunk_id,
                text=text,
                score=_distance_to_score(distance),
                metadata=metadata or {},
            )
        )

    return chunks
