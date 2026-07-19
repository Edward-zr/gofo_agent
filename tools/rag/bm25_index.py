"""BM25 sparse retrieval over the ingested SOP corpus."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from rank_bm25 import BM25Okapi

from tools.rag.corpus import load_bm25_corpus, tokenize


@dataclass(frozen=True)
class BM25Hit:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    rank: int


class BM25Retriever:
    """In-memory BM25 index built from the persisted corpus."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self._ids = [record["id"] for record in records]
        self._texts = [record["text"] for record in records]
        self._metadatas = [record.get("metadata") or {} for record in records]
        tokenized = [tokenize(text) for text in self._texts]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    @classmethod
    def from_disk(cls) -> BM25Retriever:
        return cls(load_bm25_corpus())

    def available(self) -> bool:
        return bool(self.records) and self._bm25 is not None

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[BM25Hit]:
        if not self.available():
            return []

        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )

        hits: list[BM25Hit] = []
        for index, score in ranked:
            if score <= 0:
                continue
            metadata = self._metadatas[index]
            if filters and not _matches_filters(metadata, filters):
                continue
            hits.append(
                BM25Hit(
                    chunk_id=self._ids[index],
                    text=self._texts[index],
                    metadata=metadata,
                    score=float(score),
                    rank=len(hits) + 1,
                )
            )
            if len(hits) >= top_k:
                break
        return hits


@lru_cache(maxsize=1)
def get_bm25_retriever() -> BM25Retriever:
    return BM25Retriever.from_disk()


def clear_bm25_cache() -> None:
    get_bm25_retriever.cache_clear()


def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        actual = metadata.get(key)
        if actual is None:
            return False
        if str(actual).lower() != str(expected).lower():
            return False
    return True
