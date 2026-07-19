"""
Ingest GOFO SOP PDFs into ChromaDB and the BM25 corpus for hybrid retrieval.

Reads every PDF in ./docs, splits text into chunks, embeds with OpenAI,
stores vectors in a persistent ChromaDB collection, and builds a BM25 index
from the same chunk records.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from tools.rag.corpus import (
    build_chunk_id,
    enrich_metadata,
    sanitize_metadata,
    save_bm25_corpus,
)


def _require_api_key() -> str:
    """Load environment variables and ensure OpenAI credentials are present."""
    load_dotenv()
    return config.require_openai_api_key()


def _discover_pdfs(docs_dir: Path) -> list[Path]:
    """Return all PDF files under docs_dir, sorted for deterministic processing."""
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    return sorted(
        path
        for path in docs_dir.rglob("*.pdf")
        if path.is_file() and not path.name.startswith(".")
    )


def _load_pdfs(pdf_paths: list[Path]) -> tuple[list[Document], list[str]]:
    """Extract text from each PDF using LangChain's PyPDFLoader."""
    documents: list[Document] = []
    loaded_names: list[str] = []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        if not pages:
            print(f"Warning: no text extracted from {pdf_path.name}; skipping.")
            continue

        for page in pages:
            page.metadata = enrich_metadata(
                page.metadata,
                source_path=pdf_path,
                docs_dir=config.DOCS_DIR,
            )

        documents.extend(pages)
        loaded_names.append(pdf_path.name)

    return documents, loaded_names


def _split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks sized for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(documents)


def _analyze_pdf_ingestion(
    pdf_paths: list[Path],
    documents: list[Document],
    chunks: list[Document],
) -> None:
    """Print per-PDF extraction and chunking stats to aid debugging."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
    )

    print("\nPer-PDF ingestion analysis:")
    for pdf_path in pdf_paths:
        pdf_pages = [doc for doc in documents if doc.metadata.get("filename") == pdf_path.name]
        pdf_chunks = splitter.split_documents(pdf_pages)
        total_chars = sum(len(doc.page_content) for doc in pdf_pages)
        chunk_lengths = [len(chunk.page_content) for chunk in pdf_chunks]

        print(f"\n  {pdf_path.name}")
        print(f"    Pages loaded: {len(pdf_pages)}")
        print(f"    Total characters extracted: {total_chars:,}")
        print(f"    Chunks generated: {len(pdf_chunks)}")
        if chunk_lengths:
            print(
                "    Chunk sizes (chars): "
                f"min={min(chunk_lengths)}, "
                f"max={max(chunk_lengths)}, "
                f"avg={sum(chunk_lengths) / len(chunk_lengths):.1f}"
            )

    print(
        f"\n  Combined chunks stored: {len(chunks)} "
        f"(from {sum(len(c.page_content) for c in chunks):,} total characters)"
    )


def _build_chunk_records(
    chunks: list[Document],
) -> tuple[list[str], list[str], list[dict[str, str | int | float | bool]], list[dict]]:
    """Prepare ids, texts, metadata payloads, and BM25 records."""
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, str | int | float | bool]] = []
    bm25_records: list[dict] = []

    for index, chunk in enumerate(chunks):
        source = str(chunk.metadata.get("source", "unknown"))
        chunk_id = build_chunk_id(source, index)

        metadata = sanitize_metadata(chunk.metadata)
        ids.append(chunk_id)
        texts.append(chunk.page_content)
        metadatas.append(metadata)
        bm25_records.append({"id": chunk_id, "text": chunk.page_content, "metadata": metadata})

    return ids, texts, metadatas, bm25_records


def _embed_texts(texts: list[str], api_key: str) -> list[list[float]]:
    """Generate OpenAI embeddings in batches to stay within API limits."""
    embedding_model = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=api_key,
    )

    vectors: list[list[float]] = []
    for start in range(0, len(texts), config.EMBED_BATCH_SIZE):
        batch = texts[start : start + config.EMBED_BATCH_SIZE]
        vectors.extend(embedding_model.embed_documents(batch))

    return vectors


def _reset_collection(client: chromadb.ClientAPI, collection_name: str) -> chromadb.Collection:
    """Delete an existing collection when present, then create a fresh one."""
    existing_names = {collection.name for collection in client.list_collections()}
    if collection_name in existing_names:
        client.delete_collection(name=collection_name)
    return client.create_collection(name=collection_name)


def ingest() -> None:
    """End-to-end ingestion pipeline: load PDFs -> split -> embed -> store."""
    api_key = _require_api_key()

    pdf_paths = _discover_pdfs(config.DOCS_DIR)
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {config.DOCS_DIR}. Add SOP PDFs before running ingest."
        )

    documents, loaded_pdf_names = _load_pdfs(pdf_paths)
    if not documents:
        raise RuntimeError("PDFs were found, but no text could be extracted from them.")

    chunks = _split_documents(documents)
    if not chunks:
        raise RuntimeError("Text splitting produced zero chunks.")

    ids, texts, metadatas, bm25_records = _build_chunk_records(chunks)
    embeddings = _embed_texts(texts, api_key)

    config.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
    collection = _reset_collection(chroma_client, config.COLLECTION_NAME)

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    bm25_path = save_bm25_corpus(
        records=bm25_records,
        collection_name=config.COLLECTION_NAME,
    )

    _analyze_pdf_ingestion(pdf_paths, documents, chunks)

    print(f"\nPDFs loaded: {len(loaded_pdf_names)}")
    for name in loaded_pdf_names:
        print(f"  - {name}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Collection name: {config.COLLECTION_NAME}")
    print(f"BM25 corpus: {bm25_path}")
    print("Success: GOFO SOP knowledge base ingested into ChromaDB and BM25.")


if __name__ == "__main__":
    try:
        ingest()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
