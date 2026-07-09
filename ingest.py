"""
Ingest GOFO SOP PDFs into ChromaDB for the Operations Intelligence Agent.

Reads every PDF in ./docs, splits text into chunks, embeds with OpenAI,
and stores vectors in a persistent ChromaDB collection.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "docs"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "gofo_sop"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_BATCH_SIZE = 100


def _require_api_key() -> str:
    """Load environment variables and ensure OpenAI credentials are present."""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError(
            "OPENAI_API_KEY is missing or unset. Add a valid key to .env before ingesting."
        )
    return api_key


def _discover_pdfs(docs_dir: Path) -> list[Path]:
    """Return all PDF files under docs_dir, sorted for deterministic processing."""
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    pdf_paths = sorted(
        path
        for path in docs_dir.rglob("*.pdf")
        if path.is_file() and not path.name.startswith(".")
    )
    return pdf_paths


def _load_pdfs(pdf_paths: list[Path]) -> tuple[list[Document], list[str]]:
    """
    Extract text from each PDF using LangChain's PyPDFLoader.

    Returns loaded LangChain documents and the list of PDF filenames loaded.
    """
    documents: list[Document] = []
    loaded_names: list[str] = []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        if not pages:
            print(f"Warning: no text extracted from {pdf_path.name}; skipping.")
            continue

        # Normalize metadata so every chunk can trace back to its source PDF.
        for page in pages:
            page.metadata["source"] = str(pdf_path.relative_to(DOCS_DIR))
            page.metadata["filename"] = pdf_path.name

        documents.extend(pages)
        loaded_names.append(pdf_path.name)

    return documents, loaded_names


def _split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks sized for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
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
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    print("\nPer-PDF ingestion analysis:")
    for pdf_path in pdf_paths:
        pdf_pages = [
            doc
            for doc in documents
            if doc.metadata.get("filename") == pdf_path.name
        ]
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

    # Cross-check totals against the final chunk list used for storage.
    print(
        f"\n  Combined chunks stored: {len(chunks)} "
        f"(from {sum(len(c.page_content) for c in chunks):,} total characters)"
    )


def _sanitize_metadata(metadata: dict) -> dict[str, str | int | float | bool]:
    """
    ChromaDB only accepts scalar metadata values.
    Convert missing or complex values to strings.
    """
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized


def _build_chunk_records(
    chunks: list[Document],
) -> tuple[list[str], list[str], list[dict[str, str | int | float | bool]]]:
    """Prepare ids, texts, and metadata payloads for ChromaDB insertion."""
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, str | int | float | bool]] = []

    for index, chunk in enumerate(chunks):
        source = str(chunk.metadata.get("source", "unknown"))
        chunk_id = f"{source}::chunk-{index:05d}"

        ids.append(chunk_id)
        texts.append(chunk.page_content)
        metadatas.append(_sanitize_metadata(chunk.metadata))

    return ids, texts, metadatas


def _embed_texts(texts: list[str], api_key: str) -> list[list[float]]:
    """Generate OpenAI embeddings in batches to stay within API limits."""
    embedding_model = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=api_key,
    )

    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        vectors.extend(embedding_model.embed_documents(batch))

    return vectors


def _reset_collection(client: chromadb.ClientAPI, collection_name: str) -> chromadb.Collection:
    """
    Delete an existing collection when present, then create a fresh one.
    Ensures each ingest run replaces stale vectors instead of merging them.
    """
    existing_names = {collection.name for collection in client.list_collections()}
    if collection_name in existing_names:
        client.delete_collection(name=collection_name)

    return client.create_collection(name=collection_name)


def ingest() -> None:
    """End-to-end ingestion pipeline: load PDFs -> split -> embed -> store."""
    api_key = _require_api_key()

    # -----------------------------------------------------------------------
    # 1. Discover and load PDFs from ./docs
    # -----------------------------------------------------------------------
    pdf_paths = _discover_pdfs(DOCS_DIR)
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {DOCS_DIR}. Add SOP PDFs before running ingest."
        )

    documents, loaded_pdf_names = _load_pdfs(pdf_paths)
    if not documents:
        raise RuntimeError("PDFs were found, but no text could be extracted from them.")

    # -----------------------------------------------------------------------
    # 2. Split documents into retrieval-friendly chunks
    # -----------------------------------------------------------------------
    chunks = _split_documents(documents)
    if not chunks:
        raise RuntimeError("Text splitting produced zero chunks.")

    ids, texts, metadatas = _build_chunk_records(chunks)

    # -----------------------------------------------------------------------
    # 3. Generate OpenAI embeddings for every chunk
    # -----------------------------------------------------------------------
    embeddings = _embed_texts(texts, api_key)

    # -----------------------------------------------------------------------
    # 4. Persist vectors in ChromaDB (replace collection if it already exists)
    # -----------------------------------------------------------------------
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    collection = _reset_collection(chroma_client, COLLECTION_NAME)

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # -----------------------------------------------------------------------
    # 5. Report ingestion summary and per-PDF diagnostics
    # -----------------------------------------------------------------------
    _analyze_pdf_ingestion(pdf_paths, documents, chunks)

    print(f"\nPDFs loaded: {len(loaded_pdf_names)}")
    for name in loaded_pdf_names:
        print(f"  - {name}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Collection name: {COLLECTION_NAME}")
    print("Success: GOFO SOP knowledge base ingested into ChromaDB.")


if __name__ == "__main__":
    try:
        ingest()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
