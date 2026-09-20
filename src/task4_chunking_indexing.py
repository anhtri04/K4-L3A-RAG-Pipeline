"""Task 4 - load, chunk, embed, and index the standardized corpus.

The module deliberately keeps the embedding entry point in one place.  Task 5
imports :func:`embed_texts`, which prevents query and document embeddings from
silently using different providers or dimensions.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable

from dotenv import load_dotenv

from .contracts import validate_document


load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = ROOT_DIR / "data" / "standardized"


def _configured_path(name: str, default: Path) -> Path:
    """Resolve an optional path setting relative to the repository root."""
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    configured = Path(raw_value).expanduser()
    return configured if configured.is_absolute() else ROOT_DIR / configured


CHROMA_DIR = _configured_path("CHROMA_DIR", ROOT_DIR / "chroma_db")

# Character based chunking is suitable for the mixed Markdown/PDF corpus and
# keeps the size predictable for every embedding provider.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").strip() or "BAAI/bge-m3"
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
COLLECTION_NAME = (
    os.getenv("CHROMA_COLLECTION_NAME", "rag_documents").strip()
    or "rag_documents"
)

_st_model: Any = None
_st_model_name: str | None = None
_oai_client: Any = None
_oai_client_config: tuple[str, str] | None = None
_gemini_client: Any = None
_gemini_api_key: str | None = None

_TITLE_PATTERN = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_SOURCE_PATTERN = re.compile(r"^\*\*Source:\*\*\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['\u2019-][^\W_]+)*", re.UNICODE)


def _get_openai_client():
    """Return a cached client for OpenAI or an OpenAI-compatible endpoint."""
    global _oai_client, _oai_client_config

    base_url = os.getenv(
        "EMBEDDING_BASE_URL",
        os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    api_key = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    config = (base_url, api_key)
    if _oai_client is None or _oai_client_config != config:
        from openai import OpenAI

        kwargs: dict[str, str] = {"api_key": api_key or "not-set"}
        if base_url:
            kwargs["base_url"] = base_url
        _oai_client = OpenAI(**kwargs)
        _oai_client_config = config
    return _oai_client


def _get_gemini_client():
    """Return a cached Google GenAI client."""
    global _gemini_client, _gemini_api_key

    api_key = os.getenv("EMBEDDING_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    if _gemini_client is None or _gemini_api_key != api_key:
        from google import genai

        _gemini_client = genai.Client(api_key=api_key or None)
        _gemini_api_key = api_key
    return _gemini_client


def _hash_embed(text: str, dimension: int) -> list[float]:
    """Create a deterministic, dependency-free feature-hashing embedding.

    This provider is intended for offline development and automated tests.  It
    is not a replacement for a trained embedding model, but unlike random test
    vectors it still gives identical words and bigrams matching dimensions.
    """
    if dimension <= 0:
        raise ValueError("EMBEDDING_DIM must be positive")

    tokens = _TOKEN_PATTERN.findall(text.casefold())
    features = tokens + [f"{left}\u241f{right}" for left, right in zip(tokens, tokens[1:])]
    counts = Counter(features)
    vector = [0.0] * dimension
    for feature, count in counts.items():
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
        index = int.from_bytes(digest[:8], "big") % dimension
        sign = 1.0 if digest[8] & 1 else -1.0
        vector[index] += sign * (1.0 + math.log(count))

    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def _coerce_vectors(vectors: Iterable[Iterable[Any]], expected_count: int) -> list[list[float]]:
    """Validate an embedding response and convert it to plain Python floats."""
    materialized: list[list[float]] = []
    for vector in vectors:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError) as error:
            raise ValueError("embedding provider returned a non-numeric vector") from error
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError("embedding vectors must be non-empty and finite")
        materialized.append(values)

    if len(materialized) != expected_count:
        raise ValueError(
            f"embedding provider returned {len(materialized)} vectors for "
            f"{expected_count} texts"
        )
    dimensions = {len(vector) for vector in materialized}
    if len(dimensions) > 1:
        raise ValueError("embedding provider returned inconsistent dimensions")
    return materialized


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed text with the configured provider while preserving input order.

    Supported values for ``EMBEDDING_PROVIDER`` are ``sentence_transformers``,
    ``openai``/``openai_compatible`` (including LM Studio), ``gemini``, and
    ``hash``.  ``hash`` is fully offline and deterministic, making it useful
    for a reproducible demo when no model server or API key is available.
    """
    if not isinstance(texts, list):
        raise TypeError("texts must be a list of strings")
    if not texts:
        return []
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("every text to embed must be a non-empty string")

    provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").strip().lower()
    model_name = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL).strip() or EMBEDDING_MODEL

    if provider in {"hash", "hashing", "offline", "local_hash"}:
        dimension = int(os.getenv("EMBEDDING_DIM", str(EMBEDDING_DIM)))
        return [_hash_embed(text, dimension) for text in texts]

    if provider in {"openai", "openai_compatible", "lmstudio", "lm_studio"}:
        client = _get_openai_client()
        vectors: list[list[float]] = []
        batch_size = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "32")))
        for offset in range(0, len(texts), batch_size):
            response = client.embeddings.create(
                model=model_name,
                input=texts[offset : offset + batch_size],
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
        return _coerce_vectors(vectors, len(texts))

    if provider in {"gemini", "google", "google_genai"}:
        client = _get_gemini_client()
        batch_size = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "32")))
        vectors = []
        output_dimension = os.getenv("GEMINI_EMBEDDING_DIM", "").strip()
        config = {"output_dimensionality": int(output_dimension)} if output_dimension else None
        for offset in range(0, len(texts), batch_size):
            kwargs: dict[str, Any] = {
                "model": model_name,
                "contents": texts[offset : offset + batch_size],
            }
            if config is not None:
                kwargs["config"] = config
            response = client.models.embed_content(**kwargs)
            vectors.extend(embedding.values for embedding in response.embeddings)
        return _coerce_vectors(vectors, len(texts))

    if provider in {"sentence_transformers", "sentence-transformers", "local"}:
        global _st_model, _st_model_name
        if _st_model is None or _st_model_name != model_name:
            from sentence_transformers import SentenceTransformer

            _st_model = SentenceTransformer(model_name)
            _st_model_name = model_name
        try:
            vectors = _st_model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        except TypeError:
            # Some older/fake SentenceTransformer implementations expose only
            # the basic ``encode(texts)`` call.
            vectors = _st_model.encode(texts)
        return _coerce_vectors(vectors, len(texts))

    raise ValueError(
        "Unsupported EMBEDDING_PROVIDER. Choose sentence_transformers, openai, "
        "gemini, or hash."
    )


def get_collection():
    """Open the persistent Chroma collection configured for cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _document_type(relative_path: Path) -> str:
    parts = {part.casefold() for part in relative_path.parts[:-1]}
    return "legal" if "legal" in parts else "news"


def _frontmatter_value(content: str, key: str) -> str | None:
    """Read a simple scalar from the YAML front matter emitted by Task 3."""
    if not content.startswith("---"):
        return None
    closing = content.find("\n---", 3)
    if closing < 0:
        return None
    frontmatter = content[3:closing]
    match = re.search(
        rf"^{re.escape(key)}:\s*(.*?)\s*$",
        frontmatter,
        re.MULTILINE | re.IGNORECASE,
    )
    if not match:
        return None
    value = match.group(1).strip()
    if value.casefold() in {"null", "none", "~"}:
        return None
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        parsed = value[1:-1] if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {'\"', "'"}
        ) else value
    if parsed is None:
        return None
    return str(parsed).strip() or None


def load_documents() -> list[dict]:
    """Read standardized Markdown files into deterministic Documents."""
    if not STANDARDIZED_DIR.is_dir():
        return []

    documents: list[dict] = []
    paths = sorted(
        path
        for path in STANDARDIZED_DIR.rglob("*.md")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(STANDARDIZED_DIR).parts)
    )
    for path in paths:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        relative_path = path.relative_to(STANDARDIZED_DIR)
        title_match = _TITLE_PATTERN.search(content)
        source_match = _SOURCE_PATTERN.search(content)
        title = (
            _frontmatter_value(content, "title")
            or (title_match.group(1).strip() if title_match else path.stem)
        )
        source = (
            _frontmatter_value(content, "source_file")
            or (source_match.group(1).strip() if source_match else path.name)
        )
        url = _frontmatter_value(content, "url")
        if url is None and source.lower().startswith(("http://", "https://")):
            url = source
        frontmatter_type = (_frontmatter_value(content, "doc_type") or "").casefold()
        doc_type = (
            frontmatter_type
            if frontmatter_type in {"legal", "news"}
            else _document_type(relative_path)
        )
        document = {
            "id": relative_path.as_posix(),
            "content": content,
            "metadata": {
                "source": source,
                "title": title or path.stem,
                "doc_type": doc_type,
                "url": url,
            },
        }
        validate_document(document)
        documents.append(document)
    return documents


def _fallback_split_text(text: str) -> list[str]:
    """Small character splitter used only when langchain is unavailable."""
    if CHUNK_SIZE <= 0 or CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("chunk settings require 0 <= overlap < size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        limit = min(start + CHUNK_SIZE, len(text))
        end = limit
        if limit < len(text):
            minimum_break = start + max(CHUNK_SIZE // 2, 1)
            candidates = [
                text.rfind(separator, minimum_break, limit)
                for separator in ("\n\n", "\n", ". ", " ")
            ]
            viable = [position for position in candidates if position >= minimum_break]
            if viable:
                end = max(viable) + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split Documents into non-empty chunks with stable IDs and metadata."""
    if CHUNK_SIZE <= 0 or CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("chunk settings require 0 <= overlap < size")

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        split_text = splitter.split_text
    except ImportError:
        split_text = _fallback_split_text

    chunks: list[dict] = []
    seen_document_ids: set[str] = set()
    for document in documents:
        validate_document(document)
        document_id = document["id"]
        if document_id in seen_document_ids:
            raise ValueError(f"duplicate document id: {document_id}")
        seen_document_ids.add(document_id)

        chunk_index = 0
        for raw_text in split_text(document["content"]):
            text = raw_text.strip()
            if not text:
                continue
            chunk = {
                "id": f"{document_id}::chunk-{chunk_index}",
                "content": text,
                "metadata": {**document["metadata"], "chunk_index": chunk_index},
            }
            validate_document(chunk, require_chunk=True)
            chunks.append(chunk)
            chunk_index += 1
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Return EmbeddedChunks without mutating the caller's chunk objects."""
    if not chunks:
        return []
    for chunk in chunks:
        validate_document(chunk, require_chunk=True)
    vectors = _coerce_vectors(
        embed_texts([chunk["content"] for chunk in chunks]),
        len(chunks),
    )
    return [
        {
            **chunk,
            "metadata": dict(chunk["metadata"]),
            "embedding": vector,
        }
        for chunk, vector in zip(chunks, vectors)
    ]


def _metadata_for_storage(metadata: dict) -> dict:
    """Convert contract metadata to scalar values accepted by Chroma."""
    stored: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            stored[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            stored[key] = value
        else:
            stored[key] = str(value)
    return stored


def _metadata_from_storage(metadata: dict | None) -> dict:
    """Restore metadata values serialized for Chroma."""
    restored = dict(metadata or {})
    if restored.get("url") == "":
        restored["url"] = None
    chunk_index = restored.get("chunk_index")
    if isinstance(chunk_index, float) and chunk_index.is_integer():
        restored["chunk_index"] = int(chunk_index)
    return restored


def _existing_collection_ids(collection: Any) -> set[str] | None:
    """Read IDs for stale-record cleanup; return None for older fake clients."""
    getter = getattr(collection, "get", None)
    if not callable(getter):
        return None
    try:
        response = getter(include=[])
    except (TypeError, ValueError):
        try:
            response = getter()
        except (AttributeError, TypeError, ValueError):
            return None
    if not isinstance(response, dict) or not isinstance(response.get("ids"), list):
        return None
    ids = response["ids"]
    if ids and isinstance(ids[0], list):
        ids = [item for group in ids for item in group]
    return {item for item in ids if isinstance(item, str)}


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert the complete chunk set and remove stale IDs from Chroma."""
    collection = get_collection()
    existing_ids = _existing_collection_ids(collection)
    if not chunks:
        # An empty source corpus is still a complete snapshot.  Clearing the
        # old collection prevents deleted documents from remaining searchable.
        if existing_ids:
            batch_size = max(1, int(os.getenv("CHROMA_BATCH_SIZE", "100")))
            stale_ids = sorted(existing_ids)
            for offset in range(0, len(stale_ids), batch_size):
                collection.delete(ids=stale_ids[offset : offset + batch_size])
        print("No chunks to index.")
        return

    for chunk in chunks:
        validate_document(chunk, require_chunk=True)
        if "embedding" not in chunk:
            raise ValueError(f"chunk {chunk['id']} has no embedding")
    ids = [chunk["id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("chunk IDs must be unique")
    vectors = _coerce_vectors(
        [chunk["embedding"] for chunk in chunks],
        len(chunks),
    )

    batch_size = max(1, int(os.getenv("CHROMA_BATCH_SIZE", "100")))
    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset : offset + batch_size]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=vectors[offset : offset + len(batch)],
            metadatas=[_metadata_for_storage(chunk["metadata"]) for chunk in batch],
        )

    # ``upsert`` makes repeated runs duplicate-free.  Removing IDs which are no
    # longer produced also prevents stale chunks after a source document shrinks
    # or is removed.
    if existing_ids is not None:
        stale_ids = sorted(existing_ids.difference(ids))
        if stale_ids:
            for offset in range(0, len(stale_ids), batch_size):
                collection.delete(ids=stale_ids[offset : offset + batch_size])


def run_pipeline() -> None:
    """Run load -> chunk -> embed -> index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks")


if __name__ == "__main__":
    run_pipeline()
