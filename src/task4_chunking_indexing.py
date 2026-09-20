"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import os
from pathlib import Path
import re
from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

COLLECTION_NAME = "rag_documents"

_EMBED_MODEL = None


def _get_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _EMBED_MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed danh sách văn bản thành vectors."""
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()


def _get_chroma_path() -> str:
    """Trả về path an toàn (8.3 short path trên Windows) cho ChromaDB."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    full_path = str(CHROMA_DIR.resolve())
    if os.name == "nt":
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(1024)
            if ctypes.windll.kernel32.GetShortPathNameW(full_path, buf, 1024):
                return buf.value
        except Exception:
            pass
    return full_path


def get_collection():
    """Mở hoặc tạo Chroma collection dùng cosine distance."""
    import chromadb
    client = chromadb.PersistentClient(path=_get_chroma_path())
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )


def load_documents() -> list[dict]:
    """Đọc Markdown trong standardized/ và trả về danh sách Document."""
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if path.name.startswith("."):
            continue

        doc_type = "legal" if "legal" in path.parts else "news"
        content = path.read_text(encoding="utf-8")

        # Trích xuất tiêu đề nếu có header # ...
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else path.stem.replace("_", " ").title()

        # Trích xuất URL nếu có **Source:** http...
        url_match = re.search(r"\*\*Source:\*\*\s*(https?://[^\s\n]+)", content)
        url = url_match.group(1).strip() if url_match else None

        doc_id = path.relative_to(STANDARDIZED_DIR).as_posix()
        documents.append({
            "id": doc_id,
            "content": content,
            "metadata": {
                "source": path.name,
                "title": title,
                "doc_type": doc_type,
                "url": url,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for document in documents:
        raw_chunks = splitter.split_text(document["content"])
        for index, text in enumerate(raw_chunks):
            chunk_id = f"{document['id']}::chunk-{index}"
            chunk_metadata = {
                **document["metadata"],
                "chunk_index": index,
            }
            chunks.append({
                "id": chunk_id,
                "content": text,
                "metadata": chunk_metadata,
            })
    return chunks


def embed_chunks(chunks: list[dict], batch_size: int = 64) -> list[dict]:
    """Thêm vector embedding vào từng chunk."""
    if not chunks:
        return []

    texts = [chunk["content"] for chunk in chunks]
    all_vectors: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors = embed_texts(batch)
        all_vectors.extend(vectors)

    for chunk, vector in zip(chunks, all_vectors):
        chunk["embedding"] = vector

    return chunks


def index_to_vectorstore(chunks: list[dict], batch_size: int = 250) -> None:
    """Upsert chunks vào ChromaDB an toàn theo batch."""
    if not chunks:
        return

    collection = get_collection()

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [chunk["id"] for chunk in batch]
        documents = [chunk["content"] for chunk in batch]
        embeddings = [chunk["embedding"] for chunk in batch]

        # Chuẩn hóa metadata (url thành string nếu None để ChromaDB nhận diện an toàn)
        metadatas = []
        for chunk in batch:
            meta = dict(chunk["metadata"])
            if meta.get("url") is None:
                meta["url"] = ""
            metadatas.append(meta)

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    print("Loading standardized documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")

    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Embedding chunks...")
    embedded_chunks = embed_chunks(chunks)

    print("Indexing to ChromaDB...")
    index_to_vectorstore(embedded_chunks)
    print(f"Successfully indexed {len(embedded_chunks)} chunks to ChromaDB collection '{COLLECTION_NAME}'!")


if __name__ == "__main__":
    run_pipeline()
