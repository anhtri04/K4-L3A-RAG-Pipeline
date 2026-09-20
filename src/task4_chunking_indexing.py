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

from pathlib import Path

import os

from dotenv import load_dotenv


load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

COLLECTION_NAME = "rag_documents"

_st_model = None
_oai_client = None


def _get_openai_client():
    global _oai_client
    if _oai_client is None:
        from openai import OpenAI

        _oai_client = OpenAI(
            base_url=os.getenv("EMBEDDING_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")),
            api_key=os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", "lm-studio")),
        )
    return _oai_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed bằng LM Studio (OpenAI-compatible) hoặc sentence_transformers."""
    provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").lower()
    model = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)

    if provider in ("openai", "openai_compatible", "lmstudio", "lm_studio"):
        client = _get_openai_client()
        vectors: list[list[float]] = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = client.embeddings.create(model=model, input=batch)
            # sort by index to preserve order
            ordered = sorted(resp.data, key=lambda d: d.index)
            vectors.extend([list(d.embedding) for d in ordered])
        return vectors

    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer

        _st_model = SentenceTransformer(model)
    return _st_model.encode(texts).tolist()


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if path.name.startswith("."):
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        # title: dòng # đầu tiên, fallback stem
        title = path.stem
        for line in content.splitlines():
            if line.startswith("# "):
                title = line[2:].strip() or title
                break
        doc_type = "legal" if "legal" in path.parts else "news"
        documents.append({
            "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
            "content": content,
            "metadata": {
                "source": path.name,
                "title": title,
                "doc_type": doc_type,
                "url": None,
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
        for index, text in enumerate(splitter.split_text(document["content"])):
            text = text.strip()
            if not text:
                continue
            chunks.append({
                "id": f"{document['id']}::chunk-{index}",
                "content": text,
                "metadata": {**document["metadata"], "chunk_index": index},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    if not chunks:
        return chunks
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    if not chunks:
        print("No chunks to index.")
        return
    collection = get_collection()
    batch = 100
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        collection.upsert(
            ids=[chunk["id"] for chunk in part],
            documents=[chunk["content"] for chunk in part],
            embeddings=[chunk["embedding"] for chunk in part],
            metadatas=[chunk["metadata"] for chunk in part],
        )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks")


if __name__ == "__main__":
    run_pipeline()
