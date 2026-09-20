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

from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"

_BATCH_SIZE = 32

# Cache singleton cho local model (tránh reload mỗi lần gọi embed_texts).
_ST_MODEL = None


def _get_st_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed danh sách text, giữ thứ tự input.

    Dispatch theo EMBEDDING_PROVIDER trong .env:
      - sentence_transformers (default, local BAAI/bge-m3)
      - openai
      - gemini
    Task 5 reuse hàm này nên model/dimension phải nhất quán với index.
    """
    if not texts:
        return []
    provider = os.getenv("EMBEDDING_PROVIDER", EMBEDDING_PROVIDER).strip().lower()

    if provider == "sentence_transformers":
        global _ST_MODEL
        from sentence_transformers import SentenceTransformer

        if _ST_MODEL is None:
            _ST_MODEL = SentenceTransformer(_get_st_model_name())
        vectors = _ST_MODEL.encode(
            texts, batch_size=_BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True
        )
        return [list(map(float, vec)) for vec in vectors]

    if provider == "openai":
        from openai import OpenAI

        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        client = OpenAI()  # đọc OPENAI_API_KEY từ .env
        vectors: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            resp = client.embeddings.create(model=model, input=batch)
            vectors.extend([list(map(float, item.embedding)) for item in resp.data])
        return vectors

    if provider == "gemini":
        from google import genai

        model = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
        client = genai.Client()  # đọc GEMINI_API_KEY từ .env
        vectors_g: list[list[float]] = []
        for text in texts:
            resp = client.models.embed_content(model=model, contents=text)
            # SDK mới trả về resp.embeddings[0].values
            emb = resp.embeddings[0].values if hasattr(resp, "embeddings") else resp.embedding
            vectors_g.append(list(map(float, emb)))
        return vectors_g

    raise ValueError(f"Unknown EMBEDDING_PROVIDER={provider!r}")


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
    """Đọc Markdown và trả về danh sách Document.

    - Đọc mọi .md dưới data/standardized/ (recursive, sort để ổn định).
    - id = relative posix path (ổn định, rerun không trùng).
    - doc_type = "legal" nếu "legal" trong path parts, ngược lại "news".
    - Bỏ qua file rỗng để thỏa contract content non-empty.
    """
    documents: list[dict] = []
    if not STANDARDIZED_DIR.is_dir():
        return documents
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in path.parts else "news"
        documents.append(
            {
                "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
                "content": content,
                "metadata": {
                    "source": path.name,
                    "title": path.stem,
                    "doc_type": doc_type,
                    "url": None,
                },
            }
        )
    return documents


def _simple_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Fallback thuần-Python khi chưa cài langchain-text-splitters.

    Giữ semantics gần RecursiveCharacterTextSplitter: ưu tiên cắt ở
    ["\\n\\n", "\\n", ". ", " ", ""] và ghép lại thành cửa sổ trượt
    chunk_size/overlap theo ký tự. Đảm bảo mỗi chunk <= chunk_size.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]
    # Bước 1: tách thô theo separators theo thứ tự ưu tiên.
    pieces: list[str] = [text]
    for sep in separators:
        if not sep:
            break
        next_pieces: list[str] = []
        for piece in pieces:
            if len(piece) <= chunk_size:
                next_pieces.append(piece)
            else:
                next_pieces.extend(piece.split(sep))
        # gắn lại sep (trừ sep rỗng) để không mất ký tự phân cách
        if sep != " " or True:
            pass
        pieces = [p for p in (s.strip() for s in next_pieces) if p]
        if all(len(p) <= chunk_size for p in pieces):
            break
    # Những piece còn dài (không có sep) -> cắt cứng theo ký tự.
    units: list[str] = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            units.append(piece)
        else:
            for i in range(0, len(piece), chunk_size):
                units.append(piece[i : i + chunk_size].strip())
    units = [u for u in units if u]
    # Bước 2: merge units thành chunks với overlap theo ký tự.
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = (current + " " + unit).strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
                # overlap: giữ đuôi chunk_size? ở đây giữ CHUNK_OVERLAP ký tự cuối
                overlap = current[-chunk_overlap:] if chunk_overlap > 0 else ""
                current = (overlap + " " + unit).strip()
                # nếu unit đơn lẻ vẫn quá dài (hiếm), cắt cứng
                while len(current) > chunk_size:
                    chunks.append(current[:chunk_size])
                    current = current[chunk_size - chunk_overlap :].strip() if chunk_overlap else ""
            else:
                for i in range(0, len(unit), chunk_size):
                    chunks.append(unit[i : i + chunk_size])
                current = ""
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index.

    ID ổn định: f"{doc_id}::chunk-{index}". Metadata giữ source/title/
    doc_type/url của doc gốc + chunk_index. Không trả chunk rỗng.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        split_fn = splitter.split_text
    except ImportError:
        split_fn = lambda text: _simple_split(text, CHUNK_SIZE, CHUNK_OVERLAP)

    chunks: list[dict] = []
    for document in documents:
        texts = split_fn(document["content"])
        for index, text in enumerate(texts):
            text = text.strip()
            if not text:
                continue
            chunks.append(
                {
                    "id": f"{document['id']}::chunk-{index}",
                    "content": text,
                    "metadata": {**document["metadata"], "chunk_index": index},
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk (batch, giữ nguyên field khác)."""
    if not chunks:
        return chunks
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def _sanitize_metadata(metadata: dict) -> dict:
    """Chroma chỉ nhận str/int/float/bool — convert None -> ''."""
    clean: dict = {}
    for key, value in metadata.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB (idempotent — chạy lại không trùng)."""
    if not chunks:
        return
    collection = get_collection()
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[_sanitize_metadata(chunk["metadata"]) for chunk in chunks],
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
