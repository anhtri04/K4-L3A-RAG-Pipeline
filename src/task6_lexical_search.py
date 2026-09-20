"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""


CORPUS: list[dict] = []


def _ensure_corpus() -> list[dict]:
    """Load corpus từ Chroma (đã index) nếu CORPUS rỗng. Không override khi test mock."""
    if CORPUS:
        return CORPUS
    # 1. Thử đọc từ Chroma vectorstore (cùng corpus Task 4/5)
    try:
        from .task4_chunking_indexing import get_collection

        data = get_collection().get(include=["documents", "metadatas"])
        items = []
        for item_id, content, metadata in zip(data["ids"], data["documents"], data["metadatas"]):
            items.append({"id": item_id, "content": content, "metadata": metadata})
        if items:
            CORPUS.extend(items)
            return CORPUS
    except Exception:
        pass
    # 2. Fallback: load + chunk từ standardized MD
    try:
        from .task4_chunking_indexing import chunk_documents, load_documents

        chunks = chunk_documents(load_documents())
        CORPUS.extend(chunks)
    except Exception:
        pass
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    from rank_bm25 import BM25Okapi

    tokenized = [item["content"].lower().split() for item in corpus]
    return BM25Okapi(tokenized)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    import numpy as np

    if not query or not query.strip() or top_k <= 0:
        return []
    corpus = _ensure_corpus()
    if not corpus:
        return []
    bm25 = build_bm25_index(corpus)
    scores = bm25.get_scores(query.lower().split())
    indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for index in indices:
        if scores[index] <= 0:
            continue
        item = corpus[int(index)]
        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": float(scores[index]),
            "metadata": item["metadata"],
            "retrieval_method": "bm25",
        })
    return results


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
