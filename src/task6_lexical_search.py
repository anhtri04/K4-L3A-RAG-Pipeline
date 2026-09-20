"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""

import numpy as np
from rank_bm25 import BM25Okapi

from .contracts import validate_search_results

CORPUS: list[dict] = []
_CACHED_BM25 = None
_CACHED_CORPUS_ID = None


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    tokenized = [item["content"].lower().split() for item in corpus]
    bm25 = BM25Okapi(tokenized)
    # Tránh trường hợp idf <= 0 khi tập corpus nhỏ (chuẩn hóa theo Lucene BM25)
    for word, val in bm25.idf.items():
        if val <= 0:
            bm25.idf[word] = 0.1
    return bm25


def _ensure_corpus() -> list[dict]:
    """Tự động nạp corpus nếu chưa được gán."""
    global CORPUS
    if not CORPUS:
        try:
            from .task4_chunking_indexing import chunk_documents, load_documents
            docs = load_documents()
            CORPUS = chunk_documents(docs)
        except Exception:
            CORPUS = []
    return CORPUS


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    global _CACHED_BM25, _CACHED_CORPUS_ID

    corpus = _ensure_corpus()
    if not corpus:
        return []

    corpus_id = id(corpus)
    if _CACHED_BM25 is None or _CACHED_CORPUS_ID != corpus_id:
        _CACHED_BM25 = build_bm25_index(corpus)
        _CACHED_CORPUS_ID = corpus_id

    tokenized_query = query.lower().split()
    if not tokenized_query:
        return []

    scores = _CACHED_BM25.get_scores(tokenized_query)
    sorted_indices = np.argsort(scores)[::-1]

    results = []
    seen_ids = set()
    for index in sorted_indices:
        score = float(scores[index])
        if score <= 0:
            continue
        item = corpus[index]
        if item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])

        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": score,
            "metadata": item["metadata"],
            "retrieval_method": "bm25",
        })
        if len(results) >= top_k:
            break

    validate_search_results(results, top_k=top_k, expected_method="bm25")
    return results


if __name__ == "__main__":
    for result in lexical_search("Independent Football Regulator", top_k=3):
        print(result["id"], result["score"], result["metadata"]["title"])
