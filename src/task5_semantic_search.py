"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

from .contracts import validate_search_results
from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    query_vector = embed_texts([query])[0]
    collection = get_collection()

    try:
        response = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as err:
        print(f"ChromaDB query warning: {err}")
        return []

    if not response or not response.get("ids") or not response["ids"][0]:
        return []

    results = []
    ids = response["ids"][0]
    documents = response["documents"][0]
    metadatas = response["metadatas"][0]
    distances = response["distances"][0]

    for item_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
        # Cosine distance trong Chroma thuộc [0, 2]; similarity = 1 - distance
        score = max(0.0, 1.0 - float(distance))
        results.append({
            "id": item_id,
            "content": content,
            "score": score,
            "metadata": metadata,
            "retrieval_method": "dense",
        })

    # Sắp xếp giảm dần theo score và giới hạn không vượt quá top_k
    sorted_results = sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]
    validate_search_results(sorted_results, top_k=top_k, expected_method="dense")
    return sorted_results


if __name__ == "__main__":
    for result in semantic_search("Independent Football Regulator powers", top_k=3):
        print(result["id"], result["score"], result["metadata"]["title"])
