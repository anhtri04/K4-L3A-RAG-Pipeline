"""Task 5 - cosine semantic search over the Task 4 Chroma collection."""

from __future__ import annotations

import math

from .contracts import validate_document
from .task4_chunking_indexing import (
    _metadata_from_storage,
    embed_texts,
    get_collection,
)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Return unique dense SearchResults ordered by cosine similarity."""
    if not isinstance(query, str) or not query.strip():
        return []
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        return []

    query_vectors = embed_texts([query.strip()])
    if not query_vectors:
        return []

    collection = get_collection()
    counter = getattr(collection, "count", None)
    if callable(counter):
        try:
            if counter() == 0:
                return []
        except (AttributeError, TypeError, ValueError):
            pass

    response = collection.query(
        query_embeddings=[query_vectors[0]],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if not isinstance(response, dict):
        return []

    def first_row(name: str) -> list:
        rows = response.get(name)
        if not isinstance(rows, list) or not rows:
            return []
        return rows[0] if isinstance(rows[0], list) else rows

    ids = first_row("ids")
    documents = first_row("documents")
    metadatas = first_row("metadatas")
    distances = first_row("distances")

    by_id: dict[str, dict] = {}
    for item_id, content, raw_metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        if not isinstance(item_id, str) or not item_id.strip():
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            score = 1.0 - float(distance)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue

        try:
            metadata = _metadata_from_storage(raw_metadata)
        except (TypeError, ValueError):
            continue
        result = {
            "id": item_id,
            "content": content,
            "score": score,
            "metadata": metadata,
            "retrieval_method": "dense",
        }
        try:
            validate_document(result, require_chunk=True)
        except ValueError:
            continue
        previous = by_id.get(item_id)
        if previous is None or score > previous["score"]:
            by_id[item_id] = result

    return sorted(
        by_id.values(),
        key=lambda item: (-item["score"], item["id"]),
    )[:top_k]


if __name__ == "__main__":
    for search_result in semantic_search("football governance", top_k=3):
        print(search_result)
