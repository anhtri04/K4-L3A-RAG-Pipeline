"""Task 7 - Reciprocal Rank Fusion for dense and BM25 rankings."""

from __future__ import annotations

import math

from .contracts import validate_document


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse ranked lists once using ``sum(1 / (k + rank))``.

    Duplicate IDs inside a single input list are counted only once.  This keeps
    malformed input from receiving an artificial score boost, while duplicates
    across different retrieval methods contribute as RRF intends.
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        return []
    if not isinstance(k, (int, float)) or isinstance(k, bool) or not math.isfinite(k) or k < 0:
        raise ValueError("RRF k must be a finite, non-negative number")
    if not ranked_lists:
        return []

    scores: dict[str, float] = {}
    representatives: dict[str, dict] = {}
    best_ranks: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    sequence = 0

    for ranked_list in ranked_lists:
        if not ranked_list:
            continue
        seen_in_list: set[str] = set()
        unique_rank = 0
        for item in ranked_list:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id.strip() or item_id in seen_in_list:
                continue
            try:
                validate_document(item, require_chunk=True)
            except ValueError:
                continue

            seen_in_list.add(item_id)
            unique_rank += 1
            denominator = float(k) + unique_rank
            if denominator <= 0:
                raise ValueError("RRF k + rank must be positive")
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / denominator
            best_ranks[item_id] = min(best_ranks.get(item_id, unique_rank), unique_rank)
            if item_id not in representatives:
                representatives[item_id] = item
                first_seen[item_id] = sequence
                sequence += 1

    ranked_ids = sorted(
        scores,
        key=lambda item_id: (
            -scores[item_id],
            best_ranks[item_id],
            first_seen[item_id],
            item_id,
        ),
    )
    results: list[dict] = []
    for item_id in ranked_ids[:top_k]:
        item = representatives[item_id]
        results.append(
            {
                **item,
                "metadata": dict(item["metadata"]),
                "score": float(scores[item_id]),
                "retrieval_method": "hybrid",
            }
        )
    return results


if __name__ == "__main__":
    print("Use rerank_rrf([dense_results, bm25_results]) to fuse rankings.")
