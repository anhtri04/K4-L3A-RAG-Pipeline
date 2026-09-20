"""Task 9 — resilient hybrid retrieval with a PageIndex fallback."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable

from dotenv import load_dotenv

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


LOGGER = logging.getLogger(__name__)
DEFAULT_TOP_K = 5
# Calibrated on the checked-in football golden set with the default 1024-D
# hash embedding.  Other embedding providers should override SCORE_THRESHOLD
# after repeating the in-domain/out-of-domain calibration.
DEFAULT_SCORE_THRESHOLD = 0.2128

load_dotenv()


def _configured_threshold() -> float:
    """Read a cosine threshold in [0, 1], falling back on bad config."""
    raw_value = os.getenv("SCORE_THRESHOLD", "").strip()
    if not raw_value:
        return DEFAULT_SCORE_THRESHOLD
    try:
        value = float(raw_value)
    except ValueError:
        return DEFAULT_SCORE_THRESHOLD
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return DEFAULT_SCORE_THRESHOLD
    return value


SCORE_THRESHOLD = _configured_threshold()


def _safe_search(
    name: str,
    search: Callable[..., list[dict]],
    query: str,
    top_k: int,
) -> list[dict]:
    try:
        results = search(query, top_k=top_k)
    except Exception as error:
        LOGGER.warning("%s retrieval failed: %s", name, error)
        return []
    if not isinstance(results, list):
        LOGGER.warning("%s retrieval returned a non-list result", name)
        return []
    return results


def _effective_threshold(value: float) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return SCORE_THRESHOLD
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        return SCORE_THRESHOLD
    return threshold


def _best_dense_score(results: list[dict]) -> float:
    scores: list[float] = []
    for result in results:
        try:
            score = float(result.get("score", 0.0))
        except (AttributeError, TypeError, ValueError):
            continue
        if math.isfinite(score):
            scores.append(score)
    return max(scores, default=0.0)


def _degraded_hybrid(dense: list[dict], sparse: list[dict], top_k: int) -> list[dict]:
    """Keep the pipeline useful if an optional reranker itself fails."""
    # Do not compare raw cosine and BM25 values. Prefer the dense ranking when
    # present; otherwise expose the sparse ranking with rank-derived scores.
    ranked = dense if dense else sparse
    results: list[dict] = []
    seen_ids: set[str] = set()
    for rank, result in enumerate(ranked, 1):
        item_id = result.get("id") if isinstance(result, dict) else None
        if not isinstance(item_id, str) or not item_id or item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        item = result.copy()
        item["retrieval_method"] = "hybrid"
        item["score"] = 1.0 / (60 + rank)
        results.append(item)
        if len(results) >= top_k:
            break
    return results


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Return hybrid results, or PageIndex results for low dense confidence.

    The fallback decision always uses the original dense cosine score.  RRF is
    executed at most once and its incomparable score is never thresholded.
    Provider errors degrade to whatever local evidence is still available.
    """
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []

    query = query.strip()
    candidate_k = top_k * 2
    dense = _safe_search("dense", semantic_search, query, candidate_k)
    # ``use_reranking=False`` is the required dense-only A/B baseline.  Avoid
    # doing unused BM25 work so the latency measurement is genuinely dense-only.
    sparse = (
        _safe_search("BM25", lexical_search, query, candidate_k)
        if use_reranking
        else []
    )

    if use_reranking:
        if dense or sparse:
            try:
                hybrid = rerank_rrf([dense, sparse], top_k=top_k)
                if not isinstance(hybrid, list):
                    raise TypeError("RRF returned a non-list result")
            except Exception as error:
                LOGGER.warning("RRF reranking failed: %s", error)
                hybrid = _degraded_hybrid(dense, sparse, top_k)
        else:
            hybrid = []
    else:
        # ``False`` is intentionally dense-only for the required A/B baseline.
        hybrid = dense[:top_k]

    threshold = _effective_threshold(score_threshold)
    if _best_dense_score(dense) < threshold:
        fallback = _safe_search("PageIndex", pageindex_search, query, top_k)
        if fallback:
            return fallback[:top_k]
    return hybrid[:top_k]


if __name__ == "__main__":
    for result in retrieve("football governance", top_k=3):
        print(result)
