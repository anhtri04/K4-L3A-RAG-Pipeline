"""Task 6 - deterministic BM25 lexical search over the Task 4 chunks."""

from __future__ import annotations

from collections import Counter
import math
import re

from .contracts import validate_document


CORPUS: list[dict] = []

_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['\u2019-][^\W_]+)*", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Tokenize words case-insensitively while discarding punctuation."""
    return _TOKEN_PATTERN.findall(text.casefold())


def _normalise_corpus(corpus: list[dict]) -> list[dict]:
    """Remove invalid/duplicate records so search results satisfy the contract."""
    by_id: dict[str, dict] = {}
    for item in corpus:
        try:
            validate_document(item, require_chunk=True)
        except ValueError:
            continue
        by_id.setdefault(item["id"], item)
    return sorted(by_id.values(), key=lambda item: item["id"])


def _ensure_corpus() -> list[dict]:
    """Load the indexed chunk corpus, falling back to standardized Markdown."""
    if CORPUS:
        return CORPUS

    try:
        from .task4_chunking_indexing import get_collection, _metadata_from_storage

        data = get_collection().get(include=["documents", "metadatas"])
        ids = data.get("ids", []) if isinstance(data, dict) else []
        documents = data.get("documents", []) if isinstance(data, dict) else []
        metadatas = data.get("metadatas", []) if isinstance(data, dict) else []
        indexed = [
            {
                "id": item_id,
                "content": content,
                "metadata": _metadata_from_storage(metadata),
            }
            for item_id, content, metadata in zip(ids, documents, metadatas)
        ]
        indexed = _normalise_corpus(indexed)
        if indexed:
            CORPUS.extend(indexed)
            return CORPUS
    except Exception:
        # Chroma is optional during first-time/offline setup.  The same chunks
        # can be rebuilt deterministically from standardized Markdown below.
        pass

    try:
        from .task4_chunking_indexing import chunk_documents, load_documents

        chunks = _normalise_corpus(chunk_documents(load_documents()))
        CORPUS.extend(chunks)
    except Exception:
        pass
    return CORPUS


class _BM25Index:
    """BM25 using the positive Robertson/Lucene IDF variant."""

    def __init__(self, tokenized_documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.documents = tokenized_documents
        self.k1 = k1
        self.b = b
        self.lengths = [len(document) for document in tokenized_documents]
        self.average_length = (
            sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for document in tokenized_documents:
            document_frequency.update(set(document))
        count = len(tokenized_documents)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        self.term_frequencies = [Counter(document) for document in tokenized_documents]

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        if not self.documents or not query_tokens:
            return [0.0] * len(self.documents)
        average_length = self.average_length or 1.0
        scores: list[float] = []
        for frequencies, document_length in zip(self.term_frequencies, self.lengths):
            score = 0.0
            length_normalizer = self.k1 * (
                1.0 - self.b + self.b * document_length / average_length
            )
            for term in query_tokens:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                score += self.idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1.0) / (frequency + length_normalizer)
                )
            scores.append(score)
        return scores


def build_bm25_index(corpus: list[dict]):
    """Build a BM25 index from the exact chunk contents used by dense search."""
    tokenized = [_tokenize(item["content"]) for item in corpus]
    return _BM25Index(tokenized)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Return unique BM25 SearchResults ordered by descending score."""
    if not isinstance(query, str) or not query.strip():
        return []
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []
    corpus = _normalise_corpus(_ensure_corpus())
    if not corpus:
        return []

    scores = build_bm25_index(corpus).get_scores(query_tokens)
    ranked = sorted(
        (
            (float(score), item)
            for score, item in zip(scores, corpus)
            if math.isfinite(float(score)) and float(score) > 0.0
        ),
        key=lambda pair: (-pair[0], pair[1]["id"]),
    )
    return [
        {
            "id": item["id"],
            "content": item["content"],
            "score": score,
            "metadata": dict(item["metadata"]),
            "retrieval_method": "bm25",
        }
        for score, item in ranked[:top_k]
    ]


if __name__ == "__main__":
    for search_result in lexical_search("football governance", top_k=3):
        print(search_result)
