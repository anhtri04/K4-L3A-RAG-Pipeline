"""Deterministic, offline A/B evaluation for the football RAG pipeline.

This module deliberately evaluates retrieval without PageIndex fallback so the
only changed component is the retrieval strategy:

* Config A: dense search only.
* Config B: the same dense search plus BM25, fused once with RRF.

The default hash embedding and extractive answerer make the run reproducible and
API-key free.  The four reported values are transparent lexical/evidence proxies,
not RAGAS or LLM-judge scores; their definitions are emitted with every result.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Iterable

import numpy as np
from dotenv import load_dotenv


# Load a repository-local configuration before applying offline defaults.
# Evaluation uses an isolated Chroma directory so it can never overwrite the
# application's production/local-demo index.
ROOT_DIR = Path(__file__).resolve().parent.parent
EVALUATION_DIR = ROOT_DIR / "group_project" / "evaluation"
load_dotenv(ROOT_DIR / ".env")
os.environ.setdefault("EMBEDDING_PROVIDER", "hash")
os.environ.setdefault("EMBEDDING_DIM", "1024")
os.environ["CHROMA_DIR"] = os.getenv(
    "EVALUATION_CHROMA_DIR",
    str(ROOT_DIR / ".cache" / "evaluation_chroma"),
)
os.environ["CHROMA_COLLECTION_NAME"] = "rag_evaluation"

from .task4_chunking_indexing import (  # noqa: E402
    chunk_documents,
    embed_texts,
    embed_chunks,
    index_to_vectorstore,
    load_documents,
)
from .task6_lexical_search import CORPUS as BM25_CORPUS, lexical_search  # noqa: E402
from .task7_reranking import rerank_rrf  # noqa: E402
from .task10_generation import (  # noqa: E402
    SAFE_REFUSAL,
    generate_from_chunks,
)


GOLDEN_PATH = EVALUATION_DIR / "golden_dataset.json"
RESULTS_PATH = EVALUATION_DIR / "results.json"
REPORT_PATH = EVALUATION_DIR / "RESULT.md"

TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
CITATION_PATTERN = re.compile(r"\[(\d+)]")

# Function words do not express the evidence that a football answer must find.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "in", "into", "is",
    "it", "its", "may", "of", "on", "or", "that", "the", "their", "them",
    "they", "this", "to", "was", "were", "what", "when", "which", "while",
    "who", "why", "will", "with", "would",
}

METRIC_DEFINITIONS = {
    "context_recall": (
        "Fraction of human-authored evidence anchors whose content-token recall "
        "is at least 0.60 in the concatenated top-k context."
    ),
    "context_precision": (
        "Fraction of top-k chunks from an expected source that cover at least "
        "0.35 of one evidence anchor's content tokens."
    ),
    "faithfulness": (
        "Fraction of extractive answer sentences found verbatim in their cited "
        "chunks, multiplied by citation validity; zero for an empty answer."
    ),
    "answer_relevance": (
        "Content-token F1 between the deterministic extractive answer and the "
        "human-authored expected answer."
    ),
}

OUT_OF_DOMAIN_QUERIES = [
    "How do I bake sourdough bread with a crispy crust?",
    "What is the quantum error correction threshold for superconducting qubits?",
    "Which treatment is recommended for seasonal allergic rhinitis?",
    "How should a spacecraft enter orbit around Mars?",
    "What caused the fall of the Western Roman Empire?",
]

_DENSE_CHUNKS: list[dict] = []
_DENSE_MATRIX: np.ndarray | None = None
_DENSE_NORMS: np.ndarray | None = None


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(str(text).casefold())
        if token not in STOPWORDS and len(token) > 1
    ]


def _token_recall(reference: str, candidate: str) -> float:
    reference_counts = Counter(_tokens(reference))
    if not reference_counts:
        return 0.0
    candidate_counts = Counter(_tokens(candidate))
    overlap = sum((reference_counts & candidate_counts).values())
    return overlap / sum(reference_counts.values())


def _token_f1(reference: str, candidate: str) -> float:
    reference_counts = Counter(_tokens(reference))
    candidate_counts = Counter(_tokens(candidate))
    if not reference_counts or not candidate_counts:
        return 0.0
    overlap = sum((reference_counts & candidate_counts).values())
    precision = overlap / sum(candidate_counts.values())
    recall = overlap / sum(reference_counts.values())
    return 2.0 * precision * recall / (precision + recall) if overlap else 0.0


def _clean_sentence(sentence: str) -> str:
    sentence = re.sub(r"\[[^]]*]\([^)]*\)", "", sentence)
    sentence = re.sub(r"^[#>*\-\s]+", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence


def _extractive_answer(question: str, contexts: list[dict], max_sentences: int = 2) -> dict:
    """Select answer sentences using only the question and retrieved context."""
    candidates: list[tuple[float, int, int, str]] = []
    question_tokens = set(_tokens(question))
    for source_index, result in enumerate(contexts, 1):
        for sentence_index, raw_sentence in enumerate(
            SENTENCE_SPLIT_PATTERN.split(result.get("content", ""))
        ):
            sentence = _clean_sentence(raw_sentence)
            sentence_tokens = set(_tokens(sentence))
            if len(sentence) < 35 or len(sentence) > 420 or not sentence_tokens:
                continue
            overlap = len(question_tokens & sentence_tokens)
            if overlap == 0:
                continue
            query_coverage = overlap / max(len(question_tokens), 1)
            specificity = overlap / math.sqrt(len(sentence_tokens))
            score = query_coverage + 0.20 * specificity
            candidates.append((score, source_index, sentence_index, sentence))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    selected: list[tuple[int, str]] = []
    seen_sentences: set[str] = set()
    for _, source_index, _, sentence in candidates:
        key = sentence.casefold()
        if key in seen_sentences:
            continue
        seen_sentences.add(key)
        selected.append((source_index, sentence))
        if len(selected) >= max_sentences:
            break

    answer = " ".join(f"{sentence} [{source_index}]" for source_index, sentence in selected)
    return {"answer": answer, "selected": selected}


def _context_recall(case: dict, contexts: list[dict]) -> float:
    anchors = case.get("evidence_anchors") or [case["expected_context"]]
    combined = "\n".join(result.get("content", "") for result in contexts)
    matched = sum(_token_recall(anchor, combined) >= 0.60 for anchor in anchors)
    return matched / len(anchors) if anchors else 0.0


def _context_precision(case: dict, contexts: list[dict]) -> float:
    if not contexts:
        return 0.0
    expected_sources = set(case.get("source_files", []))
    anchors = case.get("evidence_anchors") or [case["expected_context"]]
    relevant = 0
    for result in contexts:
        metadata = result.get("metadata", {})
        source_matches = not expected_sources or metadata.get("source") in expected_sources
        anchor_matches = max(
            (_token_recall(anchor, result.get("content", "")) for anchor in anchors),
            default=0.0,
        )
        if source_matches and anchor_matches >= 0.35:
            relevant += 1
    return relevant / len(contexts)


def _faithfulness(answer_data: dict, contexts: list[dict]) -> float:
    selected = answer_data["selected"]
    answer = answer_data["answer"]
    if not selected or not answer:
        return 0.0
    citations = [int(value) for value in CITATION_PATTERN.findall(answer)]
    citation_validity = (
        1.0
        if len(citations) == len(selected)
        and all(1 <= value <= len(contexts) for value in citations)
        else 0.0
    )
    grounded = 0
    for source_index, sentence in selected:
        source_text = contexts[source_index - 1].get("content", "")
        if sentence in source_text or _token_recall(sentence, source_text) >= 0.95:
            grounded += 1
    return citation_validity * grounded / len(selected)


def _serialize_result(result: dict) -> dict:
    return {
        "id": result["id"],
        "score": round(float(result["score"]), 6),
        "retrieval_method": result["retrieval_method"],
        "metadata": result["metadata"],
    }


def _evaluate_case(case: dict, contexts: list[dict], latency_ms: float) -> dict:
    answer_data = _extractive_answer(case["question"], contexts)
    metrics = {
        "faithfulness": _faithfulness(answer_data, contexts),
        "answer_relevance": _token_f1(case["expected_answer"], answer_data["answer"]),
        "context_recall": _context_recall(case, contexts),
        "context_precision": _context_precision(case, contexts),
    }
    return {
        "id": case["id"],
        "question": case["question"],
        "answer": answer_data["answer"],
        "metrics": {name: round(value, 4) for name, value in metrics.items()},
        "average": round(sum(metrics.values()) / len(metrics), 4),
        "latency_ms": round(latency_ms, 2),
        "retrieved": [_serialize_result(result) for result in contexts],
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _aggregate(cases: list[dict]) -> dict:
    metrics = {
        name: round(_mean(case["metrics"][name] for case in cases), 4)
        for name in METRIC_DEFINITIONS
    }
    metrics["average"] = round(_mean(metrics.values()), 4)
    metrics["mean_latency_ms"] = round(_mean(case["latency_ms"] for case in cases), 2)
    return metrics


def _prepare_exact_dense(embedded_chunks: list[dict]) -> None:
    """Prepare an exact cosine index from Task 4 embeddings.

    Chroma remains the application's persistent vector database.  The benchmark
    uses an exact matrix search because repeated HNSW upserts can perturb the
    ordering of near-tied hash vectors and would make a regression baseline
    needlessly non-deterministic.
    """
    global _DENSE_CHUNKS, _DENSE_MATRIX, _DENSE_NORMS
    _DENSE_CHUNKS = embedded_chunks
    _DENSE_MATRIX = np.asarray(
        [chunk["embedding"] for chunk in embedded_chunks],
        dtype=np.float64,
    )
    _DENSE_NORMS = np.linalg.norm(_DENSE_MATRIX, axis=1)


def _exact_dense_search(question: str, top_k: int) -> list[dict]:
    if not question.strip() or top_k <= 0 or _DENSE_MATRIX is None or _DENSE_NORMS is None:
        return []
    query = np.asarray(embed_texts([question.strip()])[0], dtype=np.float64)
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0.0:
        return []
    denominators = _DENSE_NORMS * query_norm
    scores = np.divide(
        _DENSE_MATRIX @ query,
        denominators,
        out=np.zeros_like(_DENSE_NORMS),
        where=denominators != 0,
    )
    ranked_indices = sorted(
        range(len(_DENSE_CHUNKS)),
        key=lambda index: (-float(scores[index]), _DENSE_CHUNKS[index]["id"]),
    )[:top_k]
    return [
        {
            "id": _DENSE_CHUNKS[index]["id"],
            "content": _DENSE_CHUNKS[index]["content"],
            "score": float(scores[index]),
            "metadata": dict(_DENSE_CHUNKS[index]["metadata"]),
            "retrieval_method": "dense",
        }
        for index in ranked_indices
    ]


def _retrieve_dense(question: str, top_k: int, candidate_k: int) -> list[dict]:
    return _exact_dense_search(question, top_k=candidate_k)[:top_k]


def _retrieve_hybrid(question: str, top_k: int, candidate_k: int) -> list[dict]:
    dense = _exact_dense_search(question, top_k=candidate_k)
    sparse = lexical_search(question, top_k=candidate_k)
    return rerank_rrf([dense, sparse], top_k=top_k)


def _top_dense_score(question: str) -> float:
    results = _exact_dense_search(question, top_k=1)
    return float(results[0]["score"]) if results else 0.0


def _calibrate_threshold(golden: list[dict]) -> dict:
    in_domain = [
        {"id": case["id"], "score": _top_dense_score(case["question"])}
        for case in golden
    ]
    out_domain = [
        {"query": query, "score": _top_dense_score(query)}
        for query in OUT_OF_DOMAIN_QUERIES
    ]
    labelled_scores = [(item["score"], 1) for item in in_domain] + [
        (item["score"], 0) for item in out_domain
    ]
    unique_scores = sorted({score for score, _ in labelled_scores})
    candidates = [0.0, 1.0]
    candidates.extend((left + right) / 2 for left, right in zip(unique_scores, unique_scores[1:]))
    best = (float("-inf"), 0.30)
    for threshold in candidates:
        positive_accuracy = _mean(score >= threshold for score, label in labelled_scores if label == 1)
        negative_accuracy = _mean(score < threshold for score, label in labelled_scores if label == 0)
        balanced_accuracy = (positive_accuracy + negative_accuracy) / 2
        candidate = (balanced_accuracy, -abs(threshold - 0.30), threshold)
        if candidate > (best[0], -abs(best[1] - 0.30), best[1]):
            best = (balanced_accuracy, threshold)
    return {
        "configured_pipeline_threshold": 0.30,
        "recommended_hash_threshold": round(best[1], 4),
        "balanced_accuracy": round(best[0], 4),
        "in_domain_top_scores": [
            {**item, "score": round(item["score"], 4)} for item in in_domain
        ],
        "out_of_domain_top_scores": [
            {**item, "score": round(item["score"], 4)} for item in out_domain
        ],
        "note": "Fallback was disabled for A/B isolation; this calibration is diagnostic only.",
    }


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or "unavailable"
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _corpus_sha256(documents: list[dict]) -> str:
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda item: item["id"]):
        digest.update(document["id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(document["content"].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _failure_reason(case: dict) -> tuple[str, str]:
    metrics = case["metrics"]
    if metrics["context_recall"] == 0:
        return "retrieval", "No expected evidence anchor appeared in the top-k context."
    if metrics["context_precision"] < 0.20:
        return "retrieval/ranking", "Most top-k chunks came from a different source or passage."
    if metrics["answer_relevance"] < 0.25:
        return "generation", "Question-only extractive ranking selected a weak answer sentence."
    return "evaluation", "The answer was grounded, but lexical proxy overlap was limited."


def _report_markdown(payload: dict) -> str:
    a = payload["configs"]["dense_only"]["aggregate"]
    b = payload["configs"]["hybrid_rrf"]["aggregate"]
    metric_rows = []
    for name, label in (
        ("faithfulness", "Faithfulness"),
        ("answer_relevance", "Answer relevance"),
        ("context_recall", "Context recall"),
        ("context_precision", "Context precision"),
        ("average", "**Average**"),
    ):
        metric_rows.append(
            f"| {label} | {a[name]:.4f} | {b[name]:.4f} | {b[name] - a[name]:+.4f} |"
        )

    combined = [
        ("Config A", case) for case in payload["configs"]["dense_only"]["cases"]
    ] + [
        ("Config B", case) for case in payload["configs"]["hybrid_rrf"]["cases"]
    ]
    worst = sorted(combined, key=lambda item: (item[1]["average"], item[0], item[1]["id"]))[:3]
    failure_rows = []
    for rank, (config, case) in enumerate(worst, 1):
        stage, reason = _failure_reason(case)
        m = case["metrics"]
        question = case["question"].replace("|", "\\|")
        failure_rows.append(
            f"| {rank} | {question} | {config} | {m['faithfulness']:.4f} | "
            f"{m['answer_relevance']:.4f} | {m['context_recall']:.4f} | "
            f"{m['context_precision']:.4f} | {stage} | {reason} |"
        )

    winner = "Config B — hybrid + RRF" if b["average"] > a["average"] else "Config A — dense-only"
    calibration = payload["threshold_calibration"]
    in_scores = [item["score"] for item in calibration["in_domain_top_scores"]]
    out_scores = [item["score"] for item in calibration["out_of_domain_top_scores"]]
    return "\n".join(
        [
            "# RAG evaluation results",
            "",
            "> These are deterministic lexical/evidence proxy metrics, not RAGAS or LLM-judge scores. "
            "They are intended as a reproducible, API-free regression baseline.",
            "",
            "## Run information",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Evaluation date | {payload['run_info']['evaluated_at_utc']} |",
            "| Framework and version | Repository evaluator schema 1.0 (deterministic proxy) |",
            "| Evaluator model | None; fixed lexical/evidence rules |",
            "| Generator model | Deterministic extractive answerer v1 |",
            f"| Embedding model | {payload['run_info']['embedding']} |",
            f"| Corpus version/commit | Git `{payload['run_info']['git_commit']}`; SHA-256 `{payload['run_info']['corpus_sha256'][:16]}…`; "
            f"{payload['run_info']['document_count']} documents / {payload['run_info']['chunk_count']} chunks |",
            f"| Golden dataset size | {payload['run_info']['golden_size']} |",
            f"| `top_k` | {payload['run_info']['top_k']} (candidate_k={payload['run_info']['candidate_k']}) |",
            f"| Fallback threshold and calibration | App default 0.30; hash diagnostic recommends {calibration['recommended_hash_threshold']:.4f} "
            f"(balanced accuracy {calibration['balanced_accuracy']:.4f}). In-domain top-score range "
            f"{min(in_scores):.4f}–{max(in_scores):.4f}; out-of-domain {min(out_scores):.4f}–{max(out_scores):.4f}. "
            "Fallback was disabled during A/B. |",
            "",
            "## Configurations",
            "",
            f"- **Config A — dense-only:** {payload['run_info']['embedding']}, exact cosine search over Task 4 embeddings, top "
            f"{payload['run_info']['candidate_k']} candidates reduced to top {payload['run_info']['top_k']}.",
            f"- **Config B — hybrid + RRF:** the same dense candidates plus BM25 candidates, fused exactly once with RRF (`k=60`) and reduced to top {payload['run_info']['top_k']}.",
            "",
            f"Both configurations use the same {payload['run_info']['golden_size']} questions, corpus, chunks, embedding function, `top_k`, extractive answerer and evaluator. "
            "No PageIndex fallback is called, so retrieval strategy is the only experimental variable. "
            "The evaluator uses exact NumPy cosine ranking instead of Chroma HNSW to keep near-tie ordering reproducible; the application still uses Chroma.",
            "",
            "## Metric definitions",
            "",
            *[f"- **{name.replace('_', ' ').title()}:** {definition}" for name, definition in METRIC_DEFINITIONS.items()],
            "",
            "## Overall scores",
            "",
            "| Metric | Config A | Config B | Delta B−A |",
            "| --- | ---: | ---: | ---: |",
            *metric_rows,
            "",
            "## A/B comparison",
            "",
            f"- Better configuration on the composite proxy: **{winner}**.",
            f"- Evidence: hybrid changed average context recall by {b['context_recall'] - a['context_recall']:+.4f}, "
            f"context precision by {b['context_precision'] - a['context_precision']:+.4f}, and answer relevance by "
            f"{b['answer_relevance'] - a['answer_relevance']:+.4f}.",
            f"- Latency/cost: mean local retrieval latency was {a['mean_latency_ms']:.2f} ms for dense and "
            f"{b['mean_latency_ms']:.2f} ms for hybrid. Both had zero API/token cost; hybrid adds local BM25 and RRF work.",
            "- Faithfulness has a ceiling effect because the common generator copies cited source sentences; retrieval-oriented metrics are more discriminative in this offline baseline.",
            "",
            "## Worst performers",
            "",
            "| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |",
            "| --: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
            *failure_rows,
            "",
            "## Recommendations",
            "",
            "| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |",
            "| --: | --- | --- | --- | --- |",
            "| 1 | Replace hash embeddings with the configured BGE-M3 or a production embedding API, then rerun the same benchmark. | Hash vectors only encode shared unigrams/bigrams and cannot model paraphrases. | Higher dense recall and fewer source misses. | Compare `results.json` against this locked baseline with the same golden IDs. |",
            "| 2 | Add heading/page-aware chunking or parent-child retrieval for long policy PDFs. | Relevant evidence is sometimes split across fixed 500-character chunks. | Better context recall and more coherent cited passages. | Measure recall/precision changes per legal-document case. |",
            "| 3 | Use the production LLM and RAGAS/LLM judges as a second, key-enabled evaluation tier. | The extractive answerer makes faithfulness nearly saturated and lexical F1 understates valid paraphrases. | More realistic generation quality and semantic grading. | Record judge model/version and compare with deterministic proxy results. |",
            "",
            "## Bonus experiments",
            "",
            "| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |",
            "| --- | --- | ---: | ---: | --- |",
            "| No bonus experiment in this offline run | Hybrid + RRF | 0.0000 | 0 | Establish the required baseline before testing HyDE or a learned reranker. |",
            "",
            "The complete per-question rankings, answers, scores, latency measurements, metric definitions and threshold calibration are stored in `results.json`.",
            "",
        ]
    )


def run_evaluation(top_k: int = 5, candidate_k: int = 10, rebuild_index: bool = True) -> dict:
    if top_k <= 0 or candidate_k < top_k:
        raise ValueError("Require candidate_k >= top_k > 0")
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if not isinstance(golden, list) or len(golden) < 15:
        raise ValueError("golden_dataset.json must contain at least 15 cases")

    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    _prepare_exact_dense(embedded_chunks)
    BM25_CORPUS.clear()
    BM25_CORPUS.extend(chunks)
    if rebuild_index:
        index_to_vectorstore(embedded_chunks)

    configurations = {
        "dense_only": _retrieve_dense,
        "hybrid_rrf": _retrieve_hybrid,
    }
    evaluated_configs: dict[str, dict] = {}
    for config_name, retriever in configurations.items():
        evaluated_cases = []
        for case in golden:
            started = time.perf_counter()
            contexts = retriever(case["question"], top_k, candidate_k)
            elapsed_ms = (time.perf_counter() - started) * 1000
            evaluated_cases.append(_evaluate_case(case, contexts, elapsed_ms))
        evaluated_configs[config_name] = {
            "aggregate": _aggregate(evaluated_cases),
            "cases": evaluated_cases,
        }

    provider = os.getenv("EMBEDDING_PROVIDER", "hash").strip().lower()
    embedding_description = (
        f"hash-blake2b-unigram-bigram-v1 ({os.getenv('EMBEDDING_DIM')} dimensions)"
        if provider in {"hash", "hashing", "offline", "local_hash"}
        else f"{provider}:{os.getenv('EMBEDDING_MODEL', 'configured-model')}"
    )
    payload = {
        "schema_version": "1.0",
        "run_info": {
            "evaluated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "git_commit": _git_commit(),
            "corpus_sha256": _corpus_sha256(documents),
            "embedding": embedding_description,
            "generator": "deterministic-extractive-v1",
            "evaluator": "deterministic-lexical-evidence-proxy-v1",
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "golden_size": len(golden),
            "top_k": top_k,
            "candidate_k": candidate_k,
            "dense_backend": "exact-numpy-cosine-over-task4-embeddings",
            "api_cost_usd": 0.0,
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "configs": evaluated_configs,
        "threshold_calibration": _calibrate_threshold(golden),
        "limitations": [
            "Hash embeddings are a reproducibility baseline, not a semantic production model.",
            "The extractive generator has a faithfulness ceiling effect.",
            "Lexical proxies may score correct paraphrases lower than a semantic judge would.",
        ],
    }
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(_report_markdown(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Reuse an existing Chroma index built with the same embedding settings.",
    )
    args = parser.parse_args()
    payload = run_evaluation(
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rebuild_index=not args.skip_index,
    )
    a = payload["configs"]["dense_only"]["aggregate"]
    b = payload["configs"]["hybrid_rrf"]["aggregate"]
    print(f"Evaluated {payload['run_info']['golden_size']} grounded questions")
    print(f"Dense-only average: {a['average']:.4f}")
    print(f"Hybrid + RRF average: {b['average']:.4f}")
    print(f"Wrote {RESULTS_PATH.relative_to(ROOT_DIR)}")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
