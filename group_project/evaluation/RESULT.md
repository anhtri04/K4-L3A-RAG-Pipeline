# RAG evaluation results

> These are deterministic lexical/evidence proxy metrics, not RAGAS or LLM-judge scores. They are intended as a reproducible, API-free regression baseline.

## Run information

| Field | Value |
| --- | --- |
| Evaluation date | 2026-09-20T10:25:12+00:00 |
| Framework and version | Repository evaluator schema 1.0 (deterministic proxy) |
| Evaluator model | None; fixed lexical/evidence rules |
| Generator model | Deterministic extractive answerer v1 |
| Embedding model | hash-blake2b-unigram-bigram-v1 (1024 dimensions) |
| Corpus version/commit | Git `eb1d6cc`; SHA-256 `421f57dfa5f937a5…`; 12 documents / 2656 chunks |
| Golden dataset size | 23 |
| `top_k` | 5 (candidate_k=10) |
| Fallback threshold and calibration | App default 0.30; hash diagnostic recommends 0.2150 (balanced accuracy 0.7783). In-domain top-score range 0.1868–0.3846; out-of-domain 0.1588–0.3348. Fallback was disabled during A/B. |

## Configurations

- **Config A — dense-only:** hash-blake2b-unigram-bigram-v1 (1024 dimensions), exact cosine search over Task 4 embeddings, top 10 candidates reduced to top 5.
- **Config B — hybrid + RRF:** the same dense candidates plus BM25 candidates, fused exactly once with RRF (`k=60`) and reduced to top 5.

Both configurations use the same 23 questions, corpus, chunks, embedding function, `top_k`, extractive answerer and evaluator. No PageIndex fallback is called, so retrieval strategy is the only experimental variable. The evaluator uses exact NumPy cosine ranking instead of Chroma HNSW to keep near-tie ordering reproducible; the application still uses Chroma.

## Metric definitions

- **Context Recall:** Fraction of human-authored evidence anchors whose content-token recall is at least 0.60 in the concatenated top-k context.
- **Context Precision:** Fraction of top-k chunks from an expected source that cover at least 0.35 of one evidence anchor's content tokens.
- **Faithfulness:** Fraction of extractive answer sentences found verbatim in their cited chunks, multiplied by citation validity; zero for an empty answer.
- **Answer Relevance:** Content-token F1 between the deterministic extractive answer and the human-authored expected answer.

## Overall scores

| Metric | Config A | Config B | Delta B−A |
| --- | ---: | ---: | ---: |
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer relevance | 0.1914 | 0.2214 | +0.0300 |
| Context recall | 0.4638 | 0.5471 | +0.0833 |
| Context precision | 0.2957 | 0.3652 | +0.0695 |
| **Average** | 0.4877 | 0.5334 | +0.0457 |

## A/B comparison

- Better configuration on the composite proxy: **Config B — hybrid + RRF**.
- Evidence: hybrid changed average context recall by +0.0833, context precision by +0.0695, and answer relevance by +0.0300.
- Latency/cost: mean local retrieval latency was 2.53 ms for dense and 160.99 ms for hybrid. Both had zero API/token cost; hybrid adds local BM25 and RRF work.
- Faithfulness has a ceiling effect because the common generator copies cited source sentences; retrieval-oriented metrics are more discriminative in this offline baseline.

## Worst performers

| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | Which formal ownership thresholds are listed in Schedule 1 guidance under the Football Governance Act 2025? | Config A | 1.0000 | 0.0000 | 0.0000 | 0.0000 | retrieval | No expected evidence anchor appeared in the top-k context. |
| 2 | What immediate health advice does the grassroots concussion guidance give after a potential concussion? | Config A | 1.0000 | 0.0000 | 0.0000 | 0.0000 | retrieval | No expected evidence anchor appeared in the top-k context. |
| 3 | How does the review define scanning in soccer, and which positive performance outcomes are linked to it? | Config A | 1.0000 | 0.0000 | 0.0000 | 0.0000 | retrieval | No expected evidence anchor appeared in the top-k context. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| --: | --- | --- | --- | --- |
| 1 | Replace hash embeddings with the configured BGE-M3 or a production embedding API, then rerun the same benchmark. | Hash vectors only encode shared unigrams/bigrams and cannot model paraphrases. | Higher dense recall and fewer source misses. | Compare `results.json` against this locked baseline with the same golden IDs. |
| 2 | Add heading/page-aware chunking or parent-child retrieval for long policy PDFs. | Relevant evidence is sometimes split across fixed 500-character chunks. | Better context recall and more coherent cited passages. | Measure recall/precision changes per legal-document case. |
| 3 | Use the production LLM and RAGAS/LLM judges as a second, key-enabled evaluation tier. | The extractive answerer makes faithfulness nearly saturated and lexical F1 understates valid paraphrases. | More realistic generation quality and semantic grading. | Record judge model/version and compare with deterministic proxy results. |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| --- | --- | ---: | ---: | --- |
| No bonus experiment in this offline run | Hybrid + RRF | 0.0000 | 0 | Establish the required baseline before testing HyDE or a learned reranker. |

The complete per-question rankings, answers, scores, latency measurements, metric definitions and threshold calibration are stored in `results.json`.
