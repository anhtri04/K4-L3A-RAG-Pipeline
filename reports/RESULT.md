# RAG evaluation results (unified — `main`)

> Báo cáo hợp nhất trên nhánh `main`. Trên `main` chưa có run đánh giá độc lập
> (xem "Tình trạng trên main"); số liệu dưới đây tổng hợp trung thực từ 2 run
> đã thực hiện trên nhánh cá nhân, giữ nguyên framework/metric của từng run,
> không gộp hai hệ metric khác nhau thành một bảng.

## Tình trạng trên main

- Commit `main`: `23803eb` (sau `b441d99 feat(data): add licensed football corpus`).
- Corpus trên `main`: 4 PDF legal + 8 JSON news trong `data/landing/`
  (chi tiết giấy phép OGL v3.0 / CC BY 4.0 trong `data/SOURCES.md`).
  Kiểm chứng ngày 21/09/2026: `data/landing/legal/` có 4 file,
  `data/landing/news/` có 8 file.
- `data/standardized/legal/` và `data/standardized/news/` trên `main` đang trống.
- `group_project/evaluation/golden_dataset.json` trên `main` đang trống (0 byte).
- Chưa có script evaluator trên `main` (`src/evaluate_rag.py` mới chỉ tồn tại
  trên nhánh `TranCaoThang-2A202602520`).
- Môi trường hiện tại chưa cài `pytest` (cả `mise` python 3.11 và `.venv`),
  nên chưa rerun contract/acceptance tests trên `main` ở lần ghi này.
- Để tái lập end-to-end trên `main`: chạy convert → chunk/index → bổ sung
  golden ≥ 15 cases → chạy evaluator → cập nhật `group_project/evaluation/RESULT.md`.

## Run information (các run nguồn)

| Field | Run 1 — offline deterministic (nhánh `TranCaoThang-2A202602520`) | Run 2 — Ragas-style (nhánh `tri`) |
| --- | --- | --- |
| Evaluation date | 2026-09-20T10:25:12+00:00 | 2026-09-20 |
| Framework and version | Repository evaluator schema 1.0 (deterministic proxy, API-free) | Ragas 0.4.3 |
| Evaluator model | None; fixed lexical/evidence rules | gpt-4o-mini |
| Generator model | Deterministic extractive answerer v1 | gpt-4o-mini |
| Embedding model | hash-blake2b-unigram-bigram-v1 (1024 dims) | sentence-transformers/all-MiniLM-L6-v2 |
| Corpus version/commit | Git `eb1d6cc`; 12 documents / 2656 chunks | football-v1.0 (branch `tri`) |
| Golden dataset size | 23 | 15 |
| `top_k` | 5 (candidate_k=10) | 5 |
| Fallback threshold and calibration | App default 0.30; hash diagnostic recommends 0.2150 (balanced accuracy 0.7783). Fallback disabled during A/B. | 0.35 |

Nguồn: `origin/TranCaoThang-2A202602520:group_project/evaluation/RESULT.md` +
`results.json` + `golden_dataset.json` (23 cases);
`origin/tri:group_project/evaluation/RESULT.md` + `golden_dataset.json` (15 cases).

## Configurations

- **Config A — dense-only:** dense cosine search (Run 1: hash embeddings, exact NumPy ranking; Run 2: ChromaDB cosine), top_k=5.
- **Config B — hybrid + RRF:** cùng dense candidates + BM25 (Run 2 dùng BM25Okapi), fuse đúng một lần bằng RRF (`k=60`), top_k=5.

Cả hai run đều giữ cùng golden/generator/evaluator/prompt/`top_k` trong nội bộ
từng run; chỉ thay retrieval strategy. Hai run dùng framework khác nhau nên
không so sánh số tuyệt đối liên run, chỉ so sánh delta B−A trong từng run.

## Overall scores

### Run 1 — deterministic proxy, 23 questions (nhánh Thắng)

| Metric | Config A | Config B | Delta B−A |
| --- | ---: | ---: | ---: |
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer relevance | 0.1914 | 0.2214 | +0.0300 |
| Context recall | 0.4638 | 0.5471 | +0.0833 |
| Context precision | 0.2957 | 0.3652 | +0.0695 |
| **Average** | 0.4877 | 0.5334 | +0.0457 |

Định nghĩa metric của Run 1: Context Recall = tỉ lệ evidence anchor đạt
content-token recall ≥ 0.60 trong top-k; Context Precision = tỉ lệ chunk đúng
nguồn phủ ≥ 0.35 token của một anchor; Faithfulness = tỉ lệ câu trích nguyên
văn trong chunk được cite nhân citation validity; Answer Relevance = F1
content-token giữa đáp án trích và expected answer.

### Run 2 — Ragas-style, 15 questions (nhánh Trí)

| Metric | Config A | Config B | Delta B−A |
| --- | ---: | ---: | ---: |
| Faithfulness | 0.88 | 0.94 | +0.06 |
| Answer relevance | 0.84 | 0.91 | +0.07 |
| Context recall | 0.80 | 0.93 | +0.13 |
| Context precision | 0.82 | 0.90 | +0.08 |
| **Average** | 0.835 | 0.920 | +0.085 |

## A/B comparison

- Cấu hình tốt hơn: **Config B — hybrid + RRF** ở cả hai run.
- Evidence:
  - Run 1: recall +0.0833, precision +0.0695, relevance +0.0300; faithfulness bão hòa 1.0 do generator sao chép câu nguồn.
  - Run 2: recall +0.13, precision +0.08, relevance +0.07, faithfulness +0.06; BM25 bắt thuật ngữ pháp lý/tên riêng (IFR, xG, SIC) tốt hơn dense đơn lẻ.
- Trade-off về latency/cost:
  - Run 1 (local): dense 2.53 ms vs hybrid 160.99 ms trung bình; zero API/token cost.
  - Run 2: hybrid chậm hơn khoảng 15–25 ms do chạy 2 retrieval + RRF; chi phí chấp nhận được so với gain.

## Worst performers

| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause | Nguồn |
| --: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | Which formal ownership thresholds are listed in Schedule 1 guidance under the Football Governance Act 2025? | Config A | 1.0000 | 0.0000 | 0.0000 | 0.0000 | retrieval | Không anchor nào lọt top-k (Run 1). | Thắng, 23Q |
| 2 | What immediate health advice does the grassroots concussion guidance give after a potential concussion? | Config A | 1.0000 | 0.0000 | 0.0000 | 0.0000 | retrieval | Không anchor nào lọt top-k (Run 1). | Thắng, 23Q |
| 3 | How does the review define scanning in soccer, and which positive performance outcomes are linked to it? | Config A | 1.0000 | 0.0000 | 0.0000 | 0.0000 | retrieval | Không anchor nào lọt top-k (Run 1). | Thắng, 23Q |
| 4 | What perceptual-cognitive skills are identified as crucial for talent development in young soccer players? | Config A | 0.72 | 0.75 | 0.70 | 0.71 | retrieval | Dense miss multi-word keywords (head-turn frequency, visual scanning). | Trí, 15Q |
| 5 | What are the recommended rest periods for grassroots athletes suspected of suffering a concussion? | Config A | 0.80 | 0.81 | 0.75 | 0.78 | retrieval | Chunk boundary cắt mốc ngày nghỉ khỏi mô tả triệu chứng. | Trí, 15Q |
| 6 | What did the research on erroneous offside calls discover regarding assistant referee errors? | Config B | 0.85 | 0.88 | 0.82 | 0.84 | generation | Prompt dài nén mất chi tiết optical illusion. | Trí, 15Q |

Chi tiết per-question của Run 1 nằm ở `results.json` trên nhánh Thắng.

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | --- | --- | --- | --- |
| 1 | Thay hash embeddings bằng BGE-M3 / embedding API production rồi rerun cùng benchmark. | Hash chỉ bắt unigram/bigram chung, miss paraphrase. | Tăng dense recall, giảm source miss. | So `results.json` mới với baseline đã lock, cùng golden IDs. |
| 2 | Chunking heading/page-aware hoặc parent-child retrieval cho PDF chính sách dài. | Evidence bị cắt qua chunk 500 ký tự cố định. | Tăng recall + passage trích mạch lạc hơn. | Đo recall/precision theo từng case legal. |
| 3 | Thêm tầng judge bằng LLM production / RAGAS key-enabled song song với proxy offline. | Extractive làm faithfulness bão hòa; lexical F1 đánh thấp paraphrase hợp lệ. | Đánh giá generation thực tế hơn. | Ghi model/version judge, đối chiếu với proxy. |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| --- | --- | ---: | ---: | --- |
| BGE-M3 Multilingual vs MiniLM (nhánh Trí) | MiniLM (Config B) | +0.02 Average | +80ms latency | BGE-M3 tốt hơn cho tiếng Việt; MiniLM đủ nhanh/hiệu quả cho corpus tiếng Anh. |
| Chưa chạy trên main | Hybrid + RRF | 0.0000 | 0 | Lock baseline trước khi thử HyDE / learned reranker. |

## Cách tái lập trên main

1. `python -m src.task3_convert_markdown` → lấp đầy `data/standardized/` (hiện trống).
2. Bổ sung `group_project/evaluation/golden_dataset.json` ≥ 15 cases có `question`, `expected_answer`, `expected_context` (hiện 0 byte).
3. Port `src/evaluate_rag.py` + `results.json` từ nhánh Thắng (hoặc chạy Ragas như nhánh Trí với API key).
4. Cập nhật `group_project/evaluation/RESULT.md` bằng số run mới, thay thế bảng tổng hợp này.
