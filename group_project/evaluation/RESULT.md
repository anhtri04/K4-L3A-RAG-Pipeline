# RAG evaluation results (nhánh `anhtri_lead` — Nguyễn Anh Trí, 2A202602730)

> Trạng thái nhánh (21/09/2026): pipeline Task 1–10 + UI đã xong; golden 23 cases
> đã reuse từ nhánh `TranCaoThang-2A202602520`; evaluator riêng của nhánh chưa có
> nên chưa có run A/B độc lập tại đây.
> Số liệu A/B chính thức của nhóm nằm ở `reports/RESULT.md` trên `main`
> (Run 1 deterministic 23 câu, Run 2 Ragas 15 câu — cả hai kết luận hybrid + RRF
> tốt hơn). File này sẽ được thay bằng số run của nhánh ngay khi golden +
> evaluator được bổ sung.

## Run information

| Field | Value |
| --- | --- |
| Evaluation date | 2026-09-21 (ghi nhận trạng thái, chưa phải run đo) |
| Framework and version | Contract/acceptance tests trong repo; `pytest` chưa được cài trong môi trường hiện tại |
| Evaluator model | Chưa có trên nhánh (dự kiến port evaluator deterministic từ nhánh Thắng hoặc Ragas như nhánh Trí) |
| Generator model | `src/task10_generation.py` qua OpenAI-compatible API; safe refusal khi thiếu evidence |
| Embedding model | `BAAI/bge-m3` 1024 dims mặc định, dùng chung `embed_texts()` |
| Corpus version/commit | Nhánh `anhtri_lead` tại `5d9f626` (Task 2: 6 URL bóng đá về `data/landing/news/`) |
| Golden dataset size | 23 (reuse từ nhánh `TranCaoThang-2A202602520`; mỗi case có `question`, `expected_answer`, `expected_context`) |
| `top_k` | 5 (UI chỉnh 3–10) |
| Fallback threshold and calibration | `SCORE_THRESHOLD=0.3` trên cosine gốc của dense; RRF `k=60`; fallback PageIndex, trả hybrid khi fallback lỗi |

## Configurations

- **Config A — dense-only:** `semantic_search` top_k (tương ứng `retrieve(..., use_reranking=False)`).
- **Config B — hybrid + RRF:** dense + BM25 → `rerank_rrf` một lần → top_k (mặc định `use_reranking=True`).

Hai config dùng cùng golden, generator, evaluator, prompt và `top_k` khi run;
chỉ thay retrieval strategy.

## Overall scores

Chưa có số đo trên nhánh. Tham chiếu nhóm: Run 1 (23 câu) Config B hơn Config A
recall +0.0833 / precision +0.0695 / relevance +0.0300; Run 2 (15 câu) hơn
recall +0.13 / precision +0.08 / relevance +0.07 / faithfulness +0.06.
Chi tiết đầy đủ ở `reports/RESULT.md` trên `main`.

| Metric | Config A | Config B | Delta B−A |
| --- | ---: | ---: | ---: |
| Faithfulness | Chưa đo trên nhánh | Chưa đo trên nhánh | Xem `main` |
| Answer relevance | Chưa đo trên nhánh | Chưa đo trên nhánh | Xem `main` |
| Context recall | Chưa đo trên nhánh | Chưa đo trên nhánh | Xem `main` |
| Context precision | Chưa đo trên nhánh | Chưa đo trên nhánh | Xem `main` |
| **Average** | Chưa đo trên nhánh | Chưa đo trên nhánh | Xem `main` |

## A/B comparison

- Cấu hình tốt hơn (theo cả hai run cấp nhóm): **Config B — hybrid + RRF**.
- Evidence: `reports/RESULT.md` trên `main` (bảng điểm + worst performers 6 cases).
- Trade-off về latency/cost: hybrid chậm hơn local nhưng zero API cost cho retrieval.

## Worst performers

| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | Ownership thresholds (Schedule 1, Football Governance Act 2025) | n/a (nhóm Run 1-A) | 1.0 | 0.0 | 0.0 | 0.0 | retrieval | Không anchor nào lọt top-k. |
| 2 | Immediate concussion health advice (grassroots) | n/a (nhóm Run 1-A) | 1.0 | 0.0 | 0.0 | 0.0 | retrieval | Không anchor nào lọt top-k. |
| 3 | Scanning definition + outcomes (talent review) | n/a (nhóm Run 1-A) | 1.0 | 0.0 | 0.0 | 0.0 | retrieval | Không anchor nào lọt top-k. |

Số liệu gốc của bảng này thuộc về run nhóm; cột Config ghi `n/a` để không nhận
vơ thành run của nhánh.

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | --- | --- | --- | --- |
| 1 | Port evaluator về nhánh rồi chạy A/B riêng | Golden đã reuse (23 cases); còn thiếu `src/evaluate_rag.py` | Có run A/B riêng | `test_golden_dataset_has_15_grounded_cases` xanh (golden đã đạt) + run evaluator mới |
| 2 | `pip install -e ".[dev]"` rồi `pytest -q` | Môi trường thiếu `pytest` | Xác nhận contracts Task 4–10 | `test_contracts.py` xanh |
| 3 | Chunking heading-aware / parent-child | Evidence bị cắt ở chunk 500 ký tự | Tăng recall legal | So recall trước/sau theo case |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| --- | --- | ---: | ---: | --- |
| Chưa chạy trên nhánh | Hybrid + RRF | 0.0000 | 0 | Có run A/B của nhánh trước khi thử HyDE / reranker học được. |
