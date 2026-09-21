# RAG evaluation results (nhánh `anhtri_lead` — Nguyễn Anh Trí, 2A202602730)

> Phạm vi nhánh: pipeline Task 1–10 + UI Streamlit. Nhánh này chưa có run A/B
> độc lập (chưa có golden riêng và evaluator riêng) nên file này ghi nhận
> trạng thái pipeline có thể kiểm chứng bằng mã nguồn/contract, đồng thời
> trỏ về số liệu A/B chính thức ở báo cáo hợp nhất trên `main`
> (`reports/RESULT.md`: Run 1 deterministic 23 câu, Run 2 Ragas 15 câu —
> cả hai đều kết luận hybrid + RRF tốt hơn dense-only).

## Run information

| Field | Value |
| --- | --- |
| Evaluation date | 2026-09-21 (ghi nhận trạng thái nhánh, chưa phải run đo metric) |
| Framework and version | Contract/acceptance tests trong repo (`tests/test_contracts.py`, `tests/test_acceptance.py`); chưa cài `pytest` trong môi trường hiện tại nên chưa rerun tại đây |
| Evaluator model | Không áp dụng trên nhánh này (chưa có evaluator riêng) |
| Generator model | `src/task10_generation.py` qua OpenAI-compatible API (`LLM_PROVIDER`/`LLM_MODEL` trong `.env`); safe refusal khi thiếu evidence |
| Embedding model | `BAAI/bge-m3` 1024 dims mặc định (`EMBEDDING_MODEL`/`EMBEDDING_DIM`), dùng chung `embed_texts()` cho index và query; hỗ trợ LM Studio/OpenAI-compatible |
| Corpus version/commit | Nhánh `anhtri_lead` tại `5d9f626`; Task 2 crawl 6 URL bóng đá (luật việt vị, chiến thuật, VAR, IFAB, Premier League, FIFA) về `data/landing/news/` |
| Golden dataset size | 0 trên nhánh này (`group_project/evaluation/golden_dataset.json` còn trống) — cần bổ sung ≥ 15 cases |
| `top_k` | 5 mặc định (`DEFAULT_TOP_K`/`TOP_K`; UI cho chỉnh 3–10) |
| Fallback threshold and calibration | `SCORE_THRESHOLD=0.3`, so với cosine score gốc của dense (không so với điểm RRF); RRF `k=60`; fallback PageIndex khi dưới ngưỡng, trả hybrid thay vì crash khi fallback lỗi |

## Configurations

- **Config A — dense-only:** `semantic_search` (ChromaDB cosine, `top_k*2` ứng viên) rồi cắt top_k — tương ứng nhánh `use_reranking=False` của `retrieve()`.
- **Config B — hybrid + RRF:** dense + `lexical_search` (BM25) → `rerank_rrf` đúng một lần (`k=60`) → top_k — tương ứng `use_reranking=True` (mặc định).

Hai config dùng chung corpus, chunk IDs, embedding function, prompt và `top_k`;
chỉ thay retrieval strategy. Đây là định nghĩa thiết kế; số đo A/B thực tế lấy
từ báo cáo hợp nhất trên `main`.

## Overall scores

Chưa có run đo trên nhánh này nên chưa có bảng điểm riêng. Kết quả đã đo ở
cấp nhóm (xem `reports/RESULT.md` trên `main`):

- Run 1 (deterministic, 23 câu): Config B hơn Config A ở recall +0.0833, precision +0.0695, relevance +0.0300; faithfulness bão hòa 1.0.
- Run 2 (Ragas, 15 câu): Config B hơn Config A ở recall +0.13, precision +0.08, relevance +0.07, faithfulness +0.06.

Việc cần làm trên nhánh này: port `src/evaluate_rag.py` từ nhánh Thắng (hoặc
chạy Ragas như nhánh Trí) với golden ≥ 15 cases rồi thay bảng này bằng số run
của nhánh.

## A/B comparison

- Cấu hình tốt hơn (theo cả hai run cấp nhóm): **Config B — hybrid + RRF**.
- Evidence: xem bảng điểm và worst performers trong `reports/RESULT.md` trên `main`.
- Trade-off về latency/cost: hybrid chậm hơn local (Run 1: ~2.5 ms vs ~161 ms; Run 2: +15–25 ms) do chạy 2 retrieval + RRF; zero API/token cost cho retrieval.

## Worst performers

Chưa có failure của nhánh này từ run đo. Failure đã biết ở cấp nhóm:

| # | Question | Nguồn | Failure stage | Root cause |
| --: | --- | --- | --- | --- |
| 1 | Ownership thresholds in Schedule 1 guidance (Football Governance Act 2025)? | Run 1, Config A | retrieval | Không anchor nào lọt top-k. |
| 2 | Immediate health advice after potential concussion? | Run 1, Config A | retrieval | Không anchor nào lọt top-k. |
| 3 | Scanning definition + linked performance outcomes? | Run 1, Config A | retrieval | Không anchor nào lọt top-k. |

Nhóm failure này củng cố quyết định giữ fallback + safe refusal trong `retrieve()`/`generate_with_citation()` của nhánh.

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | --- | --- | --- | --- |
| 1 | Bổ sung golden ≥ 15 cases + port evaluator về nhánh rồi chạy A/B của nhánh | `golden_dataset.json` trống, chưa có `src/evaluate_rag.py` trên nhánh | Có số đo riêng, hết phụ thuộc run nhánh khác | `pytest tests/test_acceptance.py -q` qua `test_golden_dataset_has_15_grounded_cases` |
| 2 | Cài dev deps rồi rerun contract tests (`pip install -e ".[dev]"`, `pytest -q`) | Môi trường hiện tại thiếu `pytest` | Xác nhận chữ ký/schema Task 4–10 còn giữ | Toàn bộ `test_contracts.py` xanh |
| 3 | Chunking heading-aware / parent-child cho PDF dài | Evidence bị cắt ở chunk 500 ký tự (cả 2 run nhóm) | Tăng recall, passage trích mạch lạc | So recall/precision theo case legal trước/sau |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| --- | --- | ---: | ---: | --- |
| Chưa chạy trên nhánh này | Hybrid + RRF | 0.0000 | 0 | Ưu tiên có run A/B của nhánh trước khi thử HyDE / learned reranker. |

## Kiểm chứng pipeline trên nhánh (không cần API key)

- `tests/test_contracts.py`: khóa chữ ký `retrieve(query, top_k, score_threshold, use_reranking)`, `generate_with_citation(query, top_k)`, `rerank_rrf(ranked_lists, top_k, k)`; RRF đúng công thức; fallback dùng dense score; generation chấp nhận safe refusal.
- Chạy UI: `streamlit run app.py`, hỏi luật (việt vị, handball, VAR, thẻ phạt) và chiến thuật; kiểm tra citation `[1][2]` mở được panel tài liệu; câu ngoài miền phải trả safe refusal.
