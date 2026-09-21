# Báo cáo đóng góp cá nhân

## Thông tin

- Họ và tên: Nguyễn Anh Trí
- Mã học viên: 2A202602730
- Nhóm: K4-L3A
- Repository: [anhtri04/K4-L3A-RAG-Pipeline](https://github.com/anhtri04/K4-L3A-RAG-Pipeline)
- Branch cá nhân: `anhtri_lead`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
| --- | --- | --- | --- |
| Task 1–3: thu thập & chuẩn hoá | Củng cố download tài liệu legal, crawl news bằng Crawl4AI (6 URL bóng đá: luật việt vị, chiến thuật, VAR, IFAB, Premier League, FIFA) với schema `url/title/date_crawled/content_markdown`, và convert legal+news sang Markdown chuẩn hoá | `src/task1_collect_legal_docs.py`, `src/task2_crawl_news.py` (`crawl_article`/`crawl_all`), `src/task3_convert_markdown.py` (`convert_legal_docs`/`convert_news_articles`/`convert_all`); commit `ac6d1f8` | Done |
| Task 4: chunking & indexing | Chia Markdown trong `data/standardized/` theo recursive strategy (`CHUNK_SIZE=500`, `CHUNK_OVERLAP=50`), ID chunk ổn định chống trùng khi rerun, embed một provider duy nhất (`embed_texts`, mặc định `BAAI/bge-m3` 1024 dims, hỗ trợ LM Studio/OpenAI-compatible), upsert ChromaDB collection `rag_documents` với cosine distance | `src/task4_chunking_indexing.py` (`load_documents`, `chunk_documents`, `embed_chunks`, `index_to_vectorstore`, `run_pipeline`); commit `5d9f626` | Done |
| Task 5–7: dense + BM25 + RRF | Dense search ChromaDB dùng chung `embed_texts()` của Task 4 trả `SearchResult` chuẩn; BM25 lexical search trên cùng chunk IDs; fuse đúng một lần bằng RRF (`k=60`), dedup, gắn nhãn `hybrid` | `src/task5_semantic_search.py` (`semantic_search`), `src/task6_lexical_search.py` (`lexical_search`, `build_bm25_index`), `src/task7_reranking.py` (`rerank_rrf`); commit `5d9f626` | Done |
| Task 8–9: fallback + retrieval pipeline | PageIndex vectorless fallback; pipeline hoàn chỉnh: dense+BM25 → RRF một lần → lấy best cosine gốc của dense so threshold (`SCORE_THRESHOLD=0.3`) → fallback khi thiếu tự tin → trả hybrid thay vì crash khi fallback lỗi | `src/task8_pageindex_vectorless.py` (`pageindex_search`), `src/task9_retrieval_pipeline.py` (`retrieve`); commit `5d9f626` | Done |
| Task 10 + UI Streamlit | Generation có citation: reorder chống lost-in-the-middle (`reorder_for_llm`), context kèm title/source (`format_context`), system prompt chỉ trả lời từ context + safe refusal khi thiếu evidence, gọi LLM qua OpenAI-compatible API; UI chat 2 cột (chat + panel tài liệu), click citation `[1][2]` xem nguồn, nút xóa hội thoại, slider số chunks | `src/task10_generation.py` (`generate_with_citation`, `call_llm`), `app.py` (103 dòng); commit `5d9f626` | Done |
| Báo cáo nhánh | Báo cáo cá nhân này + `reports/RESULT.md` phạm vi nhánh | `reports/2A202602730-nguyen-anh-tri.md`, `reports/RESULT.md` trên nhánh `anhtri_lead` | Done |

Chỉ kê khai công việc đối chiếu được bằng file/commit/test như bảng trên.
Phần corpus có giấy phép, evaluator offline và golden 23 câu là công của nhánh
`TranCaoThang-2A202602520`; golden/Ragas 15 câu là công của nhánh `tri` —
tôi không nhận là tác giả các phần đó (chi tiết hợp nhất ở `reports/RESULT.md` trên `main`).

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Dense và BM25 cùng trả về một schema `SearchResult`, RRF chỉ gộp thứ hạng và chạy đúng một lần; fallback so threshold với cosine score gốc của dense, không so với điểm RRF.  
   **Lý do/evidence:** hai thang điểm khác nhau (cosine similarity vs RRF rank score) nên so threshold với RRF score là sai ngữ nghĩa; contract test `test_retrieve_uses_dense_score_for_fallback` và `test_retrieve_fuses_once_when_dense_is_confident` trong `tests/test_contracts.py` khóa hành vi này.  
   **Trade-off:** pipeline từ chối khéo (fallback/refusal) thường xuyên hơn khi dense thiếu tự tin, đổi lại không crash và không bịa citation.

2. **Quyết định:** Chunking recursive cố định 500 ký tự / overlap 50, ID chunk ổn định, một provider embedding duy nhất cho cả index và query.  
   **Lý do/evidence:** ID ổn định giúp rerun không tạo trùng trong ChromaDB; dùng chung `embed_texts()` tránh lệch dimension giữa lúc index và query (lỗi từng gặp: khác provider/dimension là query lỗi).  
   **Trade-off:** chunk cố định đơn giản, tái lập tốt nhưng có thể cắt đứt evidence dài trong PDF chính sách (đúng failure mode mà cả hai run đánh giá đều ghi nhận) — cần heading-aware/parent-child retrieval về sau.

## Kiểm thử và kết quả

- Test tôi dựa vào: `tests/test_contracts.py` (ổn định chữ ký public API Task 4–10, validator document/search/generation, RRF đúng công thức `1/62+1/61`, fallback dùng dense score, generation chấp nhận safe refusal) và `tests/test_acceptance.py` (corpus ≥3 legal + ≥5 news, standardized phủ cả 2 loại, golden ≥15 cases, báo cáo đánh giá đã điền đầy đủ).
- Chữ ký hàm trên nhánh khớp contract: `retrieve(query, top_k, score_threshold, use_reranking)`, `generate_with_citation(query, top_k)`, `rerank_rrf(ranked_lists, top_k, k)` — kiểm bằng đọc mã nguồn (môi trường hiện tại chưa cài `pytest` ở cả `mise` python 3.11 và `.venv` nên chưa rerun được tại đây; cần `pip install -e ".[dev]"` rồi `pytest -q`).
- Query kiểm thủ công qua UI: câu hỏi luật (việt vị, handball, VAR, thẻ phạt) và chiến thuật; hành vi kỳ vọng là đáp án kèm citation click được, câu ngoài miền trả safe refusal `Tôi không thể xác minh thông tin này từ nguồn hiện có`.
- Lỗi đã xử lý trong code: fallback provider lỗi thì bắt exception và trả hybrid thay vì crash (`test_retrieve_survives_fallback_provider_error` khóa hành vi này); thiếu `LLM_MODEL` trong `.env` thì raise lỗi rõ ràng thay vì gọi API mù.

## Điều còn hạn chế

- Nhánh này chưa có golden dataset riêng (`group_project/evaluation/golden_dataset.json` còn trống) và chưa có script evaluator riêng nên chưa có run A/B độc lập — số liệu A/B chính thức nằm ở báo cáo hợp nhất trên `main` (Run 1: deterministic 23 câu; Run 2: Ragas 15 câu, cả hai đều kết luận hybrid+RRF tốt hơn).
- Nếu có thêm thời gian, việc đầu tiên tôi làm là port `src/evaluate_rag.py` từ nhánh Thắng (hoặc chạy Ragas như nhánh Trí) trên chính corpus đã convert của nhánh này, bổ sung golden ≥15 cases, rồi cập nhật `group_project/evaluation/RESULT.md` bằng run của nhánh.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 21/09/2026
- Tên thành viên: Nguyễn Anh Trí
