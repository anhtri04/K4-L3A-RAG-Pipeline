# TEAMMATES — K4-L3A-RAG-Pipeline

> Nhóm K4-L3A — dự án chatbot RAG bóng đá (hybrid retrieval + citation + Streamlit + đánh giá A/B).
> Nguồn xác minh: `git log --all`, `git branch -a`, diff từng nhánh vs `main`, và `reports/2A202602520-tran-cao-thang.md` (nhánh `TranCaoThang-2A202602520`).

| # | Họ tên | Mã học viên | Vai trò | Nhánh | Phần việc chính |
|---|--------|-------------|---------|-------|-----------------|
| 1 | Trần Cao Thắng | 2A202602520 | Data + QA / Integration | `TranCaoThang-2A202602520` | Corpus bóng đá có giấy phép (4 PDF + 8 news, `data/landing/`, `data/SOURCES.md`); tích hợp nhánh nhóm (Task 1–10); QA reproducibility (audit/convert/idempotence, contract + retrieval/generation tests); evaluation A/B + handoff (`group_project/evaluation/`, `src/evaluate_rag.py`, `reports/2A202602520-tran-cao-thang.md`) |
| 2 | Võ Đức Trí | 2A202602603 | Data standardization + Evaluation | `tri` | `update data`: bổ sung Markdown chuẩn hoá (`data/standardized/`), cập nhật `golden_dataset.json` + `RESULT.md`, tinh chỉnh `src/task1–task9` |
| 3 | Đỗ Hoàng Nam Khánh | 2A202602423 | Backend / Retrieval + Integration | `khanh` | `feat: implement RAG pipeline with FastAPI and Opencode integration`: `task4_chunking_indexing`, `task5_semantic_search`, `task6_lexical_search`, `task7_reranking`, `task9_retrieval_pipeline`, `task10_generation`, `docs/RAG_THEORY.md` |
| 4 | Phạm Minh Cương | 2A202602825 | Data collection (Task 1–3) + Test | `pmcnb` | `Implement football news crawling and processing`: `src/task1_collect_legal_docs.py`, `src/task2_crawl_news.py`, `src/task3_convert_markdown.py`, `crawl_football.ps1`, `tests/test_crawl_football.py`, `tests/test_task1_collect_legal_docs.py` |
| 5 | Nguyễn Anh Trí | 2A202602730 | Lead — RAG core (Task 4–10) + UI | `anhtri_lead` | `feat: implement RAG core tasks 4-10 and Streamlit chat with citation panel` + `enhance document handling and crawling (task 1-3)`: `src/task1–task10`, `app.py` (Streamlit chat + citation) |

## Ghi chú nhánh

- `main` (`origin/main`): hiện tại đồng bộ tới commit `b441d99 feat(data): add licensed football corpus` — corpus 4 PDF legal + 8 JSON news.
- Nhánh cá nhân chưa merge vào `main` (tính tới 21/09/2026):
  - `origin/TranCaoThang-2A202602520` — ahead `main` nhiều nhất (full pipeline + `src/evaluate_rag.py` + `results.json` + báo cáo cá nhân).
  - `origin/tri`, `origin/khanh`, `origin/pmcnb`, `origin/anhtri_lead` — mỗi nhánh 1–2 commit riêng, xem `git diff --stat main..origin/<nhánh>`.
- `khvavuong <vuong2002vp@gmail.com>` (5 commit đầu: init + templates + deps) là tác giả khung repo/template, không tính là thành viên nhóm 5 người.

## Việc cần bổ sung

- [ ] Chốt vai trò chính thức nếu có thay đổi (hiện tại: Nguyễn Anh Trí là Lead).
- [ ] Mỗi thành viên copy `reports/INDIVIDUAL_REPORT.md` thành `reports/<mssv>-<short-name>.md` như bạn Trần Cao Thắng đã làm (`reports/2A202602520-tran-cao-thang.md`).
