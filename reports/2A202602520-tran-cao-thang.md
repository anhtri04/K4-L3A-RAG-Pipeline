# Báo cáo đóng góp cá nhân

## Thông tin

- Họ và tên: Trần Cao Thắng
- Mã học viên: 2A202602520
- Nhóm: K4-L3A — nhóm dự án 5 thành viên
- Repository: [anhtri04/K4-L3A-RAG-Pipeline](https://github.com/anhtri04/K4-L3A-RAG-Pipeline)
- Branch cá nhân: `TranCaoThang-2A202602520`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
| --- | --- | --- | --- |
| Corpus bóng đá | Tìm và kiểm chứng 4 PDF chính sách chính thức cùng 8 bài viết/nghiên cứu; giữ URL, ngày, attribution và giấy phép OGL v3.0/CC BY 4.0. | `data/landing/`, `data/SOURCES.md`; commit `b441d99` | Done |
| Tích hợp nhánh nhóm | Đưa hai commit do `anhtri04` phát triển vào nhánh cá nhân để kiểm thử tích hợp; giữ nguyên authorship, không nhận là tác giả các module đó. | `bdd7659` (Task 1–3), `eb1d6cc` (Task 4–10/UI) | Done |
| Reproducibility và QA | Chạy audit/convert corpus, xác nhận đủ 12 Markdown và rerun không thay đổi; kiểm tra module contracts, retrieval/generation, acceptance và luồng offline hash. | `data/standardized/`, `tests/test_data_tasks.py`, `tests/test_generation_pipeline.py` | Done |
| Evaluation/handoff | Phối hợp hoàn thiện golden set, chạy A/B dense-only so với hybrid+RRF, rà báo cáo kết quả và hướng dẫn chạy lại trên Python 3.11. | `group_project/evaluation/`, `README.md`, `.env.example` | Done |

Phạm vi trên phản ánh vai trò tìm dữ liệu hợp pháp, tích hợp và QA. Phần code nền Task 1–10 từ commit của thành viên khác vẫn được ghi đúng tác giả trong lịch sử Git.

## Quyết định kỹ thuật quan trọng

1. **Chọn snapshot có giấy phép rõ ràng thay cho dữ liệu bóng đá sao chép tùy ý.**  
   **Lý do/evidence:** nguồn GOV.UK/PDF chính phủ có OGL v3.0; bài PLOS ONE có CC BY 4.0; toàn bộ nguồn được đối chiếu trong `data/SOURCES.md` và giữ provenance trong từng record.  
   **Trade-off:** corpus hợp pháp và tái lập tốt nhưng thiên về bóng đá Anh, nội dung tiếng Anh và không phải nguồn tin thời gian thực.

2. **Dùng hash embedding cho baseline offline tái lập, tách khỏi cấu hình model production.**  
   **Lý do/evidence:** cùng một corpus/query tạo cùng vector và evaluation có thể chạy không cần key hoặc tải model; dense và BM25 vẫn dùng chung chunk IDs để so sánh công bằng.  
   **Trade-off:** hash embedding chủ yếu bắt trùng từ/ngữ đoạn, nên semantic recall thấp hơn model như BGE-M3 hoặc embedding API.

## Kiểm thử và kết quả

- Test/query đã dùng: audit 4 PDF + 8 JSON, convert hai lần để kiểm tra idempotence, index bằng `EMBEDDING_PROVIDER=hash`, query về Football Governance Act/chấn động/xG và câu ngoài miền, sau đó chạy `python -m pytest -q`.
- Bằng chứng ở mức module trước lần chạy tổng cuối: Task 1–3 đạt **10/10 test**; nhóm contract + retrieval/generation đạt **26/26 test**.
- Kết quả A/B và bốn metric của lần chạy cuối được lưu trong `group_project/evaluation/RESULT.md`; dữ liệu chi tiết nằm ở `group_project/evaluation/results.json`.
- Lỗi đã phát hiện/xử lý: Python 3.14 không khớp ràng buộc `<3.14`, vì vậy môi trường bàn giao được cố định ở Python 3.11; index Chroma phải dùng cùng provider/dimension giữa lúc index và query.

## Điều còn hạn chế

- Phần tôi làm chưa bổ sung corpus đa ngôn ngữ, dữ liệu trận đấu trực tiếp hoặc đánh giá thủ công bởi chuyên gia bóng đá.
- Nếu có thêm thời gian, ưu tiên mở rộng golden set với hard negatives/paraphrase tiếng Việt, chạy thêm embedding semantic production và thực hiện human evaluation để đối chiếu proxy offline.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình, phân biệt rõ phần do thành viên khác phát triển, và tôi có thể giải thích hoặc chạy lại quy trình trong buổi demo.

- Ngày: 20/09/2026
- Tên thành viên: Trần Cao Thắng
