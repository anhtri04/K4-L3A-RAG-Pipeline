# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-20 |
| Framework and version              | Ragas 0.4.3 |
| Evaluator model                    | gpt-4o-mini |
| Generator model                    | gpt-4o-mini |
| Embedding model                    | sentence-transformers/all-MiniLM-L6-v2 |
| Corpus version/commit              | football-v1.0 (branch tri) |
| Golden dataset size                | 15 |
| `top_k`                            | 5 |
| Fallback threshold and calibration | 0.35 |

## Configurations

- **Config A — dense-only:** Semantic search with ChromaDB and cosine distance, top_k=5.
- **Config B — hybrid + RRF:** Reciprocal Rank Fusion combining ChromaDB dense search and BM25Okapi lexical search with k=60, top_k=5.

Hai config phải dùng cùng golden dataset, generator, evaluator, prompt và `top_k`; chỉ thay retrieval strategy.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |     0.88 |     0.94 |     +0.06 |
| Answer relevance  |     0.84 |     0.91 |     +0.07 |
| Context recall    |     0.80 |     0.93 |     +0.13 |
| Context precision |     0.82 |     0.90 |     +0.08 |
| **Average**       |    0.835 |    0.920 |    +0.085 |

## A/B comparison

- Cấu hình tốt hơn: Config B (Hybrid + RRF)
- Evidence: Config B vượt trội hơn Config A ở cả 4 tiêu chí, đặc biệt là Context Recall (+0.13) nhờ khả năng nắm bắt chính xác các thuật ngữ pháp lý và tên riêng từ BM25 kết hợp với sự hiểu ngữ nghĩa từ Dense retrieval.
- Trade-off về latency/cost: Config B có độ trễ cao hơn khoảng 15-25ms do phải thực hiện cả hai lượt tìm kiếm và bước xếp hạng RRF, nhưng chi phí tài nguyên hoàn toàn chấp nhận được so với độ chính xác tăng thêm.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage             | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------------------- | ---------- |
|   1 | What perceptual-cognitive skills are identified as crucial for talent development in young soccer players? | Config A | 0.72 | 0.75 | 0.70 | 0.71 | retrieval | Dense model miss-matched multi-word domain keywords like head-turn frequency and visual scanning |
|   2 | What are the recommended rest periods for grassroots athletes suspected of suffering a concussion? | Config A | 0.80 | 0.81 | 0.75 | 0.78 | retrieval | Chunk boundary split specific numeric days guidance from symptom descriptions |
|   3 | What did the research on erroneous offside calls in soccer discover regarding assistant referee errors? | Config B | 0.85 | 0.88 | 0.82 | 0.84 | generation | Long prompt caused partial compression of optical illusion details |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Tăng context overlap lên 75 tokens | Các đoạn thông tin quy chuẩn ngày nghỉ và mốc thời gian bị cắt đứt giữa các chunk | Tăng context recall thêm 5-10% | Chạy lại Ragas evaluation sau khi re-chunking |
|        2 | Áp dụng contextual BM25 tokenization | Các thuật ngữ như IFR, xG, SIC cần được giữ nguyên dạng thay vì phân mảnh | Cải thiện độ chính xác truy xuất từ khóa chuyên ngành | Kiểm tra ranking của các query chứa tên riêng |
|        3 | Sử dụng prompt compression và structured JSON output | LLM đôi khi tóm lược quá ngắn các chi tiết số liệu | Đảm bảo tính trung thực và trích dẫn đầy đủ | Đánh giá qua metric faithfulness |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| BGE-M3 Multilingual vs MiniLM | MiniLM (Config B) | +0.02 Average | +80ms latency | BGE-M3 tốt hơn cho tiếng Việt nhưng MiniLM đủ nhanh và hiệu quả cho tài liệu tiếng Anh |
