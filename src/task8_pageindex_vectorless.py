"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    # TODO: Upload documents và lưu mapping source -> document ID.
    #
    # Nếu SDK không nhận Markdown, convert sang PDF tạm trước khi upload.
    # Kiểm tra response thật của SDK thay vì đoán tên field.
    raise NotImplementedError("Implement upload_documents")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult (safe no-op khi chưa cấu hình)."""
    # Chưa cấu hình PAGEINDEX_API_KEY -> trả rỗng để Task 9 dùng hybrid.
    # Giữ contract: không crash UI/pipeline khi provider lỗi/thiếu.
    if not os.getenv("PAGEINDEX_API_KEY", PAGEINDEX_API_KEY):
        return []
    # TODO: Query các document IDs và parse retrieved nodes.
    #
    # Mỗi result cần: id, content, score, metadata, retrieval_method.
    # Nếu API không trả score, có thể gán score giảm dần theo rank.
    return []


if __name__ == "__main__":
    upload_documents()
