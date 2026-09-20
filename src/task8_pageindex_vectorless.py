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
    """Upload tài liệu và lưu document IDs để tái sử dụng.

    Stub an toàn: nếu không có PAGEINDEX_API_KEY thì skip thay vì crash.
    Ghi chú để báo cáo: PageIndex là optional fallback, pipeline vẫn chạy hybrid.
    """
    key = os.getenv("PAGEINDEX_API_KEY", "")
    if not key:
        print("Skip PageIndex upload: PAGEINDEX_API_KEY not set.")
        return
    # TODO(optional): implement real upload khi có key + SDK.
    # Kiểm tra response thật của SDK thay vì đoán tên field.
    print("PAGEINDEX_API_KEY is set but real upload is not implemented; skipping.")
    return


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult (stub trả [] khi chưa cấu hình)."""
    import os as _os

    if not query or not query.strip() or top_k <= 0:
        return []
    if not _os.getenv("PAGEINDEX_API_KEY", ""):
        return []
    try:
        # TODO(optional): query PageIndex tại đây, parse nodes thành SearchResult
        # với retrieval_method="pageindex", score giảm dần theo rank nếu API thiếu score.
        return []
    except Exception as error:
        print(f"PageIndex error (non-fatal): {error}")
        return []


if __name__ == "__main__":
    upload_documents()
