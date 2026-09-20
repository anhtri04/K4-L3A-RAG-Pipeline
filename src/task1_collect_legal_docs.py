"""
Task 1 — Thu thập tài liệu chính sách/quy định (Football topic).

Nguồn: IFAB Laws of the Game các mùa giải (nguồn chính thức, public PDF).
Lưu file gốc vào data/landing/legal/.
"""

from pathlib import Path

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# IFAB official PDFs — verified HTTP 200 with curl (2026-09-20).
SOURCES = {
    "laws-of-the-game-2024-25.pdf": "https://downloads.theifab.com/downloads/laws-of-the-game-2024-25?l=en",
    "laws-of-the-game-2023-24.pdf": "https://downloads.theifab.com/downloads/laws-of-the-game-2023-24?l=en",
    "laws-of-the-game-2022-23.pdf": "https://downloads.theifab.com/downloads/laws-of-the-game-2022-23?l=en",
}


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents(sources: dict | None = None) -> None:
    """Tải ít nhất 3 PDF từ nguồn công khai (skip nếu đã có, >1KB)."""
    sources = sources or SOURCES
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (RAG-Lab)"})

    for filename, url in sources.items():
        dest = DATA_DIR / filename
        if dest.exists() and dest.stat().st_size > 1024:
            print(f"Skip (exists): {dest} ({dest.stat().st_size} bytes)")
            continue
        print(f"Downloading {filename} ...")
        response = session.get(url, timeout=60, allow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and not url.endswith(".pdf") and "l=en" not in url:
            print(f"Warning: unexpected content-type {content_type} for {filename}")
        dest.write_bytes(response.content)
        if dest.stat().st_size <= 1024:
            raise RuntimeError(f"Downloaded file too small: {dest}")
        print(f"Saved: {dest} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    setup_directory()
    download_documents()
