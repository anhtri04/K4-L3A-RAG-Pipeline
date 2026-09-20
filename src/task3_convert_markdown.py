"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Hướng dẫn:
    1. Dùng MarkItDown để convert PDF/DOCX.
    2. Đọc JSON và giữ metadata ở đầu file Markdown.
    3. Giữ cấu trúc thư mục legal/ và news/.
    4. Không tạo file rỗng hoặc file trùng khi chạy lại.
"""

import json
from pathlib import Path
from markitdown import MarkItDown


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs() -> None:
    """Convert PDF/DOCX vào standardized/legal."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = MarkItDown()
    for path in sorted(legal_dir.iterdir()):
        if path.suffix.lower() in {".pdf", ".doc", ".docx"} and not path.name.startswith("."):
            target_file = output_dir / f"{path.stem}.md"
            print(f"Converting legal doc: {path.name} -> {target_file.name}...")
            result = converter.convert(str(path))
            content = result.text_content.strip()

            title = path.stem.replace("_", " ").title()
            header = (
                f"# {title}\n\n"
                f"**Source:** {path.name}\n"
                f"**Document Type:** legal\n\n"
                f"---\n\n"
            )
            target_file.write_text(header + content, encoding="utf-8")
            print(f"Saved: {target_file.name} ({len(content):,} chars)")


def convert_news_articles() -> None:
    """Convert JSON vào standardized/news."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(news_dir.glob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            target_file = output_dir / f"{path.stem}.md"
            title = data.get("title", path.stem)
            url = data.get("url", "")
            date_crawled = data.get("date_crawled", "")
            body = data.get("content_markdown", "").strip()

            header = (
                f"# {title}\n\n"
                f"**Source:** {url}\n"
                f"**Crawled:** {date_crawled}\n"
                f"**Document Type:** news\n\n"
                f"---\n\n"
            )
            target_file.write_text(header + body, encoding="utf-8")
            print(f"Saved: {target_file.name} ({len(body):,} chars)")
        except Exception as err:
            print(f"Failed to convert {path.name}: {err}")


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
