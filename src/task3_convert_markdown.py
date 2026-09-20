"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Hướng dẫn:
    1. Dùng MarkItDown để convert PDF/DOCX.
    2. Đọc JSON và giữ metadata ở đầu file Markdown.
    3. Giữ cấu trúc thư mục legal/ và news/.
    4. Không tạo file rỗng hoặc file trùng khi chạy lại.

Cài đặt:
    Dependency MarkItDown đã được khai báo trong pyproject.toml.
    
-> Hoặc dùng công cụ nào bạn quen khác Markitdown
"""

from pathlib import Path


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs() -> None:
    """Convert PDF/DOCX trong landing/legal sang standardized/legal (.md)."""
    from markitdown import MarkItDown

    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = MarkItDown()

    inputs = sorted(
        p for p in legal_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".pdf", ".doc", ".docx"}
        and not p.name.startswith(".")
    )
    if not inputs:
        raise FileNotFoundError(f"No legal docs in {legal_dir}")
    for path in inputs:
        dest = output_dir / f"{path.stem}.md"
        result = converter.convert(str(path))
        text = (result.text_content or "").strip()
        if len(text) < 200:
            print(f"Warning: {path.name} converted too short ({len(text)} chars), skip")
            continue
        header = f"# {path.stem}\n\n**Source:** {path.name}\n\n---\n\n"
        dest.write_text(header + text, encoding="utf-8")
        print(f"Saved: {dest} ({len(text)} chars)")


def convert_news_articles() -> None:
    """Convert JSON trong landing/news sang standardized/news (.md)."""
    import json

    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(news_dir.glob("*.json"))
    if not inputs:
        raise FileNotFoundError(f"No news JSON in {news_dir}")
    for path in inputs:
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("url", "title", "date_crawled", "content_markdown"):
            if not str(data.get(key, "")).strip():
                raise ValueError(f"{path.name} missing {key}")
        header = (
            f"# {data['title']}\n\n"
            f"**Source:** {data['url']}\n\n"
            f"**Crawled:** {data['date_crawled']}\n\n---\n\n"
        )
        body = header + data["content_markdown"].strip() + "\n"
        if len(body.strip()) < 200:
            print(f"Warning: {path.name} body too short, skip")
            continue
        dest = output_dir / f"{path.stem}.md"
        dest.write_text(body, encoding="utf-8")
        print(f"Saved: {dest}")


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
