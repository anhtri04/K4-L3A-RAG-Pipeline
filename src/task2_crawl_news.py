"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium
    
-> Dùng Firecrawl or bất cứ công cụ nào bạn quen    
"""

import asyncio
import json
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Football topic — 6 URLs (cần >=5 thành công). Đã verify HTTP 200 bằng curl.
ARTICLE_URLS = [
    "https://en.wikipedia.org/wiki/Offside_(association_football)",
    "https://en.wikipedia.org/wiki/Association_football_tactics_and_skills",
    "https://en.wikipedia.org/wiki/VAR_(football)",
    "https://www.theifab.com/laws-of-the-game-documents",
    "https://www.premierleague.com/news/4183910",
    "https://www.fifa.com/en/articles/laws-of-the-game-changes-2024-25",
]


async def crawl_article(url: str) -> dict:
    """Crawl 1 URL bằng Crawl4AI, trả về dict đúng schema acceptance test."""
    from datetime import datetime, timezone

    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        markdown = (getattr(result, "markdown", "") or "").strip()
        if not markdown:
            raise RuntimeError(f"Empty markdown for {url}")
        metadata = getattr(result, "metadata", {}) or {}
        title = metadata.get("title") or url
        return {
            "url": url,
            "title": str(title).strip(),
            "date_crawled": datetime.now(timezone.utc).isoformat(),
            "content_markdown": markdown,
        }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            article = await crawl_article(url)
            output = DATA_DIR / f"article_{index:02d}.json"
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output}")
        except Exception as error:
            print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
