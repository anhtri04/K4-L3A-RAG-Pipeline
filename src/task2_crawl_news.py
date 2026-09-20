"""
Task 2 — Crawl bài viết/thông báo bóng đá.

Chủ đề: Bài viết, thông báo chính phủ và nghiên cứu khoa học về bóng đá.
Nguồn: UK Government (gov.uk) và PLOS ONE.
"""

import asyncio
from datetime import datetime
import json
from pathlib import Path
import re
import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_SOURCES = [
    {
        "filename": "01_football_governance_act_2025.json",
        "url": "https://www.gov.uk/government/news/football-governance-act-becomes-law-in-historic-moment-for-english-football",
    },
    {
        "filename": "02_independent_football_regulator_fact_sheet_2024.json",
        "url": "https://www.gov.uk/government/publications/football-governance-bill-supporting-documents/fact-sheet-the-independent-football-regulator-ifr",
    },
    {
        "filename": "03_grassroots_sport_concussion_guidance_2023.json",
        "url": "https://www.gov.uk/government/news/landmark-concussion-guidance-for-grassroots-sport-published",
    },
    {
        "filename": "04_womens_football_standards_blueprint_2023.json",
        "url": "https://www.gov.uk/government/news/government-backs-karen-carneys-blueprint-to-raise-standards-in-domestic-womens-football",
    },
    {
        "filename": "05_offside_calls_research_2017.json",
        "url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0174358",
    },
    {
        "filename": "06_expected_goals_models_2024.json",
        "url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0312278",
    },
    {
        "filename": "07_soccer_talent_development_review_2025.json",
        "url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0327721",
    },
    {
        "filename": "08_soccer_injury_forecasting_ml_2018.json",
        "url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0201264",
    },
]

ARTICLE_URLS = [item["url"] for item in ARTICLE_SOURCES]


def _find_cached_article(url: str) -> dict | None:
    """Tìm article đã có trong landing/news theo URL."""
    if not DATA_DIR.exists():
        return None
    for path in DATA_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("url") == url and data.get("content_markdown"):
                return data
        except Exception:
            continue
    return None


async def crawl_article(url: str) -> dict:
    """Crawl nội dung bài viết và trả về dict đúng schema."""
    cached = _find_cached_article(url)
    if cached:
        return cached

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    html = response.text

    title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else "Football News Article"
    title = re.sub(r"\s*-\s*GOV\.UK|\s*\|\s*PLOS ONE", "", title).strip()

    body_match = re.search(r"<article[^>]*>(.*?)</article>", html, re.IGNORECASE | re.DOTALL)
    raw_content = body_match.group(1) if body_match else html
    clean_text = re.sub(r"<script.*?</script>", "", raw_content, flags=re.IGNORECASE | re.DOTALL)
    clean_text = re.sub(r"<style.*?</style>", "", clean_text, flags=re.IGNORECASE | re.DOTALL)
    clean_text = re.sub(r"<[^>]+>", " ", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": f"# {title}\n\n{clean_text}",
    }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành file JSON đúng tên theo SOURCES.md."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for item in ARTICLE_SOURCES:
        url = item["url"]
        filename = item["filename"]
        try:
            article = await crawl_article(url)
            output = DATA_DIR / filename
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {filename} — {article.get('title', '')[:50]}")
        except Exception as error:
            print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
