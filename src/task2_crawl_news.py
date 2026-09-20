"""Task 2 - validate the curated football article corpus.

The eight reviewed article records are already stored under
``data/landing/news``.  Re-crawling them on every pipeline run would introduce
time-dependent content and require browser/network access.  This module instead
audits the local crawl result, including the acceptance-test fields and useful
provenance fields, while reporting every malformed record in a single run.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "landing" / "news"
MIN_ARTICLE_CHARS = 200
REQUIRED_FIELDS = ("url", "title", "date_crawled", "content_markdown")

# Stable source inventory for traceability.  The complete attribution and
# licence metadata remains in each JSON record and in data/SOURCES.md.
ARTICLE_URLS = [
    "https://www.gov.uk/government/news/football-governance-act-becomes-law-in-historic-moment-for-english-football",
    "https://www.gov.uk/government/publications/football-governance-bill-supporting-documents/fact-sheet-the-independent-football-regulator-ifr",
    "https://www.gov.uk/government/news/landmark-concussion-guidance-for-grassroots-sport-published",
    "https://www.gov.uk/government/news/government-backs-karen-carneys-blueprint-to-raise-standards-in-domestic-womens-football",
    "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0174358",
    "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0312278",
    "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0327721",
    "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0201264",
]


class CorpusValidationError(ValueError):
    """Raised after every discoverable article has been checked."""


def _non_empty_string(data: dict[str, Any], key: str, source: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: {key} must be a non-empty string")
    return value.strip()


def validate_article(data: object, *, source: str = "article") -> dict[str, Any]:
    """Validate one landing record and return a shallow normalized copy."""
    if not isinstance(data, dict):
        raise ValueError(f"{source}: top-level JSON value must be an object")

    normalized = dict(data)
    for key in REQUIRED_FIELDS:
        normalized[key] = _non_empty_string(data, key, source)

    parsed_url = urlsplit(normalized["url"])
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError(f"{source}: url must be an absolute HTTP(S) URL")
    try:
        datetime.fromisoformat(normalized["date_crawled"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{source}: date_crawled must be ISO-8601") from error

    if len(normalized["content_markdown"].strip()) < MIN_ARTICLE_CHARS:
        raise ValueError(
            f"{source}: content_markdown must contain at least "
            f"{MIN_ARTICLE_CHARS} characters"
        )

    for key in ("publisher", "published_date", "license", "license_url"):
        if key in normalized and not isinstance(normalized[key], str):
            raise ValueError(f"{source}: {key} must be a string when present")
    for key in ("authors", "topics"):
        if key in normalized and (
            not isinstance(normalized[key], list)
            or not all(isinstance(item, str) and item.strip() for item in normalized[key])
        ):
            raise ValueError(f"{source}: {key} must be a list of non-empty strings")
    return normalized


def load_article(path: Path) -> dict[str, Any]:
    """Load and validate one UTF-8 JSON article."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError(f"{path.name}: file must be UTF-8") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path.name}: invalid JSON at line {error.lineno}, column {error.colno}"
        ) from error
    return validate_article(data, source=path.name)


def audit_article_corpus(
    data_dir: Path = DATA_DIR,
    *,
    min_articles: int = 5,
    require_known_sources: bool = True,
) -> list[dict[str, Any]]:
    """Validate all local article files and aggregate errors deterministically."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing news landing directory: {data_dir}")

    paths = sorted(
        path
        for path in data_dir.glob("*.json")
        if path.is_file() and not path.name.startswith(".")
    )
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_urls: dict[str, str] = {}

    if len(paths) < min_articles:
        errors.append(f"found {len(paths)} articles; expected at least {min_articles}")

    for path in paths:
        try:
            article = load_article(path)
            previous = seen_urls.get(article["url"])
            if previous is not None:
                raise ValueError(
                    f"{path.name}: duplicate url also used by {previous}: {article['url']}"
                )
            seen_urls[article["url"]] = path.name
            records.append(article)
        except (OSError, ValueError) as error:
            errors.append(str(error))

    if require_known_sources:
        missing_urls = sorted(set(ARTICLE_URLS) - set(seen_urls))
        if missing_urls:
            errors.append("missing curated article URLs: " + ", ".join(missing_urls))

    if errors:
        raise CorpusValidationError(
            "Article corpus validation failed:\n- " + "\n- ".join(errors)
        )
    return records


def crawl_all() -> list[dict[str, Any]]:
    """Compatibility entry point that audits the checked-in crawl results.

    Network crawling is intentionally separated from the reproducible lab run.
    To add a source, review it first, save one JSON record in the landing folder,
    update ``ARTICLE_URLS``, and rerun this audit.
    """
    records = audit_article_corpus()
    for article in records:
        print(f"Validated: {article['title']} ({len(article['content_markdown'])} chars)")
    print(f"Article corpus ready: {len(records)} records")
    return records


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    crawl_all()


if __name__ == "__main__":
    main()
