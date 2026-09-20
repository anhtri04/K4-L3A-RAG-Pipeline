import json
import os
from pathlib import Path

import pytest

from src import task1_collect_legal_docs as task1
from src import task2_crawl_news as task2
from src import task3_convert_markdown as task3


ROOT = Path(__file__).resolve().parent.parent


def fake_pdf_bytes() -> bytes:
    return b"%PDF-1.4\n" + (b"valid-local-content\n" * 80) + b"%%EOF\n"


def article(**overrides):
    record = {
        "url": "https://example.org/football/article",
        "title": "Football research article",
        "date_crawled": "2026-09-20T15:13:04+07:00",
        "publisher": "Example Publisher",
        "authors": ["A. Researcher"],
        "published_date": "2026-01-01",
        "topics": ["football"],
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "content_markdown": "# Football research\n\n" + ("Evidence about football. " * 20),
    }
    record.update(overrides)
    return record


def test_task1_audits_local_files_without_network(tmp_path):
    source = tmp_path / "policy.pdf"
    source.write_bytes(fake_pdf_bytes())

    records = task1.audit_legal_corpus(
        tmp_path, min_documents=1, require_known_sources=False
    )

    assert [record["filename"] for record in records] == ["policy.pdf"]
    assert records[0]["size_bytes"] == source.stat().st_size
    assert len(records[0]["sha256"]) == 64


def test_task1_rejects_truncated_pdf(tmp_path):
    source = tmp_path / "broken.pdf"
    source.write_bytes(b"%PDF-1.4\n" + (b"x" * 2048))

    with pytest.raises(task1.CorpusValidationError, match="truncated PDF"):
        task1.audit_legal_corpus(
            tmp_path, min_documents=1, require_known_sources=False
        )


def test_task2_reports_all_bad_records_and_duplicate_urls(tmp_path):
    (tmp_path / "01_valid.json").write_text(
        json.dumps(article()), encoding="utf-8"
    )
    (tmp_path / "02_duplicate.json").write_text(
        json.dumps(article(title="Duplicate URL")), encoding="utf-8"
    )
    (tmp_path / "03_malformed.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(task2.CorpusValidationError) as caught:
        task2.audit_article_corpus(
            tmp_path, min_articles=1, require_known_sources=False
        )

    message = str(caught.value)
    assert "duplicate url" in message
    assert "invalid JSON" in message


def test_news_conversion_preserves_metadata_and_is_idempotent(tmp_path):
    landing = tmp_path / "landing"
    output = tmp_path / "standardized"
    landing.mkdir()
    source = landing / "article.json"
    source.write_text(
        json.dumps(article(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    outputs = task3.convert_news_articles(landing, output)
    generated = outputs[0]
    first_content = generated.read_text(encoding="utf-8")
    os.utime(generated, (123, 123))

    task3.convert_news_articles(landing, output)

    assert generated.stat().st_mtime_ns == 123_000_000_000
    assert generated.read_text(encoding="utf-8") == first_content
    assert 'source_file: "article.json"' in first_content
    assert 'doc_type: "news"' in first_content
    assert 'url: "https://example.org/football/article"' in first_content
    assert 'license: "CC BY 4.0"' in first_content
    assert "source_sha256:" in first_content


def test_news_conversion_continues_before_reporting_malformed_file(tmp_path):
    landing = tmp_path / "landing"
    output = tmp_path / "standardized"
    landing.mkdir()
    (landing / "01_valid.json").write_text(
        json.dumps(article()), encoding="utf-8"
    )
    (landing / "02_bad.json").write_text("[]", encoding="utf-8")

    with pytest.raises(task3.ConversionError, match="02_bad.json"):
        task3.convert_news_articles(landing, output)

    assert (output / "01_valid.md").is_file()
    assert not (output / "02_bad.md").exists()


def test_legal_conversion_includes_source_hash_and_extraction_details(
    tmp_path, monkeypatch
):
    landing = tmp_path / "landing"
    output = tmp_path / "standardized"
    landing.mkdir()
    source = landing / "policy.pdf"
    source.write_bytes(fake_pdf_bytes())
    monkeypatch.setattr(
        task3,
        "_extract_legal_file",
        lambda path: (
            "Local policy evidence. " * 20,
            {"page_count": 2, "extraction_method": "test"},
        ),
    )

    outputs = task3.convert_legal_docs(landing, output)
    content = outputs[0].read_text(encoding="utf-8")

    assert 'source_file: "policy.pdf"' in content
    assert 'doc_type: "legal"' in content
    assert "url: null" in content
    assert "page_count: 2" in content
    assert 'extraction_method: "test"' in content
    assert "# Policy" in content


def test_checked_in_standardized_corpus_matches_every_landing_source():
    landing = ROOT / "data" / "landing"
    standardized = ROOT / "data" / "standardized"
    legal_sources = sorted(
        path
        for path in (landing / "legal").iterdir()
        if path.suffix.lower() in task1.SUPPORTED_EXTENSIONS
    )
    news_sources = sorted((landing / "news").glob("*.json"))

    assert len(legal_sources) == 4
    assert len(news_sources) == 8
    for doc_type, sources in (("legal", legal_sources), ("news", news_sources)):
        for source in sources:
            output = standardized / doc_type / f"{source.stem}.md"
            assert output.is_file(), f"missing standardized output for {source.name}"
            content = output.read_text(encoding="utf-8")
            assert len(content.strip()) >= task3.MIN_CONTENT_CHARS
            assert f'source_file: "{source.name}"' in content
            assert f'doc_type: "{doc_type}"' in content
            assert "source_sha256:" in content
