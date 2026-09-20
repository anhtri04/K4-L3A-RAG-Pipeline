"""Task 3 - convert the local football corpus to provenance-rich Markdown.

Conversion is offline, deterministic, and idempotent.  Every generated file
contains YAML front matter with the original filename, URL/licence metadata, and
a SHA-256 digest of the landing source.  PDF extraction prefers ``pypdf`` for
page-aware output and falls back to the project's ``markitdown[pdf]`` dependency.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.task1_collect_legal_docs import (
    LEGAL_DOCUMENTS,
    SUPPORTED_EXTENSIONS,
    sha256_file,
)
from src.task2_crawl_news import load_article


LANDING_DIR = Path(__file__).resolve().parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "standardized"
MIN_CONTENT_CHARS = 200


class ConversionError(RuntimeError):
    """Raised after all convertible inputs have been attempted."""


def _clean_text(text: str) -> str:
    """Normalize line endings and noisy trailing whitespace reproducibly."""
    text = text.replace("\ufeff", "").replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n")).strip()
    return re.sub(r"\n{4,}", "\n\n\n", text)


def _one_line(value: object, fallback: str) -> str:
    text = " ".join(str(value or "").split())
    return text or fallback


def _front_matter(metadata: dict[str, Any]) -> str:
    """Render JSON-compatible values as deterministic YAML front matter."""
    lines = ["---"]
    for key, value in metadata.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", key):
            rendered_key = json.dumps(key, ensure_ascii=False)
        else:
            rendered_key = key
        rendered_value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        lines.append(f"{rendered_key}: {rendered_value}")
    lines.append("---")
    return "\n".join(lines)


def _ensure_title(body: str, title: str) -> str:
    """Add a useful H1 only when the extracted body does not already have one."""
    for line in body.splitlines():
        if not line.strip():
            continue
        if line.startswith("# "):
            return body
        break
    return f"# {title}\n\n{body}"


def _render_document(metadata: dict[str, Any], body: str) -> str:
    cleaned = _clean_text(body)
    if len(cleaned) < MIN_CONTENT_CHARS:
        raise ValueError(
            f"extracted content is too short ({len(cleaned)} characters; "
            f"expected at least {MIN_CONTENT_CHARS})"
        )
    title = _one_line(metadata.get("title"), "Untitled document")
    cleaned = _ensure_title(cleaned, title)
    return f"{_front_matter(metadata)}\n\n{cleaned}\n"


def _write_if_changed(path: Path, content: str) -> bool:
    """Atomically replace an output only when its exact UTF-8 bytes changed."""
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)
    return True


def _extract_pdf_with_pypdf(path: Path) -> tuple[str, dict[str, Any]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if reader.is_encrypted and reader.decrypt("") == 0:
        raise ValueError("encrypted PDF cannot be opened without a password")

    sections: list[str] = []
    text_pages = 0
    for index, page in enumerate(reader.pages, 1):
        text = _clean_text(page.extract_text() or "")
        if not text:
            continue
        text_pages += 1
        sections.append(f"## Page {index}\n\n{text}")
    return "\n\n".join(sections), {
        "page_count": len(reader.pages),
        "text_page_count": text_pages,
        "extraction_method": "pypdf",
    }


def _extract_with_markitdown(path: Path) -> tuple[str, dict[str, Any]]:
    from markitdown import MarkItDown

    result = MarkItDown().convert(str(path))
    return _clean_text(result.text_content or ""), {
        "extraction_method": "markitdown"
    }


def _extract_legal_file(path: Path) -> tuple[str, dict[str, Any]]:
    """Extract a legal source locally and retain actionable backend errors."""
    errors: list[str] = []
    if path.suffix.lower() == ".pdf":
        try:
            text, details = _extract_pdf_with_pypdf(path)
            if len(text.strip()) >= MIN_CONTENT_CHARS:
                return text, details
            errors.append(f"pypdf returned only {len(text.strip())} characters")
        except (ImportError, OSError, ValueError) as error:
            errors.append(f"pypdf: {error}")
        except Exception as error:  # malformed PDF internals vary by parser version
            errors.append(f"pypdf: {type(error).__name__}: {error}")

    try:
        text, details = _extract_with_markitdown(path)
        if len(text.strip()) >= MIN_CONTENT_CHARS:
            return text, details
        errors.append(f"markitdown returned only {len(text.strip())} characters")
    except (ImportError, OSError, ValueError) as error:
        errors.append(f"markitdown: {error}")
    except Exception as error:  # third-party converter exceptions are not stable
        errors.append(f"markitdown: {type(error).__name__}: {error}")

    raise ConversionError(
        f"could not extract {path.name}; " + "; ".join(errors)
    )


def _legal_metadata(path: Path, extraction: dict[str, Any]) -> dict[str, Any]:
    known = dict(LEGAL_DOCUMENTS.get(path.name, {}))
    title = _one_line(known.pop("title", None), path.stem.replace("_", " ").title())
    metadata: dict[str, Any] = {
        "source_file": path.name,
        "doc_type": "legal",
        "title": title,
        "url": known.pop("url", None),
        **known,
        "source_sha256": sha256_file(path),
        **extraction,
    }
    return metadata


def _news_metadata(path: Path, article: dict[str, Any]) -> dict[str, Any]:
    preferred_order = (
        "title",
        "url",
        "publisher",
        "authors",
        "published_date",
        "date_crawled",
        "language",
        "source_type",
        "topics",
        "license",
        "license_url",
        "attribution",
        "changes",
    )
    metadata: dict[str, Any] = {
        "source_file": path.name,
        "doc_type": "news",
    }
    for key in preferred_order:
        if key in article:
            metadata[key] = article[key]
    for key in sorted(set(article) - set(preferred_order) - {"content_markdown"}):
        metadata[key] = article[key]
    metadata["source_sha256"] = sha256_file(path)
    return metadata


def _legal_inputs(legal_dir: Path) -> list[Path]:
    if not legal_dir.is_dir():
        raise FileNotFoundError(f"Missing legal landing directory: {legal_dir}")
    return sorted(
        path
        for path in legal_dir.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _remove_stale_markdown(output_dir: Path, outputs: list[Path]) -> None:
    """Remove generated Markdown whose landing source no longer exists.

    Cleanup only runs after an error-free conversion, so a malformed or
    temporarily unreadable input cannot cause its last good output to vanish.
    """
    expected = {path.resolve() for path in outputs}
    for path in sorted(output_dir.glob("*.md")):
        if path.is_file() and path.resolve() not in expected:
            path.unlink()
            print(f"Removed stale output: {path}")


def convert_legal_docs(
    legal_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    strict: bool = True,
) -> list[Path]:
    """Convert every legal landing document and continue past bad inputs."""
    legal_dir = legal_dir or LANDING_DIR / "legal"
    output_dir = output_dir or OUTPUT_DIR / "legal"
    inputs = _legal_inputs(legal_dir)
    if not inputs:
        raise FileNotFoundError(f"No legal documents in {legal_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    errors: list[str] = []
    output_names: set[str] = set()
    for path in inputs:
        destination = output_dir / f"{path.stem}.md"
        if destination.name in output_names:
            errors.append(
                f"{path.name}: output name collides with another source: "
                f"{destination.name}"
            )
            continue
        output_names.add(destination.name)
        try:
            body, extraction = _extract_legal_file(path)
            content = _render_document(_legal_metadata(path, extraction), body)
            changed = _write_if_changed(destination, content)
            outputs.append(destination)
            status = "Saved" if changed else "Unchanged"
            print(f"{status}: {destination} ({len(content)} chars)")
        except (OSError, ValueError, ConversionError) as error:
            errors.append(f"{path.name}: {error}")

    if errors:
        message = "Legal conversion completed with errors:\n- " + "\n- ".join(errors)
        if strict:
            raise ConversionError(message)
        print(message)
    else:
        _remove_stale_markdown(output_dir, outputs)
    return outputs


def convert_news_articles(
    news_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    strict: bool = True,
) -> list[Path]:
    """Convert every valid article JSON while preserving all provenance fields."""
    news_dir = news_dir or LANDING_DIR / "news"
    output_dir = output_dir or OUTPUT_DIR / "news"
    if not news_dir.is_dir():
        raise FileNotFoundError(f"Missing news landing directory: {news_dir}")
    inputs = sorted(
        path
        for path in news_dir.glob("*.json")
        if path.is_file() and not path.name.startswith(".")
    )
    if not inputs:
        raise FileNotFoundError(f"No news JSON files in {news_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    errors: list[str] = []
    for path in inputs:
        destination = output_dir / f"{path.stem}.md"
        try:
            article = load_article(path)
            body = str(article["content_markdown"])
            content = _render_document(_news_metadata(path, article), body)
            changed = _write_if_changed(destination, content)
            outputs.append(destination)
            status = "Saved" if changed else "Unchanged"
            print(f"{status}: {destination} ({len(content)} chars)")
        except (OSError, ValueError) as error:
            errors.append(f"{path.name}: {error}")

    if errors:
        message = "News conversion completed with errors:\n- " + "\n- ".join(errors)
        if strict:
            raise ConversionError(message)
        print(message)
    else:
        _remove_stale_markdown(output_dir, outputs)
    return outputs


def convert_all(*, strict: bool = True) -> dict[str, list[Path]]:
    """Convert the complete checked-in corpus without network access."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    converted: dict[str, list[Path]] = {"legal": [], "news": []}
    try:
        converted["legal"] = convert_legal_docs(strict=strict)
    except (OSError, ValueError, ConversionError) as error:
        errors.append(str(error))
    try:
        converted["news"] = convert_news_articles(strict=strict)
    except (OSError, ValueError, ConversionError) as error:
        errors.append(str(error))

    if errors and strict:
        raise ConversionError("Corpus conversion failed:\n" + "\n".join(errors))
    print(
        f"Standardized {len(converted['legal'])} legal documents and "
        f"{len(converted['news'])} news articles in {OUTPUT_DIR}"
    )
    return converted


if __name__ == "__main__":
    convert_all()
