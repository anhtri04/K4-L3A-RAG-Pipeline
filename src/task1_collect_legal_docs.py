"""Task 1 - validate the curated football policy corpus.

The source documents are intentionally committed in ``data/landing/legal``.
Running this module is therefore deterministic and performs no network access:
it verifies that the expected files are present, non-trivial, and look like the
declared document type.  Source metadata is kept here as structured data so the
Markdown conversion step can preserve provenance.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "landing" / "legal"
MIN_DOCUMENT_BYTES = 1024
SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx"}

# These records mirror data/SOURCES.md and are consumed by Task 3.  Keeping the
# URL and licence beside the filename prevents provenance from being lost when
# the binary source is converted to Markdown.
LEGAL_DOCUMENTS: dict[str, dict[str, Any]] = {
    "football_governance_significant_influence_control_guidance_2025.pdf": {
        "title": (
            "Meaning of Significant Influence or Control in the Context of "
            "the Football Governance Act 2025"
        ),
        "url": (
            "https://assets.publishing.service.gov.uk/media/693315275b5198836f30415a/"
            "E03461288_Statutory_Guidance_About_the_Meaning_of__Significant_"
            "Influence_or_Control__final_accessible_December_2025__1_.pdf"
        ),
        "publisher": "UK Department for Culture, Media and Sport",
        "published_date": "2025-12",
        "retrieved_date": "2026-09-20",
        "license": "Open Government Licence v3.0",
        "license_url": (
            "https://www.nationalarchives.gov.uk/doc/open-government-licence/"
            "version/3/"
        ),
        "attribution": (
            "Contains public sector information licensed under the Open "
            "Government Licence v3.0."
        ),
    },
    "fan_led_review_football_governance_2021.pdf": {
        "title": "Fan-Led Review of Football Governance",
        "url": (
            "https://assets.publishing.service.gov.uk/media/"
            "63e4d010d3bf7f05b871200d/"
            "Football_Fan_led_Governance_Review_v8Web_Accessible.pdf"
        ),
        "publisher": "UK Department for Digital, Culture, Media and Sport",
        "published_date": "2021-11-24",
        "retrieved_date": "2026-09-20",
        "license": "Open Government Licence v3.0",
        "license_url": (
            "https://www.nationalarchives.gov.uk/doc/open-government-licence/"
            "version/3/"
        ),
        "attribution": (
            "Contains public sector information licensed under the Open "
            "Government Licence v3.0."
        ),
    },
    "club_football_governance_white_paper_2023.pdf": {
        "title": "A Sustainable Future: Reforming Club Football Governance",
        "url": (
            "https://assets.publishing.service.gov.uk/media/"
            "63f65d3de90e077bb0c92853/"
            "Reform_of_club_football_governance_-_White_Paper.pdf"
        ),
        "publisher": "UK Department for Culture, Media and Sport",
        "published_date": "2023-02-23",
        "retrieved_date": "2026-09-20",
        "license": "Open Government Licence v3.0",
        "license_url": (
            "https://www.nationalarchives.gov.uk/doc/open-government-licence/"
            "version/3/"
        ),
        "attribution": (
            "Contains public sector information licensed under the Open "
            "Government Licence v3.0."
        ),
    },
    "government_response_womens_football_2023.pdf": {
        "title": "Government Response to the Independent Review of Women's Football",
        "url": (
            "https://assets.publishing.service.gov.uk/media/"
            "656a1e710f12ef070e3e0104/"
            "Government_response_to_independent_review_-_reframing_the_"
            "opportunity_in_women_s_football.pdf"
        ),
        "publisher": "UK Department for Culture, Media and Sport",
        "published_date": "2023-12-04",
        "retrieved_date": "2026-09-20",
        "license": "Open Government Licence v3.0",
        "license_url": (
            "https://www.nationalarchives.gov.uk/doc/open-government-licence/"
            "version/3/"
        ),
        "attribution": (
            "Contains public sector information licensed under the Open "
            "Government Licence v3.0."
        ),
    },
}

# Backward-compatible public name used by the previous implementation.
SOURCES = {name: metadata["url"] for name, metadata in LEGAL_DOCUMENTS.items()}


class CorpusValidationError(ValueError):
    """Raised after every discoverable source has been checked."""


def setup_directory(data_dir: Path = DATA_DIR) -> None:
    """Create the landing directory when bootstrapping a fresh checkout."""
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {data_dir}")


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading a large PDF in RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_legal_document(
    path: Path, *, min_bytes: int = MIN_DOCUMENT_BYTES
) -> dict[str, Any]:
    """Validate one local PDF/DOC/DOCX and return stable inventory metadata."""
    if not path.is_file():
        raise ValueError(f"not a file: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported document type: {path.name}")

    size = path.stat().st_size
    if size <= min_bytes:
        raise ValueError(
            f"document is too small ({size} bytes, expected > {min_bytes}): {path.name}"
        )

    with path.open("rb") as source:
        prefix = source.read(8)
        if suffix == ".pdf":
            source.seek(max(size - 4096, 0))
            suffix_bytes = source.read()
        else:
            suffix_bytes = b""

    if suffix == ".pdf" and (
        not prefix.startswith(b"%PDF-") or b"%%EOF" not in suffix_bytes
    ):
        raise ValueError(f"invalid or truncated PDF signature: {path.name}")
    if suffix == ".docx" and not prefix.startswith(b"PK"):
        raise ValueError(f"invalid DOCX container signature: {path.name}")
    if suffix == ".doc" and not prefix.startswith(bytes.fromhex("D0CF11E0")):
        raise ValueError(f"invalid legacy DOC signature: {path.name}")

    return {
        "filename": path.name,
        "size_bytes": size,
        "sha256": sha256_file(path),
        "metadata": dict(LEGAL_DOCUMENTS.get(path.name, {})),
    }


def audit_legal_corpus(
    data_dir: Path = DATA_DIR,
    *,
    min_documents: int = 3,
    require_known_sources: bool = True,
) -> list[dict[str, Any]]:
    """Validate all local sources and report all problems in one exception."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing legal landing directory: {data_dir}")

    paths = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    errors: list[str] = []
    records: list[dict[str, Any]] = []

    if len(paths) < min_documents:
        errors.append(
            f"found {len(paths)} legal documents; expected at least {min_documents}"
        )
    if require_known_sources:
        missing = sorted(set(LEGAL_DOCUMENTS) - {path.name for path in paths})
        if missing:
            errors.append("missing curated sources: " + ", ".join(missing))

    for path in paths:
        try:
            records.append(validate_legal_document(path))
        except (OSError, ValueError) as error:
            errors.append(f"{path.name}: {error}")

    if errors:
        raise CorpusValidationError("Legal corpus validation failed:\n- " + "\n- ".join(errors))
    return records


def download_documents(sources: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Compatibility entry point: validate the vendored corpus without a download.

    The lab repository already contains the licensed source files.  Silently
    downloading replacements would make runs non-reproducible, so custom remote
    source maps are rejected and the checked-in corpus is audited instead.
    """
    if sources is not None and sources != SOURCES:
        raise ValueError(
            "Remote downloads are disabled; place reviewed sources in "
            "data/landing/legal and update LEGAL_DOCUMENTS explicitly."
        )
    return audit_legal_corpus()


def main() -> None:
    setup_directory()
    records = download_documents()
    for record in records:
        print(
            f"Validated: {record['filename']} "
            f"({record['size_bytes']} bytes, sha256={record['sha256'][:12]}...)"
        )
    print(f"Legal corpus ready: {len(records)} documents")


if __name__ == "__main__":
    main()
