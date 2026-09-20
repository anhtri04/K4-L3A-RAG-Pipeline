"""Task 8 — PageIndex vectorless fallback.

PageIndex is optional.  The hybrid retriever remains usable when the API key is
missing, a document is still being indexed, or the remote service fails.

The version pinned by this project accepts PDF documents and exposes an
asynchronous retrieval API.  Run this module once to upload the legal PDFs and
cache their document IDs locally before enabling the fallback in the UI.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any, Iterator

from dotenv import load_dotenv

from .task1_collect_legal_docs import LEGAL_DOCUMENTS


load_dotenv()

LOGGER = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = ROOT_DIR / "data" / "standardized"
LEGAL_PDF_DIR = ROOT_DIR / "data" / "landing" / "legal"
PAGEINDEX_CACHE_PATH = ROOT_DIR / "pageindex_doc_ids.json"

DEFAULT_RETRIEVAL_TIMEOUT = 30.0
DEFAULT_POLL_INTERVAL = 0.5
DEFAULT_HTTP_TIMEOUT = 15.0


def _positive_env_float(name: str, default: float) -> float:
    """Read a positive finite float without making configuration fatal."""
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        return default
    if value <= 0 or value == float("inf") or value != value:
        return default
    return value


def _make_client(api_key: str):
    """Construct the pinned PageIndex cloud client lazily.

    Keeping the import and construction behind this seam makes the optional
    provider straightforward to mock in offline tests.
    """
    from pageindex import PageIndexClient

    return PageIndexClient(api_key=api_key)


def _call_with_timeout(function: Any, timeout: float, *args: Any, **kwargs: Any) -> Any:
    """Bound SDK calls because PageIndex 0.2.8 does not expose HTTP timeouts."""
    responses: Queue[tuple[bool, Any]] = Queue(maxsize=1)

    def run() -> None:
        try:
            responses.put((True, function(*args, **kwargs)))
        except BaseException as error:  # propagate the provider's original error
            responses.put((False, error))

    worker = Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"PageIndex request exceeded {timeout:.1f}s")
    try:
        succeeded, payload = responses.get_nowait()
    except Empty as error:
        raise RuntimeError("PageIndex request ended without a response") from error
    if not succeeded:
        raise payload
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_source(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return path.name


def _load_cache() -> dict[str, dict[str, Any]]:
    """Load the cache while accepting the early ``source -> doc_id`` format."""
    if not PAGEINDEX_CACHE_PATH.is_file():
        return {}
    try:
        payload = json.loads(PAGEINDEX_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        LOGGER.warning("Ignoring an unreadable PageIndex cache at %s", PAGEINDEX_CACHE_PATH)
        return {}

    if not isinstance(payload, dict):
        return {}
    raw_documents = payload.get("documents", payload)
    if not isinstance(raw_documents, dict):
        return {}

    documents: dict[str, dict[str, Any]] = {}
    for source, entry in raw_documents.items():
        if not isinstance(source, str) or not source.strip():
            continue
        if isinstance(entry, str):
            entry = {"doc_id": entry}
        if not isinstance(entry, dict):
            continue
        doc_id = entry.get("doc_id") or entry.get("id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            continue
        documents[source] = {
            **entry,
            "doc_id": doc_id.strip(),
            "source": str(entry.get("source") or Path(source).name),
            "title": str(entry.get("title") or Path(source).stem.replace("_", " ")),
            "doc_type": str(entry.get("doc_type") or "legal"),
            "url": entry.get("url") if isinstance(entry.get("url"), str) else None,
        }
    return documents


def _write_cache(documents: dict[str, dict[str, Any]]) -> None:
    PAGEINDEX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "documents": documents}
    temporary_path = PAGEINDEX_CACHE_PATH.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(PAGEINDEX_CACHE_PATH)


def _discover_pdfs() -> list[Path]:
    """Return provider-supported documents in a stable order."""
    configured = os.getenv("PAGEINDEX_DOCUMENT_DIR", "").strip()
    source_dir = Path(configured).expanduser() if configured else LEGAL_PDF_DIR
    if not source_dir.is_dir():
        return []
    return sorted(
        path for path in source_dir.rglob("*.pdf")
        if path.is_file() and not path.name.startswith(".")
    )


def upload_documents() -> None:
    """Upload uncached PDFs and persist source-to-document-ID mappings.

    A content digest prevents duplicate uploads when this command is rerun.
    Individual failures are non-fatal so a transient error on one document does
    not discard IDs that were already cached successfully.
    """
    api_key = os.getenv("PAGEINDEX_API_KEY", "").strip()
    if not api_key:
        print("Skip PageIndex upload: PAGEINDEX_API_KEY is not configured.")
        return

    pdf_paths = _discover_pdfs()
    if not pdf_paths:
        print("Skip PageIndex upload: no PDF documents were found.")
        return

    try:
        client = _make_client(api_key)
    except Exception as error:
        LOGGER.warning("PageIndex client is unavailable: %s", error)
        return

    cache = _load_cache()
    request_timeout = _positive_env_float(
        "PAGEINDEX_HTTP_TIMEOUT_SECONDS", DEFAULT_HTTP_TIMEOUT
    )
    changed = False
    uploaded = 0
    for path in pdf_paths:
        source_key = _relative_source(path)
        digest = _file_sha256(path)
        cached = cache.get(source_key, {})
        if cached.get("doc_id") and cached.get("sha256") == digest:
            continue

        try:
            response = _call_with_timeout(
                client.submit_document,
                request_timeout,
                str(path),
            )
            doc_id = response.get("doc_id") if isinstance(response, dict) else None
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise RuntimeError("PageIndex response did not include doc_id")
        except Exception as error:
            LOGGER.warning("PageIndex upload failed for %s: %s", path.name, error)
            continue

        source_metadata = LEGAL_DOCUMENTS.get(path.name, {})
        cache[source_key] = {
            "doc_id": doc_id.strip(),
            "sha256": digest,
            "source": path.name,
            "title": str(
                source_metadata.get("title") or path.stem.replace("_", " ")
            ),
            "doc_type": "legal",
            "url": source_metadata.get("url"),
            "publisher": source_metadata.get("publisher"),
            "published_date": source_metadata.get("published_date"),
            "license": source_metadata.get("license"),
            "license_url": source_metadata.get("license_url"),
        }
        uploaded += 1
        changed = True

    if changed:
        try:
            _write_cache(cache)
        except OSError as error:
            LOGGER.warning("Could not persist PageIndex document IDs: %s", error)
            return
    print(f"PageIndex upload complete: {uploaded} uploaded, {len(cache)} cached.")


def _wait_for_retrieval(client: Any, retrieval_id: str, deadline: float) -> dict[str, Any]:
    poll_interval = _positive_env_float(
        "PAGEINDEX_POLL_INTERVAL_SECONDS", DEFAULT_POLL_INTERVAL
    )
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("PageIndex retrieval timed out")
        request_timeout = min(
            _positive_env_float("PAGEINDEX_HTTP_TIMEOUT_SECONDS", DEFAULT_HTTP_TIMEOUT),
            max(deadline - time.monotonic(), 0.001),
        )
        response = _call_with_timeout(
            client.get_retrieval,
            request_timeout,
            retrieval_id,
        )
        if not isinstance(response, dict):
            raise RuntimeError("PageIndex returned an invalid retrieval response")
        status = str(response.get("status", "")).strip().lower()
        if status == "completed":
            return response
        if status in {"failed", "error", "cancelled", "canceled"}:
            raise RuntimeError(f"PageIndex retrieval ended with status={status}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("PageIndex retrieval timed out")
        time.sleep(min(poll_interval, remaining))


def _flatten_relevant_contents(value: Any) -> Iterator[dict[str, Any]]:
    """Flatten both response shapes emitted by PageIndex 0.2.x."""
    if isinstance(value, str):
        if value.strip():
            yield {"relevant_content": value.strip()}
        return
    if isinstance(value, list):
        for item in value:
            yield from _flatten_relevant_contents(item)
        return
    if not isinstance(value, dict):
        return

    content = value.get("relevant_content") or value.get("content") or value.get("text")
    if isinstance(content, str) and content.strip():
        yield value
        return
    for key in ("items", "contents", "relevant_content", "relevant_contents", "nodes"):
        if key in value:
            yield from _flatten_relevant_contents(value[key])


def _page_number(value: Any, fallback: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return max(value, 0)
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group())
    return fallback


def _parse_retrieval(
    response: dict[str, Any],
    document: dict[str, Any],
) -> list[dict[str, Any]]:
    nodes = response.get("retrieved_nodes") or response.get("nodes") or []
    if not isinstance(nodes, list):
        return []

    parsed: list[dict[str, Any]] = []
    for node_index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        node_title = str(node.get("title") or document["title"])
        relevant = list(_flatten_relevant_contents(node.get("relevant_contents", [])))
        if not relevant:
            relevant = list(_flatten_relevant_contents(node))
        for content_index, item in enumerate(relevant):
            content = item.get("relevant_content") or item.get("content") or item.get("text")
            if not isinstance(content, str) or not content.strip():
                continue
            page_value = (
                item.get("page_index")
                or item.get("page")
                or item.get("page_idx")
                or item.get("physical_index")
                or node.get("page_index")
            )
            page_number = _page_number(page_value, node_index)
            title = str(item.get("section_title") or node_title)
            content = content.strip()
            identity = "|".join(
                (
                    document["doc_id"],
                    str(node.get("node_id", node_index)),
                    str(page_value or content_index),
                    hashlib.sha1(content.encode("utf-8")).hexdigest()[:12],
                )
            )
            metadata = {
                "source": document["source"],
                "title": title,
                "doc_type": document.get("doc_type", "legal"),
                "url": document.get("url"),
                "chunk_index": page_number,
                "page_index": page_value,
            }
            for key in ("publisher", "published_date", "license", "license_url"):
                value = document.get(key)
                if isinstance(value, str) and value.strip():
                    metadata[key] = value.strip()
            parsed.append(
                {
                    "id": f"pageindex:{identity}",
                    "content": content,
                    "score": 0.0,
                    "metadata": metadata,
                    "retrieval_method": "pageindex",
                }
            )
    return parsed


def _interleave(ranked_lists: list[list[dict[str, Any]]]) -> Iterator[dict[str, Any]]:
    """Merge per-document rankings without letting the first PDF dominate."""
    max_length = max((len(items) for items in ranked_lists), default=0)
    for rank in range(max_length):
        for items in ranked_lists:
            if rank < len(items):
                yield items[rank]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve grounded PageIndex snippets as ``SearchResult`` objects.

    Missing configuration and all provider failures deliberately return an
    empty list.  Task 9 can then continue with its already-computed hybrid
    results instead of crashing the application.
    """
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []
    api_key = os.getenv("PAGEINDEX_API_KEY", "").strip()
    if not api_key:
        return []

    documents = list(_load_cache().values())
    if not documents:
        return []
    try:
        client = _make_client(api_key)
    except Exception as error:
        LOGGER.warning("PageIndex client is unavailable: %s", error)
        return []

    timeout = _positive_env_float(
        "PAGEINDEX_RETRIEVAL_TIMEOUT_SECONDS", DEFAULT_RETRIEVAL_TIMEOUT
    )
    http_timeout = _positive_env_float(
        "PAGEINDEX_HTTP_TIMEOUT_SECONDS", DEFAULT_HTTP_TIMEOUT
    )
    deadline = time.monotonic() + timeout
    per_document_results: list[list[dict[str, Any]]] = []
    for document in documents:
        if time.monotonic() >= deadline:
            LOGGER.warning("PageIndex fallback reached its %.1fs timeout", timeout)
            break
        doc_id = document["doc_id"]
        try:
            call_timeout = min(http_timeout, max(deadline - time.monotonic(), 0.001))
            if not _call_with_timeout(client.is_retrieval_ready, call_timeout, doc_id):
                continue
            call_timeout = min(http_timeout, max(deadline - time.monotonic(), 0.001))
            submission = _call_with_timeout(
                client.submit_query,
                call_timeout,
                doc_id,
                query.strip(),
                thinking=True,
            )
            retrieval_id = (
                submission.get("retrieval_id") if isinstance(submission, dict) else None
            )
            if not isinstance(retrieval_id, str) or not retrieval_id.strip():
                raise RuntimeError("PageIndex response did not include retrieval_id")
            response = _wait_for_retrieval(client, retrieval_id, deadline)
            parsed = _parse_retrieval(response, document)
            if parsed:
                per_document_results.append(parsed)
        except Exception as error:
            LOGGER.warning("PageIndex retrieval failed for %s: %s", document["source"], error)

    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in _interleave(per_document_results):
        if item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])
        item["score"] = 1.0 / (len(results) + 1)
        results.append(item)
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    upload_documents()
