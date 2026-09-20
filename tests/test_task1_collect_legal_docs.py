"""Offline tests for downloading, source metadata and preserving valid files."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

from src import task1_collect_legal_docs as legal


# A PDF-envelope fixture, not a full PDF document (structural QA uses real PDFs).
PDF_BYTES = b"%PDF-1.7\n%" + b"x" * 2048 + b"\n%%EOF\n"


def response(body=PDF_BYTES, length=None):
    result = MagicMock()
    result.__enter__.return_value = result
    result.headers = {"Content-Length": str(len(body) if length is None else length)}
    result.read.return_value = body
    result.geturl.return_value = "https://example.org/resolved.pdf"
    return result


class LegalDocumentsTests(unittest.TestCase):
    def test_download_keeps_original_bytes_metadata_and_skips_verified_files(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(legal, "urlopen", side_effect=lambda *a, **k: response()) as request, \
                patch.object(legal.time, "sleep"):
            root = Path(directory)
            report = legal.download_documents(data_dir=root)
            self.assertEqual(report["downloaded"], 3)
            self.assertEqual(report["errors"], [])
            manifest = json.loads((root / "sources.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["documents"]), 3)
            for source, metadata in zip(legal.SOURCES, manifest["documents"]):
                self.assertEqual((root / source.filename).read_bytes(), PDF_BYTES)
                self.assertEqual(metadata["url"], source.url)
                self.assertEqual(metadata["edition"], source.edition)
                self.assertEqual(metadata["doc_type"], "legal")
                self.assertEqual(metadata["size_bytes"], len(PDF_BYTES))
                self.assertEqual(len(metadata["sha256"]), 64)
            second = legal.download_documents(data_dir=root)
            self.assertEqual(second["skipped"], 3)
            self.assertEqual(request.call_count, 3)
            self.assertEqual(json.loads((root / "sources.json").read_text(encoding="utf-8")), manifest)
            self.assertEqual(len(list(root.glob("*.pdf"))), 3)
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_html_truncated_pdf_and_empty_responses_are_rejected(self):
        for body in (b"", b"<html>Access denied</html>" * 100, PDF_BYTES[:-10]):
            with self.subTest(body_size=len(body)), \
                    patch.object(legal, "urlopen", return_value=response(body)), \
                    self.assertRaises(ValueError):
                legal.fetch_pdf("https://example.org/test.pdf", timeout=5, retries=0)

    def test_mismatched_content_length_is_rejected(self):
        with patch.object(legal, "urlopen", return_value=response(length=9999)), \
                self.assertRaisesRegex(ValueError, "Content-Length"):
            legal.fetch_pdf("https://example.org/test.pdf", timeout=5, retries=0)

    def test_forbidden_download_is_not_retried(self):
        error = HTTPError("https://example.org/test.pdf", 403, "Forbidden", {}, None)
        with patch.object(legal, "urlopen", side_effect=error) as request, self.assertRaises(HTTPError):
            legal.fetch_pdf(error.url, timeout=5, retries=2)
        self.assertEqual(request.call_count, 1)

    def test_retryable_error_respects_retry_after(self):
        error = HTTPError("https://example.org/test.pdf", 503, "Unavailable", {"Retry-After": "3"}, None)
        with patch.object(legal, "urlopen", side_effect=[error, response()]) as request, \
                patch.object(legal.time, "sleep") as sleep:
            body, url = legal.fetch_pdf(error.url, timeout=5, retries=1)
        self.assertEqual(body, PDF_BYTES)
        self.assertEqual(url, "https://example.org/resolved.pdf")
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once_with(3.0)

    def test_changed_checksum_triggers_redownload(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(legal, "urlopen", side_effect=lambda *a, **k: response()) as request, \
                patch.object(legal.time, "sleep"):
            root = Path(directory)
            legal.download_documents(data_dir=root)
            changed = root / legal.SOURCES[0].filename
            changed.write_bytes(PDF_BYTES.replace(b"xx", b"yy", 1))
            report = legal.download_documents(data_dir=root)
            self.assertEqual(report["downloaded"], 1)
            self.assertEqual(report["skipped"], 2)
            self.assertEqual(request.call_count, 4)
            self.assertEqual(changed.read_bytes(), PDF_BYTES)

    def test_failed_refresh_preserves_existing_file_and_source_metadata(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(legal.time, "sleep"):
            root = Path(directory)
            with patch.object(legal, "urlopen", side_effect=lambda *a, **k: response()):
                legal.download_documents(data_dir=root)
            manifest = (root / "sources.json").read_bytes()
            with patch.object(legal, "urlopen", return_value=response(b"<html>error</html>")):
                report = legal.download_documents(data_dir=root, force=True)
            self.assertEqual(len(report["errors"]), 3)
            self.assertEqual((root / "sources.json").read_bytes(), manifest)
            self.assertTrue(all((root / source.filename).read_bytes() == PDF_BYTES for source in legal.SOURCES))

    def test_one_source_failure_does_not_stop_other_sources(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(legal, "urlopen", side_effect=[URLError("offline"), response(), response()]), \
                patch.object(legal.time, "sleep"):
            root = Path(directory)
            report = legal.download_documents(data_dir=root, retries=0)
            self.assertEqual(report["downloaded"], 2)
            self.assertEqual(len(report["errors"]), 1)
            self.assertFalse((root / legal.SOURCES[0].filename).exists())
            self.assertEqual(len(list(root.glob("*.pdf"))), 2)

    def test_invalid_manifest_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sources.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                legal.download_documents(data_dir=root)
            self.assertEqual(path.read_text(encoding="utf-8"), "[]")

    def test_cli_reports_partial_failure_and_rejects_bad_arguments(self):
        for report, expected in (
            ({"total_valid": 3, "downloaded": 3, "skipped": 0, "errors": []}, 0),
            ({"total_valid": 2, "downloaded": 2, "skipped": 0, "errors": [{}]}, 1),
        ):
            with patch.object(legal, "download_documents", return_value=report), \
                    patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(legal.main([]), expected)
        for args in (["--timeout", "nan"], ["--timeout", "0"], ["--retries", "-1"]):
            with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit) as error:
                legal.main(args)
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
