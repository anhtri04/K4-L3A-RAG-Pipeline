"""Task 1 — Tải tài liệu luật/quy định bóng đá gốc từ IFAB và VFF.

Chạy: python -m src.task1_collect_legal_docs
File PDF và metadata nguồn nằm trực tiếp trong data/landing/legal/.
Chỉ dùng thư viện chuẩn Python; bước chuyển Markdown thuộc Task 3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "landing" / "legal"
MAX_BYTES = 64 * 1024 * 1024
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentSource:
    filename: str
    title: str
    issuer: str
    edition: str
    language: str
    url: str
    source_page: str


SOURCES = (
    DocumentSource(
        filename="ifab_laws_of_the_game_2026_2027_en.pdf",
        title="Laws of the Game 2026/27",
        issuer="The IFAB", edition="2026/27", language="en",
        url="https://downloads.theifab.com/downloads/laws-of-the-game-202627-single-pages?l=en",
        source_page="https://www.theifab.com/laws-of-the-game-documents/",
    ),
    DocumentSource(
        filename="vff_quy_dinh_ky_luat_2026_vi.pdf",
        title="Quy định về kỷ luật của LĐBĐVN (sửa đổi, bổ sung năm 2026)",
        issuer="Liên đoàn Bóng đá Việt Nam (VFF)", edition="2026", language="vi",
        url="https://vff.org.vn/wp-content/uploads/2026/02/Quy-%C4%91%E1%BB%8Bnh-v%E1%BB%81-k%E1%BB%B7-lu%E1%BA%ADt-c%E1%BB%A7a-L%C4%90B%C4%90VN-s%E1%BB%ADa-%C4%91%E1%BB%95i-b%E1%BB%95-sung-n%C4%83m-2026.pdf",
        source_page="https://vff.org.vn/luat-van-ban/quy-dinh-ve-ky-luat-cua-ldbdvn-sua-doi-bo-sung-nam-2026/",
    ),
    DocumentSource(
        filename="vff_dieu_le_vleague_2025_2026_vi.pdf",
        title="Điều lệ Giải bóng đá Vô địch Quốc gia LPBank 2025/26",
        issuer="Liên đoàn Bóng đá Việt Nam (VFF)", edition="2025/26", language="vi",
        url="https://vff.org.vn/wp-content/uploads/2025/08/2025-08-04-Dieu-le-Giai-VDQG-LPBANK-2025-26.pdf",
        source_page="https://vff.org.vn/luat-van-ban/dieu-le-giai-vo-dich-quoc-gia-lpbank-2025-26/",
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_directory(data_dir: Path | None = None) -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    (data_dir if data_dir is not None else DATA_DIR).mkdir(parents=True, exist_ok=True)


def validate_pdf(body: bytes) -> None:
    """Check the PDF envelope; a PDF reader can validate its internal structure."""
    if len(body) <= 1024:
        raise ValueError("File PDF quá nhỏ hoặc rỗng (phải lớn hơn 1 KiB)")
    if len(body) > MAX_BYTES:
        raise ValueError("File vượt giới hạn 64 MiB")
    if not body.startswith(b"%PDF-"):
        raise ValueError("Nội dung không phải PDF; có thể là trang lỗi HTML")
    if b"%%EOF" not in body[-1024:]:
        raise ValueError("PDF thiếu dấu kết thúc, có thể chưa tải đủ")


def atomic_write(path: Path, body: bytes) -> None:
    """Replace a complete file only after writing its temporary copy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(body)
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fetch_pdf(url: str, *, timeout: float, retries: int) -> tuple[bytes, str]:
    for attempt in range(retries + 1):
        request = Request(url, headers={
            "User-Agent": "FootballRAGCollector/1.0 (educational document downloader)",
            "Accept": "application/pdf, application/octet-stream",
        })
        try:
            with urlopen(request, timeout=timeout) as response:
                length = response.headers.get("Content-Length")
                if length and int(length) > MAX_BYTES:
                    raise ValueError("File vượt giới hạn 64 MiB")
                body = response.read(MAX_BYTES + 1)
                if length and len(body) != int(length):
                    raise ValueError("Số byte tải về khác Content-Length")
                validate_pdf(body)
                return body, response.geturl()
        except (OSError, HTTPException) as error:
            wait = float(2 ** attempt)
            if isinstance(error, HTTPError):
                header = error.headers.get("Retry-After")
                error.close()
                # Do not retry forbidden/blocked or missing resources.
                if error.code not in {429, 500, 502, 503, 504}:
                    raise
                if header:
                    try:
                        requested_wait = float(header)
                    except ValueError:
                        try:
                            requested_wait = (parsedate_to_datetime(header)
                                              - datetime.now(timezone.utc)).total_seconds()
                        except (ValueError, TypeError, OverflowError):
                            requested_wait = 0
                    if math.isfinite(requested_wait):
                        wait = max(wait, requested_wait)
            if attempt == retries or wait > 60:
                raise
            LOG.warning("Thử lại sau %.1f giây: %s", wait, error)
            time.sleep(wait)
    raise RuntimeError("Download did not complete")


def load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(documents, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("filename"), str)
        for item in documents
    ):
        raise ValueError(f"Metadata nguồn không hợp lệ: {path}")
    return {item["filename"]: item for item in documents}


def is_cached(path: Path, source: DocumentSource, previous: dict) -> bool:
    if not path.is_file() or previous.get("url") != source.url:
        return False
    if path.stat().st_size > MAX_BYTES:
        return False
    body = path.read_bytes()
    try:
        validate_pdf(body)
    except ValueError:
        return False
    return (previous.get("size_bytes") == len(body)
            and previous.get("sha256") == hashlib.sha256(body).hexdigest())


def download_documents(*, data_dir: Path | None = None, force: bool = False,
                       timeout: float = 45, retries: int = 2) -> dict:
    """Download original PDFs, preserving their source metadata for Task 3."""
    directory = data_dir if data_dir is not None else DATA_DIR
    setup_directory(directory)
    manifest_path = directory / "sources.json"
    manifest = load_manifest(manifest_path)
    report = {"started_at": utc_now(), "downloaded": 0, "skipped": 0,
              "errors": [], "files": []}
    for index, source in enumerate(SOURCES):
        path = directory / source.filename
        try:
            previous = manifest.get(source.filename, {})
            if not force and is_cached(path, source, previous):
                report["skipped"] += 1
                report["files"].append(source.filename)
                LOG.info("Đã có file hợp lệ: %s", source.filename)
                continue
            if index:
                time.sleep(1.1)
            body, resolved_url = fetch_pdf(source.url, timeout=timeout, retries=retries)
            atomic_write(path, body)
            manifest[source.filename] = {
                **asdict(source), "doc_type": "legal", "date_downloaded": utc_now(),
                "download_url": resolved_url, "size_bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
            report["downloaded"] += 1
            report["files"].append(source.filename)
            LOG.info("Đã tải: %s (%d byte)", source.filename, len(body))
        except (OSError, HTTPException, ValueError) as error:
            LOG.error("Không tải được %s: %s", source.filename, error)
            report["errors"].append({"filename": source.filename, "url": source.url,
                                     "error": str(error)})
    report["finished_at"] = utc_now()
    report["total_valid"] = report["downloaded"] + report["skipped"]
    for name, payload in (
        ("sources.json", {"schema_version": 1, "documents": list(manifest.values())}),
        ("download_report.json", report),
    ):
        atomic_write(directory / name, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return report


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Task 1: thu thập PDF luật/quy định bóng đá.")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--force", action="store_true", help="Tải lại cả các file đã có metadata hợp lệ")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0 or not 0 <= args.retries <= 5:
        parser.error("--timeout phải là số hữu hạn > 0; --retries phải trong khoảng 0..5")
    try:
        report = download_documents(data_dir=args.output_dir, force=args.force,
                                    timeout=args.timeout, retries=args.retries)
    except (OSError, ValueError) as error:
        LOG.error("Task 1 thất bại: %s", error)
        return 1
    print(f"PDF hợp lệ: {report['total_valid']}; tải mới: {report['downloaded']}; "
          f"đã có: {report['skipped']}; lỗi: {len(report['errors'])}")
    print(f"Nguồn: {args.output_dir / 'sources.json'}")
    return 0 if not report["errors"] and report["total_valid"] >= 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
