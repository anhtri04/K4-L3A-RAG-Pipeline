"""Task 2 — Crawl tin tức và dữ liệu bóng đá thành JSON và Markdown.

Dùng thư viện chuẩn Python, không cần API key hoặc browser.
Chạy: python -m src.task2_crawl_news --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
import tempfile
import time
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent.parent
API = "https://api.openligadb.de"
KINDS = ("news", "matches", "teams", "standings", "players")
FEEDS = {
    "vnexpress": ("VnExpress", "https://vnexpress.net/rss/the-thao.rss"),
    "bbc": ("BBC Sport", "https://feeds.bbci.co.uk/sport/football/rss.xml"),
}
ENDPOINTS = {
    "matches": "getmatchdata",
    "teams": "getavailableteams",
    "standings": "getbltable",
    "players": "getgoalgetters",
}
FOOTBALL_TERMS = (
    "bóng đá", "cầu thủ", "bàn thắng", "tiền đạo", "hậu vệ", "thủ môn",
    "tiền vệ", "ngoại hạng anh", "champions league", "europa league",
    "bundesliga", "la liga", "laliga", "serie a", "ligue 1", "v-league",
    "v league", "v.league", "fifa", "uefa", "arsenal", "liverpool",
    "man utd", "man city", "chelsea", "barcelona", "real madrid",
    "bayern", "messi", "ronaldo", "haaland", "mbappe", "futsal",
)
OTHER_SPORTS = ("bóng chuyền", "bóng rổ", "tennis", "quần vợt", "cầu lông", "golf", "cờ vua")
LOG = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"URL không hợp lệ: {url}")
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith(("utm_", "at_")) and k.lower() != "fbclid"]
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path, urlencode(query), ""))


class TextExtractor(HTMLParser):
    """Strip RSS markup, including script/style content, without extra packages."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        elif tag in {"br", "p", "div", "li"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        elif tag in {"p", "div", "li"}:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def plain_text(value: str) -> str:
    parser = TextExtractor()
    parser.feed(value)
    return " ".join("".join(parser.parts).split())


class HttpClient:
    """Sequential requests, timeouts and bounded retries for temporary errors."""

    def __init__(self, timeout: float = 25, delay: float = 1.1, retries: int = 2):
        self.timeout = timeout
        self.delay = delay
        self.retries = retries
        self.last_request: float | None = None

    def get(self, url: str) -> bytes:
        canonical_url(url)  # Reject non-HTTP schemes.
        for attempt in range(self.retries + 1):
            if self.last_request is not None:
                time.sleep(max(0, self.delay - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            request = Request(url, headers={
                "User-Agent": "FootballRAGCrawler/1.0 (educational RSS/API client)",
                "Accept": "application/json, application/rss+xml, application/xml, text/xml",
            })
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    body = response.read(10 * 1024 * 1024 + 1)
                    if len(body) > 10 * 1024 * 1024:
                        raise ValueError("Phản hồi lớn hơn giới hạn 10 MiB")
                    return body
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                retry_after = 0.0
                if isinstance(error, HTTPError):
                    if error.code not in {429, 500, 502, 503, 504}:
                        error.close()
                        raise
                    header = error.headers.get("Retry-After", "0")
                    error.close()
                    try:
                        retry_after = float(header)
                    except ValueError:
                        try:
                            retry_after = (parsedate_to_datetime(header)
                                           - datetime.now(timezone.utc)).total_seconds()
                        except (ValueError, TypeError, OverflowError):
                            pass
                if attempt == self.retries or retry_after > 60:
                    raise
                wait = max(2 ** attempt, retry_after)
                LOG.warning("Thử lại %s sau %.1f giây: %s", url, wait, error)
                time.sleep(wait)
        raise RuntimeError("Request did not complete")

    def get_json(self, url: str) -> list[dict]:
        data = json.loads(self.get(url))
        if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
            raise ValueError("API phải trả về một danh sách JSON object")
        return data


def record(kind: str, identity: str, title: str, url: str, source: str,
           fields: dict, raw: dict, *, league: str | None = None,
           season: int | None = None) -> dict:
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Thiếu tiêu đề")
    content = "\n".join(f"- **{key}:** {value if value is not None else 'Chưa có dữ liệu'}"
                        for key, value in fields.items())
    return {
        "id": identity, "record_type": kind, "title": title,
        "url": canonical_url(url), "source": source, "date_crawled": utc_now(),
        "league": league, "season": season, "content_markdown": content,
        "data": raw,
    }


def parse_news(body: bytes, feed_name: str, limit: int) -> list[dict]:
    root = ET.fromstring(body)
    if root.tag != "rss" or root.find("channel") is None:
        raise ValueError("Nguồn không trả về RSS 2.0 hợp lệ")
    source, feed_url = FEEDS[feed_name]
    output, seen = [], set()
    for item in root.findall("./channel/item"):
        title = plain_text(item.findtext("title", ""))
        summary = plain_text(item.findtext("description", ""))
        link = item.findtext("link", "").strip()
        if not title or not link or not summary:
            continue
        text = f"{title} {summary}".casefold()
        if feed_name == "vnexpress" and (
            any(term in text for term in OTHER_SPORTS)
            or not any(term in text for term in FOOTBALL_TERMS)
        ):
            continue
        try:
            url = canonical_url(link)
        except ValueError:
            continue
        if url in seen:
            continue
        seen.add(url)
        published = item.findtext("pubDate", "").strip()
        try:
            published = parsedate_to_datetime(published).isoformat()
        except (ValueError, TypeError, OverflowError):
            pass  # Keep the original date rather than inventing a timestamp.
        article = record("news", url, title, url, source, {
            "Ngày đăng": published or None, "Tóm tắt RSS": summary,
            "Phạm vi nội dung": "Tiêu đề và tóm tắt do RSS cung cấp",
        }, {"published_at": published or None, "summary": summary, "feed_url": feed_url})
        output.append(article)
        if len(output) >= limit:
            break
    return output


def normalize_match(item: dict, league: str, season: int) -> dict:
    match_id = item["matchID"]
    home = (item.get("team1") or {}).get("teamName") or "Chưa xác định"
    away = (item.get("team2") or {}).get("teamName") or "Chưa xác định"
    finished = item.get("matchIsFinished") is True
    # Never treat a half-time result or an unplayed placeholder 0-0 as final.
    results = [r for r in (item.get("matchResults") or [])
               if r.get("pointsTeam1") is not None and r.get("pointsTeam2") is not None
               and (str(r.get("resultTypeKind") or "").startswith("After")
                    or (not r.get("resultTypeKind") and r.get("resultTypeID") == 2))]
    final = max(results, key=lambda r: r.get("resultOrderID") or 0) if finished and results else None
    score = f"{final['pointsTeam1']} - {final['pointsTeam2']}" if final else None
    goals = item.get("goals") or []
    goal_text = "; ".join(
        f"{g.get('goalGetterName') or 'Chưa xác định'} ({g.get('matchMinute', '?')} phút"
        f"{' · phản lưới' if g.get('isOwnGoal') else ''}"
        f"{' · phạt đền' if g.get('isPenalty') else ''})" for g in goals
    )
    output = record("matches", f"openligadb:{league}:{season}:match:{match_id}",
                    f"{home} – {away}", f"{API}/getmatchdata/{match_id}", "OpenLigaDB", {
        "Giải đấu": item.get("leagueName") or league, "Mùa bắt đầu": season,
        "Vòng": (item.get("group") or {}).get("groupName"),
        "Đội nhà": home, "Đội khách": away,
        "Giờ thi đấu (UTC)": item.get("matchDateTimeUTC"),
        "Trạng thái": "Đã kết thúc" if finished else "Chưa kết thúc / chưa thi đấu",
        "Kết quả": score,
        "Loại kết quả": (final.get("resultTypeKind") or final.get("resultName")) if final else None,
        "Cầu thủ ghi bàn": goal_text or None,
        "Sân": (item.get("location") or {}).get("locationStadium"),
    }, item, league=league, season=season)
    output["status"] = "finished" if finished else "not_finished"
    output["score"] = {"home": final["pointsTeam1"], "away": final["pointsTeam2"]} if final else None
    return output


def normalize_rows(kind: str, rows: list[dict], league: str, season: int,
                   url: str):
    for position, item in enumerate(rows, 1):
        if kind == "matches":
            yield normalize_match(item, league, season)
            continue
        fields = {"Giải đấu": league, "Mùa bắt đầu": season}
        if kind == "teams":
            identity, title = item["teamId"], item["teamName"]
            fields.update({"Đội bóng": title, "Tên ngắn": item.get("shortName"),
                           "Nhóm": item.get("teamGroupName")})
        elif kind == "standings":
            identity, title = item["teamInfoId"], item["teamName"]
            fields.update({"Đội bóng": title, "Hạng theo nguồn": position})
            fields.update({label: item.get(key) for key, label in {
                "points": "Điểm", "matches": "Số trận", "won": "Thắng", "draw": "Hòa",
                "lost": "Thua", "goals": "Bàn thắng", "opponentGoals": "Bàn thua",
                "goalDiff": "Hiệu số",
            }.items()})
            title = f"Bảng xếp hạng: {title} ({league}, {season})"
        else:
            identity, title = item["goalGetterId"], item["goalGetterName"]
            fields.update({"Cầu thủ": title, "Số bàn thắng": item.get("goalCount"),
                           "Phạm vi": "Danh sách ghi bàn của giải; không phải toàn bộ đội hình"})
        yield record(kind, f"openligadb:{league}:{season}:{kind}:{identity}", title,
                     url, "OpenLigaDB", fields, item, league=league, season=season)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_record(item: dict, data_dir: Path) -> None:
    kind = item["record_type"]
    filename = hashlib.sha256(item["id"].encode("utf-8")).hexdigest()[:24]
    # Task 3 and acceptance tests read news/*.json directly, without recursion.
    relative = (Path("news") / f"football_{filename}" if kind == "news"
                else Path("football") / kind / filename)
    atomic_write(data_dir / "landing" / relative.with_suffix(".json"),
                 json.dumps(item, ensure_ascii=False, indent=2) + "\n")
    markdown = (f"# {item['title']}\n\n**Source:** {item['url']}\n\n"
                f"**Provider:** {item['source']}\n\n**Crawled:** {item['date_crawled']}\n\n"
                f"**Type:** {kind}\n\n---\n\n{item['content_markdown']}\n")
    atomic_write(data_dir / "standardized" / relative.with_suffix(".md"), markdown)


def within_dates(item: dict, start: date | None, end: date | None) -> bool:
    if start is None and end is None:
        return True
    value = item.get("matchDateTimeUTC")
    if not value:
        return False
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError("matchDateTimeUTC thiếu múi giờ")
    day = moment.astimezone(timezone.utc).date()
    return (start is None or day >= start) and (end is None or day <= end)


def crawl(args: argparse.Namespace, client: HttpClient | None = None) -> dict:
    client = client or HttpClient(args.timeout, args.delay, args.retries)
    report = {"started_at": utc_now(), "season": args.season, "leagues": args.leagues,
              "counts": {kind: 0 for kind in KINDS}, "errors": [], "warnings": [], "sources": []}

    def collect(kind: str, url: str, loader):
        try:
            # Materialize before writing, so a malformed row cannot silently truncate a source.
            items = list(loader())
            for item in items:
                save_record(item, args.data_dir)
                report["counts"][kind] += 1
            report["sources"].append({"url": url, "record_type": kind, "count": len(items)})
            LOG.info("%s: %d bản ghi (%s)", kind, len(items), url)
            if not items:
                report["warnings"].append(f"Không có dữ liệu {kind}: {url}")
        except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as error:
            report["errors"].append({"url": url, "record_type": kind, "error": str(error)})
            LOG.error("Không lấy được %s: %s", url, error)

    if "news" in args.categories:
        for name in dict.fromkeys(args.news_sources):
            url = FEEDS[name][1]
            collect("news", url, lambda: parse_news(client.get(url), name, args.news_limit))
    for league in dict.fromkeys(args.leagues):
        for kind, endpoint in ENDPOINTS.items():
            if kind not in args.categories:
                continue
            url = f"{API}/{endpoint}/{quote(league, safe='')}/{args.season}"

            def load_rows():
                rows = client.get_json(url)
                if kind == "matches":
                    rows = [r for r in rows if within_dates(r, args.start_date, args.end_date)]
                return normalize_rows(kind, rows, league, args.season, url)

            collect(kind, url, load_rows)
    report["finished_at"] = utc_now()
    report["total"] = sum(report["counts"].values())
    atomic_write(args.data_dir / "football_crawl_report.json",
                 json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    today = date.today()
    parser = argparse.ArgumentParser(description="Crawl tin RSS và dữ liệu bóng đá OpenLigaDB.")
    parser.add_argument("--leagues", nargs="+", default=["bl1"], help="Mã giải OpenLigaDB; mặc định bl1 (Bundesliga)")
    parser.add_argument("--season", type=int, default=today.year if today.month >= 7 else today.year - 1,
                        help="Năm bắt đầu mùa giải; mặc định mùa hiện tại theo mốc tháng 7")
    parser.add_argument("--categories", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--news-sources", nargs="+", choices=FEEDS, default=list(FEEDS))
    parser.add_argument("--news-limit", type=int, default=20, help="Tối đa số tin cho MỖI nguồn RSS")
    parser.add_argument("--start-date", type=date.fromisoformat, help="Lọc trận từ ngày UTC YYYY-MM-DD")
    parser.add_argument("--end-date", type=date.fromisoformat, help="Lọc trận đến ngày UTC YYYY-MM-DD (bao gồm)")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--timeout", type=float, default=25)
    parser.add_argument("--delay", type=float, default=1.1, help="Khoảng cách request, tối thiểu 1.1 giây")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--list-leagues", action="store_true", help="Liệt kê giải bóng đá cho --season rồi thoát")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1900 <= args.season <= 2200:
        parser.error("--season phải nằm trong khoảng 1900..2200")
    if any(not league.strip() or not re.fullmatch(r"[\w -]+", league) for league in args.leagues):
        parser.error("Mã giải chỉ được chứa chữ, số, khoảng trắng, dấu _ hoặc -")
    if args.news_limit < 1 or args.retries < 0:
        parser.error("--news-limit phải > 0 và --retries phải >= 0")
    if not math.isfinite(args.timeout) or args.timeout <= 0 or not math.isfinite(args.delay) or args.delay < 1.1:
        parser.error("--timeout phải > 0 và --delay phải >= 1.1 (số hữu hạn)")
    if args.start_date and args.end_date and args.start_date > args.end_date:
        parser.error("--start-date phải <= --end-date")
    try:
        if args.list_leagues:
            rows = HttpClient(args.timeout, args.delay, args.retries).get_json(
                f"{API}/getavailableleagues/{args.season}")
            for row in rows:
                sport = row.get("sport") or {}
                if sport.get("sportId") == 1 or str(sport.get("sportName", "")).casefold() in {"fußball", "football", "soccer"}:
                    print(f"{row.get('leagueShortcut')}\t{row.get('leagueSeason')}\t{row.get('leagueName')}")
            return 0
        report = crawl(args)
    except (OSError, ValueError) as error:
        LOG.error("Crawl thất bại: %s", error)
        return 1
    print(json.dumps({"counts": report["counts"], "total": report["total"],
                      "errors": len(report["errors"]), "warnings": len(report["warnings"])}, ensure_ascii=False))
    print(f"Report: {args.data_dir / 'football_crawl_report.json'}")
    # Partial failures must be visible to scripts that run this crawler.
    return 1 if report["errors"] or not report["total"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
