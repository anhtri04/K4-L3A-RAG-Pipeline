"""Offline regression tests; also runnable without pytest via unittest."""

import copy
import io
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from src import task2_crawl_news as football


MATCH = {
    "matchID": 10, "matchIsFinished": True,
    "matchDateTimeUTC": "2026-09-20T14:00:00Z",
    "team1": {"teamId": 1, "teamName": "Bayern"},
    "team2": {"teamId": 2, "teamName": "Dortmund"},
    "matchResults": [
        {"resultTypeID": 2, "resultOrderID": 2, "pointsTeam1": 3, "pointsTeam2": 1},
        {"resultTypeID": 1, "resultOrderID": 1, "pointsTeam1": 0, "pointsTeam2": 1},
    ],
    "goals": [{"goalGetterName": "Player A", "matchMinute": 55, "isOwnGoal": True}],
}
RSS = b'''<rss version="2.0"><channel>
<item><title>Arsenal win</title><link>https://example.org/football?utm_source=feed</link>
<description>&lt;p&gt;Arsenal score twice.&lt;/p&gt;&lt;script&gt;bad()&lt;/script&gt;</description>
<pubDate>Sun, 20 Sep 2026 13:35:05 +0700</pubDate></item>
<item><title>Arsenal win duplicate</title><link>https://example.org/football#top</link>
<description>Arsenal score twice.</description></item>
<item><title>Tennis final</title><link>https://example.org/tennis</link>
<description>Tennis final finished.</description></item>
<item><title>Invalid link</title><link>file:///etc/passwd</link>
<description>Arsenal</description></item>
</channel></rss>'''


class FootballTests(unittest.TestCase):
    def test_news_filters_other_sports_cleans_html_and_deduplicates(self):
        items = football.parse_news(RSS, "vnexpress", 10)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["url"], "https://example.org/football")
        self.assertEqual(items[0]["data"]["summary"], "Arsenal score twice.")
        self.assertEqual(items[0]["data"]["published_at"], "2026-09-20T13:35:05+07:00")
        self.assertIn("RSS", items[0]["content_markdown"])

    def test_invalid_feed_is_reported_not_silently_empty(self):
        for body in (b"<html>Access denied</html>", b"<rss><channel>"):
            with self.subTest(body=body), self.assertRaises((ValueError, football.ET.ParseError)):
                football.parse_news(body, "bbc", 5)

    def test_final_score_does_not_depend_on_result_array_order(self):
        result = football.normalize_match(MATCH, "bl1", 2026)
        self.assertEqual(result["score"], {"home": 3, "away": 1})
        self.assertIn("phản lưới", result["content_markdown"])

    def test_half_time_and_unfinished_zero_score_are_not_final_results(self):
        for finished, results in ((False, MATCH["matchResults"]),
                                  (True, [MATCH["matchResults"][1]]), (True, None)):
            item = {**MATCH, "matchIsFinished": finished, "matchResults": results}
            with self.subTest(finished=finished, results=results):
                self.assertIsNone(football.normalize_match(item, "bl1", 2026)["score"])

    def test_extra_time_result_keeps_its_meaning(self):
        item = copy.deepcopy(MATCH)
        item["matchResults"].append({"resultTypeKind": "AfterExtraTime", "resultOrderID": 3,
                                      "pointsTeam1": 4, "pointsTeam2": 2})
        result = football.normalize_match(item, "bl1", 2026)
        self.assertEqual(result["score"], {"home": 4, "away": 2})
        self.assertIn("AfterExtraTime", result["content_markdown"])

    def test_inclusive_utc_date_filter(self):
        day = date(2026, 9, 20)
        self.assertTrue(football.within_dates(MATCH, day, day))
        self.assertFalse(football.within_dates(MATCH, date(2026, 9, 21), None))
        self.assertFalse(football.within_dates({}, day, None))
        self.assertTrue(football.within_dates({}, None, None))
        self.assertTrue(football.within_dates(
            {"matchDateTimeUTC": "2026-09-21T01:00:00+07:00"}, day, day))

    def test_rerun_updates_same_match_file_and_preserves_utf8(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pending = {**MATCH, "matchIsFinished": False}
            football.save_record(football.normalize_match(pending, "bl1", 2026), root)
            football.save_record(football.normalize_match(MATCH, "bl1", 2026), root)
            files = list(root.rglob("*.json"))
            self.assertEqual(len(files), 1)
            self.assertEqual(json.loads(files[0].read_text(encoding="utf-8"))["status"], "finished")
            self.assertIn("Đã kết thúc", next(root.rglob("*.md")).read_text(encoding="utf-8"))
            self.assertEqual(list(root.rglob("*.tmp")), [])

    def test_team_and_player_records_are_scoped_by_league_and_season(self):
        for kind, rows in (
            ("teams", [{"teamId": 1, "teamName": "Club"}]),
            ("players", [{"goalGetterId": 1, "goalGetterName": "Player", "goalCount": 0}]),
        ):
            records = [next(football.normalize_rows(kind, rows, league, season, football.API))
                       for league, season in (("bl1", 2025), ("bl1", 2026), ("bl2", 2026))]
            self.assertEqual(len({r["id"] for r in records}), 3)

    def test_all_categories_export_and_report_partial_failure(self):
        class FakeClient:
            def get(self, url):
                return RSS

            def get_json(self, url):
                if "getmatchdata" in url:
                    return [MATCH]
                if "getavailableteams" in url:
                    return [{"teamId": 1, "teamName": "Club"}]
                if "getbltable" in url:
                    raise URLError("service unavailable")
                return [{"goalGetterId": 1, "goalGetterName": "Player", "goalCount": 2}]

        with tempfile.TemporaryDirectory() as temporary:
            args = football.build_parser().parse_args([
                "--data-dir", temporary, "--season", "2026", "--news-sources", "vnexpress"])
            report = football.crawl(args, FakeClient())
            self.assertEqual(report["counts"], {"news": 1, "matches": 1, "teams": 1,
                                               "standings": 0, "players": 1})
            self.assertEqual(len(report["errors"]), 1)
            self.assertEqual(len(list(Path(temporary).rglob("*.md"))), 4)
            self.assertTrue((Path(temporary) / "football_crawl_report.json").exists())
            news_files = list((Path(temporary) / "landing" / "news").glob("*.json"))
            self.assertEqual(len(news_files), 1)
            self.assertTrue((Path(temporary) / "standardized" / "news"
                             / news_files[0].with_suffix(".md").name).exists())
            self.assertFalse((Path(temporary) / "landing" / "football" / "news").exists())

    def test_http_retries_429_and_respects_retry_after(self):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"[]"
        error = HTTPError(football.API, 429, "Too many requests", {"Retry-After": "3"}, None)
        with patch.object(football, "urlopen", side_effect=[error, response]) as request, \
                patch.object(football.time, "sleep") as sleep:
            self.assertEqual(football.HttpClient().get_json(football.API), [])
            self.assertEqual(request.call_count, 2)
            sleep.assert_any_call(3.0)

    def test_http_404_does_not_retry(self):
        error = HTTPError(football.API, 404, "Not found", {}, None)
        with patch.object(football, "urlopen", side_effect=error) as request:
            with self.assertRaises(HTTPError):
                football.HttpClient().get(football.API)
            self.assertEqual(request.call_count, 1)

    def test_bad_api_shape_is_rejected(self):
        for body in (b'{}', b'[1]', b'not json'):
            with patch.object(football.HttpClient, "get", return_value=body), self.assertRaises(ValueError):
                football.HttpClient().get_json(football.API)

    def test_cli_failure_exit_codes_and_validation(self):
        for report in ({"counts": {}, "total": 0, "errors": [], "warnings": []},
                       {"counts": {}, "total": 5, "errors": ["failure"], "warnings": []}):
            with patch.object(football, "crawl", return_value=report), patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(football.main([]), 1)
        for args in (["--news-limit", "0"], ["--delay", "nan"], ["--leagues", "../bad"],
                     ["--start-date", "2026-10-01", "--end-date", "2026-09-01"]):
            with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit) as error:
                football.main(args)
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
