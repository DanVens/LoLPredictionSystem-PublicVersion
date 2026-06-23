#!/usr/bin/env python3
"""
Scrape Games of Legends lane gold timelines with champion matchup context.

The script can start from:
- a single tournament URL such as /tournament/tournament-matchlist/.../
- a higher-level page that links to many tournaments

It supports both GOL match list styles:
- direct links to /page-game/
- series links to /page-summary/ that must be expanded into per-game pages

Games or tournaments with missing timeline data are skipped and reported.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import quote, urljoin, urlparse

import requests


ROLES = ["TOP", "JGL", "MID", "BOT", "SPT"]
HEADERS = [
    "game_id",
    "match",
    "tournament",
    "patch",
    "date",
    "minute",
    "role",
    "blue_team",
    "red_team",
    "blue_champion",
    "red_champion",
    "blue_gold",
    "red_gold",
    "gold_lead_blue",
    "leading_side",
    "winner_side",
    "winner_team",
    "game_url",
    "timeline_url",
]

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/135.0.0.0 Safari/537.36"
)
GOL_TOURNAMENT_BASE_URL = "https://gol.gg/tournament/"

TOURNAMENT_LINK_RE = re.compile(
    r"""href=['"]([^'"]*?/tournament/(?:tournament-stats|tournament-matchlist)/[^'"]+/)['"]""",
    re.IGNORECASE,
)
SUMMARY_LINK_RE = re.compile(
    r"""href=['"]([^'"]*?/game/stats/\d+/page-summary/?[^'"]*)['"]""",
    re.IGNORECASE,
)
PREVIEW_LINK_RE = re.compile(
    r"""href=['"]([^'"]*?/game/stats/\d+/page-preview/?[^'"]*)['"]""",
    re.IGNORECASE,
)
DIRECT_GAME_ID_RE = re.compile(r"/game/stats/(\d+)/page-game(?:/|\b)", re.IGNORECASE)
SUMMARY_GAME_ID_RE = re.compile(r"/game/stats/(\d+)/page-summary(?:/|\b)", re.IGNORECASE)
DATE_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")
PATCH_RE = re.compile(r">\s*v([0-9.]+)\s*<")
OVERVIEW_GAMES_RE = re.compile(
    r"""Number of games:</td><td[^>]*>\s*(\d+)\s*</td>""",
    re.IGNORECASE,
)
HEADER_BLOCK_RE = re.compile(
    r"""<div class=['"]col-12 (blue|red)-line-header['"]>(.*?)</div>""",
    re.IGNORECASE | re.DOTALL,
)
H1_RE = re.compile(r"<h1>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
TOURNAMENT_ANCHOR_RE = re.compile(
    r"""<a href=['"]([^'"]*/tournament/tournament-stats/[^'"]+/)['"][^>]*>(.*?)</a>""",
    re.IGNORECASE | re.DOTALL,
)
GOLDDATAS_BLOCK_RE = re.compile(
    r"var\s+golddatas\s*=\s*\{(.*?)\}\s*;\s*var\s+csdatas",
    re.IGNORECASE | re.DOTALL,
)
LABELS_RE = re.compile(r"labels:\s*\[(.*?)\]\s*,\s*datasets\s*:", re.IGNORECASE | re.DOTALL)
DATASET_RE = re.compile(
    r"label:\s*'([^']+)'.*?data:\s*\[(.*?)\]",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class GameMeta:
    game_id: str
    match: str
    tournament: str
    patch: str
    date: str
    blue_team: str
    red_team: str
    winner_side: str
    winner_team: str


class PlayersTableParser(HTMLParser):
    """Extract champion order from the two top-level players tables."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[str]] = []
        self._in_players_table = False
        self._nested_table_depth = 0
        self._current_table: list[str] = []
        self._in_row = False
        self._current_td_index = 0
        self._row_champion: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        class_names = set((attr_map.get("class") or "").split())

        if tag == "table":
            if not self._in_players_table and "playersInfosLine" in class_names and len(self.tables) < 2:
                self._in_players_table = True
                self._nested_table_depth = 0
                self._current_table = []
                return
            if self._in_players_table:
                self._nested_table_depth += 1
                return

        if not self._in_players_table or self._nested_table_depth != 0:
            return

        if tag == "tr":
            self._in_row = True
            self._current_td_index = 0
            self._row_champion = None
            return

        if not self._in_row:
            return

        if tag == "td":
            self._current_td_index += 1
            return

        if tag == "img" and self._current_td_index == 1 and self._row_champion is None:
            if "champion_icon" in class_names:
                alt_text = (attr_map.get("alt") or "").strip()
                if alt_text:
                    self._row_champion = alt_text

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._in_players_table:
            if self._nested_table_depth > 0:
                self._nested_table_depth -= 1
                return

            self.tables.append(self._current_table[:])
            self._in_players_table = False
            self._current_table = []
            self._in_row = False
            self._current_td_index = 0
            self._row_champion = None
            return

        if not self._in_players_table or self._nested_table_depth != 0:
            return

        if tag == "tr" and self._in_row:
            if self._row_champion:
                self._current_table.append(self._row_champion)
            self._in_row = False
            self._current_td_index = 0
            self._row_champion = None


@dataclass
class TournamentDiscovery:
    direct_game_ids: list[str]
    summary_urls: list[str]
    preview_urls: list[str]


def parse_overview_game_count(stats_html: str) -> int | None:
    match = OVERVIEW_GAMES_RE.search(stats_html)
    if not match:
        return None
    return int(match.group(1))


def clean_text(raw: str) -> str:
    no_tags = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(no_tags)).strip()


def normalize_matchlist_url(url: str) -> str:
    return url.replace("/tournament/tournament-stats/", "/tournament/tournament-matchlist/")


def infer_output_path(start_url: str) -> str:
    parsed = urlparse(start_url)
    path = parsed.path.strip("/")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", path).strip("_") or "gol_export"
    return f"{slug}_gold_with_champions.csv"


def load_seen_game_ids_from_csv(csv_path: str) -> set[str]:
    seen: set[str] = set()
    if not os.path.exists(csv_path):
        return seen

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "game_id" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} does not contain a game_id column")
        for row in reader:
            game_id = (row.get("game_id") or "").strip()
            if game_id:
                seen.add(game_id)
    return seen


def sleep_with_jitter(delay: float) -> None:
    time.sleep(delay + random.uniform(0.0, min(0.35, delay)))


def split_js_array(raw: str) -> list[str]:
    values: list[str] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if len(item) >= 2 and item[0] == item[-1] and item[0] in {"'", '"'}:
            item = html.unescape(item[1:-1])
        values.append(item)
    return values


def parse_numeric(value: str) -> int | float | str:
    if value == "":
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def fetch_text(session: requests.Session, url: str, timeout: float, retries: int) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if shutil.which("curl"):
                result = subprocess.run(
                    [
                        "curl",
                        "-sS",
                        "-L",
                        "-A",
                        USER_AGENT,
                        "--max-time",
                        str(int(timeout)),
                        url,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    return result.stdout
                raise RuntimeError(result.stderr.strip() or f"curl exited with {result.returncode}")

            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or response.encoding
            return response.text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == retries:
                break
            time.sleep(0.6 * attempt)
    raise RuntimeError(f"request failed for {url}: {last_error}") from last_error


def post_json(
    session: requests.Session,
    url: str,
    data: list[tuple[str, str]],
    timeout: float,
    retries: int,
) -> object:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if shutil.which("curl"):
                cmd = [
                    "curl",
                    "-sS",
                    "-L",
                    "-A",
                    USER_AGENT,
                    "--max-time",
                    str(int(timeout)),
                    "-X",
                    "POST",
                ]
                for key, value in data:
                    cmd.extend(["--data-urlencode", f"{key}={value}"])
                cmd.append(url)

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    return json.loads(result.stdout)
                raise RuntimeError(result.stderr.strip() or f"curl exited with {result.returncode}")

            response = session.post(url, data=data, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == retries:
                break
            time.sleep(0.6 * attempt)
    raise RuntimeError(f"request failed for {url}: {last_error}") from last_error


def extract_tournament_urls(start_url: str, html_text: str) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for href in TOURNAMENT_LINK_RE.findall(html_text):
        absolute = normalize_matchlist_url(
            urljoin(GOL_TOURNAMENT_BASE_URL, html.unescape(href))
        )
        if absolute not in seen:
            seen.add(absolute)
            urls.append(absolute)
    return urls


def extract_tournament_urls_from_ajax(
    session: requests.Session,
    start_url: str,
    season: str,
    leagues: list[str],
    timeout: float,
    retries: int,
) -> list[str]:
    ajax_url = urljoin(start_url, "../ajax.trlist.php")
    tournament_base = urljoin(start_url, "../")
    payload: list[tuple[str, str]] = [("season", season)]
    for league in leagues:
        payload.append(("league[]", league))

    data = post_json(session, ajax_url, payload, timeout=timeout, retries=retries)
    if not isinstance(data, list):
        raise ValueError(f"unexpected ajax response type: {type(data).__name__}")

    urls: list[str] = []
    seen: set[str] = set()
    for row in data:
        if not isinstance(row, dict):
            continue
        trname = str(row.get("trname", "")).strip()
        if not trname:
            continue
        encoded_name = quote(trname, safe="()'-")
        url = normalize_matchlist_url(
            urljoin(tournament_base, f"tournament-stats/{encoded_name}/")
        )
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def extract_matchlist_game_sources(matchlist_url: str, html_text: str) -> TournamentDiscovery:
    direct_game_ids: list[str] = []
    seen_game_ids: set[str] = set()
    for game_id in DIRECT_GAME_ID_RE.findall(html_text):
        if game_id not in seen_game_ids:
            seen_game_ids.add(game_id)
            direct_game_ids.append(game_id)

    summary_urls: list[str] = []
    seen_summaries: set[str] = set()
    for href in SUMMARY_LINK_RE.findall(html_text):
        absolute = urljoin(GOL_TOURNAMENT_BASE_URL, html.unescape(href))
        if absolute not in seen_summaries:
            seen_summaries.add(absolute)
            summary_urls.append(absolute)

    preview_urls: list[str] = []
    seen_previews: set[str] = set()
    for href in PREVIEW_LINK_RE.findall(html_text):
        absolute = urljoin(GOL_TOURNAMENT_BASE_URL, html.unescape(href))
        if absolute not in seen_previews:
            seen_previews.add(absolute)
            preview_urls.append(absolute)

    return TournamentDiscovery(
        direct_game_ids=direct_game_ids,
        summary_urls=summary_urls,
        preview_urls=preview_urls,
    )


def extract_game_ids_from_summary(html_text: str) -> list[str]:
    game_ids: list[str] = []
    seen: set[str] = set()
    for game_id in DIRECT_GAME_ID_RE.findall(html_text):
        if game_id not in seen:
            seen.add(game_id)
            game_ids.append(game_id)
    return game_ids


def parse_header_teams(game_html: str) -> tuple[str, str, str, str]:
    blue_team = ""
    red_team = ""
    blue_result = ""
    red_result = ""

    for side, block in HEADER_BLOCK_RE.findall(game_html):
        cleaned = clean_text(block)
        match = re.match(r"(.+?)\s*-\s*(WIN|LOSS)\s*$", cleaned, re.IGNORECASE)
        if not match:
            continue
        team_name = match.group(1).strip()
        result = match.group(2).upper()
        if side.lower() == "blue":
            blue_team = team_name
            blue_result = result
        else:
            red_team = team_name
            red_result = result

    return blue_team, red_team, blue_result, red_result


def parse_game_meta(game_id: str, game_url: str, game_html: str) -> GameMeta:
    h1_match = H1_RE.search(game_html)
    match_name = clean_text(h1_match.group(1)) if h1_match else ""

    tournament_name = ""
    tournament_anchor = TOURNAMENT_ANCHOR_RE.search(game_html)
    if tournament_anchor:
        tournament_name = clean_text(tournament_anchor.group(2))

    date_match = DATE_RE.search(game_html)
    patch_match = PATCH_RE.search(game_html)

    blue_team, red_team, blue_result, red_result = parse_header_teams(game_html)

    winner_side = ""
    if blue_result == "WIN":
        winner_side = "blue"
    elif red_result == "WIN":
        winner_side = "red"

    winner_team = ""
    if winner_side == "blue":
        winner_team = blue_team
    elif winner_side == "red":
        winner_team = red_team

    return GameMeta(
        game_id=game_id,
        match=match_name,
        tournament=tournament_name,
        patch=patch_match.group(1) if patch_match else "",
        date=date_match.group(0) if date_match else "",
        blue_team=blue_team,
        red_team=red_team,
        winner_side=winner_side,
        winner_team=winner_team,
    )


def parse_champions_by_role(game_html: str) -> tuple[list[str], list[str]]:
    parser = PlayersTableParser()
    parser.feed(game_html)
    parser.close()

    if len(parser.tables) < 2:
        raise ValueError("could not find both players tables")

    blue = parser.tables[0][:5]
    red = parser.tables[1][:5]

    if len(blue) < 5 or len(red) < 5:
        raise ValueError(
            f"not enough champion rows found (blue={len(blue)}, red={len(red)})"
        )

    return blue, red


def parse_gold_datasets(timeline_html: str) -> tuple[list[str], list[list[int | float | str]]]:
    block_match = GOLDDATAS_BLOCK_RE.search(timeline_html)
    if not block_match:
        raise ValueError("missing golddatas block")

    block = block_match.group(1)
    labels_match = LABELS_RE.search(block)
    if not labels_match:
        raise ValueError("missing golddatas labels")

    labels = split_js_array(labels_match.group(1))
    datasets: list[list[int | float | str]] = []
    dataset_labels: list[str] = []

    for dataset_match in DATASET_RE.finditer(block):
        dataset_labels.append(dataset_match.group(1).strip())
        data_values = [parse_numeric(v) for v in split_js_array(dataset_match.group(2))]
        datasets.append(data_values)

    if len(datasets) < 10:
        raise ValueError(f"expected 10 gold datasets, found {len(datasets)}")

    expected_roles = ROLES + ROLES
    if dataset_labels[:10] != expected_roles:
        # GOL has been consistent here, but this warning makes future weirdness visible.
        print(
            f"warning: unexpected dataset labels {dataset_labels[:10]}",
            file=sys.stderr,
        )

    return labels, datasets[:10]


def iter_game_rows(
    meta: GameMeta,
    blue_champs: list[str],
    red_champs: list[str],
    labels: list[str],
    datasets: list[list[int | float | str]],
    game_url: str,
    timeline_url: str,
) -> Iterable[dict[str, object]]:
    blue_gold_by_role = {role: datasets[idx] for idx, role in enumerate(ROLES)}
    red_gold_by_role = {role: datasets[idx + 5] for idx, role in enumerate(ROLES)}

    for minute_index, minute in enumerate(labels):
        for role_index, role in enumerate(ROLES):
            blue_gold = blue_gold_by_role[role][minute_index]
            red_gold = red_gold_by_role[role][minute_index]

            gold_lead_blue: int | float | str = ""
            if isinstance(blue_gold, (int, float)) and isinstance(red_gold, (int, float)):
                gold_lead_blue = blue_gold - red_gold

            if gold_lead_blue == "":
                leading_side = ""
            elif gold_lead_blue > 0:
                leading_side = "blue"
            elif gold_lead_blue < 0:
                leading_side = "red"
            else:
                leading_side = "tied"

            yield {
                "game_id": meta.game_id,
                "match": meta.match,
                "tournament": meta.tournament,
                "patch": meta.patch,
                "date": meta.date,
                "minute": minute,
                "role": role,
                "blue_team": meta.blue_team,
                "red_team": meta.red_team,
                "blue_champion": blue_champs[role_index],
                "red_champion": red_champs[role_index],
                "blue_gold": blue_gold,
                "red_gold": red_gold,
                "gold_lead_blue": gold_lead_blue,
                "leading_side": leading_side,
                "winner_side": meta.winner_side,
                "winner_team": meta.winner_team,
                "game_url": game_url,
                "timeline_url": timeline_url,
            }


def scrape_game(
    session: requests.Session,
    game_id: str,
    timeout: float,
    retries: int,
) -> tuple[GameMeta, list[str], list[str], list[str], list[list[int | float | str]]]:
    game_url = f"https://gol.gg/game/stats/{game_id}/page-game/"
    timeline_url = f"https://gol.gg/game/stats/{game_id}/page-timeline/"

    game_html = fetch_text(session, game_url, timeout=timeout, retries=retries)
    timeline_html = fetch_text(session, timeline_url, timeout=timeout, retries=retries)

    meta = parse_game_meta(game_id, game_url, game_html)
    blue_champs, red_champs = parse_champions_by_role(game_html)
    labels, datasets = parse_gold_datasets(timeline_html)

    return meta, blue_champs, red_champs, labels, datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape GOL lane gold timelines with champions and winners.",
    )
    parser.add_argument(
        "start_url",
        help=(
            "A GOL tournament match list/stats URL, or a higher-level page that links "
            "to multiple tournaments."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV path. Defaults to a name derived from the start URL.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Base delay in seconds between requests. Default: 0.35",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=25.0,
        help="Request timeout in seconds. Default: 25",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Request retries per page. Default: 3",
    )
    parser.add_argument(
        "--tournament-limit",
        type=int,
        default=0,
        help="Optional cap on tournaments to process for testing.",
    )
    parser.add_argument(
        "--game-limit",
        type=int,
        default=0,
        help="Optional cap on games to process for testing.",
    )
    parser.add_argument(
        "--season",
        help=(
            "Season code for /tournament/list/ pages, for example S16. "
            "Required if you want to scrape all tournaments from the tournament list for one season."
        ),
    )
    parser.add_argument(
        "--league",
        action="append",
        default=[],
        help=(
            "Optional league filter for /tournament/list/ AJAX loading, for example "
            "--league LEC --league LCK. Can be repeated."
        ),
    )
    parser.add_argument(
        "--summary-output",
        help=(
            "Optional CSV path for a per-tournament audit report showing overview, "
            "discovered, attempted, processed, and skipped games."
        ),
    )
    parser.add_argument(
        "--debug-discovery-output",
        help=(
            "Optional JSON path that records the discovered game ids for each tournament "
            "before scraping individual game pages."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from an existing output CSV by skipping any game_id values already "
            "present there."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output or infer_output_path(args.start_url)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
        }
    )

    print(f"start url: {args.start_url}")
    print(f"output: {output_path}")

    start_html = fetch_text(session, args.start_url, timeout=args.timeout, retries=args.retries)

    parsed_start = urlparse(args.start_url)
    if "/tournament/list/" in parsed_start.path:
        if not args.season:
            print(
                "For /tournament/list/ you need to pass --season, for example --season S16.",
                file=sys.stderr,
            )
            return 1
        tournament_urls = extract_tournament_urls_from_ajax(
            session,
            args.start_url,
            season=args.season,
            leagues=args.league,
            timeout=args.timeout,
            retries=args.retries,
        )
    elif "/tournament/tournament-matchlist/" in parsed_start.path or "/tournament/tournament-stats/" in parsed_start.path:
        tournament_urls = [normalize_matchlist_url(args.start_url)]
    else:
        tournament_urls = extract_tournament_urls(args.start_url, start_html)

    if args.tournament_limit > 0:
        tournament_urls = tournament_urls[: args.tournament_limit]

    if not tournament_urls:
        print("No tournament URLs found from the provided page.", file=sys.stderr)
        return 1

    print(f"tournaments found: {len(tournament_urls)}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    seen_game_ids: set[str] = set()
    append_mode = False
    if args.resume and os.path.exists(output_path):
        seen_game_ids = load_seen_game_ids_from_csv(output_path)
        append_mode = True
        print(f"resume mode: loaded {len(seen_game_ids)} existing game ids from {output_path}")

    total_rows = 0
    skipped_tournaments = 0
    skipped_games = 0
    processed_games = 0
    tournament_reports: list[dict[str, object]] = []
    discovery_debug: list[dict[str, object]] = []

    with open(output_path, "a" if append_mode else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        if not append_mode:
            writer.writeheader()

        for tournament_index, tournament_url in enumerate(tournament_urls, start=1):
            print(f"[tournament {tournament_index}/{len(tournament_urls)}] {tournament_url}")

            try:
                stats_url = tournament_url.replace(
                    "/tournament/tournament-matchlist/",
                    "/tournament/tournament-stats/",
                )
                stats_html = fetch_text(
                    session,
                    stats_url,
                    timeout=args.timeout,
                    retries=args.retries,
                )
                matchlist_html = fetch_text(
                    session,
                    tournament_url,
                    timeout=args.timeout,
                    retries=args.retries,
                )
            except Exception as exc:  # noqa: BLE001
                skipped_tournaments += 1
                print(f"  skip tournament fetch failed: {exc}", file=sys.stderr)
                continue

            discovery = extract_matchlist_game_sources(tournament_url, matchlist_html)
            overview_game_count = parse_overview_game_count(stats_html)
            discovered_game_ids: list[str] = []
            seen_discovered_game_ids: set[str] = set()

            for game_id in discovery.direct_game_ids:
                if game_id not in seen_discovered_game_ids:
                    seen_discovered_game_ids.add(game_id)
                    discovered_game_ids.append(game_id)

            print(
                "  "
                f"overview number of games: {overview_game_count if overview_game_count is not None else 'unknown'}"
            )

            for summary_url in discovery.summary_urls:
                try:
                    summary_html = fetch_text(
                        session,
                        summary_url,
                        timeout=args.timeout,
                        retries=args.retries,
                    )
                    for game_id in extract_game_ids_from_summary(summary_html):
                        if game_id not in seen_discovered_game_ids:
                            seen_discovered_game_ids.add(game_id)
                            discovered_game_ids.append(game_id)
                except Exception as exc:  # noqa: BLE001
                    print(f"  skip summary page {summary_url}: {exc}", file=sys.stderr)
                sleep_with_jitter(args.delay)

            new_game_ids = [game_id for game_id in discovered_game_ids if game_id not in seen_game_ids]
            duplicate_game_count = len(discovered_game_ids) - len(new_game_ids)
            discovered_new_game_count = len(new_game_ids)

            print(
                "  "
                f"discovered game ids: {len(discovered_game_ids)} | "
                f"new this run: {discovered_new_game_count} | "
                f"already seen: {duplicate_game_count}"
            )
            if overview_game_count is not None and len(discovered_game_ids) != overview_game_count:
                print(
                    "  "
                    f"warning: overview says {overview_game_count} games, "
                    f"but discovery found {len(discovered_game_ids)}",
                    file=sys.stderr,
                )

            discovery_debug.append(
                {
                    "tournament_url": tournament_url,
                    "overview_games": overview_game_count,
                    "discovered_games": len(discovered_game_ids),
                    "discovered_game_ids": discovered_game_ids,
                }
            )

            tournament_processed = 0
            tournament_skipped = 0

            if not new_game_ids:
                skipped_tournaments += 1
                print("  no completed games with usable links found")
                tournament_reports.append(
                    {
                        "tournament_url": tournament_url,
                        "overview_games": overview_game_count if overview_game_count is not None else "",
                        "discovered_games": len(discovered_game_ids),
                        "missing_from_discovery": (
                            overview_game_count - len(discovered_game_ids)
                            if overview_game_count is not None
                            else ""
                        ),
                        "new_games": discovered_new_game_count,
                        "attempted_games": 0,
                        "processed_games": tournament_processed,
                        "skipped_games": tournament_skipped,
                        "duplicate_games": duplicate_game_count,
                    }
                )
                continue

            if args.game_limit > 0:
                remaining = max(args.game_limit - processed_games, 0)
                if remaining == 0:
                    break
                new_game_ids = new_game_ids[:remaining]

            attempted_game_count = len(new_game_ids)

            for game_index, game_id in enumerate(new_game_ids, start=1):
                game_url = f"https://gol.gg/game/stats/{game_id}/page-game/"
                timeline_url = f"https://gol.gg/game/stats/{game_id}/page-timeline/"
                print(f"  [game {game_index}/{len(new_game_ids)}] {game_id}")

                try:
                    meta, blue_champs, red_champs, labels, datasets = scrape_game(
                        session,
                        game_id,
                        timeout=args.timeout,
                        retries=args.retries,
                    )

                    row_count = 0
                    for row in iter_game_rows(
                        meta,
                        blue_champs,
                        red_champs,
                        labels,
                        datasets,
                        game_url,
                        timeline_url,
                    ):
                        writer.writerow(row)
                        row_count += 1

                    handle.flush()
                    total_rows += row_count
                    processed_games += 1
                    tournament_processed += 1
                    seen_game_ids.add(game_id)
                    print(
                        f"    wrote {row_count} rows | "
                        f"{meta.blue_team} vs {meta.red_team} | {meta.date}"
                    )
                except Exception as exc:  # noqa: BLE001
                    skipped_games += 1
                    tournament_skipped += 1
                    print(f"    skip game {game_id}: {exc}", file=sys.stderr)

                sleep_with_jitter(args.delay)

                if args.game_limit > 0 and processed_games >= args.game_limit:
                    break

            tournament_reports.append(
                {
                    "tournament_url": tournament_url,
                    "overview_games": overview_game_count if overview_game_count is not None else "",
                    "discovered_games": len(discovered_game_ids),
                    "missing_from_discovery": (
                        overview_game_count - len(discovered_game_ids)
                        if overview_game_count is not None
                        else ""
                    ),
                    "new_games": discovered_new_game_count,
                    "attempted_games": attempted_game_count,
                    "processed_games": tournament_processed,
                    "skipped_games": tournament_skipped,
                    "duplicate_games": duplicate_game_count,
                }
            )

            if args.game_limit > 0 and processed_games >= args.game_limit:
                break

            sleep_with_jitter(args.delay)

    if args.summary_output:
        summary_dir = os.path.dirname(args.summary_output) or "."
        os.makedirs(summary_dir, exist_ok=True)
        with open(args.summary_output, "w", newline="", encoding="utf-8") as summary_handle:
            summary_writer = csv.DictWriter(
                summary_handle,
                fieldnames=[
                    "tournament_url",
                    "overview_games",
                    "discovered_games",
                    "missing_from_discovery",
                    "new_games",
                    "attempted_games",
                    "processed_games",
                    "skipped_games",
                    "duplicate_games",
                ],
            )
            summary_writer.writeheader()
            summary_writer.writerows(tournament_reports)

    if args.debug_discovery_output:
        debug_dir = os.path.dirname(args.debug_discovery_output) or "."
        os.makedirs(debug_dir, exist_ok=True)
        with open(args.debug_discovery_output, "w", encoding="utf-8") as debug_handle:
            json.dump(discovery_debug, debug_handle, ensure_ascii=False, indent=2)

    print("")
    print("done")
    print(f"rows written: {total_rows}")
    print(f"games processed: {processed_games}")
    print(f"games skipped: {skipped_games}")
    print(f"tournaments skipped: {skipped_tournaments}")
    if args.summary_output:
        print(f"summary saved to: {args.summary_output}")
    if args.debug_discovery_output:
        print(f"discovery debug saved to: {args.debug_discovery_output}")
    print(f"csv saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
