#!/usr/bin/env python3
"""
Build a clean training table from GOL export CSV files.

The output keeps one row per game + minute + role, optionally restricted to
specific checkpoint minutes such as 10/15/20.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import Counter


BASE_COLUMNS = [
    "source_file",
    "source_season",
    "year",
    "game_id",
    "match",
    "tournament",
    "league",
    "tournament_stage",
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
    "gold_lead_red",
    "leading_side",
    "winner_side",
    "winner_team",
    "blue_win",
    "is_leading_blue",
    "champion_matchup",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine GOL exports into a model-ready training CSV.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "CSV files or glob patterns to include. If omitted, defaults to "
            "'s15_all.csv' plus 'data/*.csv' when present."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/training_table_10_15_20.csv",
        help="Output CSV path. Default: data/training_table_10_15_20.csv",
    )
    parser.add_argument(
        "--checkpoints",
        default="10,15,20",
        help=(
            "Comma-separated minutes to keep. Use 'all' to keep every minute. "
            "Default: 10,15,20"
        ),
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Keep duplicate game_id+minute+role rows instead of deduplicating them.",
    )
    return parser.parse_args()


def resolve_input_paths(raw_inputs: list[str]) -> list[str]:
    if not raw_inputs:
        raw_inputs = []
        if os.path.exists("s15_all.csv"):
            raw_inputs.append("s15_all.csv")
        raw_inputs.append("data/*.csv")

    paths: list[str] = []
    seen: set[str] = set()
    for entry in raw_inputs:
        matches = sorted(glob.glob(entry))
        if not matches and os.path.isfile(entry):
            matches = [entry]
        for path in matches:
            if path.endswith(":Zone.Identifier"):
                continue
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def parse_checkpoints(raw: str) -> set[str] | None:
    if raw.strip().lower() == "all":
        return None
    values = {part.strip() for part in raw.split(",") if part.strip()}
    if not values:
        raise ValueError("No checkpoint minutes were provided.")
    return values


def infer_source_season(path: str, row: dict[str, str]) -> str:
    lower_name = os.path.basename(path).lower()
    if lower_name.startswith("s15"):
        return "S15"
    if lower_name.startswith("s16"):
        return "S16"

    year = (row.get("date") or "")[:4]
    if year == "2025":
        return "S15"
    if year == "2026":
        return "S16"
    return ""


def to_int_string(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    return str(int(float(value)))


def normalize_gold(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(number)


def normalize_tournament_name(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(raw).lower()).strip()


def classify_tournament_stage(raw: str) -> str:
    name = normalize_tournament_name(raw)
    if not name:
        return "other"

    international_markers = (
        "worlds",
        "world championship",
        "esports world cup",
        "ewc",
        "mid season invitational",
        "msi",
        "first stand",
        "all star",
        "rift rivals",
        "international",
        "emea masters",
        "european masters",
    )
    if any(marker in name for marker in international_markers):
        return "international"

    playoff_markers = (
        "playoff",
        "play in",
        "regional finals",
        "season finals",
        "final four",
        "finals",
        "main event",
        "championship",
        "lcq",
        "knockout",
        "gauntlet",
    )
    if any(marker in name for marker in playoff_markers):
        return "playoffs"

    regular_markers = (
        "season",
        "split",
        "summer",
        "spring",
        "winter",
        "rounds",
        "regular",
        "versus",
    )
    if any(marker in name for marker in regular_markers):
        return "regular_season"

    return "other"


def extract_league_name(raw: str) -> str:
    text = " ".join(str(raw).split()).strip()
    if not text:
        return "OTHER"

    if classify_tournament_stage(text) == "international":
        return "INTERNATIONAL"

    year_match = re.search(r"\b20\d{2}\b", text)
    if year_match:
        prefix = text[: year_match.start()].strip(" -")
        if prefix:
            return prefix

    split_match = re.search(
        r"\b(playoffs?|play[- ]?in|main event|season finals|regional finals|split|season|spring|summer|winter|versus|cup|kick[- ]?off|invitational|championship)\b",
        text,
        flags=re.IGNORECASE,
    )
    if split_match:
        prefix = text[: split_match.start()].strip(" -")
        if prefix:
            return prefix

    return text


def build_row(source_path: str, row: dict[str, str]) -> dict[str, str]:
    minute = to_int_string(row.get("minute", ""))
    blue_gold = normalize_gold(row.get("blue_gold", ""))
    red_gold = normalize_gold(row.get("red_gold", ""))
    gold_lead_blue = normalize_gold(row.get("gold_lead_blue", ""))
    gold_lead_red = ""
    if gold_lead_blue:
        gold_lead_red = normalize_gold(str(-float(gold_lead_blue)))

    winner_side = (row.get("winner_side") or "").strip()
    leading_side = (row.get("leading_side") or "").strip()

    return {
        "source_file": source_path,
        "source_season": infer_source_season(source_path, row),
        "year": (row.get("date") or "")[:4],
        "game_id": (row.get("game_id") or "").strip(),
        "match": (row.get("match") or "").strip(),
        "tournament": (row.get("tournament") or "").strip(),
        "league": extract_league_name((row.get("tournament") or "").strip()),
        "tournament_stage": classify_tournament_stage((row.get("tournament") or "").strip()),
        "patch": (row.get("patch") or "").strip(),
        "date": (row.get("date") or "").strip(),
        "minute": minute,
        "role": (row.get("role") or "").strip(),
        "blue_team": (row.get("blue_team") or "").strip(),
        "red_team": (row.get("red_team") or "").strip(),
        "blue_champion": (row.get("blue_champion") or "").strip(),
        "red_champion": (row.get("red_champion") or "").strip(),
        "blue_gold": blue_gold,
        "red_gold": red_gold,
        "gold_lead_blue": gold_lead_blue,
        "gold_lead_red": gold_lead_red,
        "leading_side": leading_side,
        "winner_side": winner_side,
        "winner_team": (row.get("winner_team") or "").strip(),
        "blue_win": "1" if winner_side == "blue" else "0",
        "is_leading_blue": "1" if leading_side == "blue" else "0",
        "champion_matchup": (
            f"{(row.get('blue_champion') or '').strip()}_vs_{(row.get('red_champion') or '').strip()}"
        ),
    }


def main() -> int:
    args = parse_args()
    checkpoints = parse_checkpoints(args.checkpoints)
    input_paths = resolve_input_paths(args.inputs)

    if not input_paths:
        raise SystemExit("No input CSV files found.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    seen_keys: set[tuple[str, str, str]] = set()
    files_used = 0
    rows_written = 0
    duplicates_skipped = 0
    minute_counter: Counter[str] = Counter()
    season_counter: Counter[str] = Counter()
    game_ids: set[str] = set()

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASE_COLUMNS)
        writer.writeheader()

        for path in input_paths:
            with open(path, newline="", encoding="utf-8") as source_handle:
                reader = csv.DictReader(source_handle)
                if not reader.fieldnames:
                    continue

                files_used += 1
                for row in reader:
                    built = build_row(path, row)
                    minute = built["minute"]
                    if checkpoints is not None and minute not in checkpoints:
                        continue

                    dedupe_key = (built["game_id"], minute, built["role"])
                    if not args.allow_duplicates and dedupe_key in seen_keys:
                        duplicates_skipped += 1
                        continue

                    seen_keys.add(dedupe_key)
                    writer.writerow(built)
                    rows_written += 1
                    minute_counter[minute] += 1
                    season_counter[built["source_season"]] += 1
                    if built["game_id"]:
                        game_ids.add(built["game_id"])

    print(f"files used: {files_used}")
    print(f"rows written: {rows_written}")
    print(f"unique games: {len(game_ids)}")
    print(f"duplicates skipped: {duplicates_skipped}")
    print(f"minutes kept: {dict(sorted(minute_counter.items(), key=lambda item: int(item[0])))}")
    print(f"rows by source season: {dict(season_counter)}")
    print(f"saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
