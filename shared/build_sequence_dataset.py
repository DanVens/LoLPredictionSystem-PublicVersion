#!/usr/bin/env python3
"""
Build a sequence-ready dataset from GOL export CSV files.

Each game becomes one sequence over minutes. Each timestep contains all five
roles together, which is a much better shape for time-series models than
treating each role-row as a separate example.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np


ROLES = ["TOP", "JGL", "MID", "BOT", "SPT"]
ROLE_TO_INDEX = {role: idx for idx, role in enumerate(ROLES)}

TIMESTEP_FEATURE_NAMES = (
    [f"blue_gold_{role.lower()}" for role in ROLES]
    + [f"red_gold_{role.lower()}" for role in ROLES]
    + [f"gold_lead_{role.lower()}" for role in ROLES]
    + ["blue_gold_total", "red_gold_total", "gold_lead_total", "blue_lanes_ahead"]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a time-series dataset from GOL export CSV files.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "CSV files or glob patterns to include. If omitted, defaults to "
            "'s15_all.csv' plus 'data/*.csv' excluding generated training tables."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default="data/sequence_dataset",
        help="Output prefix for .npz and .json files. Default: data/sequence_dataset",
    )
    parser.add_argument(
        "--max-minute",
        type=int,
        default=45,
        help="Maximum minute to keep in each sequence. Default: 45",
    )
    parser.add_argument(
        "--min-sequence-length",
        type=int,
        default=10,
        help="Minimum number of timesteps required to keep a game. Default: 10",
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
            base = os.path.basename(path)
            if path.endswith(":Zone.Identifier"):
                continue
            if base.startswith("training_table"):
                continue
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def infer_source_season(path: str, date_text: str) -> str:
    base = os.path.basename(path).lower()
    if base.startswith("s15"):
        return "S15"
    if base.startswith("s16"):
        return "S16"
    if date_text.startswith("2025"):
        return "S15"
    if date_text.startswith("2026"):
        return "S16"
    return ""


def safe_float(raw: str) -> float:
    value = str(raw).strip()
    return float(value) if value else 0.0


def safe_int(raw: str) -> int | None:
    value = str(raw).strip()
    if not value:
        return None
    return int(float(value))


def parse_patch(raw: str) -> tuple[int, int]:
    raw = str(raw).strip()
    if not raw:
        return 0, 0
    parts = raw.split(".")
    try:
        major = int(parts[0])
    except ValueError:
        major = 0
    try:
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        minor = 0
    return major, minor


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


def get_or_add(mapping: dict[str, int], value: str) -> int:
    if value not in mapping:
        mapping[value] = len(mapping) + 1
    return mapping[value]


def main() -> int:
    args = parse_args()
    input_paths = resolve_input_paths(args.inputs)
    if not input_paths:
        raise SystemExit("No input CSV files found.")

    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)

    # Group raw rows by game_id and minute.
    games: dict[str, dict[str, object]] = {}
    duplicate_game_ids: set[str] = set()
    input_file_counter = Counter()

    for path in input_paths:
        input_file_counter[path] += 1
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                game_id = (row.get("game_id") or "").strip()
                minute = safe_int(row.get("minute", ""))
                role = (row.get("role") or "").strip()

                if not game_id or minute is None or role not in ROLE_TO_INDEX:
                    continue
                if minute > args.max_minute:
                    continue

                game = games.setdefault(
                    game_id,
                    {
                        "date": (row.get("date") or "").strip(),
                        "match": (row.get("match") or "").strip(),
                        "tournament": (row.get("tournament") or "").strip(),
                        "winner_side": (row.get("winner_side") or "").strip(),
                        "winner_team": (row.get("winner_team") or "").strip(),
                        "patch": (row.get("patch") or "").strip(),
                        "source_file": path,
                        "source_season": infer_source_season(path, (row.get("date") or "").strip()),
                        "blue_team": (row.get("blue_team") or "").strip(),
                        "red_team": (row.get("red_team") or "").strip(),
                        "blue_champions": [""] * 5,
                        "red_champions": [""] * 5,
                        "minutes": defaultdict(lambda: {"blue": [0.0] * 5, "red": [0.0] * 5, "lead": [0.0] * 5}),
                    },
                )

                if game["source_file"] != path:
                    duplicate_game_ids.add(game_id)
                    continue

                role_index = ROLE_TO_INDEX[role]
                game["blue_champions"][role_index] = (row.get("blue_champion") or "").strip()
                game["red_champions"][role_index] = (row.get("red_champion") or "").strip()

                minute_entry = game["minutes"][minute]
                minute_entry["blue"][role_index] = safe_float(row.get("blue_gold", ""))
                minute_entry["red"][role_index] = safe_float(row.get("red_gold", ""))
                minute_entry["lead"][role_index] = safe_float(row.get("gold_lead_blue", ""))

    # Remove duplicate games that came from multiple files to keep one clean source.
    for game_id in duplicate_game_ids:
        games.pop(game_id, None)

    champion_to_id: dict[str, int] = {}
    tournament_to_id: dict[str, int] = {}
    season_to_id: dict[str, int] = {}
    stage_to_id: dict[str, int] = {}
    league_to_id: dict[str, int] = {}
    team_to_id: dict[str, int] = {}

    kept_games: list[dict[str, object]] = []
    length_counter = Counter()

    for game_id, game in games.items():
        minute_keys = sorted(game["minutes"].keys())
        if len(minute_keys) < args.min_sequence_length:
            continue
        kept_games.append({"game_id": game_id, **game, "minute_keys": minute_keys})
        length_counter[len(minute_keys)] += 1

    kept_games.sort(key=lambda item: (item["date"], item["game_id"]))

    num_games = len(kept_games)
    max_len = max(len(game["minute_keys"]) for game in kept_games) if kept_games else 0
    num_features = len(TIMESTEP_FEATURE_NAMES)

    X = np.zeros((num_games, max_len, num_features), dtype=np.float32)
    mask = np.zeros((num_games, max_len), dtype=np.float32)
    minutes = np.full((num_games, max_len), -1, dtype=np.int16)
    y = np.zeros(num_games, dtype=np.int8)
    patch = np.zeros((num_games, 2), dtype=np.int16)
    blue_champion_ids = np.zeros((num_games, 5), dtype=np.int16)
    red_champion_ids = np.zeros((num_games, 5), dtype=np.int16)
    tournament_ids = np.zeros(num_games, dtype=np.int16)
    season_ids = np.zeros(num_games, dtype=np.int16)
    stage_ids = np.zeros(num_games, dtype=np.int16)
    league_ids = np.zeros(num_games, dtype=np.int16)
    blue_team_ids = np.zeros(num_games, dtype=np.int16)
    red_team_ids = np.zeros(num_games, dtype=np.int16)
    game_ids = np.empty(num_games, dtype="<U16")
    dates = np.empty(num_games, dtype="<U10")

    for game_idx, game in enumerate(kept_games):
        game_ids[game_idx] = game["game_id"]
        dates[game_idx] = game["date"]
        y[game_idx] = 1 if game["winner_side"] == "blue" else 0
        patch_major, patch_minor = parse_patch(game["patch"])
        patch[game_idx] = [patch_major, patch_minor]
        tournament_ids[game_idx] = get_or_add(tournament_to_id, game["tournament"])
        season_ids[game_idx] = get_or_add(season_to_id, game["source_season"])
        stage_ids[game_idx] = get_or_add(stage_to_id, classify_tournament_stage(game["tournament"]))
        league_ids[game_idx] = get_or_add(league_to_id, extract_league_name(game["tournament"]))
        blue_team_ids[game_idx] = get_or_add(team_to_id, game["blue_team"])
        red_team_ids[game_idx] = get_or_add(team_to_id, game["red_team"])

        for role_idx, champion in enumerate(game["blue_champions"]):
            blue_champion_ids[game_idx, role_idx] = get_or_add(champion_to_id, champion)
        for role_idx, champion in enumerate(game["red_champions"]):
            red_champion_ids[game_idx, role_idx] = get_or_add(champion_to_id, champion)

        for step_idx, minute in enumerate(game["minute_keys"]):
            minute_entry = game["minutes"][minute]
            blue = minute_entry["blue"]
            red = minute_entry["red"]
            lead = minute_entry["lead"]

            feature_values = (
                blue
                + red
                + lead
                + [
                    float(sum(blue)),
                    float(sum(red)),
                    float(sum(lead)),
                    float(sum(1 for value in lead if value > 0)),
                ]
            )
            X[game_idx, step_idx, :] = np.asarray(feature_values, dtype=np.float32)
            mask[game_idx, step_idx] = 1.0
            minutes[game_idx, step_idx] = minute

    np.savez_compressed(
        f"{args.output_prefix}.npz",
        X=X,
        mask=mask,
        minutes=minutes,
        y=y,
        patch=patch,
        blue_champion_ids=blue_champion_ids,
        red_champion_ids=red_champion_ids,
        tournament_ids=tournament_ids,
        season_ids=season_ids,
        stage_ids=stage_ids,
        league_ids=league_ids,
        blue_team_ids=blue_team_ids,
        red_team_ids=red_team_ids,
        game_ids=game_ids,
        dates=dates,
    )

    metadata = {
        "input_paths": input_paths,
        "num_games": num_games,
        "max_sequence_length": max_len,
        "num_timestep_features": num_features,
        "timestep_feature_names": TIMESTEP_FEATURE_NAMES,
        "roles": ROLES,
        "max_minute": args.max_minute,
        "min_sequence_length": args.min_sequence_length,
        "champion_to_id": champion_to_id,
        "tournament_to_id": tournament_to_id,
        "season_to_id": season_to_id,
        "stage_to_id": stage_to_id,
        "league_to_id": league_to_id,
        "team_to_id": team_to_id,
        "sequence_length_distribution": dict(sorted(length_counter.items())),
        "duplicate_games_skipped": len(duplicate_game_ids),
    }
    with open(f"{args.output_prefix}.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"games kept: {num_games}")
    print(f"max sequence length: {max_len}")
    print(f"timestep features: {num_features}")
    print(f"duplicate games skipped: {len(duplicate_game_ids)}")
    print(f"sequence length distribution: {dict(length_counter.most_common(15))}")
    print(f"saved arrays to: {args.output_prefix}.npz")
    print(f"saved metadata to: {args.output_prefix}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
