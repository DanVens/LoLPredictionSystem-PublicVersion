#!/usr/bin/env python3
"""
Shared sequence-aware tabular feature building for LogReg, XGBoost, and MLP.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np


ROLE_ORDER = ["TOP", "JGL", "MID", "BOT", "SPT"]
ROLE_ORDER_LOWER = [role.lower() for role in ROLE_ORDER]

VARIANT_CONFIGS = {
    "gold_only": {
        "use_champions": False,
        "use_context": False,
    },
    "gold_champions": {
        "use_champions": True,
        "use_context": False,
    },
    "gold_champions_context": {
        "use_champions": True,
        "use_context": True,
    },
}

PER_MINUTE_BASE_FEATURE_NAMES = [
    "blue_gold_total_k",
    "red_gold_total_k",
    "gold_lead_total_k",
    "gold_lead_abs_k",
    "blue_lanes_ahead",
]

ROLE_NUMERIC_SUFFIXES = [
    "blue_gold_k",
    "red_gold_k",
    "gold_lead_k",
]


@dataclass
class GamePrefixRecord:
    game_id: str
    game_date: str
    prefix_minute: int
    label: int
    numeric_features: dict[str, float]
    categorical_tokens: list[str]
    observed_minutes: list[int]


@dataclass
class GameSnapshotRecord:
    game_id: str
    game_date: str
    minute: int
    label: int
    numeric_features: dict[str, float]
    categorical_tokens: list[str]


def safe_float(raw: str) -> float:
    value = str(raw or "").strip()
    return float(value) if value else 0.0


def parse_patch(raw: str) -> tuple[float, float]:
    text = str(raw or "").strip()
    if not text:
        return 0.0, 0.0
    parts = text.split(".")
    try:
        major = float(parts[0])
    except ValueError:
        major = 0.0
    try:
        minor = float(parts[1]) if len(parts) > 1 else 0.0
    except ValueError:
        minor = 0.0
    return major, minor


def minute_numeric_feature_names() -> list[str]:
    names = PER_MINUTE_BASE_FEATURE_NAMES[:]
    for role in ROLE_ORDER:
        for suffix in ROLE_NUMERIC_SUFFIXES:
            names.append(f"{role.lower()}_{suffix}")
    return names


def snapshot_numeric_feature_names(use_context: bool, include_bias: bool = False) -> list[str]:
    names: list[str] = []
    if include_bias:
        names.append("bias")
    names.extend(["current_minute"])
    names.extend(minute_numeric_feature_names())
    if use_context:
        names.extend(["patch_major", "patch_minor"])
    return names


def numeric_feature_names(prefix_minute: int, use_context: bool, include_bias: bool = False) -> list[str]:
    names: list[str] = []
    if include_bias:
        names.append("bias")
    names.extend(
        [
            "observed_minutes_count",
            "observed_minutes_fraction",
            "last_observed_minute",
        ]
    )
    base_minute_names = minute_numeric_feature_names()
    for minute in range(prefix_minute + 1):
        names.append(f"minute_{minute}_observed")
        for base_name in base_minute_names:
            names.append(f"minute_{minute}_{base_name}")
    if use_context:
        names.extend(["patch_major", "patch_minor"])
    return names


def build_minute_numeric_features(rows_by_role: dict[str, dict[str, str]]) -> dict[str, float]:
    blue_total = 0.0
    red_total = 0.0
    role_leads: list[float] = []
    numeric: dict[str, float] = {}

    for role in ROLE_ORDER:
        row = rows_by_role[role]
        blue_gold = safe_float(row["blue_gold"]) / 1000.0
        red_gold = safe_float(row["red_gold"]) / 1000.0
        lead = blue_gold - red_gold
        blue_total += blue_gold
        red_total += red_gold
        role_leads.append(lead)
        role_prefix = role.lower()
        numeric[f"{role_prefix}_blue_gold_k"] = blue_gold
        numeric[f"{role_prefix}_red_gold_k"] = red_gold
        numeric[f"{role_prefix}_gold_lead_k"] = lead

    numeric.update(
        {
            "blue_gold_total_k": blue_total,
            "red_gold_total_k": red_total,
            "gold_lead_total_k": blue_total - red_total,
            "gold_lead_abs_k": abs(blue_total - red_total),
            "blue_lanes_ahead": float(sum(1 for lead in role_leads if lead > 0.0)),
        }
    )
    return numeric


def zero_minute_numeric_features() -> dict[str, float]:
    return {name: 0.0 for name in minute_numeric_feature_names()}


def champion_tokens(rows_by_role: dict[str, dict[str, str]]) -> list[str]:
    tokens: list[str] = []
    for role in ROLE_ORDER:
        row = rows_by_role[role]
        blue = row["blue_champion"]
        red = row["red_champion"]
        matchup = row.get("champion_matchup") or f"{blue}_vs_{red}"
        tokens.extend(
            [
                f"blue_champion_{role}={blue}",
                f"red_champion_{role}={red}",
                f"matchup_{role}={matchup}",
                f"role_blue_{role}={role}|{blue}",
                f"role_red_{role}={role}|{red}",
            ]
        )
    return tokens


def context_tokens(base_row: dict[str, str], include_patch_token: bool) -> list[str]:
    tokens = [
        f"source_season={base_row['source_season']}",
        f"league={base_row['league']}",
        f"tournament_stage={base_row['tournament_stage']}",
        f"tournament={base_row['tournament']}",
        f"blue_team={base_row['blue_team']}",
        f"red_team={base_row['red_team']}",
        f"team_matchup={base_row['blue_team']}|{base_row['red_team']}",
    ]
    if include_patch_token:
        tokens.append(f"patch={base_row['patch']}")
    return tokens


def build_prefix_record(
    game_rows_by_minute: dict[int, dict[str, dict[str, str]]],
    prefix_minute: int,
    use_champions: bool,
    use_context: bool,
    include_patch_token: bool,
) -> GamePrefixRecord | None:
    complete_minutes = sorted(
        minute
        for minute, rows_by_role in game_rows_by_minute.items()
        if minute <= prefix_minute and all(role in rows_by_role for role in ROLE_ORDER)
    )
    if prefix_minute not in complete_minutes:
        return None

    base_row = game_rows_by_minute[prefix_minute][ROLE_ORDER[0]]
    numeric: dict[str, float] = {
        "observed_minutes_count": float(len(complete_minutes)),
        "observed_minutes_fraction": float(len(complete_minutes)) / float(prefix_minute + 1),
        "last_observed_minute": float(complete_minutes[-1]),
    }

    last_features: dict[str, float] | None = None
    zero_features = zero_minute_numeric_features()
    complete_minute_set = set(complete_minutes)
    for minute in range(prefix_minute + 1):
        minute_prefix = f"minute_{minute}"
        if minute in complete_minute_set:
            minute_features = build_minute_numeric_features(game_rows_by_minute[minute])
            last_features = minute_features
            observed = 1.0
        elif last_features is not None:
            minute_features = last_features
            observed = 0.0
        else:
            minute_features = zero_features
            observed = 0.0

        numeric[f"{minute_prefix}_observed"] = observed
        for name, value in minute_features.items():
            numeric[f"{minute_prefix}_{name}"] = value

    if use_context:
        patch_major, patch_minor = parse_patch(base_row["patch"])
        numeric["patch_major"] = patch_major
        numeric["patch_minor"] = patch_minor

    tokens: list[str] = []
    if use_champions:
        tokens.extend(champion_tokens(game_rows_by_minute[prefix_minute]))
    if use_context:
        tokens.extend(context_tokens(base_row, include_patch_token=include_patch_token))

    return GamePrefixRecord(
        game_id=base_row["game_id"],
        game_date=base_row["date"],
        prefix_minute=prefix_minute,
        label=int(base_row["blue_win"]),
        numeric_features=numeric,
        categorical_tokens=tokens,
        observed_minutes=complete_minutes,
    )


def build_snapshot_record(
    game_id: str,
    minute: int,
    rows_by_role: dict[str, dict[str, str]],
    *,
    use_champions: bool,
    use_context: bool,
    include_patch_token: bool,
    include_bias: bool,
) -> GameSnapshotRecord:
    numeric = build_minute_numeric_features(rows_by_role)
    numeric["current_minute"] = float(minute)
    if include_bias:
        numeric["bias"] = 1.0

    base_row = rows_by_role[ROLE_ORDER[0]]
    if use_context:
        patch_major, patch_minor = parse_patch(base_row["patch"])
        numeric["patch_major"] = patch_major
        numeric["patch_minor"] = patch_minor

    tokens: list[str] = []
    if use_champions:
        tokens.extend(champion_tokens(rows_by_role))
    if use_context:
        tokens.extend(context_tokens(base_row, include_patch_token=include_patch_token))

    return GameSnapshotRecord(
        game_id=game_id,
        game_date=base_row["date"],
        minute=minute,
        label=int(base_row["blue_win"]),
        numeric_features=numeric,
        categorical_tokens=tokens,
    )


def load_prefix_records(
    csv_path: str,
    prefix_minute: int,
    use_champions: bool,
    use_context: bool,
    include_patch_token: bool,
) -> list[GamePrefixRecord]:
    grouped: dict[str, dict[int, dict[str, dict[str, str]]]] = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            minute_text = str(row.get("minute") or "").strip()
            if not minute_text:
                continue
            minute = int(float(minute_text))
            if minute > prefix_minute:
                continue
            role = row["role"].strip().upper()
            if role not in ROLE_ORDER:
                continue
            game_id = row["game_id"]
            grouped.setdefault(game_id, {}).setdefault(minute, {})[role] = row

    records: list[GamePrefixRecord] = []
    for game_id in sorted(grouped):
        record = build_prefix_record(
            grouped[game_id],
            prefix_minute=prefix_minute,
            use_champions=use_champions,
            use_context=use_context,
            include_patch_token=include_patch_token,
        )
        if record is not None:
            records.append(record)
    return records


def load_snapshot_records(
    csv_path: str,
    *,
    use_champions: bool,
    use_context: bool,
    include_patch_token: bool,
    include_bias: bool = False,
    min_minute: int | None = None,
    max_minute: int | None = None,
) -> list[GameSnapshotRecord]:
    grouped: dict[str, dict[int, dict[str, dict[str, str]]]] = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            minute_text = str(row.get("minute") or "").strip()
            if not minute_text:
                continue
            minute = int(float(minute_text))
            if min_minute is not None and minute < min_minute:
                continue
            if max_minute is not None and minute > max_minute:
                continue
            role = str(row.get("role") or "").strip().upper()
            if role not in ROLE_ORDER:
                continue
            game_id = row["game_id"]
            grouped.setdefault(game_id, {}).setdefault(minute, {})[role] = row

    records: list[GameSnapshotRecord] = []
    for game_id in sorted(grouped):
        minute_map = grouped[game_id]
        for minute in sorted(minute_map):
            rows_by_role = minute_map[minute]
            if not all(role in rows_by_role for role in ROLE_ORDER):
                continue
            records.append(
                build_snapshot_record(
                    game_id,
                    minute,
                    rows_by_role,
                    use_champions=use_champions,
                    use_context=use_context,
                    include_patch_token=include_patch_token,
                    include_bias=include_bias,
                )
            )
    return records


def split_game_ids(records: list[GamePrefixRecord], train_fraction: float) -> tuple[set[str], set[str]]:
    game_to_date: dict[str, str] = {}
    for record in records:
        game_to_date[record.game_id] = record.game_date
    ordered_games = sorted(game_to_date.items(), key=lambda item: (item[1], item[0]))
    split_index = max(1, min(len(ordered_games) - 1, int(len(ordered_games) * train_fraction)))
    train_games = {game_id for game_id, _ in ordered_games[:split_index]}
    test_games = {game_id for game_id, _ in ordered_games[split_index:]}
    return train_games, test_games


def build_feature_space(
    train_records: list[GamePrefixRecord],
    numeric_names: list[str],
) -> tuple[list[str], dict[str, int]]:
    categorical_vocab: dict[str, int] = {}
    feature_names = numeric_names[:]
    for record in train_records:
        for token in record.categorical_tokens:
            if token not in categorical_vocab:
                categorical_vocab[token] = len(feature_names)
                feature_names.append(token)
    return feature_names, categorical_vocab


def records_to_matrix(
    records: list[GamePrefixRecord],
    numeric_names: list[str],
    categorical_vocab: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    num_rows = len(records)
    num_cols = len(numeric_names) + len(categorical_vocab)
    X = np.zeros((num_rows, num_cols), dtype=np.float32)
    y = np.zeros(num_rows, dtype=np.float32)
    numeric_index = {name: idx for idx, name in enumerate(numeric_names)}

    for row_idx, record in enumerate(records):
        y[row_idx] = float(record.label)
        for name, value in record.numeric_features.items():
            col_idx = numeric_index.get(name)
            if col_idx is not None:
                X[row_idx, col_idx] = value
        for token in record.categorical_tokens:
            col_idx = categorical_vocab.get(token)
            if col_idx is not None:
                X[row_idx, col_idx] = 1.0
    return X, y
