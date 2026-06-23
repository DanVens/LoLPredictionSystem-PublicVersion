#!/usr/bin/env python3
"""
Evaluate the trained models on unseen GOL-style holdout CSV files.

This script:
1. normalizes raw GOL exports into a clean table,
2. replays each game minute-by-minute through the live predictor suite,
3. saves per-minute prediction rows, and
4. reports holdout metrics at 10/15/20 minutes for every model.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

EXPORT_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = EXPORT_ROOT / "shared"
LIVE_APP_DIR = EXPORT_ROOT / "live_app"
for path in (str(SHARED_DIR), str(LIVE_APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from build_training_table import BASE_COLUMNS, build_row
from live_inference import LiveModelSuitePredictor


ROLE_ORDER = ("top", "jgl", "mid", "bot", "spt")
MODEL_KEYS = ("gru", "logistic_regression", "xgboost", "mlp", "consensus")
ROLE_MAP = {
    "TOP": "top",
    "JGL": "jgl",
    "MID": "mid",
    "BOT": "bot",
    "SPT": "spt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained models on holdout GOL exports.")
    parser.add_argument(
        "inputs",
        nargs="*",
        help="CSV files or glob patterns to include. Default: university_exports/data/testing_data/*.csv",
    )
    parser.add_argument(
        "--table-output",
        default=str(EXPORT_ROOT / "data/testing_holdout_table_all.csv"),
        help="Normalized holdout table output path inside university_exports/data.",
    )
    parser.add_argument(
        "--predictions-output",
        default=str(EXPORT_ROOT / "artifacts/holdout_eval/predictions_all_minutes.csv"),
        help="Per-minute prediction CSV output inside university_exports/artifacts/holdout_eval.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(EXPORT_ROOT / "artifacts/holdout_eval/summary.json"),
        help="Summary JSON output inside university_exports/artifacts/holdout_eval.",
    )
    parser.add_argument(
        "--game-limit",
        type=int,
        default=0,
        help="Optional limit on number of games to evaluate. Default: 0 (no limit).",
    )
    parser.add_argument(
        "--checkpoints",
        default="5,10,15,20,25",
        help="Comma-separated actual game minutes to summarize. Default: 5,10,15,20,25",
    )
    parser.add_argument(
        "--sequence-model-root",
        default=str(EXPORT_ROOT / "artifacts/sequence_gru_team_context_mixedlength_ls005/gold_champions_context"),
        help="GRU model root inside university_exports/artifacts.",
    )
    parser.add_argument(
        "--logistic-model-root",
        default=str(EXPORT_ROOT / "artifacts/logistic_regression_no_prefix_std1/gold_champions_context"),
        help="Logistic regression model root inside university_exports/artifacts.",
    )
    parser.add_argument(
        "--xgboost-model-root",
        default=str(EXPORT_ROOT / "artifacts/xgboost_no_prefix_fullgame/gold_champions_context"),
        help="XGBoost model root inside university_exports/artifacts.",
    )
    parser.add_argument(
        "--mlp-model-root",
        default=str(EXPORT_ROOT / "artifacts/mlp_snapshot_embedding_lr1e4_trial/gold_champions_context"),
        help="MLP model root inside university_exports/artifacts.",
    )
    return parser.parse_args()


def parse_checkpoints(raw: str) -> tuple[int, ...]:
    values = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        values.append(int(float(text)))
    if not values:
        raise ValueError("No checkpoint minutes were provided.")
    return tuple(sorted(set(values)))


def resolve_input_paths(raw_inputs: list[str]) -> list[str]:
    if not raw_inputs:
        raw_inputs = [str(EXPORT_ROOT / "data/testing_data/*.csv")]

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


def normalize_holdout_rows(input_paths: list[str], table_output: str) -> int:
    row_count = 0
    seen_keys: set[tuple[str, str, str]] = set()

    os.makedirs(os.path.dirname(table_output) or ".", exist_ok=True)
    with open(table_output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASE_COLUMNS)
        writer.writeheader()

        for path in input_paths:
            with open(path, newline="", encoding="utf-8") as source_handle:
                reader = csv.DictReader(source_handle)
                if not reader.fieldnames:
                    continue
                for row in reader:
                    built = build_row(path, row)
                    dedupe_key = (built["game_id"], built["minute"], built["role"])
                    if dedupe_key in seen_keys:
                        continue
                    seen_keys.add(dedupe_key)
                    writer.writerow(built)
                    row_count += 1
    return row_count


def group_games_from_csv(table_path: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    with open(table_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            game_id = row["game_id"]
            minute = int(row["minute"])
            role = ROLE_MAP.get(row["role"].strip().upper())
            if role is None:
                continue
            game = grouped.setdefault(
                game_id,
                {
                    "game_id": game_id,
                    "base": row,
                    "minutes": {},
                },
            )
            game["minutes"].setdefault(minute, {})[role] = row

    games: list[dict[str, Any]] = []
    for game_id, game in grouped.items():
        complete_minutes = [
            minute
            for minute, rows_by_role in sorted(game["minutes"].items())
            if all(role in rows_by_role for role in ROLE_ORDER)
        ]
        if not complete_minutes:
            continue
        games.append(
            {
                "game_id": game_id,
                "base": game["base"],
                "minutes": game["minutes"],
                "complete_minutes": complete_minutes,
            }
        )
    games.sort(key=lambda item: (item["base"]["date"], item["game_id"]))
    return games


def build_snapshot(base_row: dict[str, str], game_id: str, minute: int, rows_by_role: dict[str, dict[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source": "holdout_eval",
        "status": "in_progress",
        "status_message": "Holdout evaluation replay",
        "source_file": base_row.get("source_file"),
        "source_season": base_row.get("source_season"),
        "league": base_row.get("league"),
        "tournament": base_row.get("tournament"),
        "tournament_stage": base_row.get("tournament_stage"),
        "match_id": f"holdout_{Path(base_row.get('source_file') or 'file').stem}_{game_id}",
        "game_id": int(game_id),
        "game_number": 1,
        "game_state": "replay",
        "team_left": base_row.get("blue_team"),
        "team_right": base_row.get("red_team"),
        "patch_version": base_row.get("patch"),
        "date": base_row.get("date"),
        "time_s": minute * 60,
        "time_source": "holdout_file",
        "winner_side": base_row.get("winner_side"),
        "winner_team": base_row.get("winner_team"),
        "blue_win": int(base_row.get("blue_win") or 0),
    }

    blue_total = 0
    red_total = 0
    for role in ROLE_ORDER:
        row = rows_by_role[role]
        blue_gold = int(float(row["blue_gold"]))
        red_gold = int(float(row["red_gold"]))
        result[f"blue_champion_{role}"] = row["blue_champion"]
        result[f"red_champion_{role}"] = row["red_champion"]
        result[f"blue_gold_{role}"] = blue_gold
        result[f"red_gold_{role}"] = red_gold
        blue_total += blue_gold
        red_total += red_gold

    result["gold_left"] = blue_total
    result["gold_right"] = red_total
    result["kills_left"] = 0
    result["kills_right"] = 0
    return result


def binary_log_loss(y_true: list[int], y_prob: list[float]) -> float:
    if not y_true:
        return float("nan")
    total = 0.0
    for truth, prob in zip(y_true, y_prob):
        clipped = min(max(prob, 1e-15), 1.0 - 1e-15)
        total += truth * math.log(clipped) + (1 - truth) * math.log(1.0 - clipped)
    return -total / len(y_true)


def binary_accuracy(y_true: list[int], y_prob: list[float]) -> float:
    if not y_true:
        return float("nan")
    correct = 0
    for truth, prob in zip(y_true, y_prob):
        if (1 if prob >= 0.5 else 0) == truth:
            correct += 1
    return correct / len(y_true)


def brier_score(y_true: list[int], y_prob: list[float]) -> float:
    if not y_true:
        return float("nan")
    return sum((prob - truth) ** 2 for truth, prob in zip(y_true, y_prob)) / len(y_true)


def roc_auc(y_true: list[int], y_prob: list[float]) -> float:
    if not y_true:
        return float("nan")
    paired = sorted(zip(y_prob, y_true), key=lambda item: item[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    rank_sum = 0.0
    index = 0
    while index < len(paired):
        start = index
        score = paired[index][0]
        while index < len(paired) and paired[index][0] == score:
            index += 1
        avg_rank = (start + 1 + index) / 2.0
        positives = sum(label for _, label in paired[start:index])
        rank_sum += positives * avg_rank
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate_probs(y_true: list[int], y_prob: list[float]) -> dict[str, float]:
    positive_rate = sum(y_true) / len(y_true) if y_true else float("nan")
    return {
        "rows": len(y_true),
        "positive_rate": positive_rate,
        "accuracy": binary_accuracy(y_true, y_prob),
        "log_loss": binary_log_loss(y_true, y_prob),
        "brier_score": brier_score(y_true, y_prob),
        "roc_auc": roc_auc(y_true, y_prob),
    }


def main() -> int:
    args = parse_args()
    checkpoint_minutes = parse_checkpoints(args.checkpoints)
    input_paths = resolve_input_paths(args.inputs)
    if not input_paths:
        raise SystemExit("No holdout CSV files found.")

    print(f"holdout inputs: {len(input_paths)} files")
    normalized_row_count = normalize_holdout_rows(input_paths, args.table_output)
    print(f"normalized rows: {normalized_row_count}")
    print(f"normalized table: {args.table_output}")

    games = group_games_from_csv(args.table_output)
    if args.game_limit > 0:
        games = games[: args.game_limit]
    print(f"holdout games with complete minute snapshots: {len(games)}")

    predictor = LiveModelSuitePredictor(
        sequence_model_root=args.sequence_model_root,
        logistic_model_root=args.logistic_model_root,
        xgboost_model_root=args.xgboost_model_root,
        mlp_model_root=args.mlp_model_root,
        use_sequence_prefix_switching=True,
    )
    checkpoint_truths: dict[int, list[int]] = {minute: [] for minute in checkpoint_minutes}
    checkpoint_probs: dict[int, dict[str, list[float]]] = {
        minute: {model_key: [] for model_key in MODEL_KEYS}
        for minute in checkpoint_minutes
    }
    os.makedirs(os.path.dirname(args.predictions_output) or ".", exist_ok=True)
    prediction_writer: csv.DictWriter | None = None
    prediction_handle = open(args.predictions_output, "w", newline="", encoding="utf-8")

    try:
        for game in games:
            predictor.reset()
            base_row = game["base"]
            for minute in game["complete_minutes"]:
                snapshot = build_snapshot(base_row, game["game_id"], minute, game["minutes"][minute])
                enriched = predictor.enrich(snapshot)
                model_predictions = enriched.get("model_predictions") or {}
                row = {
                    "source_file": enriched.get("source_file"),
                    "source_season": enriched.get("source_season"),
                    "date": enriched.get("date"),
                    "game_id": enriched.get("game_id"),
                    "league": enriched.get("league"),
                    "tournament": enriched.get("tournament"),
                    "tournament_stage": enriched.get("tournament_stage"),
                    "team_left": enriched.get("team_left"),
                    "team_right": enriched.get("team_right"),
                    "minute": minute,
                    "winner_side": enriched.get("winner_side"),
                    "winner_team": enriched.get("winner_team"),
                    "blue_win": enriched.get("blue_win"),
                    "consensus_blue_win_prob": enriched.get("prediction_consensus_blue_win_prob"),
                }
                for key in ("gru", "logistic_regression", "xgboost", "mlp"):
                    model_entry = model_predictions.get(key) or {}
                    row[f"{key}_blue_win_prob"] = model_entry.get("blue_win_prob")
                    row[f"{key}_status"] = model_entry.get("status")

                if prediction_writer is None:
                    prediction_writer = csv.DictWriter(prediction_handle, fieldnames=list(row.keys()))
                    prediction_writer.writeheader()
                prediction_writer.writerow(row)

                if minute in checkpoint_minutes:
                    truth = int(enriched.get("blue_win") or 0)
                    checkpoint_truths[minute].append(truth)
                    checkpoint_probs[minute]["consensus"].append(float(enriched["prediction_consensus_blue_win_prob"]))
                    for key in ("gru", "logistic_regression", "xgboost", "mlp"):
                        model_entry = model_predictions.get(key) or {}
                        checkpoint_probs[minute][key].append(float(model_entry["blue_win_prob"]))
    finally:
        prediction_handle.close()

    print(f"minute-by-minute predictions saved to: {args.predictions_output}")

    summary: dict[str, Any] = {
        "inputs": input_paths,
        "normalized_table": args.table_output,
        "predictions_output": args.predictions_output,
        "sequence_model_root": args.sequence_model_root,
        "logistic_model_root": args.logistic_model_root,
        "xgboost_model_root": args.xgboost_model_root,
        "mlp_model_root": args.mlp_model_root,
        "games_evaluated": len(games),
        "checkpoint_minutes": list(checkpoint_minutes),
        "minutes": {},
    }
    for minute in checkpoint_minutes:
        minute_summary: dict[str, Any] = {}
        truths = checkpoint_truths[minute]
        for key in MODEL_KEYS:
            minute_summary[key] = evaluate_probs(truths, checkpoint_probs[minute][key])
        summary["minutes"][str(minute)] = minute_summary

    os.makedirs(os.path.dirname(args.summary_output) or ".", exist_ok=True)
    with open(args.summary_output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"holdout summary saved to: {args.summary_output}")

    for minute in checkpoint_minutes:
        print(f"[minute {minute}]")
        for key in MODEL_KEYS:
            metrics = summary["minutes"][str(minute)][key]
            print(
                f"  {key:>20}  acc={metrics['accuracy']:.4f} "
                f"auc={metrics['roc_auc']:.4f} logloss={metrics['log_loss']:.4f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
