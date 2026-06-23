#!/usr/bin/env python3
"""
Export graph-ready data from holdout evaluation outputs.

This script converts the current holdout summary and prediction CSV into tidy
CSV files that are easy to plot in Python, R, Excel, or a notebook.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

EXPORT_ROOT = Path(__file__).resolve().parents[1]


MODEL_KEYS = ("gru", "logistic_regression", "xgboost", "mlp", "consensus")
MODEL_LABELS = {
    "gru": "GRU",
    "logistic_regression": "LogReg",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "consensus": "Consensus",
}
SUMMARY_METRICS = ("accuracy", "roc_auc", "log_loss", "brier_score", "positive_rate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export graph-ready holdout evaluation data.")
    parser.add_argument(
        "--summary-json",
        default=str(EXPORT_ROOT / "artifacts/holdout_eval/summary.json"),
        help="Holdout summary JSON path inside university_exports/artifacts/holdout_eval.",
    )
    parser.add_argument(
        "--predictions-csv",
        default=str(EXPORT_ROOT / "artifacts/holdout_eval/predictions_all_minutes.csv"),
        help="Holdout per-minute predictions CSV inside university_exports/artifacts/holdout_eval.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(EXPORT_ROOT / "artifacts/holdout_eval/graph_data"),
        help="Output directory inside university_exports/artifacts/holdout_eval.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary_metric_rows(summary: dict) -> list[dict]:
    rows: list[dict] = []
    for minute_key in sorted(summary["minutes"].keys(), key=int):
        minute_block = summary["minutes"][minute_key]
        for model_key in MODEL_KEYS:
            metrics = minute_block[model_key]
            base = {
                "minute": int(minute_key),
                "model_key": model_key,
                "model_label": MODEL_LABELS[model_key],
                "rows": int(metrics["rows"]),
            }
            for metric_name in SUMMARY_METRICS:
                base[metric_name] = float(metrics[metric_name])
            rows.append(base)
    return rows


def build_metric_long_rows(summary_metric_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in summary_metric_rows:
        for metric_name in SUMMARY_METRICS:
            rows.append(
                {
                    "minute": row["minute"],
                    "model_key": row["model_key"],
                    "model_label": row["model_label"],
                    "metric": metric_name,
                    "value": row[metric_name],
                }
            )
    return rows


def build_prediction_long_rows(prediction_rows: list[dict[str, str]]) -> list[dict]:
    long_rows: list[dict] = []
    for row in prediction_rows:
        base = {
            "source_file": row["source_file"],
            "source_season": row["source_season"],
            "date": row["date"],
            "game_id": row["game_id"],
            "league": row["league"],
            "tournament": row["tournament"],
            "tournament_stage": row["tournament_stage"],
            "team_left": row["team_left"],
            "team_right": row["team_right"],
            "minute": int(row["minute"]),
            "winner_side": row["winner_side"],
            "winner_team": row["winner_team"],
            "blue_win": int(row["blue_win"]),
        }
        for model_key in MODEL_KEYS:
            if model_key == "consensus":
                prob = row["consensus_blue_win_prob"]
                status = "ready"
            else:
                prob = row[f"{model_key}_blue_win_prob"]
                status = row[f"{model_key}_status"]
            long_rows.append(
                {
                    **base,
                    "model_key": model_key,
                    "model_label": MODEL_LABELS[model_key],
                    "status": status,
                    "blue_win_prob": float(prob) if prob not in ("", None) else None,
                }
            )
    return long_rows


def build_prediction_aggregate_rows(prediction_long_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in prediction_long_rows:
        if row["blue_win_prob"] is None:
            continue
        grouped[(int(row["minute"]), row["model_key"])].append(row)

    output: list[dict] = []
    for (minute, model_key), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        probs = [float(row["blue_win_prob"]) for row in rows]
        blue_win_probs = [float(row["blue_win_prob"]) for row in rows if int(row["blue_win"]) == 1]
        blue_loss_probs = [float(row["blue_win_prob"]) for row in rows if int(row["blue_win"]) == 0]
        output.append(
            {
                "minute": minute,
                "model_key": model_key,
                "model_label": MODEL_LABELS[model_key],
                "rows": len(rows),
                "mean_blue_win_prob": statistics.fmean(probs),
                "stdev_blue_win_prob": statistics.pstdev(probs) if len(probs) > 1 else 0.0,
                "median_blue_win_prob": statistics.median(probs),
                "mean_prob_when_blue_wins": statistics.fmean(blue_win_probs) if blue_win_probs else math.nan,
                "mean_prob_when_blue_loses": statistics.fmean(blue_loss_probs) if blue_loss_probs else math.nan,
            }
        )
    return output


def build_checkpoint_prediction_rows(
    prediction_long_rows: list[dict],
    checkpoint_minutes: set[int],
) -> list[dict]:
    return [row for row in prediction_long_rows if int(row["minute"]) in checkpoint_minutes]


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary_json)
    predictions_path = Path(args.predictions_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_path)
    prediction_rows = load_prediction_rows(predictions_path)

    summary_metric_rows = build_summary_metric_rows(summary)
    metric_long_rows = build_metric_long_rows(summary_metric_rows)
    prediction_long_rows = build_prediction_long_rows(prediction_rows)
    prediction_aggregate_rows = build_prediction_aggregate_rows(prediction_long_rows)
    checkpoint_minutes = {int(minute_key) for minute_key in summary.get("minutes", {})}
    checkpoint_prediction_rows = build_checkpoint_prediction_rows(prediction_long_rows, checkpoint_minutes)

    write_csv(
        output_dir / "summary_metrics.csv",
        summary_metric_rows,
        ["minute", "model_key", "model_label", "rows", *SUMMARY_METRICS],
    )
    write_csv(
        output_dir / "summary_metrics_long.csv",
        metric_long_rows,
        ["minute", "model_key", "model_label", "metric", "value"],
    )
    write_csv(
        output_dir / "predictions_long.csv",
        prediction_long_rows,
        [
            "source_file",
            "source_season",
            "date",
            "game_id",
            "league",
            "tournament",
            "tournament_stage",
            "team_left",
            "team_right",
            "minute",
            "winner_side",
            "winner_team",
            "blue_win",
            "model_key",
            "model_label",
            "status",
            "blue_win_prob",
        ],
    )
    write_csv(
        output_dir / "predictions_checkpoint_long.csv",
        checkpoint_prediction_rows,
        [
            "source_file",
            "source_season",
            "date",
            "game_id",
            "league",
            "tournament",
            "tournament_stage",
            "team_left",
            "team_right",
            "minute",
            "winner_side",
            "winner_team",
            "blue_win",
            "model_key",
            "model_label",
            "status",
            "blue_win_prob",
        ],
    )
    write_csv(
        output_dir / "prediction_aggregates_by_minute.csv",
        prediction_aggregate_rows,
        [
            "minute",
            "model_key",
            "model_label",
            "rows",
            "mean_blue_win_prob",
            "stdev_blue_win_prob",
            "median_blue_win_prob",
            "mean_prob_when_blue_wins",
            "mean_prob_when_blue_loses",
        ],
    )

    manifest = {
        "summary_json": str(summary_path),
        "predictions_csv": str(predictions_path),
        "games_evaluated": int(summary.get("games_evaluated", 0)),
        "outputs": {
            "summary_metrics": str(output_dir / "summary_metrics.csv"),
            "summary_metrics_long": str(output_dir / "summary_metrics_long.csv"),
            "predictions_long": str(output_dir / "predictions_long.csv"),
            "predictions_checkpoint_long": str(output_dir / "predictions_checkpoint_long.csv"),
            "prediction_aggregates_by_minute": str(output_dir / "prediction_aggregates_by_minute.csv"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
