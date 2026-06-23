#!/usr/bin/env python3
"""
Train a no-prefix XGBoost baseline on snapshot features from all minutes.

This variant intentionally avoids sequence-history feature blocks and minute-specific
checkpoints. It learns one model per variant across all available minute snapshots,
with the current minute included as an explicit numeric feature.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np

from tabular_sequence_features import (
    VARIANT_CONFIGS,
    build_feature_space,
    load_snapshot_records,
    records_to_sparse_matrix,
    snapshot_numeric_feature_names,
    split_game_ids,
)

EXPORT_ROOT = Path(__file__).resolve().parents[1]


try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "xgboost is not installed. Install it with: ./.venv/bin/pip install xgboost"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a no-prefix XGBoost baseline on snapshot features.",
    )
    parser.add_argument(
        "--input",
        default=str(EXPORT_ROOT / "data/training_table_all.csv"),
        help="Training table CSV. Default: data/training_table_all.csv",
    )
    parser.add_argument(
        "--variants",
        default="gold_only,gold_champions,gold_champions_context",
        help=(
            "Comma-separated variants. Supported: gold_only, gold_champions, "
            "gold_champions_context."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(EXPORT_ROOT / "artifacts/xgboost_no_prefix"),
        help="Directory for checkpoints and metrics. Default: artifacts/xgboost_no_prefix",
    )
    parser.add_argument(
        "--num-round",
        type=int,
        default=700,
        help="Maximum boosting rounds. Default: 700",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.03,
        help="Learning rate. Default: 0.03",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Tree max depth. Default: 4",
    )
    parser.add_argument(
        "--min-child-weight",
        type=float,
        default=3.0,
        help="XGBoost min_child_weight. Default: 3.0",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Row subsampling. Default: 0.8",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="Column subsampling. Default: 0.8",
    )
    parser.add_argument(
        "--lambda-l2",
        type=float,
        default=1.0,
        help="L2 regularization. Default: 1.0",
    )
    parser.add_argument(
        "--alpha-l1",
        type=float,
        default=0.0,
        help="L1 regularization. Default: 0.0",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=50,
        help="Early stopping rounds. Default: 50",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Chronological train split fraction. Default: 0.8",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    clipped = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
    return float(-(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped)).mean())


def binary_accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    preds = (y_prob >= 0.5).astype(np.int8)
    return float((preds == y_true).mean())


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    order = np.argsort(y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    rank_sum = 0.0
    idx = 0
    while idx < len(y_true):
        start = idx
        score = y_prob[idx]
        while idx < len(y_true) and y_prob[idx] == score:
            idx += 1
        avg_rank = (start + 1 + idx) / 2.0
        positives = int(y_true[start:idx].sum())
        rank_sum += positives * avg_rank
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    majority_prob = float(y_true.mean())
    majority_label = 1 if majority_prob >= 0.5 else 0
    majority_probs = np.full_like(y_prob, majority_prob)
    return {
        "rows": int(len(y_true)),
        "positive_rate": majority_prob,
        "accuracy": binary_accuracy(y_true, y_prob),
        "log_loss": binary_log_loss(y_true, y_prob),
        "brier_score": brier_score(y_true, y_prob),
        "roc_auc": roc_auc(y_true, y_prob),
        "majority_accuracy": float((y_true == majority_label).mean()),
        "majority_log_loss": binary_log_loss(y_true, majority_probs),
        "majority_brier_score": brier_score(y_true, majority_probs),
    }


def importance_map(model: xgb.Booster, feature_names: list[str], top_n: int = 30) -> list[dict[str, object]]:
    raw_scores = model.get_score(importance_type="gain")
    items: list[dict[str, object]] = []
    for key, score in raw_scores.items():
        if not key.startswith("f"):
            continue
        try:
            feature_idx = int(key[1:])
        except ValueError:
            continue
        if 0 <= feature_idx < len(feature_names):
            items.append({"feature": feature_names[feature_idx], "gain": float(score)})
    items.sort(key=lambda item: item["gain"], reverse=True)
    return items[:top_n]


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    variants = [part.strip() for part in args.variants.split(",") if part.strip()]
    unknown_variants = [variant for variant in variants if variant not in VARIANT_CONFIGS]
    if unknown_variants:
        raise ValueError(f"unsupported variants: {', '.join(unknown_variants)}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"input: {args.input}")
    print(f"variants: {variants}")
    print(f"output_dir: {args.output_dir}")

    all_summaries: dict[str, dict[str, object]] = {}

    for variant in variants:
        variant_config = VARIANT_CONFIGS[variant]
        variant_dir = os.path.join(args.output_dir, variant)
        os.makedirs(variant_dir, exist_ok=True)

        print(
            f"[variant {variant}] use_champions={variant_config['use_champions']} "
            f"use_context={variant_config['use_context']}"
        )

        numeric_names = snapshot_numeric_feature_names(
            use_context=variant_config["use_context"],
            include_bias=False,
        )
        records = load_snapshot_records(
            csv_path=args.input,
            use_champions=variant_config["use_champions"],
            use_context=variant_config["use_context"],
            include_patch_token=True,
            include_bias=False,
        )
        if not records:
            print(f"[variant {variant}] no records found, skipping")
            continue

        record_count = len(records)
        minutes = [record.minute for record in records]
        train_games, test_games = split_game_ids(records, args.train_fraction)
        train_records = [record for record in records if record.game_id in train_games]
        test_records = [record for record in records if record.game_id in test_games]
        feature_names, categorical_vocab = build_feature_space(train_records, numeric_names)
        X_train, y_train = records_to_sparse_matrix(train_records, numeric_names, categorical_vocab)
        X_test, y_test = records_to_sparse_matrix(test_records, numeric_names, categorical_vocab)
        model_dir = os.path.join(variant_dir, "all_minutes")
        os.makedirs(model_dir, exist_ok=True)
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
        del records, train_records, test_records, X_train, X_test

        params = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "auc"],
            "eta": args.eta,
            "max_depth": args.max_depth,
            "min_child_weight": args.min_child_weight,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "lambda": args.lambda_l2,
            "alpha": args.alpha_l1,
            "seed": args.seed,
            "tree_method": "hist",
        }
        evals_result: dict[str, dict[str, list[float]]] = {}

        print(
            f"[variant {variant}] rows={record_count} train_games={len(train_games)} "
            f"test_games={len(test_games)} minute_min={min(minutes)} "
            f"minute_max={max(minutes)} minute_avg={sum(minutes) / len(minutes):.2f} "
            f"features={len(feature_names)}"
        )

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=args.num_round,
            evals=[(dtrain, "train"), (dtest, "test")],
            early_stopping_rounds=args.early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=False,
        )

        best_iteration = int(booster.best_iteration)
        train_prob = booster.predict(dtrain, iteration_range=(0, best_iteration + 1))
        test_prob = booster.predict(dtest, iteration_range=(0, best_iteration + 1))
        train_metrics = evaluate(y_train, train_prob)
        test_metrics = evaluate(y_test, test_prob)

        print(
            f"[variant {variant}] best_round={best_iteration} "
            f"test_accuracy={test_metrics['accuracy']:.4f} "
            f"test_auc={test_metrics['roc_auc']:.4f} "
            f"test_log_loss={test_metrics['log_loss']:.4f} "
            f"baseline_accuracy={test_metrics['majority_accuracy']:.4f}"
        )

        booster.save_model(os.path.join(model_dir, "best_model.json"))
        summary = {
            "params": vars(args),
            "xgboost_params": params,
            "variant": variant,
            "variant_config": variant_config,
            "model_scope": "all_minutes_snapshot",
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "numeric_feature_names": numeric_names,
            "train_games": len(train_games),
            "test_games": len(test_games),
            "minute_min": min(minutes),
            "minute_max": max(minutes),
            "minute_avg": sum(minutes) / len(minutes),
            "best_iteration": best_iteration,
            "best_score": float(booster.best_score) if booster.best_score is not None else None,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "evals_result": evals_result,
            "top_feature_gain": importance_map(booster, feature_names),
        }
        with open(os.path.join(model_dir, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        all_summaries[variant] = summary
        print(f"[variant {variant}] best checkpoint saved to: {os.path.join(model_dir, 'best_model.json')}")

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(all_summaries, handle, indent=2)
    print(f"combined summary saved to: {os.path.join(args.output_dir, 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
