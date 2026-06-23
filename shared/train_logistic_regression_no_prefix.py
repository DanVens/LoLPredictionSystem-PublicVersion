#!/usr/bin/env python3
"""
Train a no-prefix logistic-regression baseline on snapshot features from all minutes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
from typing import Iterable

import numpy as np

from tabular_sequence_features import (
    VARIANT_CONFIGS,
    load_snapshot_records,
    snapshot_numeric_feature_names,
    split_game_ids,
)


EXPORT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a no-prefix logistic-regression baseline on snapshot features.",
    )
    parser.add_argument(
        "--input",
        default=str(EXPORT_ROOT / "data/training_table_all.csv"),
        help="Training table CSV. Default: data/training_table_all.csv",
    )
    parser.add_argument(
        "--variants",
        default="gold_champions_context",
        help=(
            "Comma-separated variants. Supported: gold_only, gold_champions, "
            "gold_champions_context."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(EXPORT_ROOT / "artifacts/logistic_regression_no_prefix_std1"),
        help="Directory for checkpoints and metrics. Default: artifacts/logistic_regression_no_prefix_std1",
    )
    parser.add_argument(
        "--hash-dim",
        type=int,
        default=262144,
        help="Hashed categorical feature dimension. Default: 262144",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs. Default: 20",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Initial SGD learning rate. Default: 0.001",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=1e-5,
        help="L2 regularization. Default: 1e-5",
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


def stable_hash(text: str) -> int:
    value = 2166136261
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def hash_index(token: str, hash_dim: int, offset: int) -> int:
    return offset + (stable_hash(token) % hash_dim)


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def dot(weights: np.ndarray, features: Iterable[tuple[int, float]]) -> float:
    total = 0.0
    for idx, value in features:
        total += float(weights[idx]) * value
    return total


def compute_numeric_stats(
    train_records: list[object],
    numeric_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    numeric_index = {name: idx for idx, name in enumerate(numeric_names)}
    values = np.zeros((len(train_records), len(numeric_names)), dtype=np.float64)

    for row_idx, record in enumerate(train_records):
        for name, value in record.numeric_features.items():
            col_idx = numeric_index.get(name)
            if col_idx is not None:
                values[row_idx, col_idx] = float(value)

    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    bias_idx = numeric_index.get("bias")
    if bias_idx is not None:
        mean[bias_idx] = 0.0
        std[bias_idx] = 1.0

    return mean.astype(np.float64), std.astype(np.float64)


def build_features(
    record: object,
    numeric_names: list[str],
    numeric_mean: np.ndarray,
    numeric_std: np.ndarray,
    hash_dim: int,
) -> list[tuple[int, float]]:
    numeric_index = {name: idx for idx, name in enumerate(numeric_names)}
    features: list[tuple[int, float]] = []

    for name, value in record.numeric_features.items():
        idx = numeric_index.get(name)
        if idx is None:
            continue
        normalized = (float(value) - numeric_mean[idx]) / numeric_std[idx]
        if normalized != 0.0 or name == "bias":
            features.append((idx, normalized))

    offset = len(numeric_names)
    for token in record.categorical_tokens:
        features.append((hash_index(token, hash_dim, offset), 1.0))
    return features


def build_examples(
    records: list[object],
    numeric_names: list[str],
    numeric_mean: np.ndarray,
    numeric_std: np.ndarray,
    hash_dim: int,
) -> list[tuple[str, str, int, int, list[tuple[int, float]]]]:
    examples = []
    for record in records:
        examples.append(
            (
                record.game_id,
                record.game_date,
                int(record.minute),
                int(record.label),
                build_features(record, numeric_names, numeric_mean, numeric_std, hash_dim),
            )
        )
    return examples


def train_one_epoch(
    train_examples: list[tuple[str, str, int, int, list[tuple[int, float]]]],
    weights: np.ndarray,
    learning_rate: float,
    l2: float,
    rng: random.Random,
) -> float:
    indices = list(range(len(train_examples)))
    rng.shuffle(indices)

    total_loss = 0.0
    for idx in indices:
        _, _, _, label, features = train_examples[idx]
        prediction = sigmoid(dot(weights, features))
        clipped = min(max(prediction, 1e-15), 1.0 - 1e-15)
        total_loss += -(label * math.log(clipped) + (1 - label) * math.log(1.0 - clipped))
        error = prediction - label
        for feature_idx, value in features:
            gradient = error * value + l2 * weights[feature_idx]
            weights[feature_idx] -= learning_rate * gradient
    return total_loss / max(len(train_examples), 1)


def binary_log_loss(y_true: list[int], y_prob: list[float]) -> float:
    total = 0.0
    for truth, prob in zip(y_true, y_prob):
        clipped = min(max(prob, 1e-15), 1.0 - 1e-15)
        total += truth * math.log(clipped) + (1 - truth) * math.log(1.0 - clipped)
    return -total / max(len(y_true), 1)


def binary_accuracy(y_true: list[int], y_prob: list[float]) -> float:
    correct = 0
    for truth, prob in zip(y_true, y_prob):
        pred = 1 if prob >= 0.5 else 0
        if pred == truth:
            correct += 1
    return correct / max(len(y_true), 1)


def brier_score(y_true: list[int], y_prob: list[float]) -> float:
    total = 0.0
    for truth, prob in zip(y_true, y_prob):
        total += (prob - truth) ** 2
    return total / max(len(y_true), 1)


def roc_auc(y_true: list[int], y_prob: list[float]) -> float:
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


def evaluate(
    examples: list[tuple[str, str, int, int, list[tuple[int, float]]]],
    weights: np.ndarray,
) -> dict[str, float]:
    y_true = [label for _, _, _, label, _ in examples]
    y_prob = [sigmoid(dot(weights, features)) for _, _, _, _, features in examples]
    majority_prob = sum(y_true) / len(y_true)
    majority_label = 1 if majority_prob >= 0.5 else 0
    majority_probs = [majority_prob] * len(y_true)

    return {
        "rows": len(examples),
        "positive_rate": majority_prob,
        "accuracy": binary_accuracy(y_true, y_prob),
        "log_loss": binary_log_loss(y_true, y_prob),
        "brier_score": brier_score(y_true, y_prob),
        "roc_auc": roc_auc(y_true, y_prob),
        "majority_accuracy": sum(1 for label in y_true if label == majority_label) / len(y_true),
        "majority_log_loss": binary_log_loss(y_true, majority_probs),
        "majority_brier_score": brier_score(y_true, majority_probs),
    }


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

        records = load_snapshot_records(
            args.input,
            use_champions=variant_config["use_champions"],
            use_context=variant_config["use_context"],
            include_patch_token=False,
            include_bias=True,
        )
        if not records:
            print(f"[variant {variant}] no records found, skipping")
            continue

        minutes = [record.minute for record in records]
        numeric_names = snapshot_numeric_feature_names(
            use_context=variant_config["use_context"],
            include_bias=True,
        )
        train_games, test_games = split_game_ids(records, args.train_fraction)
        train_records = [record for record in records if record.game_id in train_games]
        test_records = [record for record in records if record.game_id in test_games]

        numeric_mean, numeric_std = compute_numeric_stats(train_records, numeric_names)
        train_examples = build_examples(
            train_records,
            numeric_names,
            numeric_mean,
            numeric_std,
            args.hash_dim,
        )
        test_examples = build_examples(
            test_records,
            numeric_names,
            numeric_mean,
            numeric_std,
            args.hash_dim,
        )

        vector_size = len(numeric_names) + args.hash_dim
        weights = np.zeros(vector_size, dtype=np.float64)
        rng = random.Random(args.seed)
        best_test_log_loss = float("inf")
        best_weights = weights.copy()
        best_epoch_metrics: dict[str, float] | None = None
        history: list[dict[str, float]] = []

        model_dir = os.path.join(variant_dir, "all_minutes")
        os.makedirs(model_dir, exist_ok=True)

        print(
            f"[variant {variant}] rows={len(records)} train_games={len(train_games)} "
            f"test_games={len(test_games)} minute_min={min(minutes)} minute_max={max(minutes)} "
            f"minute_avg={sum(minutes) / len(minutes):.2f} features={vector_size}"
        )

        for epoch in range(1, args.epochs + 1):
            current_lr = args.learning_rate / math.sqrt(epoch)
            train_loss = train_one_epoch(
                train_examples=train_examples,
                weights=weights,
                learning_rate=current_lr,
                l2=args.l2,
                rng=rng,
            )
            train_metrics = evaluate(train_examples, weights)
            test_metrics = evaluate(test_examples, weights)
            epoch_metrics = {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_auc": train_metrics["roc_auc"],
                "train_log_loss": train_metrics["log_loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_auc": test_metrics["roc_auc"],
                "test_log_loss": test_metrics["log_loss"],
            }
            history.append(epoch_metrics)

            print(
                f"[variant {variant} epoch {epoch}] "
                f"train_loss={train_loss:.4f} "
                f"test_accuracy={test_metrics['accuracy']:.4f} "
                f"test_auc={test_metrics['roc_auc']:.4f} "
                f"test_log_loss={test_metrics['log_loss']:.4f} "
                f"baseline_accuracy={test_metrics['majority_accuracy']:.4f}"
            )

            if test_metrics["log_loss"] < best_test_log_loss:
                best_test_log_loss = test_metrics["log_loss"]
                best_weights = weights.copy()
                best_epoch_metrics = dict(epoch_metrics)

        weights = best_weights
        train_metrics = evaluate(train_examples, weights)
        test_metrics = evaluate(test_examples, weights)

        np.savez_compressed(
            os.path.join(model_dir, "best_model.npz"),
            weights=weights,
            args=np.asarray(json.dumps(vars(args)), dtype=object),
            variant=np.asarray(variant, dtype=object),
            variant_config=np.asarray(json.dumps(variant_config), dtype=object),
            model_scope=np.asarray("all_minutes_snapshot", dtype=object),
            numeric_feature_names=np.asarray(numeric_names, dtype=object),
            hash_dim=np.asarray(args.hash_dim, dtype=np.int32),
            numeric_mean=numeric_mean.astype(np.float32),
            numeric_std=numeric_std.astype(np.float32),
        )

        summary = {
            "params": vars(args),
            "variant": variant,
            "variant_config": variant_config,
            "model_scope": "all_minutes_snapshot",
            "hash_dim": args.hash_dim,
            "feature_count": vector_size,
            "numeric_feature_names": numeric_names,
            "numeric_mean": numeric_mean.tolist(),
            "numeric_std": numeric_std.tolist(),
            "train_games": len(train_games),
            "test_games": len(test_games),
            "minute_min": min(minutes),
            "minute_max": max(minutes),
            "minute_avg": sum(minutes) / len(minutes),
            "best_epoch_metrics": best_epoch_metrics,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "history": history,
        }
        with open(os.path.join(model_dir, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        all_summaries[variant] = summary
        print(f"[variant {variant}] best checkpoint saved to: {os.path.join(model_dir, 'best_model.npz')}")

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(all_summaries, handle, indent=2)
    print(f"combined summary saved to: {os.path.join(args.output_dir, 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
