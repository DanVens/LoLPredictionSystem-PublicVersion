#!/usr/bin/env python3
"""
Train logistic-regression baselines on sequence-aware tabular prefix features.
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
    GamePrefixRecord,
    load_prefix_records,
    numeric_feature_names,
    split_game_ids,
)

EXPORT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train logistic-regression baselines on sequence-aware tabular data.",
    )
    parser.add_argument(
        "--input",
        default=str(EXPORT_ROOT / "data/training_table_10_15_20.csv"),
        help=(
            "Training table CSV. Default: data/training_table_10_15_20.csv. "
            "For full minute-history features, prefer an all-minutes table."
        ),
    )
    parser.add_argument(
        "--minutes",
        default="10,15,20",
        help="Comma-separated prefix minutes to train. Default: 10,15,20",
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
        default=str(EXPORT_ROOT / "artifacts/logistic_regression"),
        help="Directory for checkpoints and metrics. Default: artifacts/logistic_regression",
    )
    parser.add_argument(
        "--hash-dim",
        type=int,
        default=131072,
        help="Hashed categorical feature dimension. Default: 131072",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Training epochs. Default: 12",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.005,
        help="Initial SGD learning rate. Default: 0.005",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=1e-6,
        help="L2 regularization. Default: 1e-6",
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


def build_features(
    record: GamePrefixRecord,
    numeric_names: list[str],
    hash_dim: int,
) -> list[tuple[int, float]]:
    numeric_index = {name: idx for idx, name in enumerate(numeric_names)}
    features: list[tuple[int, float]] = []

    for name, value in record.numeric_features.items():
        idx = numeric_index.get(name)
        if idx is None:
            continue
        if value != 0.0 or name == "bias":
            features.append((idx, value))

    offset = len(numeric_names)
    for token in record.categorical_tokens:
        features.append((hash_index(token, hash_dim, offset), 1.0))
    return features


def load_examples(
    csv_path: str,
    prefix_minute: int,
    hash_dim: int,
    use_champions: bool,
    use_context: bool,
) -> tuple[list[GamePrefixRecord], list[tuple[str, str, int, int, list[tuple[int, float]]]]]:
    numeric_names = numeric_feature_names(prefix_minute, use_context=use_context, include_bias=True)
    records = load_prefix_records(
        csv_path=csv_path,
        prefix_minute=prefix_minute,
        use_champions=use_champions,
        use_context=use_context,
        include_patch_token=False,
    )
    examples = [
        (
            record.game_id,
            record.game_date,
            record.prefix_minute,
            record.label,
            build_features(record, numeric_names=numeric_names, hash_dim=hash_dim),
        )
        for record in records
    ]
    return records, examples


def dot(weights: np.ndarray, features: Iterable[tuple[int, float]]) -> float:
    total = 0.0
    for idx, value in features:
        total += weights[idx] * value
    return total


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


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
    return total_loss / len(train_examples)


def binary_log_loss(y_true: list[int], y_prob: list[float]) -> float:
    total = 0.0
    for truth, prob in zip(y_true, y_prob):
        clipped = min(max(prob, 1e-15), 1.0 - 1e-15)
        total += truth * math.log(clipped) + (1 - truth) * math.log(1.0 - clipped)
    return -total / len(y_true)


def binary_accuracy(y_true: list[int], y_prob: list[float]) -> float:
    correct = 0
    for truth, prob in zip(y_true, y_prob):
        pred = 1 if prob >= 0.5 else 0
        if pred == truth:
            correct += 1
    return correct / len(y_true)


def brier_score(y_true: list[int], y_prob: list[float]) -> float:
    total = 0.0
    for truth, prob in zip(y_true, y_prob):
        total += (prob - truth) ** 2
    return total / len(y_true)


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

    minutes = [part.strip() for part in args.minutes.split(",") if part.strip()]
    variants = [part.strip() for part in args.variants.split(",") if part.strip()]
    unknown_variants = [variant for variant in variants if variant not in VARIANT_CONFIGS]
    if unknown_variants:
        raise ValueError(f"unsupported variants: {', '.join(unknown_variants)}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"input: {args.input}")
    print(f"minutes: {minutes}")
    print(f"variants: {variants}")
    print(f"output_dir: {args.output_dir}")

    all_summaries: dict[str, dict[str, object]] = {}

    for variant in variants:
        variant_config = VARIANT_CONFIGS[variant]
        variant_dir = os.path.join(args.output_dir, variant)
        os.makedirs(variant_dir, exist_ok=True)
        all_summaries[variant] = {}

        print(
            f"[variant {variant}] use_champions={variant_config['use_champions']} "
            f"use_context={variant_config['use_context']}"
        )

        for minute in minutes:
            prefix_minute = int(float(minute))
            record_numeric_names = numeric_feature_names(
                prefix_minute,
                use_context=variant_config["use_context"],
                include_bias=True,
            )
            vector_size = len(record_numeric_names) + args.hash_dim
            records, examples = load_examples(
                csv_path=args.input,
                prefix_minute=prefix_minute,
                hash_dim=args.hash_dim,
                use_champions=variant_config["use_champions"],
                use_context=variant_config["use_context"],
            )
            if not examples:
                print(f"[variant {variant} minute {minute}] no examples found, skipping")
                continue

            observed_counts = [len(record.observed_minutes) for record in records]
            avg_observed = sum(observed_counts) / len(observed_counts)
            max_observed = max(observed_counts)
            min_observed = min(observed_counts)

            train_games, test_games = split_game_ids(records, args.train_fraction)
            train_examples = [example for example in examples if example[0] in train_games]
            test_examples = [example for example in examples if example[0] in test_games]

            weights = np.zeros(vector_size, dtype=np.float64)
            rng = random.Random(args.seed)
            best_test_log_loss = float("inf")
            best_weights = weights.copy()
            best_epoch_metrics: dict[str, float] | None = None
            history: list[dict[str, float]] = []

            minute_dir = os.path.join(variant_dir, f"minute_{minute}")
            os.makedirs(minute_dir, exist_ok=True)

            print(
                f"[variant {variant} minute {minute}] games={len(examples)} "
                f"train_games={len(train_games)} test_games={len(test_games)} "
                f"avg_observed_minutes={avg_observed:.2f} "
                f"min_observed={min_observed} max_observed={max_observed}"
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
                    f"[variant {variant} minute {minute} epoch {epoch}] "
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
                os.path.join(minute_dir, "best_model.npz"),
                weights=weights,
                args=np.asarray(json.dumps(vars(args)), dtype=object),
                variant=np.asarray(variant, dtype=object),
                variant_config=np.asarray(json.dumps(variant_config), dtype=object),
                minute=np.asarray(minute, dtype=object),
                prefix_minute=np.asarray(prefix_minute, dtype=np.int32),
                numeric_feature_names=np.asarray(record_numeric_names, dtype=object),
                hash_dim=np.asarray(args.hash_dim, dtype=np.int32),
            )

            summary = {
                "params": vars(args),
                "variant": variant,
                "variant_config": variant_config,
                "minute": minute,
                "prefix_minute": prefix_minute,
                "hash_dim": args.hash_dim,
                "feature_count": vector_size,
                "numeric_feature_names": record_numeric_names,
                "train_games": len(train_games),
                "test_games": len(test_games),
                "avg_observed_minutes": avg_observed,
                "min_observed_minutes": min_observed,
                "max_observed_minutes": max_observed,
                "best_epoch_metrics": best_epoch_metrics,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "history": history,
            }
            with open(os.path.join(minute_dir, "summary.json"), "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
            all_summaries[variant][minute] = summary
            print(
                f"[variant {variant} minute {minute}] best checkpoint saved to: "
                f"{os.path.join(minute_dir, 'best_model.npz')}"
            )

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(all_summaries, handle, indent=2)
    print(f"combined summary saved to: {os.path.join(args.output_dir, 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
