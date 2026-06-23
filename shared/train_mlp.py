#!/usr/bin/env python3
"""
Train MLP baselines on sequence-aware tabular prefix features.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tabular_sequence_features import (
    VARIANT_CONFIGS,
    build_feature_space,
    load_prefix_records,
    load_snapshot_records,
    numeric_feature_names,
    snapshot_numeric_feature_names,
    records_to_matrix,
    split_game_ids,
)

EXPORT_ROOT = Path(__file__).resolve().parents[1]


class RecordBatchCollator:
    """Build a dense or embedding-ready batch without changing feature content."""

    def __init__(
        self,
        numeric_names: list[str],
        categorical_vocab: dict[str, int],
        numeric_mean: np.ndarray,
        numeric_std: np.ndarray,
        categorical_encoding: str = "one_hot",
    ) -> None:
        self.numeric_names = numeric_names
        self.categorical_vocab = categorical_vocab
        self.numeric_mean = numeric_mean
        self.numeric_std = numeric_std
        self.categorical_encoding = categorical_encoding
        self.embedding_vocab = {
            token: index - len(numeric_names) + 1
            for token, index in categorical_vocab.items()
        }

    def __call__(
        self,
        records: list[object],
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        if self.categorical_encoding == "embedding":
            numeric = np.asarray(
                [
                    [record.numeric_features.get(name, 0.0) for name in self.numeric_names]
                    for record in records
                ],
                dtype=np.float32,
            )
            numeric = (numeric - self.numeric_mean) / self.numeric_std
            categorical_ids: list[int] = []
            categorical_offsets: list[int] = []
            for record in records:
                categorical_offsets.append(len(categorical_ids))
                token_ids = [
                    self.embedding_vocab.get(token, 0)
                    for token in record.categorical_tokens
                ]
                categorical_ids.extend(token_ids or [0])
            inputs = (
                torch.from_numpy(numeric),
                torch.tensor(categorical_ids, dtype=torch.long),
                torch.tensor(categorical_offsets, dtype=torch.long),
            )
            y = torch.tensor([record.label for record in records], dtype=torch.float32)
            return inputs, y

        X, y = records_to_matrix(records, self.numeric_names, self.categorical_vocab)
        numeric_dim = len(self.numeric_names)
        X[:, :numeric_dim] = (X[:, :numeric_dim] - self.numeric_mean) / self.numeric_std
        return torch.from_numpy(X), torch.from_numpy(y)


class MLPModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
        *,
        numeric_dim: int | None = None,
        categorical_vocab_size: int = 0,
        categorical_embedding_dim: int = 0,
    ) -> None:
        super().__init__()
        self.categorical_embedding_dim = int(categorical_embedding_dim)
        self.uses_categorical_embeddings = self.categorical_embedding_dim > 0
        if self.uses_categorical_embeddings:
            if numeric_dim is None:
                raise ValueError("numeric_dim is required when categorical embeddings are enabled")
            self.categorical_embedding = nn.EmbeddingBag(
                categorical_vocab_size + 1,
                self.categorical_embedding_dim,
                mode="mean",
                padding_idx=0,
            )
            input_dim = int(numeric_dim) + self.categorical_embedding_dim

        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        X: torch.Tensor,
        categorical_ids: torch.Tensor | None = None,
        categorical_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.uses_categorical_embeddings:
            if categorical_ids is None or categorical_offsets is None:
                raise ValueError("categorical IDs and offsets are required for embedded MLP input")
            categorical = self.categorical_embedding(categorical_ids, categorical_offsets)
            X = torch.cat((X, categorical), dim=1)
        return self.net(X).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP baselines on tabular game-state data.")
    parser.add_argument(
        "--input",
        default=str(EXPORT_ROOT / "data/training_table_all.csv"),
        help=(
            "Training table CSV. Default: university_exports/data/training_table_all.csv. "
            "Use an all-minutes table for snapshot training."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("snapshot", "prefix"),
        default="snapshot",
        help="Training mode. 'snapshot' trains one model per variant, 'prefix' keeps checkpoint training. Default: snapshot",
    )
    parser.add_argument(
        "--minutes",
        default="10,15,20",
        help="Comma-separated prefix minutes to train when --mode prefix is used. Default: 10,15,20",
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
        default=str(EXPORT_ROOT / "artifacts/mlp"),
        help="Directory for checkpoints and metrics. Default: university_exports/artifacts/mlp",
    )
    parser.add_argument(
        "--hidden-dims",
        default="128,64",
        help="Comma-separated hidden layer sizes. Default: 128,64",
    )
    parser.add_argument(
        "--categorical-encoding",
        choices=("one_hot", "embedding"),
        default="one_hot",
        help=(
            "Categorical feature encoding. 'one_hot' preserves legacy artifacts; "
            "'embedding' keeps all tokens in a compact learned representation. "
            "Default: one_hot"
        ),
    )
    parser.add_argument(
        "--categorical-embedding-dim",
        type=int,
        default=16,
        help="Embedding size used with --categorical-encoding embedding. Default: 16",
    )
    parser.add_argument(
        "--snapshot-train-sampling",
        choices=("all", "one_per_game"),
        default="all",
        help=(
            "Training-row sampling in snapshot mode. 'one_per_game' selects one "
            "random minute per training game in each epoch while evaluation still "
            "uses all minute snapshots. Default: all"
        ),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate. Default: 0.3",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Training epochs. Default: 40",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size. Default: 128",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="AdamW learning rate. Default: 1e-3",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
        help="AdamW weight decay. Default: 1e-3",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=6,
        help="Epoch patience on test log loss. Default: 6",
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


def parse_hidden_dims(raw: str) -> list[int]:
    dims = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not dims:
        raise ValueError("No hidden dims provided.")
    return dims


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_one_snapshot_per_game(records: list[object]) -> list[object]:
    by_game: dict[str, list[object]] = {}
    for record in records:
        by_game.setdefault(record.game_id, []).append(record)
    return [
        random.choice(by_game[game_id])
        for game_id in sorted(by_game)
    ]


def numeric_standardization(
    records: list[object],
    numeric_names: list[str],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(len(numeric_names), dtype=np.float64)
    squared_total = np.zeros(len(numeric_names), dtype=np.float64)
    row_count = 0
    for offset in range(0, len(records), batch_size):
        batch = records[offset : offset + batch_size]
        numeric = np.asarray(
            [
                [record.numeric_features.get(name, 0.0) for name in numeric_names]
                for record in batch
            ],
            dtype=np.float64,
        )
        total += numeric.sum(axis=0)
        squared_total += np.square(numeric).sum(axis=0)
        row_count += len(batch)

    mean = total / max(row_count, 1)
    variance = np.maximum(squared_total / max(row_count, 1) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.reshape(1, -1).astype(np.float32), std.reshape(1, -1).astype(np.float32)


def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    clipped = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
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


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
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


def move_inputs_to_device(
    inputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(inputs, tuple):
        return tuple(value.to(device) for value in inputs)
    return inputs.to(device)


def run_model(
    model: MLPModel,
    inputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    if isinstance(inputs, tuple):
        return model(*inputs)
    return model(inputs)


def predict_probs(model: MLPModel, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, _ in loader:
            logits = run_model(model, move_inputs_to_device(inputs, device))
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs)


def train_one_epoch(
    model: MLPModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for inputs, y_batch in loader:
        inputs = move_inputs_to_device(inputs, device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = run_model(model, inputs)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        rows = int(y_batch.shape[0])
        total_loss += float(loss.item()) * rows
        total_rows += rows
    return total_loss / max(total_rows, 1)


def train_variant(
    *,
    args: argparse.Namespace,
    device: torch.device,
    hidden_dims: list[int],
    variant: str,
    variant_config: dict[str, bool],
    output_dir: str,
    label: str,
    records: list[object],
    record_numeric_names: list[str],
    metadata: dict[str, object],
) -> dict[str, object]:
    train_games, test_games = split_game_ids(records, args.train_fraction)
    train_records = [record for record in records if record.game_id in train_games]
    test_records = [record for record in records if record.game_id in test_games]
    feature_names, categorical_vocab = build_feature_space(train_records, record_numeric_names)
    categorical_encoding = args.categorical_encoding
    categorical_embedding_dim = (
        args.categorical_embedding_dim
        if categorical_encoding == "embedding" and categorical_vocab
        else 0
    )
    embedding_vocab = {
        token: index - len(record_numeric_names) + 1
        for token, index in categorical_vocab.items()
    }
    y_train = np.asarray([record.label for record in train_records], dtype=np.float32)
    y_test = np.asarray([record.label for record in test_records], dtype=np.float32)
    numeric_mean, numeric_std = numeric_standardization(
        train_records,
        record_numeric_names,
        args.batch_size,
    )
    collate_batch = RecordBatchCollator(
        record_numeric_names,
        categorical_vocab,
        numeric_mean,
        numeric_std,
        categorical_encoding=categorical_encoding,
    )

    eval_train_loader = DataLoader(
        train_records,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_records,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = MLPModel(
        input_dim=len(feature_names),
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        numeric_dim=len(record_numeric_names),
        categorical_vocab_size=len(categorical_vocab),
        categorical_embedding_dim=categorical_embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    best_test_log_loss = float("inf")
    best_epoch_metrics: dict[str, float] | None = None
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 0
    patience_left = args.early_stopping_patience
    history: list[dict[str, float]] = []
    use_game_sampled_snapshot_training = (
        metadata.get("mode") == "snapshot"
        and args.snapshot_train_sampling == "one_per_game"
    )
    rows_per_epoch = len(train_games) if use_game_sampled_snapshot_training else len(train_records)

    print(
        f"[variant {variant} {label}] games={len(records)} "
        f"train_games={len(train_games)} test_games={len(test_games)} "
        f"features={len(feature_names)} categorical_encoding={categorical_encoding} "
        f"parameters={sum(parameter.numel() for parameter in model.parameters())} "
        f"train_rows_per_epoch={rows_per_epoch}"
    )

    for epoch in range(1, args.epochs + 1):
        epoch_train_records = (
            sample_one_snapshot_per_game(train_records)
            if use_game_sampled_snapshot_training
            else train_records
        )
        train_loader = DataLoader(
            epoch_train_records,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
        )
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_prob = predict_probs(model, eval_train_loader, device)
        test_prob = predict_probs(model, test_loader, device)
        train_metrics = evaluate_probs(y_train, train_prob)
        test_metrics = evaluate_probs(y_test, test_prob)
        epoch_metrics = {
            "epoch": epoch,
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
            f"[variant {variant} {label} epoch {epoch}] "
            f"train_loss={train_loss:.4f} "
            f"test_accuracy={test_metrics['accuracy']:.4f} "
            f"test_auc={test_metrics['roc_auc']:.4f} "
            f"test_log_loss={test_metrics['log_loss']:.4f} "
            f"baseline_accuracy={test_metrics['majority_accuracy']:.4f}"
        )

        if test_metrics["log_loss"] < best_test_log_loss:
            best_test_log_loss = test_metrics["log_loss"]
            best_epoch_metrics = dict(epoch_metrics)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            patience_left = args.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    train_prob = predict_probs(model, eval_train_loader, device)
    test_prob = predict_probs(model, test_loader, device)
    train_metrics = evaluate_probs(y_train, train_prob)
    test_metrics = evaluate_probs(y_test, test_prob)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "variant": variant,
            "variant_config": variant_config,
            "hidden_dims": hidden_dims,
            "input_dim": len(feature_names),
            "feature_names": feature_names,
            "numeric_feature_names": record_numeric_names,
            "numeric_mean": numeric_mean,
            "numeric_std": numeric_std,
            "categorical_encoding": categorical_encoding,
            "categorical_vocab": embedding_vocab if categorical_embedding_dim else {},
            "categorical_embedding_dim": categorical_embedding_dim,
            "categorical_vocab_size": len(categorical_vocab),
            **metadata,
        },
        os.path.join(output_dir, "best_model.pt"),
    )

    summary = {
        "params": vars(args),
        "variant": variant,
        "variant_config": variant_config,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "numeric_feature_names": record_numeric_names,
        "categorical_encoding": categorical_encoding,
        "categorical_embedding_dim": categorical_embedding_dim,
        "categorical_vocab_size": len(categorical_vocab),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_rows_per_epoch": rows_per_epoch,
        "train_games": len(train_games),
        "test_games": len(test_games),
        "best_epoch": best_epoch,
        "best_epoch_metrics": best_epoch_metrics,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "history": history,
        **metadata,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[variant {variant} {label}] best checkpoint saved to: {os.path.join(output_dir, 'best_model.pt')}")
    return summary


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    variants = [part.strip() for part in args.variants.split(",") if part.strip()]
    unknown_variants = [variant for variant in variants if variant not in VARIANT_CONFIGS]
    if unknown_variants:
        raise ValueError(f"unsupported variants: {', '.join(unknown_variants)}")
    minutes = [part.strip() for part in args.minutes.split(",") if part.strip()] if args.mode == "prefix" else []

    device = torch.device("cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"input: {args.input}")
    print(f"mode: {args.mode}")
    if minutes:
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

        if args.mode == "snapshot":
            record_numeric_names = snapshot_numeric_feature_names(
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
                print(f"[variant {variant} snapshot] no records found, skipping")
                continue

            minute_values = [record.minute for record in records]
            variant_summary = train_variant(
                args=args,
                device=device,
                hidden_dims=hidden_dims,
                variant=variant,
                variant_config=variant_config,
                output_dir=variant_dir,
                label="snapshot",
                records=records,
                record_numeric_names=record_numeric_names,
                metadata={
                    "mode": "snapshot",
                    "min_minute": min(minute_values),
                    "max_minute": max(minute_values),
                    "row_count": len(records),
                },
            )
            all_summaries[variant] = variant_summary
            continue

        for minute in minutes:
            prefix_minute = int(float(minute))
            record_numeric_names = numeric_feature_names(
                prefix_minute,
                use_context=variant_config["use_context"],
                include_bias=False,
            )
            records = load_prefix_records(
                csv_path=args.input,
                prefix_minute=prefix_minute,
                use_champions=variant_config["use_champions"],
                use_context=variant_config["use_context"],
                include_patch_token=True,
            )
            if not records:
                print(f"[variant {variant} minute {minute}] no records found, skipping")
                continue

            observed_counts = [len(record.observed_minutes) for record in records]
            avg_observed = sum(observed_counts) / len(observed_counts)
            max_observed = max(observed_counts)
            min_observed = min(observed_counts)
            minute_dir = os.path.join(variant_dir, f"minute_{minute}")
            os.makedirs(minute_dir, exist_ok=True)
            all_summaries[variant][minute] = train_variant(
                args=args,
                device=device,
                hidden_dims=hidden_dims,
                variant=variant,
                variant_config=variant_config,
                output_dir=minute_dir,
                label=f"minute {minute}",
                records=records,
                record_numeric_names=record_numeric_names,
                metadata={
                    "mode": "prefix",
                    "minute": minute,
                    "prefix_minute": prefix_minute,
                    "avg_observed_minutes": avg_observed,
                    "min_observed_minutes": min_observed,
                    "max_observed_minutes": max_observed,
                },
            )

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(all_summaries, handle, indent=2)
    print(f"combined summary saved to: {os.path.join(args.output_dir, 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
