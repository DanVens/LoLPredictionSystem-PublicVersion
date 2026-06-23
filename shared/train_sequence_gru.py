#!/usr/bin/env python3
"""
Train a GRU-based sequence model on the prepared GOL sequence dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GRU model on GOL sequence data.",
    )
    parser.add_argument(
        "--input-prefix",
        default=str(EXPORT_ROOT / "data/sequence_dataset"),
        help="Sequence dataset prefix without extension. Default: data/sequence_dataset",
    )
    parser.add_argument(
        "--output-dir",
        default=str(EXPORT_ROOT / "artifacts/sequence_gru"),
        help="Directory for checkpoints and metrics. Default: artifacts/sequence_gru",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Training epochs. Default: 12",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size. Default: 64",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="GRU hidden size. Default: 64",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=16,
        help="Embedding dimension for champions and categorical features. Default: 16",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate. Default: 1e-3",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay. Default: 1e-4",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate. Default: 0.2",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Chronological fraction of games for training. Default: 0.8",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    parser.add_argument(
        "--prefix-minutes",
        default="10,15,20",
        help=(
            "Comma-separated prefix cutoffs. Each model only sees timesteps up to that "
            "minute. Default: 10,15,20"
        ),
    )
    parser.add_argument(
        "--variants",
        default="gold_only,gold_champions,gold_champions_context",
        help=(
            "Comma-separated ablation variants. Supported: gold_only, "
            "gold_champions, gold_champions_context. Default: "
            "gold_only,gold_champions,gold_champions_context"
        ),
    )
    parser.add_argument(
        "--mixed-prefix-training",
        action="store_true",
        help=(
            "Train a single checkpoint on a mixture of truncated sequence lengths "
            "instead of only one exact prefix cutoff."
        ),
    )
    parser.add_argument(
        "--mixed-prefix-source",
        default="5,10,15,20,25,30,35,40,45",
        help=(
            "Comma-separated source cutoffs to mix when --mixed-prefix-training is "
            "enabled. Only cutoffs <= the target prefix are used. "
            "Default: 5,10,15,20,25,30,35,40,45"
        ),
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help=(
            "Optional binary label smoothing amount in [0, 1). "
            "Targets are moved toward 0.5 during training only. Default: 0.0"
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class SequenceSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray


def chronological_split(dates: np.ndarray, game_ids: np.ndarray, train_fraction: float) -> SequenceSplit:
    order = sorted(range(len(game_ids)), key=lambda idx: (dates[idx], game_ids[idx]))
    split_idx = max(1, min(len(order) - 1, int(len(order) * train_fraction)))
    return SequenceSplit(
        train_indices=np.asarray(order[:split_idx], dtype=np.int64),
        test_indices=np.asarray(order[split_idx:], dtype=np.int64),
    )


def parse_minute_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        values.append(int(text))
    if not values:
        raise ValueError("No prefix minutes were provided.")
    return sorted(set(values))


def apply_prefix_cutoff(arrays: dict[str, np.ndarray], prefix_minute: int) -> dict[str, np.ndarray]:
    minute_mask = (arrays["minutes"] >= 0) & (arrays["minutes"] <= prefix_minute)
    effective_mask = arrays["mask"] * minute_mask.astype(arrays["mask"].dtype)
    lengths = effective_mask.sum(axis=1).astype(np.int64)
    keep_rows = lengths > 0

    trimmed = {
        "X": arrays["X"][keep_rows].copy(),
        "mask": effective_mask[keep_rows].copy(),
        "minutes": arrays["minutes"][keep_rows].copy(),
        "y": arrays["y"][keep_rows].copy(),
        "patch": arrays["patch"][keep_rows].copy(),
        "blue_champion_ids": arrays["blue_champion_ids"][keep_rows].copy(),
        "red_champion_ids": arrays["red_champion_ids"][keep_rows].copy(),
        "tournament_ids": arrays["tournament_ids"][keep_rows].copy(),
        "season_ids": arrays["season_ids"][keep_rows].copy(),
        "stage_ids": (
            arrays["stage_ids"][keep_rows].copy()
            if "stage_ids" in arrays
            else np.zeros(int(keep_rows.sum()), dtype=np.int16)
        ),
        "league_ids": (
            arrays["league_ids"][keep_rows].copy()
            if "league_ids" in arrays
            else np.zeros(int(keep_rows.sum()), dtype=np.int16)
        ),
        "blue_team_ids": (
            arrays["blue_team_ids"][keep_rows].copy()
            if "blue_team_ids" in arrays
            else np.zeros(int(keep_rows.sum()), dtype=np.int16)
        ),
        "red_team_ids": (
            arrays["red_team_ids"][keep_rows].copy()
            if "red_team_ids" in arrays
            else np.zeros(int(keep_rows.sum()), dtype=np.int16)
        ),
        "game_ids": arrays["game_ids"][keep_rows].copy(),
        "dates": arrays["dates"][keep_rows].copy(),
    }
    trimmed["mask"] = trimmed["mask"].astype(np.float32)
    trimmed["X"] *= trimmed["mask"][..., None]
    trimmed["lengths"] = lengths[keep_rows]
    return trimmed


def subset_arrays(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[indices].copy() for key, value in arrays.items()}


def concatenate_arrays(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("No array parts were provided for concatenation.")
    keys = parts[0].keys()
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}


def build_training_arrays(
    arrays: dict[str, np.ndarray],
    target_prefix_minute: int,
    *,
    mixed_prefix_training: bool,
    mixed_prefix_source: list[int],
) -> tuple[dict[str, np.ndarray], list[int]]:
    if not mixed_prefix_training:
        return apply_prefix_cutoff(arrays, target_prefix_minute), [target_prefix_minute]

    source_prefixes = [minute for minute in mixed_prefix_source if minute <= target_prefix_minute]
    if not source_prefixes:
        raise ValueError(
            f"no mixed prefix source minutes are <= target prefix {target_prefix_minute}"
        )
    parts = [apply_prefix_cutoff(arrays, prefix_minute) for prefix_minute in source_prefixes]
    return concatenate_arrays(parts), source_prefixes


class SequenceDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray) -> None:
        self.X = torch.tensor(arrays["X"][indices], dtype=torch.float32)
        self.mask = torch.tensor(arrays["mask"][indices], dtype=torch.float32)
        self.y = torch.tensor(arrays["y"][indices], dtype=torch.float32)
        self.patch = torch.tensor(arrays["patch"][indices], dtype=torch.float32)
        self.blue_champion_ids = torch.tensor(arrays["blue_champion_ids"][indices], dtype=torch.long)
        self.red_champion_ids = torch.tensor(arrays["red_champion_ids"][indices], dtype=torch.long)
        self.tournament_ids = torch.tensor(arrays["tournament_ids"][indices], dtype=torch.long)
        self.season_ids = torch.tensor(arrays["season_ids"][indices], dtype=torch.long)
        self.stage_ids = torch.tensor(arrays["stage_ids"][indices], dtype=torch.long)
        self.league_ids = torch.tensor(arrays["league_ids"][indices], dtype=torch.long)
        self.blue_team_ids = torch.tensor(arrays["blue_team_ids"][indices], dtype=torch.long)
        self.red_team_ids = torch.tensor(arrays["red_team_ids"][indices], dtype=torch.long)
        self.lengths = torch.tensor(arrays["lengths"][indices], dtype=torch.long)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "X": self.X[idx],
            "mask": self.mask[idx],
            "y": self.y[idx],
            "patch": self.patch[idx],
            "blue_champion_ids": self.blue_champion_ids[idx],
            "red_champion_ids": self.red_champion_ids[idx],
            "tournament_ids": self.tournament_ids[idx],
            "season_ids": self.season_ids[idx],
            "stage_ids": self.stage_ids[idx],
            "league_ids": self.league_ids[idx],
            "blue_team_ids": self.blue_team_ids[idx],
            "red_team_ids": self.red_team_ids[idx],
            "lengths": self.lengths[idx],
        }


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    lengths = torch.tensor([int(item["lengths"]) for item in batch], dtype=torch.long)
    order = torch.argsort(lengths, descending=True)

    def stack(key: str) -> torch.Tensor:
        return torch.stack([batch[i][key] for i in order], dim=0)

    return {
        "X": stack("X"),
        "mask": stack("mask"),
        "y": stack("y"),
        "patch": stack("patch"),
        "blue_champion_ids": stack("blue_champion_ids"),
        "red_champion_ids": stack("red_champion_ids"),
        "tournament_ids": stack("tournament_ids"),
        "season_ids": stack("season_ids"),
        "stage_ids": stack("stage_ids"),
        "league_ids": stack("league_ids"),
        "blue_team_ids": stack("blue_team_ids"),
        "red_team_ids": stack("red_team_ids"),
        "lengths": lengths[order],
    }


class SequenceGRUModel(nn.Module):
    def __init__(
        self,
        timestep_dim: int,
        num_champions: int,
        num_tournaments: int,
        num_seasons: int,
        num_stages: int,
        num_leagues: int,
        num_teams: int,
        hidden_size: int,
        embedding_dim: int,
        dropout: float,
        use_champions: bool,
        use_context: bool,
    ) -> None:
        super().__init__()
        self.use_champions = use_champions
        self.use_context = use_context
        self.season_embedding_dim = max(4, embedding_dim // 2)

        if use_champions:
            self.champion_embedding = nn.Embedding(num_champions + 1, embedding_dim, padding_idx=0)
        else:
            self.champion_embedding = None

        if use_context:
            self.tournament_embedding = nn.Embedding(num_tournaments + 1, embedding_dim)
            self.league_embedding = nn.Embedding(num_leagues + 1, embedding_dim)
            self.season_embedding = nn.Embedding(num_seasons + 1, self.season_embedding_dim)
            self.stage_embedding = nn.Embedding(num_stages + 1, self.season_embedding_dim)
            self.team_embedding = nn.Embedding(num_teams + 1, embedding_dim)
        else:
            self.tournament_embedding = None
            self.league_embedding = None
            self.season_embedding = None
            self.stage_embedding = None
            self.team_embedding = None

        self.input_norm = nn.LayerNorm(timestep_dim)
        self.gru = nn.GRU(
            input_size=timestep_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        static_dim = 0
        if use_champions:
            static_dim += embedding_dim * 2
        if use_context:
            static_dim += (
                embedding_dim
                + embedding_dim
                + self.season_embedding_dim
                + self.season_embedding_dim
                + embedding_dim * 2
                + 2
            )

        self.head = nn.Sequential(
            nn.Linear(hidden_size + static_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        X: torch.Tensor,
        lengths: torch.Tensor,
        patch: torch.Tensor,
        blue_champion_ids: torch.Tensor,
        red_champion_ids: torch.Tensor,
        tournament_ids: torch.Tensor,
        season_ids: torch.Tensor,
        stage_ids: torch.Tensor,
        league_ids: torch.Tensor,
        blue_team_ids: torch.Tensor,
        red_team_ids: torch.Tensor,
    ) -> torch.Tensor:
        X = self.input_norm(X)
        packed = pack_padded_sequence(X, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, hidden = self.gru(packed)
        seq_repr = hidden[-1]

        static_parts: list[torch.Tensor] = []
        if self.use_champions and self.champion_embedding is not None:
            blue_embed = self.champion_embedding(blue_champion_ids).mean(dim=1)
            red_embed = self.champion_embedding(red_champion_ids).mean(dim=1)
            static_parts.extend([blue_embed, red_embed])

        if (
            self.use_context
            and self.tournament_embedding is not None
            and self.league_embedding is not None
            and self.season_embedding is not None
            and self.stage_embedding is not None
            and self.team_embedding is not None
        ):
            tournament_embed = self.tournament_embedding(tournament_ids)
            league_embed = self.league_embedding(league_ids)
            season_embed = self.season_embedding(season_ids)
            stage_embed = self.stage_embedding(stage_ids)
            blue_team_embed = self.team_embedding(blue_team_ids)
            red_team_embed = self.team_embedding(red_team_ids)
            static_parts.extend(
                [
                    tournament_embed,
                    league_embed,
                    season_embed,
                    stage_embed,
                    blue_team_embed,
                    red_team_embed,
                    patch,
                ]
            )

        if static_parts:
            combined = torch.cat([seq_repr, *static_parts], dim=1)
        else:
            combined = seq_repr

        logits = self.head(combined).squeeze(1)
        return logits


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


def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(np.float64, copy=False)
    y_prob = y_prob.astype(np.float64, copy=False)
    eps = 1e-12
    clipped = np.clip(y_prob, eps, 1.0 - eps)
    return float(-(y_true * np.log(clipped) + (1 - y_true) * np.log(1.0 - clipped)).mean())


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


def evaluate_model(
    model: SequenceGRUModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                X=batch["X"].to(device),
                lengths=batch["lengths"],
                patch=batch["patch"].to(device),
                blue_champion_ids=batch["blue_champion_ids"].to(device),
                red_champion_ids=batch["red_champion_ids"].to(device),
                tournament_ids=batch["tournament_ids"].to(device),
                season_ids=batch["season_ids"].to(device),
                stage_ids=batch["stage_ids"].to(device),
                league_ids=batch["league_ids"].to(device),
                blue_team_ids=batch["blue_team_ids"].to(device),
                red_team_ids=batch["red_team_ids"].to(device),
            )
            all_logits.append(logits.cpu().numpy())
            all_targets.append(batch["y"].cpu().numpy())

    logits = np.concatenate(all_logits)
    y_true = np.concatenate(all_targets)
    logits = logits.astype(np.float64, copy=False)
    y_true = y_true.astype(np.float64, copy=False)
    stabilized_logits = np.clip(logits, -60.0, 60.0)
    y_prob = 1.0 / (1.0 + np.exp(-stabilized_logits))
    majority_prob = np.full_like(y_prob, y_true.mean())

    return {
        "rows": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "accuracy": binary_accuracy(y_true, y_prob),
        "log_loss": binary_log_loss(y_true, y_prob),
        "brier_score": brier_score(y_true, y_prob),
        "roc_auc": roc_auc(y_true, y_prob),
        "majority_accuracy": float(max(y_true.mean(), 1.0 - y_true.mean())),
        "majority_log_loss": binary_log_loss(y_true, majority_prob),
        "majority_brier_score": brier_score(y_true, majority_prob),
    }


def train_one_epoch(
    model: SequenceGRUModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    label_smoothing: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0

    for batch in loader:
        optimizer.zero_grad()
        logits = model(
            X=batch["X"].to(device),
            lengths=batch["lengths"],
            patch=batch["patch"].to(device),
            blue_champion_ids=batch["blue_champion_ids"].to(device),
            red_champion_ids=batch["red_champion_ids"].to(device),
            tournament_ids=batch["tournament_ids"].to(device),
            season_ids=batch["season_ids"].to(device),
            stage_ids=batch["stage_ids"].to(device),
            league_ids=batch["league_ids"].to(device),
            blue_team_ids=batch["blue_team_ids"].to(device),
            red_team_ids=batch["red_team_ids"].to(device),
        )
        targets = batch["y"].to(device)
        if label_smoothing > 0.0:
            targets_for_loss = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
        else:
            targets_for_loss = targets
        loss = criterion(logits, targets_for_loss)
        loss.backward()
        optimizer.step()

        rows = targets.shape[0]
        total_loss += float(loss.item()) * rows
        total_rows += rows

    return total_loss / total_rows


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in the range [0, 1).")

    base_arrays = dict(np.load(f"{args.input_prefix}.npz", allow_pickle=False))
    with open(f"{args.input_prefix}.json", encoding="utf-8") as handle:
        metadata = json.load(handle)

    device = torch.device("cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"input_prefix: {args.input_prefix}")
    print(f"output_dir: {args.output_dir}")
    prefix_minutes = parse_minute_list(args.prefix_minutes)
    mixed_prefix_source = parse_minute_list(args.mixed_prefix_source)
    variants = [part.strip() for part in args.variants.split(",") if part.strip()]
    unknown_variants = [name for name in variants if name not in VARIANT_CONFIGS]
    if unknown_variants:
        raise ValueError(f"unsupported variants: {', '.join(unknown_variants)}")

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

        base_split = chronological_split(base_arrays["dates"], base_arrays["game_ids"], args.train_fraction)
        train_base_arrays = subset_arrays(base_arrays, base_split.train_indices)
        test_base_arrays = subset_arrays(base_arrays, base_split.test_indices)

        for prefix_minute in prefix_minutes:
            train_arrays, train_source_prefixes = build_training_arrays(
                train_base_arrays,
                prefix_minute,
                mixed_prefix_training=args.mixed_prefix_training,
                mixed_prefix_source=mixed_prefix_source,
            )
            test_arrays, test_source_prefixes = build_training_arrays(
                test_base_arrays,
                prefix_minute,
                mixed_prefix_training=args.mixed_prefix_training,
                mixed_prefix_source=mixed_prefix_source,
            )
            train_indices = np.arange(len(train_arrays["y"]), dtype=np.int64)
            test_indices = np.arange(len(test_arrays["y"]), dtype=np.int64)
            train_dataset = SequenceDataset(train_arrays, train_indices)
            test_dataset = SequenceDataset(test_arrays, test_indices)

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_batch,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_batch,
            )

            model = SequenceGRUModel(
                timestep_dim=int(train_arrays["X"].shape[2]),
                num_champions=len(metadata["champion_to_id"]),
                num_tournaments=len(metadata["tournament_to_id"]),
                num_seasons=max(1, len(metadata["season_to_id"])),
                num_stages=max(1, len(metadata.get("stage_to_id", {}))),
                num_leagues=max(1, len(metadata.get("league_to_id", {}))),
                num_teams=max(1, len(metadata.get("team_to_id", {}))),
                hidden_size=args.hidden_size,
                embedding_dim=args.embedding_dim,
                dropout=args.dropout,
                use_champions=variant_config["use_champions"],
                use_context=variant_config["use_context"],
            ).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            criterion = nn.BCEWithLogitsLoss()
            best_test_log_loss = float("inf")
            best_epoch_metrics: dict[str, float] | None = None
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            history: list[dict[str, float]] = []
            prefix_dir = os.path.join(variant_dir, f"prefix_{prefix_minute}")
            os.makedirs(prefix_dir, exist_ok=True)

            print(
                f"[variant {variant} prefix {prefix_minute}] "
                f"train_games={len(base_split.train_indices)} "
                f"test_games={len(base_split.test_indices)} "
                f"max_len={train_arrays['X'].shape[1]} "
                f"train_rows={len(train_arrays['game_ids'])} "
                f"test_rows={len(test_arrays['game_ids'])} "
                f"source_prefixes={train_source_prefixes}"
            )

            for epoch in range(1, args.epochs + 1):
                train_loss = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                    label_smoothing=args.label_smoothing,
                )
                train_metrics = evaluate_model(model, train_loader, device)
                test_metrics = evaluate_model(model, test_loader, device)

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
                    f"[variant {variant} prefix {prefix_minute} epoch {epoch}] "
                    f"train_loss={train_loss:.4f} "
                    f"test_accuracy={test_metrics['accuracy']:.4f} "
                    f"test_auc={test_metrics['roc_auc']:.4f} "
                    f"test_log_loss={test_metrics['log_loss']:.4f} "
                    f"baseline_accuracy={test_metrics['majority_accuracy']:.4f}"
                )

                if test_metrics["log_loss"] < best_test_log_loss:
                    best_test_log_loss = test_metrics["log_loss"]
                    best_epoch_metrics = dict(epoch_metrics)
                    best_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "args": vars(args),
                            "variant": variant,
                            "variant_config": variant_config,
                            "prefix_minute": prefix_minute,
                            "training_prefix_minutes": train_source_prefixes,
                            "metadata": {
                                "timestep_dim": int(train_arrays["X"].shape[2]),
                                "num_champions": len(metadata["champion_to_id"]),
                                "num_tournaments": len(metadata["tournament_to_id"]),
                                "num_seasons": max(1, len(metadata["season_to_id"])),
                                "num_stages": max(1, len(metadata.get("stage_to_id", {}))),
                                "num_leagues": max(1, len(metadata.get("league_to_id", {}))),
                                "num_teams": max(1, len(metadata.get("team_to_id", {}))),
                            },
                        },
                        os.path.join(prefix_dir, "best_model.pt"),
                    )

            final_epoch_train_metrics = evaluate_model(model, train_loader, device)
            final_epoch_test_metrics = evaluate_model(model, test_loader, device)

            model.load_state_dict(best_state_dict)
            best_train_metrics = evaluate_model(model, train_loader, device)
            best_test_metrics = evaluate_model(model, test_loader, device)

            summary = {
                "params": vars(args),
                "variant": variant,
                "variant_config": variant_config,
                "prefix_minute": prefix_minute,
                "training_prefix_minutes": train_source_prefixes,
                "evaluation_prefix_minutes": test_source_prefixes,
                "train_games": int(len(base_split.train_indices)),
                "test_games": int(len(base_split.test_indices)),
                "best_epoch_metrics": best_epoch_metrics,
                "train_metrics": best_train_metrics,
                "test_metrics": best_test_metrics,
                "final_epoch_train_metrics": final_epoch_train_metrics,
                "final_epoch_test_metrics": final_epoch_test_metrics,
                "history": history,
            }
            with open(os.path.join(prefix_dir, "summary.json"), "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
            all_summaries[variant][str(prefix_minute)] = summary
            print(
                f"[variant {variant} prefix {prefix_minute}] summary saved to: "
                f"{os.path.join(prefix_dir, 'summary.json')}"
            )
            print(
                f"[variant {variant} prefix {prefix_minute}] best checkpoint saved to: "
                f"{os.path.join(prefix_dir, 'best_model.pt')}"
            )

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(all_summaries, handle, indent=2)
    print(f"combined summary saved to: {os.path.join(args.output_dir, 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
EXPORT_ROOT = Path(__file__).resolve().parents[1]
