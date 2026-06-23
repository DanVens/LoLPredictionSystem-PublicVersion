from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

try:
    from scipy import sparse as scipy_sparse
except ImportError:  # pragma: no cover - runtime dependency guard
    scipy_sparse = None

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - runtime dependency guard
    xgb = None

EXPORT_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = EXPORT_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from train_logistic_regression import hash_index as logistic_hash_index
from train_logistic_regression import sigmoid as logistic_sigmoid
from train_mlp import MLPModel
from train_sequence_gru import SequenceGRUModel
from tabular_sequence_features import minute_numeric_feature_names


ROLE_ORDER = ["top", "jgl", "mid", "bot", "spt"]
ROLE_FIELD_LABELS = {
    "top": "TOP",
    "jgl": "JGL",
    "mid": "MID",
    "bot": "BOT",
    "spt": "SPT",
}
MODEL_ORDER = ["gru", "logistic_regression", "xgboost", "mlp"]
MODEL_LABELS = {
    "gru": "GRU",
    "logistic_regression": "Logistic Regression",
    "xgboost": "XGBoost",
    "mlp": "MLP",
}


def normalize_name(raw: Any) -> str:
    text = " ".join(str(raw or "").strip().lower().split())
    return re.sub(r"[^a-z0-9]+", "", text)


def parse_patch_version(raw: Any) -> tuple[int, int]:
    text = str(raw or "").strip()
    if not text:
        return 0, 0
    parts = text.split(".")
    try:
        major = int(parts[0])
    except ValueError:
        major = 0
    try:
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        minor = 0
    return major, minor


@dataclass
class LoadedSequenceModel:
    prefix_minute: int
    model: SequenceGRUModel


@dataclass
class LoadedLogisticModel:
    minute: int
    weights: np.ndarray
    numeric_feature_names: list[str]
    numeric_mean: np.ndarray
    numeric_std: np.ndarray
    hash_dim: int
    variant: str
    model_scope: str


@dataclass
class LoadedXGBoostModel:
    minute: int
    booster: Any
    feature_names: list[str]
    numeric_feature_names: list[str]
    best_iteration: int
    model_scope: str
    prefer_sparse_dmatrix: bool = False


@dataclass
class LoadedMLPModel:
    minute: int
    model: MLPModel
    feature_names: list[str]
    numeric_feature_names: list[str]
    numeric_mean: np.ndarray
    numeric_std: np.ndarray
    categorical_encoding: str = "one_hot"
    categorical_vocab: dict[str, int] | None = None
    model_scope: str = "minute_prefix"


class LiveModelSuitePredictor:
    def __init__(
        self,
        metadata_path: str = str(EXPORT_ROOT / "data/sequence_dataset.json"),
        sequence_model_root: str = str(EXPORT_ROOT / "artifacts/sequence_gru_team_context_mixedlength_ls005/gold_champions_context"),
        logistic_model_root: str = str(EXPORT_ROOT / "artifacts/logistic_regression_no_prefix_std1/gold_champions_context"),
        xgboost_model_root: str = str(EXPORT_ROOT / "artifacts/xgboost_no_prefix_fullgame/gold_champions_context"),
        mlp_model_root: str = str(EXPORT_ROOT / "artifacts/mlp_snapshot_embedding_lr1e4_trial/gold_champions_context"),
        use_sequence_prefix_switching: bool = False,
    ) -> None:
        self.device = torch.device("cpu")
        self.metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        logistic_model_root = self._resolve_model_root(
            logistic_model_root,
            fallback_root=str(EXPORT_ROOT / "artifacts/logistic_regression/gold_champions_context"),
        )
        xgboost_model_root = self._resolve_xgboost_model_root(
            xgboost_model_root,
            fallback_root=str(EXPORT_ROOT / "artifacts/xgboost_all_minutes_tune_1/gold_champions_context"),
            required_minutes=(5, 10, 15, 20, 25),
        )
        self.sequence_models = self._load_sequence_models(sequence_model_root)
        self.logistic_models = self._load_logistic_models(logistic_model_root)
        self.xgboost_models = self._load_xgboost_models(xgboost_model_root)
        self.mlp_models = self._load_mlp_models(mlp_model_root)
        self.sequence_checkpoint_minutes = sorted(self.sequence_models)
        self.logistic_checkpoint_minutes = sorted(self.logistic_models)
        self.xgboost_checkpoint_minutes = sorted(self.xgboost_models)
        self.mlp_checkpoint_minutes = sorted(self.mlp_models)
        self.use_sequence_prefix_switching = bool(use_sequence_prefix_switching)

        self.champion_lookup = self._build_lookup(self.metadata.get("champion_to_id", {}))
        self.team_lookup = self._build_lookup(self.metadata.get("team_to_id", {}))
        self.league_lookup = self._build_lookup(self.metadata.get("league_to_id", {}))
        self.season_lookup = self._build_lookup(self.metadata.get("season_to_id", {}))

        self.current_game_id: int | None = None
        self.minute_history: dict[int, dict[str, int]] = {}

    def _build_lookup(self, mapping: dict[str, int]) -> dict[str, int]:
        lookup: dict[str, int] = {}
        for raw_key, value in mapping.items():
            if not isinstance(raw_key, str):
                continue
            lookup[raw_key] = value
            normalized = normalize_name(raw_key)
            if normalized:
                lookup[normalized] = value
        return lookup

    def _resolve_model_root(self, preferred_root: str, *, fallback_root: str) -> str:
        preferred = Path(preferred_root)
        if self._root_has_logistic_checkpoints(preferred):
            return str(preferred)
        fallback = Path(fallback_root)
        return str(fallback if self._root_has_logistic_checkpoints(fallback) else preferred)

    def _root_has_logistic_checkpoints(self, root: Path) -> bool:
        if not root.exists():
            return False
        if (root / "all_minutes" / "best_model.npz").exists():
            return True
        return any((child / "best_model.npz").exists() for child in root.glob("minute_*"))

    def _resolve_xgboost_model_root(
        self,
        preferred_root: str,
        *,
        fallback_root: str,
        required_minutes: tuple[int, ...] = (),
    ) -> str:
        preferred = Path(preferred_root)
        if self._root_has_xgboost_checkpoints(preferred, required_minutes=required_minutes):
            return str(preferred)
        fallback = Path(fallback_root)
        if self._root_has_xgboost_checkpoints(fallback, required_minutes=required_minutes):
            return str(fallback)
        return str(preferred)

    def _root_has_xgboost_checkpoints(self, root: Path, *, required_minutes: tuple[int, ...] = ()) -> bool:
        if not root.exists():
            return False
        if (root / "all_minutes" / "best_model.json").exists():
            return True
        available_minutes: set[int] = set()
        for child in root.glob("minute_*"):
            if not child.is_dir() or not (child / "best_model.json").exists():
                continue
            raw_minute = child.name[len("minute_") :]
            try:
                available_minutes.add(int(raw_minute))
            except ValueError:
                continue
        if not available_minutes:
            return False
        if required_minutes:
            return set(required_minutes).issubset(available_minutes)
        return True

    def _load_sequence_models(self, model_root: str) -> dict[int, LoadedSequenceModel]:
        models: dict[int, LoadedSequenceModel] = {}
        for prefix_minute in self._discover_checkpoint_minutes(model_root, prefix=True):
            checkpoint_path = Path(model_root) / f"prefix_{prefix_minute}" / "best_model.pt"
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            metadata = checkpoint["metadata"]
            variant_config = checkpoint.get(
                "variant_config",
                {"use_champions": True, "use_context": True},
            )

            model = SequenceGRUModel(
                timestep_dim=int(metadata["timestep_dim"]),
                num_champions=int(metadata["num_champions"]),
                num_tournaments=int(metadata["num_tournaments"]),
                num_seasons=int(metadata["num_seasons"]),
                num_stages=int(metadata.get("num_stages", 1)),
                num_leagues=int(metadata.get("num_leagues", 1)),
                num_teams=int(metadata.get("num_teams", 1)),
                hidden_size=int(checkpoint["args"]["hidden_size"]),
                embedding_dim=int(checkpoint["args"]["embedding_dim"]),
                dropout=float(checkpoint["args"]["dropout"]),
                use_champions=bool(variant_config.get("use_champions", True)),
                use_context=bool(variant_config.get("use_context", True)),
            ).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            models[prefix_minute] = LoadedSequenceModel(prefix_minute=prefix_minute, model=model)
        return models

    def _load_logistic_models(self, model_root: str) -> dict[int, LoadedLogisticModel]:
        models: dict[int, LoadedLogisticModel] = {}
        all_minutes_checkpoint = Path(model_root) / "all_minutes" / "best_model.npz"
        if all_minutes_checkpoint.exists():
            checkpoint = np.load(all_minutes_checkpoint, allow_pickle=True)
            model_scope = (
                str(np.asarray(checkpoint["model_scope"]).item())
                if "model_scope" in checkpoint
                else "all_minutes_snapshot"
            )
            models[-1] = LoadedLogisticModel(
                minute=-1,
                weights=checkpoint["weights"].astype(np.float64),
                numeric_feature_names=list(checkpoint["numeric_feature_names"].tolist()),
                numeric_mean=np.asarray(
                    checkpoint["numeric_mean"] if "numeric_mean" in checkpoint else np.zeros(len(checkpoint["numeric_feature_names"])),
                    dtype=np.float64,
                ).reshape(-1),
                numeric_std=np.asarray(
                    checkpoint["numeric_std"] if "numeric_std" in checkpoint else np.ones(len(checkpoint["numeric_feature_names"])),
                    dtype=np.float64,
                ).reshape(-1),
                hash_dim=int(np.asarray(checkpoint["hash_dim"]).item()),
                variant=str(np.asarray(checkpoint["variant"]).item()),
                model_scope=model_scope,
            )
            return models

        for minute in self._discover_checkpoint_minutes(model_root):
            checkpoint_path = Path(model_root) / f"minute_{minute}" / "best_model.npz"
            checkpoint = np.load(checkpoint_path, allow_pickle=True)
            models[minute] = LoadedLogisticModel(
                minute=minute,
                weights=checkpoint["weights"].astype(np.float64),
                numeric_feature_names=list(checkpoint["numeric_feature_names"].tolist()),
                numeric_mean=np.zeros(len(checkpoint["numeric_feature_names"]), dtype=np.float64),
                numeric_std=np.ones(len(checkpoint["numeric_feature_names"]), dtype=np.float64),
                hash_dim=int(np.asarray(checkpoint["hash_dim"]).item()),
                variant=str(np.asarray(checkpoint["variant"]).item()),
                model_scope="minute_prefix",
            )
        return models

    def _load_xgboost_models(self, model_root: str) -> dict[int, LoadedXGBoostModel]:
        if xgb is None:  # pragma: no cover - guarded by dependency install
            return {}

        models: dict[int, LoadedXGBoostModel] = {}
        all_minutes_dir = Path(model_root) / "all_minutes"
        all_minutes_checkpoint = all_minutes_dir / "best_model.json"
        if all_minutes_checkpoint.exists():
            summary = json.loads((all_minutes_dir / "summary.json").read_text(encoding="utf-8"))
            booster = xgb.Booster()
            booster.load_model(str(all_minutes_checkpoint))
            feature_names = list(summary.get("feature_names") or [])
            numeric_feature_names = list(summary.get("numeric_feature_names") or [])
            params = summary.get("params") or {}
            inference_matrix_format = str(summary.get("inference_matrix_format") or "").strip().lower()
            output_dir = str(params.get("output_dir") or "")
            prefer_sparse_dmatrix = inference_matrix_format == "sparse" or (
                not inference_matrix_format and "xgboost_no_prefix" in output_dir
            )
            if not numeric_feature_names:
                numeric_feature_names = [
                    name for name in feature_names if "=" not in name and "|" not in name
                ]
            models[-1] = LoadedXGBoostModel(
                minute=-1,
                booster=booster,
                feature_names=feature_names,
                numeric_feature_names=numeric_feature_names,
                best_iteration=int(summary.get("best_iteration") or 0),
                model_scope=str(summary.get("model_scope") or "all_minutes_snapshot"),
                prefer_sparse_dmatrix=prefer_sparse_dmatrix,
            )
            return models

        for minute in self._discover_checkpoint_minutes(model_root):
            minute_dir = Path(model_root) / f"minute_{minute}"
            summary = json.loads((minute_dir / "summary.json").read_text(encoding="utf-8"))
            booster = xgb.Booster()
            booster.load_model(str(minute_dir / "best_model.json"))
            feature_names = list(summary.get("feature_names") or [])
            numeric_feature_names = list(summary.get("numeric_feature_names") or [])
            if not numeric_feature_names:
                numeric_feature_names = [
                    name for name in feature_names if "=" not in name and "|" not in name
                ]
            models[minute] = LoadedXGBoostModel(
                minute=minute,
                booster=booster,
                feature_names=feature_names,
                numeric_feature_names=numeric_feature_names,
                best_iteration=int(summary.get("best_iteration") or 0),
                model_scope="minute_prefix",
                prefer_sparse_dmatrix=False,
            )
        return models

    def _load_mlp_models(self, model_root: str) -> dict[int, LoadedMLPModel]:
        models: dict[int, LoadedMLPModel] = {}

        def load_model(checkpoint_path: Path, minute: int) -> LoadedMLPModel:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            categorical_encoding = str(checkpoint.get("categorical_encoding") or "one_hot")
            categorical_vocab = {
                str(token): int(index)
                for token, index in dict(checkpoint.get("categorical_vocab") or {}).items()
            }
            categorical_embedding_dim = int(checkpoint.get("categorical_embedding_dim") or 0)
            model = MLPModel(
                input_dim=int(checkpoint["input_dim"]),
                hidden_dims=[int(value) for value in checkpoint["hidden_dims"]],
                dropout=float(checkpoint["args"]["dropout"]),
                numeric_dim=len(checkpoint["numeric_feature_names"]),
                categorical_vocab_size=int(checkpoint.get("categorical_vocab_size") or len(categorical_vocab)),
                categorical_embedding_dim=categorical_embedding_dim,
            ).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            return LoadedMLPModel(
                minute=minute,
                model=model,
                feature_names=list(checkpoint["feature_names"]),
                numeric_feature_names=list(checkpoint["numeric_feature_names"]),
                numeric_mean=np.asarray(checkpoint["numeric_mean"], dtype=np.float32).reshape(-1),
                numeric_std=np.asarray(checkpoint["numeric_std"], dtype=np.float32).reshape(-1),
                categorical_encoding=categorical_encoding,
                categorical_vocab=categorical_vocab,
                model_scope="all_minutes_snapshot" if checkpoint.get("mode") == "snapshot" else "minute_prefix",
            )

        snapshot_path = Path(model_root) / "best_model.pt"
        if snapshot_path.exists():
            models[-1] = load_model(snapshot_path, -1)

        for minute in self._discover_checkpoint_minutes(model_root):
            checkpoint_path = Path(model_root) / f"minute_{minute}" / "best_model.pt"
            models[minute] = load_model(checkpoint_path, minute)
        return models

    def reset(self) -> None:
        self.current_game_id = None
        self.minute_history = {}

    def _discover_checkpoint_minutes(self, model_root: str, *, prefix: bool = False) -> list[int]:
        root = Path(model_root)
        if not root.exists():
            return []
        stem = "prefix_" if prefix else "minute_"
        minutes: list[int] = []
        for child in root.iterdir():
            if not child.is_dir() or not child.name.startswith(stem):
                continue
            raw = child.name[len(stem) :]
            try:
                minutes.append(int(raw))
            except ValueError:
                continue
        return sorted(set(minutes))

    def _infer_season_key(self, result: dict[str, Any]) -> str:
        explicit = str(result.get("source_season") or "").strip()
        if explicit:
            return explicit
        current_year = datetime.now(timezone.utc).year
        if current_year == 2025:
            return "S15"
        if current_year == 2026:
            return "S16"
        return f"S{max(current_year - 2010, 0)}" if current_year >= 2011 else ""

    def _infer_season_id(self, result: dict[str, Any]) -> int:
        return self.season_lookup.get(self._infer_season_key(result), 0)

    def _lookup_id(self, lookup: dict[str, int], raw: Any) -> int:
        text = str(raw or "").strip()
        if not text:
            return 0
        return lookup.get(text, lookup.get(normalize_name(text), 0))

    def _choose_prefix(self, minute_value: int, available_minutes: list[int]) -> int:
        if not available_minutes:
            raise ValueError("no checkpoint minutes are available")
        for checkpoint_minute in available_minutes:
            if minute_value <= checkpoint_minute:
                return checkpoint_minute
        return available_minutes[-1]

    def _choose_sequence_prefix(self, minute_value: int) -> int:
        if self.use_sequence_prefix_switching:
            return self._choose_prefix(minute_value, self.sequence_checkpoint_minutes)
        if not self.sequence_checkpoint_minutes:
            raise ValueError("no sequence checkpoint minutes are available")
        return self.sequence_checkpoint_minutes[-1]

    def _choose_logistic_checkpoint(self, minute_value: int) -> int:
        if not self.logistic_models:
            raise ValueError("no logistic regression checkpoints are available")
        if -1 in self.logistic_models:
            return -1
        return self._choose_prefix(minute_value, self.logistic_checkpoint_minutes)

    def _choose_xgboost_checkpoint(self, minute_value: int) -> int:
        if not self.xgboost_models:
            raise ValueError("no XGBoost checkpoints are available")
        if -1 in self.xgboost_models:
            return -1
        return self._choose_prefix(minute_value, self.xgboost_checkpoint_minutes)

    def _choose_mlp_checkpoint(self, minute_value: int) -> int:
        if not self.mlp_models:
            raise ValueError("no MLP checkpoints are available")
        if -1 in self.mlp_models:
            return -1
        return self._choose_prefix(minute_value, self.mlp_checkpoint_minutes)

    def _infer_minute_floor_from_gold(self, result: dict[str, Any]) -> int:
        blue_total = result.get("gold_left")
        red_total = result.get("gold_right")
        if blue_total is None or red_total is None:
            blue_total = 0
            red_total = 0
            for role in ROLE_ORDER:
                blue_value = result.get(f"blue_gold_{role}")
                red_value = result.get(f"red_gold_{role}")
                if blue_value is None or red_value is None:
                    return 0
                blue_total += int(blue_value)
                red_total += int(red_value)

        total_gold = int(blue_total) + int(red_total)
        # Guard against broken live timers by requiring obviously late-game gold states
        # to use at least the corresponding checkpoint family seen in training data.
        if total_gold >= 60000:
            return 20
        if total_gold >= 40000:
            return 15
        if total_gold >= 30000:
            return 10
        return 0

    def _sanitize_minute_value(self, result: dict[str, Any], minute_value: int) -> int:
        minute_floor = self._infer_minute_floor_from_gold(result)
        if self.minute_history:
            minute_floor = max(minute_floor, max(self.minute_history))
        return max(0, min(45, max(minute_value, minute_floor)))

    def _extract_minute_snapshot(self, result: dict[str, Any]) -> dict[str, int] | None:
        values: dict[str, int] = {}
        for role in ROLE_ORDER:
            blue_gold = result.get(f"blue_gold_{role}")
            red_gold = result.get(f"red_gold_{role}")
            if blue_gold is None or red_gold is None:
                return None
            values[f"blue_gold_{role}"] = int(blue_gold)
            values[f"red_gold_{role}"] = int(red_gold)
            values[f"gold_lead_{role}"] = int(blue_gold) - int(red_gold)
        return values

    def _missing_gold_fields(self, result: dict[str, Any]) -> list[str]:
        missing: list[str] = []
        for role in ROLE_ORDER:
            if result.get(f"blue_gold_{role}") is None:
                missing.append(f"blue_{ROLE_FIELD_LABELS[role]}_gold")
            if result.get(f"red_gold_{role}") is None:
                missing.append(f"red_{ROLE_FIELD_LABELS[role]}_gold")
        return missing

    def _build_minute_numeric_features_from_snapshot(self, snapshot: dict[str, int]) -> dict[str, float]:
        numeric: dict[str, float] = {}
        blue_total = 0.0
        red_total = 0.0
        lane_leads: list[float] = []

        for role in ROLE_ORDER:
            blue_gold = float(snapshot.get(f"blue_gold_{role}") or 0.0) / 1000.0
            red_gold = float(snapshot.get(f"red_gold_{role}") or 0.0) / 1000.0
            lead = blue_gold - red_gold
            numeric[f"{role}_blue_gold_k"] = blue_gold
            numeric[f"{role}_red_gold_k"] = red_gold
            numeric[f"{role}_gold_lead_k"] = lead
            blue_total += blue_gold
            red_total += red_gold
            lane_leads.append(lead)

        numeric["blue_gold_total_k"] = blue_total
        numeric["red_gold_total_k"] = red_total
        numeric["gold_lead_total_k"] = blue_total - red_total
        numeric["gold_lead_abs_k"] = abs(blue_total - red_total)
        numeric["blue_lanes_ahead"] = float(sum(1 for lead in lane_leads if lead > 0.0))
        return numeric

    def _extract_snapshot_numeric_feature_map(
        self,
        result: dict[str, Any],
        current_minute: int | None = None,
    ) -> dict[str, float]:
        snapshot = self._extract_minute_snapshot(result)
        if snapshot is None:
            return {}
        numeric = self._build_minute_numeric_features_from_snapshot(snapshot)
        if current_minute is None:
            raw_minute = result.get("minute")
            try:
                current_minute = int(float(raw_minute)) if raw_minute is not None else 0
            except (TypeError, ValueError):
                current_minute = 0
        numeric["current_minute"] = float(current_minute)
        patch_major, patch_minor = parse_patch_version(result.get("patch_version"))
        numeric["patch_major"] = float(patch_major)
        numeric["patch_minor"] = float(patch_minor)
        return numeric

    def _extract_prefix_numeric_feature_map(
        self,
        result: dict[str, Any],
        prefix_minute: int,
    ) -> dict[str, float]:
        if prefix_minute < 0:
            return {}
        observed_minutes = sorted(minute for minute in self.minute_history if minute <= prefix_minute)
        numeric: dict[str, float] = {
            "observed_minutes_count": float(len(observed_minutes)),
            "observed_minutes_fraction": float(len(observed_minutes)) / float(prefix_minute + 1),
            "last_observed_minute": float(observed_minutes[-1]) if observed_minutes else 0.0,
        }

        minute_feature_names = minute_numeric_feature_names()
        zero_features = {name: 0.0 for name in minute_feature_names}
        observed_set = set(observed_minutes)
        last_features: dict[str, float] | None = None

        for minute in range(prefix_minute + 1):
            minute_prefix = f"minute_{minute}"
            if minute in observed_set:
                minute_features = self._build_minute_numeric_features_from_snapshot(self.minute_history[minute])
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

        patch_major, patch_minor = parse_patch_version(result.get("patch_version"))
        numeric["patch_major"] = float(patch_major)
        numeric["patch_minor"] = float(patch_minor)
        return numeric

    def _uses_sequence_tabular_features(self, numeric_names: list[str]) -> bool:
        return any(
            name.startswith("minute_") or name in {"observed_minutes_count", "observed_minutes_fraction", "last_observed_minute"}
            for name in numeric_names
        )

    def _observed_prefix_minutes(self, prefix_minute: int) -> list[int]:
        return sorted(minute for minute in self.minute_history if minute <= prefix_minute)

    def _select_mlp_model(self, prefix_minute: int) -> tuple[LoadedMLPModel, str]:
        dense_loaded = self.mlp_models[prefix_minute]
        if not self._uses_sequence_tabular_features(dense_loaded.numeric_feature_names):
            return dense_loaded, "snapshot"

        # The all-minutes dense artifacts were trained with explicit observed/minute
        # features and forward-filled gaps, so they already model sparse live history.
        # Falling back into the legacy mlp/ root mixes incompatible checkpoint families
        # (snapshot-only for some minutes, sequence-aware for others), which can create
        # unstable live probabilities. Keep live on one consistent artifact family.
        return dense_loaded, "dense_prefix"

    def _extract_snapshot_tokens(
        self,
        result: dict[str, Any],
        *,
        include_champions: bool,
        include_context: bool,
        include_patch_token: bool,
    ) -> list[str]:
        tokens: list[str] = []

        if include_champions:
            for role in ROLE_ORDER:
                role_label = ROLE_FIELD_LABELS[role]
                blue = str(result.get(f"blue_champion_{role}") or "").strip()
                red = str(result.get(f"red_champion_{role}") or "").strip()
                if blue:
                    tokens.append(f"blue_champion_{role_label}={blue}")
                    tokens.append(f"role_blue_{role_label}={role_label}|{blue}")
                if red:
                    tokens.append(f"red_champion_{role_label}={red}")
                    tokens.append(f"role_red_{role_label}={role_label}|{red}")
                if blue and red:
                    tokens.append(f"matchup_{role_label}={blue}_vs_{red}")

        if include_context:
            source_season = self._infer_season_key(result)
            league = str(result.get("league") or "").strip()
            tournament_stage = str(
                result.get("tournament_stage")
                or result.get("stage")
                or result.get("match_stage")
                or ""
            ).strip()
            tournament = str(result.get("tournament") or league or "").strip()
            blue_team = str(result.get("team_left") or "").strip()
            red_team = str(result.get("team_right") or "").strip()
            patch = str(result.get("patch_version") or "").strip()

            if source_season:
                tokens.append(f"source_season={source_season}")
            if league:
                tokens.append(f"league={league}")
            if tournament_stage:
                tokens.append(f"tournament_stage={tournament_stage}")
            if tournament:
                tokens.append(f"tournament={tournament}")
            if blue_team:
                tokens.append(f"blue_team={blue_team}")
            if red_team:
                tokens.append(f"red_team={red_team}")
            if blue_team and red_team:
                tokens.append(f"team_matchup={blue_team}|{red_team}")
            if include_patch_token and patch:
                tokens.append(f"patch={patch}")

        return tokens

    def _build_sequence_inputs(self, result: dict[str, Any]) -> dict[str, torch.Tensor] | None:
        if not self.minute_history:
            return None

        minute_keys = sorted(self.minute_history)
        timestep_dim = len(self.metadata["timestep_feature_names"])
        X = np.zeros((1, len(minute_keys), timestep_dim), dtype=np.float32)

        for step_idx, minute_key in enumerate(minute_keys):
            snapshot = self.minute_history[minute_key]
            blue_values = [snapshot[f"blue_gold_{role}"] for role in ROLE_ORDER]
            red_values = [snapshot[f"red_gold_{role}"] for role in ROLE_ORDER]
            lead_values = [snapshot[f"gold_lead_{role}"] for role in ROLE_ORDER]
            feature_values = (
                blue_values
                + red_values
                + lead_values
                + [
                    float(sum(blue_values)),
                    float(sum(red_values)),
                    float(sum(lead_values)),
                    float(sum(1 for value in lead_values if value > 0)),
                ]
            )
            X[0, step_idx, :] = np.asarray(feature_values, dtype=np.float32)

        blue_champion_ids = np.zeros((1, 5), dtype=np.int64)
        red_champion_ids = np.zeros((1, 5), dtype=np.int64)
        for role_idx, role in enumerate(ROLE_ORDER):
            blue_champion_ids[0, role_idx] = self._lookup_id(
                self.champion_lookup,
                result.get(f"blue_champion_{role}"),
            )
            red_champion_ids[0, role_idx] = self._lookup_id(
                self.champion_lookup,
                result.get(f"red_champion_{role}"),
            )

        patch_major, patch_minor = parse_patch_version(result.get("patch_version"))

        return {
            "X": torch.tensor(X, dtype=torch.float32, device=self.device),
            "lengths": torch.tensor([len(minute_keys)], dtype=torch.long),
            "patch": torch.tensor([[patch_major, patch_minor]], dtype=torch.float32, device=self.device),
            "blue_champion_ids": torch.tensor(blue_champion_ids, dtype=torch.long, device=self.device),
            "red_champion_ids": torch.tensor(red_champion_ids, dtype=torch.long, device=self.device),
            "tournament_ids": torch.tensor([0], dtype=torch.long, device=self.device),
            "season_ids": torch.tensor([self._infer_season_id(result)], dtype=torch.long, device=self.device),
            "stage_ids": torch.tensor([0], dtype=torch.long, device=self.device),
            "league_ids": torch.tensor(
                [self._lookup_id(self.league_lookup, result.get("league"))],
                dtype=torch.long,
                device=self.device,
            ),
            "blue_team_ids": torch.tensor(
                [self._lookup_id(self.team_lookup, result.get("team_left"))],
                dtype=torch.long,
                device=self.device,
            ),
            "red_team_ids": torch.tensor(
                [self._lookup_id(self.team_lookup, result.get("team_right"))],
                dtype=torch.long,
                device=self.device,
            ),
        }

    def _build_logistic_features(
        self,
        loaded: LoadedLogisticModel,
        numeric_map: dict[str, float],
        tokens: list[str],
    ) -> list[tuple[int, float]]:
        features: list[tuple[int, float]] = []
        for idx, name in enumerate(loaded.numeric_feature_names):
            value = 1.0 if name == "bias" else float(numeric_map.get(name, 0.0))
            value = (value - float(loaded.numeric_mean[idx])) / float(loaded.numeric_std[idx])
            if value != 0.0 or name == "bias":
                features.append((idx, value))

        offset = len(loaded.numeric_feature_names)
        for token in tokens:
            features.append((logistic_hash_index(token, loaded.hash_dim, offset), 1.0))
        return features

    def _build_dense_vector(
        self,
        feature_names: list[str],
        numeric_feature_names: list[str],
        numeric_map: dict[str, float],
        tokens: list[str],
    ) -> np.ndarray:
        vector = np.zeros(len(feature_names), dtype=np.float32)
        numeric_index = {name: idx for idx, name in enumerate(numeric_feature_names)}
        for name, value in numeric_map.items():
            idx = numeric_index.get(name)
            if idx is not None and idx < len(vector):
                vector[idx] = float(value)

        categorical_index = {name: idx for idx, name in enumerate(feature_names[len(numeric_feature_names):], start=len(numeric_feature_names))}
        for token in tokens:
            idx = categorical_index.get(token)
            if idx is not None:
                vector[idx] = 1.0
        return vector

    def _empty_prediction(
        self,
        key: str,
        status: str = "not_ready",
        reason: str | None = None,
        prefix_minute: int | None = None,
        current_minute: int | None = None,
    ) -> dict[str, Any]:
        return {
            "model_key": key,
            "model_label": MODEL_LABELS[key],
            "status": status,
            "reason": reason,
            "blue_win_prob": None,
            "red_win_prob": None,
            "prefix_minute": prefix_minute,
            "current_minute": current_minute,
            "timesteps": None,
        }

    def _predict_gru(
        self,
        model_inputs: dict[str, torch.Tensor],
        prefix_minute: int,
        current_minute: int,
    ) -> dict[str, Any]:
        loaded = self.sequence_models[prefix_minute]
        with torch.no_grad():
            logits = loaded.model(
                X=model_inputs["X"],
                lengths=model_inputs["lengths"],
                patch=model_inputs["patch"],
                blue_champion_ids=model_inputs["blue_champion_ids"],
                red_champion_ids=model_inputs["red_champion_ids"],
                tournament_ids=model_inputs["tournament_ids"],
                season_ids=model_inputs["season_ids"],
                stage_ids=model_inputs["stage_ids"],
                league_ids=model_inputs["league_ids"],
                blue_team_ids=model_inputs["blue_team_ids"],
                red_team_ids=model_inputs["red_team_ids"],
            )
            blue_prob = float(torch.sigmoid(logits)[0].cpu().item())

        return {
            "model_key": "gru",
            "model_label": MODEL_LABELS["gru"],
            "status": "ready",
            "reason": None,
            "blue_win_prob": blue_prob,
            "red_win_prob": 1.0 - blue_prob,
            "prefix_minute": prefix_minute,
            "current_minute": current_minute,
            "timesteps": len(self.minute_history),
        }

    def _predict_logistic(
        self,
        snapshot_numeric_map: dict[str, float],
        prefix_numeric_map: dict[str, float],
        tokens: list[str],
        checkpoint_key: int,
        current_minute: int,
    ) -> dict[str, Any]:
        loaded = self.logistic_models[checkpoint_key]
        numeric_map = (
            prefix_numeric_map
            if self._uses_sequence_tabular_features(loaded.numeric_feature_names)
            else snapshot_numeric_map
        )
        features = self._build_logistic_features(loaded, numeric_map, tokens)
        total = 0.0
        for idx, value in features:
            total += float(loaded.weights[idx]) * value
        blue_prob = float(logistic_sigmoid(total))
        return {
            "model_key": "logistic_regression",
            "model_label": MODEL_LABELS["logistic_regression"],
            "status": "ready",
            "reason": None,
            "blue_win_prob": blue_prob,
            "red_win_prob": 1.0 - blue_prob,
            "prefix_minute": None if loaded.model_scope == "all_minutes_snapshot" else loaded.minute,
            "current_minute": current_minute,
            "timesteps": None,
        }

    def _predict_xgboost(
        self,
        snapshot_numeric_map: dict[str, float],
        prefix_numeric_map: dict[str, float],
        tokens: list[str],
        checkpoint_key: int,
        current_minute: int,
    ) -> dict[str, Any]:
        if xgb is None:
            return self._empty_prediction("xgboost", status="error", reason="xgboost is not installed.")

        loaded = self.xgboost_models[checkpoint_key]
        numeric_map = (
            prefix_numeric_map
            if self._uses_sequence_tabular_features(loaded.numeric_feature_names)
            else snapshot_numeric_map
        )
        vector = self._build_dense_vector(
            loaded.feature_names,
            loaded.numeric_feature_names,
            numeric_map,
            tokens,
        )
        if loaded.prefer_sparse_dmatrix and scipy_sparse is not None:
            matrix_input = scipy_sparse.csr_matrix(vector.reshape(1, -1), dtype=np.float32)
        else:
            matrix_input = vector.reshape(1, -1)
        matrix = xgb.DMatrix(matrix_input, feature_names=loaded.feature_names)
        blue_prob = float(
            loaded.booster.predict(
                matrix,
                iteration_range=(0, loaded.best_iteration + 1),
            )[0]
        )
        return {
            "model_key": "xgboost",
            "model_label": MODEL_LABELS["xgboost"],
            "status": "ready",
            "reason": None,
            "blue_win_prob": blue_prob,
            "red_win_prob": 1.0 - blue_prob,
            "prefix_minute": None if loaded.model_scope == "all_minutes_snapshot" else loaded.minute,
            "current_minute": current_minute,
            "timesteps": None,
        }

    def _predict_mlp(
        self,
        snapshot_numeric_map: dict[str, float],
        prefix_numeric_map: dict[str, float],
        tokens: list[str],
        prefix_minute: int,
        current_minute: int,
    ) -> dict[str, Any]:
        loaded, artifact_mode = self._select_mlp_model(prefix_minute)
        numeric_map = (
            prefix_numeric_map
            if self._uses_sequence_tabular_features(loaded.numeric_feature_names)
            else snapshot_numeric_map
        )
        with torch.no_grad():
            if loaded.categorical_encoding == "embedding":
                numeric = np.asarray(
                    [[numeric_map.get(name, 0.0) for name in loaded.numeric_feature_names]],
                    dtype=np.float32,
                )
                numeric = (numeric - loaded.numeric_mean.reshape(1, -1)) / loaded.numeric_std.reshape(1, -1)
                categorical_vocab = loaded.categorical_vocab or {}
                categorical_ids = [categorical_vocab.get(token, 0) for token in tokens] or [0]
                logits = loaded.model(
                    torch.tensor(numeric, dtype=torch.float32, device=self.device),
                    torch.tensor(categorical_ids, dtype=torch.long, device=self.device),
                    torch.tensor([0], dtype=torch.long, device=self.device),
                )
            else:
                vector = self._build_dense_vector(
                    loaded.feature_names,
                    loaded.numeric_feature_names,
                    numeric_map,
                    tokens,
                )
                numeric_dim = len(loaded.numeric_feature_names)
                vector[:numeric_dim] = (vector[:numeric_dim] - loaded.numeric_mean) / loaded.numeric_std
                X = torch.tensor(vector.reshape(1, -1), dtype=torch.float32, device=self.device)
                logits = loaded.model(X)
            blue_prob = float(torch.sigmoid(logits)[0].cpu().item())
        return {
            "model_key": "mlp",
            "model_label": MODEL_LABELS["mlp"],
            "status": "ready",
            "reason": None,
            "blue_win_prob": blue_prob,
            "red_win_prob": 1.0 - blue_prob,
            "prefix_minute": None if loaded.model_scope == "all_minutes_snapshot" else prefix_minute,
            "current_minute": current_minute,
            "timesteps": None,
            "artifact_mode": artifact_mode,
        }

    def _build_comparison(self, model_predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        ready_items = [
            prediction
            for key, prediction in model_predictions.items()
            if key in MODEL_ORDER and prediction.get("status") == "ready" and prediction.get("blue_win_prob") is not None
        ]
        if not ready_items:
            return {
                "ready_model_count": 0,
                "total_model_count": len(MODEL_ORDER),
                "consensus_blue_win_prob": None,
                "consensus_red_win_prob": None,
                "min_blue_win_prob": None,
                "max_blue_win_prob": None,
                "spread_blue_win_prob": None,
                "leader_model": None,
                "trailer_model": None,
            }

        ready_items = sorted(ready_items, key=lambda item: item["blue_win_prob"])
        values = [float(item["blue_win_prob"]) for item in ready_items]
        consensus = float(sum(values) / len(values))
        return {
            "ready_model_count": len(ready_items),
            "total_model_count": len(MODEL_ORDER),
            "consensus_blue_win_prob": consensus,
            "consensus_red_win_prob": 1.0 - consensus,
            "min_blue_win_prob": values[0],
            "max_blue_win_prob": values[-1],
            "spread_blue_win_prob": values[-1] - values[0],
            "leader_model": ready_items[-1]["model_key"],
            "trailer_model": ready_items[0]["model_key"],
        }

    def _apply_primary_prediction(
        self,
        result: dict[str, Any],
        model_predictions: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        primary = None
        for key in MODEL_ORDER:
            prediction = model_predictions.get(key)
            if prediction and prediction.get("status") == "ready":
                primary = prediction
                break

        if primary is None:
            first_reason = next(
                (
                    prediction.get("reason")
                    for prediction in model_predictions.values()
                    if prediction.get("reason")
                ),
                None,
            )
            result["prediction_status"] = "not_ready"
            result["prediction_reason"] = first_reason
            return result

        result["prediction_blue_win_prob"] = primary["blue_win_prob"]
        result["prediction_red_win_prob"] = primary["red_win_prob"]
        result["prediction_model"] = primary["model_key"]
        result["prediction_prefix_minute"] = primary.get("prefix_minute")
        result["prediction_timesteps"] = primary.get("timesteps") or 0
        result["prediction_current_minute"] = primary.get("current_minute")
        result["prediction_status"] = primary["status"]
        result["prediction_reason"] = primary.get("reason")
        return result

    def enrich(self, result: dict[str, Any]) -> dict[str, Any]:
        result = dict(result)
        result["prediction_blue_win_prob"] = None
        result["prediction_red_win_prob"] = None
        result["prediction_model"] = None
        result["prediction_prefix_minute"] = None
        result["prediction_timesteps"] = 0
        result["prediction_current_minute"] = None
        result["prediction_status"] = "not_ready"
        result["prediction_reason"] = None
        result["prediction_consensus_blue_win_prob"] = None
        result["prediction_consensus_red_win_prob"] = None

        model_predictions = {key: self._empty_prediction(key) for key in MODEL_ORDER}
        result["model_predictions"] = model_predictions

        if result.get("status") != "in_progress":
            self.reset()
            reason = f"game status is {result.get('status') or 'unknown'}"
            for key in MODEL_ORDER:
                model_predictions[key] = self._empty_prediction(key, reason=reason)
            result["prediction_reason"] = reason
            result["prediction_comparison"] = self._build_comparison(model_predictions)
            return result

        game_id = result.get("game_id")
        time_s = result.get("time_s")
        if game_id is None or time_s is None:
            reason = "missing game_id or time_s"
            for key in MODEL_ORDER:
                model_predictions[key] = self._empty_prediction(key, reason=reason)
            result["prediction_reason"] = reason
            result["prediction_comparison"] = self._build_comparison(model_predictions)
            return result

        try:
            game_id_int = int(game_id)
            minute_value = max(0, min(45, int(int(time_s) // 60)))
        except (TypeError, ValueError):
            reason = f"invalid game_id/time_s: game_id={game_id} time_s={time_s}"
            for key in MODEL_ORDER:
                model_predictions[key] = self._empty_prediction(key, reason=reason)
            result["prediction_reason"] = reason
            result["prediction_comparison"] = self._build_comparison(model_predictions)
            return result

        if self.current_game_id != game_id_int:
            self.reset()
            self.current_game_id = game_id_int

        minute_value = self._sanitize_minute_value(result, minute_value)

        missing_gold_fields = self._missing_gold_fields(result)
        if missing_gold_fields:
            reason = "missing lane gold fields: " + ", ".join(missing_gold_fields)
            for key in MODEL_ORDER:
                model_predictions[key] = self._empty_prediction(
                    key,
                    reason=reason,
                    prefix_minute=self._choose_sequence_prefix(minute_value) if key == "gru" else None,
                    current_minute=minute_value,
                )
            result["prediction_reason"] = reason
            result["prediction_current_minute"] = minute_value
            result["prediction_comparison"] = self._build_comparison(model_predictions)
            return result

        minute_snapshot = self._extract_minute_snapshot(result)
        if minute_snapshot is None:
            reason = "could not build lane-gold snapshot"
            for key in MODEL_ORDER:
                model_predictions[key] = self._empty_prediction(key, reason=reason)
            result["prediction_reason"] = reason
            result["prediction_current_minute"] = minute_value
            result["prediction_comparison"] = self._build_comparison(model_predictions)
            return result
        self.minute_history[minute_value] = minute_snapshot

        model_inputs = self._build_sequence_inputs(result)
        if model_inputs is None:
            reason = "could not build model inputs"
            for key in MODEL_ORDER:
                model_predictions[key] = self._empty_prediction(key, reason=reason)
            result["prediction_reason"] = reason
            result["prediction_current_minute"] = minute_value
            result["prediction_comparison"] = self._build_comparison(model_predictions)
            return result

        gru_prefix_minute = self._choose_sequence_prefix(minute_value)
        logistic_prefix_minute = self._choose_logistic_checkpoint(minute_value)
        xgboost_checkpoint_key = self._choose_xgboost_checkpoint(minute_value)
        mlp_prefix_minute = self._choose_mlp_checkpoint(minute_value)
        snapshot_numeric_map = self._extract_snapshot_numeric_feature_map(result, current_minute=minute_value)
        prefix_numeric_maps = {
            gru_prefix_minute: self._extract_prefix_numeric_feature_map(result, gru_prefix_minute),
            logistic_prefix_minute: self._extract_prefix_numeric_feature_map(result, logistic_prefix_minute),
            xgboost_checkpoint_key: self._extract_prefix_numeric_feature_map(result, xgboost_checkpoint_key),
            mlp_prefix_minute: self._extract_prefix_numeric_feature_map(result, mlp_prefix_minute),
        }
        logistic_tokens = self._extract_snapshot_tokens(
            result,
            include_champions=True,
            include_context=True,
            include_patch_token=False,
        )
        dense_tokens = self._extract_snapshot_tokens(
            result,
            include_champions=True,
            include_context=True,
            include_patch_token=True,
        )

        model_predictions["gru"] = self._predict_gru(model_inputs, gru_prefix_minute, minute_value)
        model_predictions["logistic_regression"] = self._predict_logistic(
            snapshot_numeric_map,
            prefix_numeric_maps[logistic_prefix_minute],
            logistic_tokens,
            logistic_prefix_minute,
            minute_value,
        )
        model_predictions["xgboost"] = self._predict_xgboost(
            snapshot_numeric_map,
            prefix_numeric_maps[xgboost_checkpoint_key],
            dense_tokens,
            xgboost_checkpoint_key,
            minute_value,
        )
        model_predictions["mlp"] = self._predict_mlp(
            snapshot_numeric_map,
            prefix_numeric_maps[mlp_prefix_minute],
            dense_tokens,
            mlp_prefix_minute,
            minute_value,
        )

        result["model_predictions"] = model_predictions
        result = self._apply_primary_prediction(result, model_predictions)
        comparison = self._build_comparison(model_predictions)
        result["prediction_comparison"] = comparison
        result["prediction_consensus_blue_win_prob"] = comparison["consensus_blue_win_prob"]
        result["prediction_consensus_red_win_prob"] = comparison["consensus_red_win_prob"]
        return result


LiveSequencePredictor = LiveModelSuitePredictor
