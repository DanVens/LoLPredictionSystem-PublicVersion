#!/usr/bin/env python3
"""
Plot holdout graph-data exports into a small set of readable figures.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

EXPORT_ROOT = Path(__file__).resolve().parents[1]


MODEL_ORDER = ["gru", "logistic_regression", "xgboost", "mlp", "consensus"]
MODEL_LABELS = {
    "gru": "GRU",
    "logistic_regression": "LogReg",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "consensus": "Konsensusas",
}
MODEL_COLORS = {
    "gru": "#3a86ff",
    "logistic_regression": "#8ac926",
    "xgboost": "#ff006e",
    "mlp": "#8338ec",
    "consensus": "#ffd166",
}
METRICS = [
    ("accuracy", "Tikslumas", None),
    ("roc_auc", "ROC AUC", None),
    ("log_loss", "Logaritminis nuostolis", None),
    ("brier_score", "Brier kriterijus", None),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot holdout graph-data CSV exports.")
    parser.add_argument(
        "--graph-dir",
        default=str(EXPORT_ROOT / "artifacts/holdout_eval/graph_data"),
        help="Directory containing exported graph-data CSVs inside university_exports/artifacts/holdout_eval.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(EXPORT_ROOT / "artifacts/holdout_eval/plots"),
        help="Directory to save plots inside university_exports/artifacts/holdout_eval.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dynamic_limits(
    values: list[float],
    *,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
    min_span: float = 0.03,
    pad_ratio: float = 0.08,
) -> tuple[float, float]:
    ymin = min(values)
    ymax = max(values)
    span = ymax - ymin
    if span < min_span:
        center = (ymin + ymax) / 2.0
        ymin = center - min_span / 2.0
        ymax = center + min_span / 2.0
        span = ymax - ymin
    pad = span * pad_ratio
    lower = ymin - pad
    upper = ymax + pad
    if clamp_min is not None:
        lower = max(clamp_min, lower)
    if clamp_max is not None:
        upper = min(clamp_max, upper)
    if upper <= lower:
        if clamp_min is not None and clamp_max is not None:
            return clamp_min, clamp_max
        return ymin, ymax
    return lower, upper


def plot_summary_metrics(rows: list[dict[str, str]], output_dir: Path, plt) -> Path:
    by_metric: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        minute = int(row["minute"])
        model_key = row["model_key"]
        for metric_key, _, _ in METRICS:
            by_metric[metric_key][model_key].append((minute, float(row[metric_key])))

    fig, axes = plt.subplots(1, len(METRICS), figsize=(22, 5.8), sharex=True)
    if len(METRICS) == 1:
        axes = [axes]

    for ax, (metric_key, title, limits) in zip(axes, METRICS):
        for model_key in MODEL_ORDER:
            points = sorted(by_metric[metric_key].get(model_key, []))
            if not points:
                continue
            minutes = [minute for minute, _ in points]
            values = [value for _, value in points]
            ax.plot(
                minutes,
                values,
                marker="o",
                linewidth=2.2,
                markersize=6,
                color=MODEL_COLORS[model_key],
                label=MODEL_LABELS[model_key],
            )
        ax.set_title(title)
        ax.set_xlabel("Minutė")
        ax.set_xticks(sorted({int(row["minute"]) for row in rows}))
        ax.grid(True, alpha=0.25, linestyle="--")
        if limits is not None:
            ax.set_ylim(*limits)
        else:
            values = [float(row[metric_key]) for row in rows]
            clamp_max = 1.0 if metric_key in {"accuracy", "roc_auc"} else None
            clamp_min = None if metric_key in {"accuracy", "roc_auc"} else 0.0
            min_span = 0.03 if metric_key in {"accuracy", "roc_auc"} else 0.02
            ax.set_ylim(*dynamic_limits(values, clamp_min=clamp_min, clamp_max=clamp_max, min_span=min_span))

    axes[0].set_ylabel("Reikšmė")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(MODEL_ORDER), frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Modelių palyginimas pagal kontrolinius laiko taškus", y=1.04)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    output_path = output_dir / "holdout_summary_metrics.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return output_path


def plot_prediction_aggregates(rows: list[dict[str, str]], output_dir: Path, plt) -> Path:
    grouped: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        minute = int(row["minute"])
        model_key = row["model_key"]
        grouped["mean_blue_win_prob"][model_key].append((minute, float(row["mean_blue_win_prob"])))
        if row["mean_prob_when_blue_wins"] not in ("", "nan", "NaN"):
            grouped["mean_prob_when_blue_wins"][model_key].append((minute, float(row["mean_prob_when_blue_wins"])))
        if row["mean_prob_when_blue_loses"] not in ("", "nan", "NaN"):
            grouped["mean_prob_when_blue_loses"][model_key].append((minute, float(row["mean_prob_when_blue_loses"])))

    panels = [
        ("mean_blue_win_prob", "Vidutinė prognozuojama mėlynos pusės pergalės tikimybė"),
        ("mean_prob_when_blue_wins", "Vidutinė tikimybė, kai mėlyna pusė iš tikrųjų laimi"),
        ("mean_prob_when_blue_loses", "Vidutinė tikimybė, kai mėlyna pusė iš tikrųjų pralaimi"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 12), sharex=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (key, title) in zip(axes, panels):
        for model_key in MODEL_ORDER:
            points = sorted(grouped[key].get(model_key, []))
            if not points:
                continue
            minutes = [minute for minute, _ in points]
            values = [value for _, value in points]
            ax.plot(
                minutes,
                values,
                linewidth=2.0,
                color=MODEL_COLORS[model_key],
                label=MODEL_LABELS[model_key],
            )
        ax.set_title(title)
        ax.set_ylabel("Tikimybė")
        all_values: list[float] = []
        for model_points in grouped[key].values():
            all_values.extend(value for _, value in model_points)
        if all_values:
            ax.set_ylim(*dynamic_limits(all_values, clamp_min=0.0, clamp_max=1.0, min_span=0.08, pad_ratio=0.08))
        ax.axhline(0.5, color="#666666", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.grid(True, alpha=0.25, linestyle="--")

    axes[-1].set_xlabel("Minutė")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(MODEL_ORDER), frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_path = output_dir / "holdout_prediction_aggregates.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return output_path


def plot_model_checkpoint_metrics(
    rows: list[dict[str, str]],
    model_key: str,
    model_label: str,
    output_dir: Path,
    plt,
) -> Path:
    model_rows = [row for row in rows if row["model_key"] == model_key]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 4.4), sharex=True)
    if len(METRICS) == 1:
        axes = [axes]

    minutes = [int(row["minute"]) for row in model_rows]
    minute_label = ", ".join(str(minute) for minute in minutes)
    for ax, (metric_key, title, limits) in zip(axes, METRICS):
        values = [float(row[metric_key]) for row in model_rows]
        ax.plot(
            minutes,
            values,
            marker="o",
            linewidth=2.4,
            markersize=7,
            color=MODEL_COLORS[model_key],
        )
        ax.set_title(title)
        ax.set_xlabel("Minutė")
        ax.set_xticks(minutes)
        ax.grid(True, alpha=0.25, linestyle="--")
        if limits is not None:
            ax.set_ylim(*limits)
        else:
            clamp_max = 1.0 if metric_key in {"accuracy", "roc_auc"} else None
            clamp_min = None if metric_key in {"accuracy", "roc_auc"} else 0.0
            min_span = 0.03 if metric_key in {"accuracy", "roc_auc"} else 0.02
            ax.set_ylim(*dynamic_limits(values, clamp_min=clamp_min, clamp_max=clamp_max, min_span=min_span))

    axes[0].set_ylabel("Reikšmė")
    fig.suptitle(f"{model_label}: metrikos {minute_label} minutę", y=1.02)
    fig.tight_layout()
    output_path = output_dir / "01_kontroliniu_tasku_metrikos.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return output_path


def plot_model_probability_timeline(
    rows: list[dict[str, str]],
    model_key: str,
    model_label: str,
    output_dir: Path,
    plt,
) -> Path:
    model_rows = [row for row in rows if row["model_key"] == model_key]
    minutes = [int(row["minute"]) for row in model_rows]
    mean_prob = [float(row["mean_blue_win_prob"]) for row in model_rows]
    win_prob = []
    loss_prob = []
    for row in model_rows:
        win_raw = row["mean_prob_when_blue_wins"]
        loss_raw = row["mean_prob_when_blue_loses"]
        win_prob.append(None if win_raw in ("", "nan", "NaN") else float(win_raw))
        loss_prob.append(None if loss_raw in ("", "nan", "NaN") else float(loss_raw))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        minutes,
        mean_prob,
        linewidth=2.4,
        color=MODEL_COLORS[model_key],
        label="Visų būsenų vidurkis",
    )
    ax.plot(
        minutes,
        win_prob,
        linewidth=2.0,
        linestyle="--",
        color="#2a9d8f",
        label="Kai mėlyna pusė laimi",
    )
    ax.plot(
        minutes,
        loss_prob,
        linewidth=2.0,
        linestyle=":",
        color="#e76f51",
        label="Kai mėlyna pusė pralaimi",
    )
    ax.set_title(f"{model_label}: prognozuojamos tikimybės eiga per laiką")
    ax.set_xlabel("Minutė")
    ax.set_ylabel("Tikimybė")
    visible_values = mean_prob + [value for value in win_prob if value is not None] + [value for value in loss_prob if value is not None]
    ax.set_ylim(*dynamic_limits(visible_values, clamp_min=0.0, clamp_max=1.0, min_span=0.08, pad_ratio=0.08))
    ax.axhline(0.5, color="#666666", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    output_path = output_dir / "02_tikimybes_eiga.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return output_path


def main() -> int:
    args = parse_args()
    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Install it with: ./.venv/bin/pip install matplotlib")
        return 1

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 22,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "figure.titlesize": 24,
        }
    )

    summary_rows = read_csv_rows(graph_dir / "summary_metrics.csv")
    aggregate_rows = read_csv_rows(graph_dir / "prediction_aggregates_by_minute.csv")

    overview_dir = output_dir / "overview"
    ensure_output_dir(overview_dir)
    saved_paths = [plot_summary_metrics(summary_rows, overview_dir, plt)]
    with plt.rc_context(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.titlesize": 15,
        }
    ):
        saved_paths.append(plot_prediction_aggregates(aggregate_rows, overview_dir, plt))

    for model_key in MODEL_ORDER:
        model_dir = output_dir / model_key
        ensure_output_dir(model_dir)
        model_label = MODEL_LABELS[model_key]
        saved_paths.append(
            plot_model_checkpoint_metrics(summary_rows, model_key, model_label, model_dir, plt)
        )
        saved_paths.append(
            plot_model_probability_timeline(aggregate_rows, model_key, model_label, model_dir, plt)
        )

    for path in saved_paths:
        print(f"saved plot: {path}")

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
