# `university_exports` usage instructions

> This repository contains the prediction/ML research components of a
> larger system, including 4 prediction models (GRU, Logistic Regression,
> XGBoost, MLP).

## Contents

This folder contains:

- model artifacts,
- training scripts,
- holdout evaluation scripts,
- a web interface.

Main model artifacts:

- `GRU`: `artifacts/sequence_gru_team_context_mixedlength_ls005/gold_champions_context/prefix_45`
- `Logistic Regression`: `artifacts/logistic_regression_no_prefix_std1/gold_champions_context/all_minutes`
- `XGBoost`: `artifacts/xgboost_no_prefix_fullgame/gold_champions_context/all_minutes`
- `MLP`: `artifacts/mlp_snapshot_embedding_lr1e4_trial/gold_champions_context/best_model.pt`

## Environment

Commands are run from the `university_exports` folder:

```bash
cd university_exports
```

```bash
python3 -m venv .venv
source .venv/bin/activate
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m playwright install chromium
```

## Dependencies

Python packages are listed in:

- `requirements.txt`

## Folders

```text
university_exports/
  README.md
  requirements.txt
  data/
  artifacts/
  shared/
  evaluation/
  live_app/
  profiles/
```

- `data/` – training and testing data
- `artifacts/` – already-trained models
- `shared/` – training scripts
- `evaluation/` – holdout evaluation and plotting scripts
- `live_app/` – web interface and live inference code

The `data/` folder contains tables already prepared for model training and
testing. Raw match CSV files are not included in this package, since the
data volume is large.

## Model training

### 1. GRU

```bash
./.venv/bin/python shared/train_sequence_gru.py \
  --input-prefix data/sequence_dataset \
  --output-dir artifacts/sequence_gru_team_context_mixedlength_ls005 \
  --epochs 10 \
  --batch-size 64 \
  --hidden-size 96 \
  --embedding-dim 16 \
  --learning-rate 0.0005 \
  --weight-decay 0.0005 \
  --dropout 0.3 \
  --train-fraction 0.8 \
  --seed 42 \
  --prefix-minutes 45 \
  --variants gold_champions_context \
  --mixed-prefix-training \
  --mixed-prefix-source 5,10,15,20,25,30,35,40,45 \
  --label-smoothing 0.05
```

### 2. XGBoost

```bash
./.venv/bin/python shared/train_xgboost_no_prefix.py \
  --input data/training_table_all.csv \
  --output-dir artifacts/xgboost_no_prefix_fullgame \
  --variants gold_champions_context \
  --num-round 700 \
  --eta 0.03 \
  --max-depth 4 \
  --min-child-weight 3.0 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --lambda-l2 1.0 \
  --alpha-l1 0.0 \
  --early-stopping-rounds 50 \
  --train-fraction 0.8 \
  --seed 42
```

### 3. MLP

```bash
./.venv/bin/python shared/train_mlp.py \
  --input data/training_table_all.csv \
  --mode snapshot \
  --variants gold_champions_context \
  --output-dir artifacts/mlp_snapshot_embedding_lr1e4_trial \
  --hidden-dims 128,64 \
  --categorical-encoding embedding \
  --categorical-embedding-dim 16 \
  --snapshot-train-sampling all \
  --dropout 0.3 \
  --epochs 100 \
  --batch-size 128 \
  --learning-rate 0.0001 \
  --weight-decay 0.001 \
  --early-stopping-patience 10 \
  --train-fraction 0.8 \
  --seed 42
```

### 4. Logistic Regression

```bash
./.venv/bin/python shared/train_logistic_regression_no_prefix.py \
  --input data/training_table_all.csv \
  --output-dir artifacts/logistic_regression_no_prefix_std1 \
  --variants gold_champions_context \
  --hash-dim 262144 \
  --epochs 20 \
  --learning-rate 0.001 \
  --l2 1e-5 \
  --train-fraction 0.8 \
  --seed 42
```

## Holdout evaluation

```bash
./.venv/bin/python evaluation/evaluate_holdout_models.py \
  "data/testing_data/*.csv" \
  --table-output data/testing_holdout_table_all.csv \
  --predictions-output artifacts/holdout_eval/predictions_all_minutes.csv \
  --summary-output artifacts/holdout_eval/summary.json \
  --checkpoints 5,10,15,20,25
```

Result files:

- `data/testing_holdout_table_all.csv`
- `artifacts/holdout_eval/predictions_all_minutes.csv`
- `artifacts/holdout_eval/summary.json`

## Plot generation

```bash
./.venv/bin/python evaluation/export_holdout_graph_data.py \
  --summary-json artifacts/holdout_eval/summary.json \
  --predictions-csv artifacts/holdout_eval/predictions_all_minutes.csv \
  --output-dir artifacts/holdout_eval/graph_data

./.venv/bin/python evaluation/plot_holdout_graph_data.py \
  --graph-dir artifacts/holdout_eval/graph_data \
  --output-dir artifacts/holdout_eval/plots
```

Result folders:

- `artifacts/holdout_eval/plots/overview/`
- `artifacts/holdout_eval/plots/gru/`
- `artifacts/holdout_eval/plots/logistic_regression/`
- `artifacts/holdout_eval/plots/xgboost/`
- `artifacts/holdout_eval/plots/mlp/`
- `artifacts/holdout_eval/plots/consensus/`

## Web interface

Before running in API mode, `LOLESPORTS_API_KEY` must be set.
You can use a shell environment variable or a `.env` file in the
`university_exports/` folder.

```bash
./.venv/bin/python live_app/web_app.py \
  --source api \
  --host 127.0.0.1 \
  --port 5051 \
  --interval 5
```

Parameters:

- `--source api` – data source
- `--host 127.0.0.1` – address at which the web interface is reachable on this machine
- `--port 5051` – port on which the web interface runs
- `--interval 5` – data refresh interval, in seconds

Address in browser:

```text
http://127.0.0.1:5051
```
