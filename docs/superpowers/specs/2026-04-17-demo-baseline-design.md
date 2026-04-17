# Demo Baseline Design

## Goal

Make the repository demoable with the smallest possible code change set by
training the existing 7-feature logistic regression baseline from the processed
dataset in the repo, saving the trained pipeline as a reusable artifact, and
providing a tiny inference helper for frontend use.

## Constraints

- Reuse the existing sklearn logistic regression baseline only.
- Do not redesign the ML approach or feature set.
- Keep repo structure mostly unchanged.
- Prefer additive changes over refactors.

## Inputs

The demo model contract is limited to these seven numeric inputs:

1. `approval_delay`
2. `estimated_delivery_time`
3. `purchase_day_of_week`
4. `purchase_hour`
5. `total_items`
6. `total_price`
7. `total_freight_value`

## Dataset Source

Training reads the processed dataset already committed to the repo:

- `data/processed/final_poc_dataset.zip`

The zip contains `final_poc_dataset.csv`, which already includes the target
column `is_delayed`.

## Minimal Components

### `requirements.txt`

Declares the runtime packages needed to train the baseline and load the saved
artifact:

- `pandas`
- `scikit-learn`
- `joblib`
- `numpy`

### `model_contract.py`

Shared, small reusable module that:

- defines the ordered feature contract
- loads the processed dataset from the zip in the repo
- loads the saved model artifact
- exposes `predict_delay(features)`

### `train_baseline_model.py`

Standalone reproducible training entry point that:

- loads the processed dataset from the repo
- selects the existing seven baseline features
- trains the same `StandardScaler -> LogisticRegression` pipeline
- saves the trained pipeline artifact

## Artifact

Committed model artifact path:

- `artifacts/baseline_logreg_pipeline.joblib`

## Non-Goals

- No frontend implementation yet.
- No API/server layer.
- No new model architecture or tuning work.
- No migration of notebook-style scripts into a package.

