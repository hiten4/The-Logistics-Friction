from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping
import zipfile

import joblib
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
PROCESSED_DATASET_PATH = REPO_ROOT / "data" / "processed" / "final_poc_dataset.zip"
MODEL_ARTIFACT_PATH = REPO_ROOT / "artifacts" / "baseline_logreg_pipeline.joblib"

FEATURE_NAMES = [
    "approval_delay",
    "estimated_delivery_time",
    "purchase_day_of_week",
    "purchase_hour",
    "total_items",
    "total_price",
    "total_freight_value",
]

FEATURE_CONTRACT = {
    "approval_delay": "Days between purchase timestamp and order approval.",
    "estimated_delivery_time": "Days between purchase timestamp and estimated delivery.",
    "purchase_day_of_week": "Purchase weekday encoded as 0=Monday through 6=Sunday.",
    "purchase_hour": "Purchase hour in 24-hour local time, from 0 to 23.",
    "total_items": "Total number of items in the order.",
    "total_price": "Total order item price before freight.",
    "total_freight_value": "Total freight amount for the order.",
}

TARGET_COLUMN = "is_delayed"


def load_processed_dataset(dataset_path: Path | None = None) -> pd.DataFrame:
    zip_path = Path(dataset_path or PROCESSED_DATASET_PATH)
    if not zip_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {zip_path}.")
    with zipfile.ZipFile(zip_path) as zipped_dataset:
        csv_members = [name for name in zipped_dataset.namelist() if name.endswith(".csv")]
        if len(csv_members) != 1:
            raise ValueError(
                f"Expected exactly one CSV inside {zip_path}, found {len(csv_members)}."
            )
        with zipped_dataset.open(csv_members[0]) as csv_file:
            return pd.read_csv(csv_file)


def build_feature_frame(features: Mapping[str, object]) -> pd.DataFrame:
    missing = [name for name in FEATURE_NAMES if name not in features]
    extra = sorted(set(features) - set(FEATURE_NAMES))

    if missing or extra:
        problems = []
        if missing:
            problems.append(f"missing={missing}")
        if extra:
            problems.append(f"extra={extra}")
        raise ValueError("Invalid feature payload: " + ", ".join(problems))

    ordered_row = {name: float(features[name]) for name in FEATURE_NAMES}
    return pd.DataFrame([ordered_row], columns=FEATURE_NAMES)


@lru_cache(maxsize=4)
def _load_model_cached(artifact_path: str):
    return joblib.load(artifact_path)


def load_model(model_path: Path | None = None):
    artifact_path = Path(model_path or MODEL_ARTIFACT_PATH)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {artifact_path}. Run train_baseline_model.py first."
        )
    return _load_model_cached(str(artifact_path.resolve()))


def predict_delay(features: Mapping[str, object], model_path: Path | None = None) -> dict[str, float | int]:
    model = load_model(model_path=model_path)
    feature_frame = build_feature_frame(features)
    delay_probability = float(model.predict_proba(feature_frame)[0, 1])
    predicted_class = int(model.predict(feature_frame)[0])
    return {
        "is_delayed": predicted_class,
        "delay_probability": delay_probability,
        "on_time_probability": 1.0 - delay_probability,
    }
