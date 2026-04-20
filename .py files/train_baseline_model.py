from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_contract import (
    FEATURE_NAMES,
    MODEL_ARTIFACT_PATH,
    TARGET_COLUMN,
    load_processed_dataset,
)


def train_and_save_model(artifact_path: Path | None = None):
    dataset = load_processed_dataset()
    x = dataset[FEATURE_NAMES]
    y = dataset[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    destination = Path(artifact_path or MODEL_ARTIFACT_PATH)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)

    print(f"Rows    : {dataset.shape[0]:,}")
    print(f"Delay % : {dataset[TARGET_COLUMN].mean() * 100:.1f}%")
    print(f"Train   : {len(x_train):,}  |  Test : {len(x_test):,}")
    print(f"Saved   : {destination}")
    print(classification_report(y_test, y_pred, target_names=["On-Time", "Delayed"]))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"PR-AUC  : {average_precision_score(y_test, y_prob):.4f}  <- main metric")

    return model


if __name__ == "__main__":
    train_and_save_model()
