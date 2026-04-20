from __future__ import annotations

import pytest

from model_contract import (
    FEATURE_NAMES,
    MODEL_ARTIFACT_PATH,
    build_feature_frame,
    load_processed_dataset,
    predict_delay,
)


def test_load_processed_dataset_contains_expected_columns() -> None:
    dataset = load_processed_dataset()

    assert set(FEATURE_NAMES).issubset(dataset.columns)
    assert "is_delayed" in dataset.columns


def test_build_feature_frame_rejects_missing_fields() -> None:
    with pytest.raises(ValueError, match="missing"):
        build_feature_frame({"approval_delay": 1})


def test_predict_delay_returns_probability_payload() -> None:
    result = predict_delay(
        {
            "approval_delay": 0,
            "estimated_delivery_time": 30,
            "purchase_day_of_week": 1,
            "purchase_hour": 8,
            "total_items": 4,
            "total_price": 220,
            "total_freight_value": 5,
        },
        model_path=MODEL_ARTIFACT_PATH,
    )

    assert result["is_delayed"] in {0, 1}
    assert 0.0 <= float(result["delay_probability"]) <= 1.0
    assert 0.0 <= float(result["on_time_probability"]) <= 1.0
