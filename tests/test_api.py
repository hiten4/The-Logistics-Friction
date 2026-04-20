from __future__ import annotations

from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def test_health_endpoint_reports_runtime_state() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert "model_ready" in payload
    assert "feature_contract" in payload
    assert "demo_scenarios" in payload


def test_predict_endpoint_returns_prediction_payload() -> None:
    response = client.post(
        "/predict",
        json={
            "approval_delay": 3,
            "estimated_delivery_time": 7,
            "purchase_day_of_week": 5,
            "purchase_hour": 20,
            "total_items": 1,
            "total_price": 30,
            "total_freight_value": 45,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_class"] in {"Delayed", "Not delayed"}
    assert payload["risk_band"] in {"Low", "Medium", "High"}
    assert 0.0 <= payload["delay_probability"] <= 1.0
    assert 0.0 <= payload["on_time_probability"] <= 1.0


def test_predict_endpoint_rejects_invalid_hour() -> None:
    response = client.post(
        "/predict",
        json={
            "approval_delay": 1,
            "estimated_delivery_time": 10,
            "purchase_day_of_week": 2,
            "purchase_hour": 30,
            "total_items": 2,
            "total_price": 90,
            "total_freight_value": 18,
        },
    )

    assert response.status_code == 422
