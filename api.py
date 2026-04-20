from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from demo_config import DEMO_SCENARIOS, get_risk_band
from model_contract import FEATURE_CONTRACT, MODEL_ARTIFACT_PATH, load_model, predict_delay


app = FastAPI(
    title="Supply Chain Delay Oracle API",
    version="0.1.0",
    description="Thin prediction API for the local Streamlit demo.",
)


class PredictionRequest(BaseModel):
    approval_delay: float = Field(ge=0)
    estimated_delivery_time: float = Field(gt=0)
    purchase_day_of_week: int = Field(ge=0, le=6)
    purchase_hour: int = Field(ge=0, le=23)
    total_items: float = Field(gt=0)
    total_price: float = Field(ge=0)
    total_freight_value: float = Field(ge=0)


class PredictionResponse(BaseModel):
    is_delayed: int
    predicted_class: str
    delay_probability: float
    on_time_probability: float
    risk_band: str
    recommended_action: str
    model_artifact: str


def get_model_status() -> tuple[bool, str | None]:
    try:
        load_model(MODEL_ARTIFACT_PATH)
    except Exception as exc:  # pragma: no cover - exercised through health endpoint
        return False, str(exc)
    return True, None


@app.get("/health")
def health() -> dict[str, Any]:
    model_ready, error = get_model_status()
    return {
        "status": "ok" if model_ready else "degraded",
        "model_ready": model_ready,
        "model_artifact": str(MODEL_ARTIFACT_PATH),
        "feature_contract": FEATURE_CONTRACT,
        "demo_scenarios": list(DEMO_SCENARIOS),
        "error": error,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    model_ready, error = get_model_status()
    if not model_ready:
        raise HTTPException(status_code=503, detail=error or "Model artifact unavailable.")

    result = predict_delay(payload.model_dump(), model_path=MODEL_ARTIFACT_PATH)
    risk_band, action = get_risk_band(float(result["delay_probability"]))
    predicted_class = "Delayed" if int(result["is_delayed"]) == 1 else "Not delayed"

    return PredictionResponse(
        is_delayed=int(result["is_delayed"]),
        predicted_class=predicted_class,
        delay_probability=float(result["delay_probability"]),
        on_time_probability=float(result["on_time_probability"]),
        risk_band=risk_band,
        recommended_action=action,
        model_artifact=MODEL_ARTIFACT_PATH.name,
    )
