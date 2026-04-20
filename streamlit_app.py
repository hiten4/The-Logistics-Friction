from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

from demo_config import (
    APP_SUBTITLE,
    APP_TITLE,
    BACKEND_DEFAULT_URL,
    DAY_LABELS,
    DEMO_SCENARIOS,
)
from model_contract import FEATURE_CONTRACT


API_TIMEOUT_SECONDS = 5.0
BACKEND_URL = os.getenv("LOGISTICS_BACKEND_URL", BACKEND_DEFAULT_URL).rstrip("/")


def apply_scenario(name: str) -> None:
    for feature_name, value in DEMO_SCENARIOS[name].items():
        st.session_state[feature_name] = value


def initialize_state() -> None:
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "prediction_error" not in st.session_state:
        st.session_state.prediction_error = None
    if "approval_delay" not in st.session_state:
        apply_scenario("Routine order")


def render_prediction(result: dict[str, Any]) -> None:
    st.subheader("Prediction")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted class", str(result["predicted_class"]))
    col2.metric("Delay probability", f"{float(result['delay_probability']):.1%}")
    col3.metric("Risk band", str(result["risk_band"]))
    st.info(f"Recommended action: {result['recommended_action']}")
    st.caption(f"Served by backend artifact: {result['model_artifact']}")


@st.cache_data(ttl=10, show_spinner=False)
def get_backend_health(backend_url: str) -> dict[str, Any]:
    with httpx.Client(timeout=API_TIMEOUT_SECONDS) as client:
        response = client.get(f"{backend_url}/health")
        response.raise_for_status()
        return response.json()


def request_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    with httpx.Client(timeout=API_TIMEOUT_SECONDS) as client:
        response = client.post(f"{BACKEND_URL}/predict", json=payload)
        if response.is_error:
            detail = response.text
            try:
                detail = response.json().get("detail", detail)
            except ValueError:
                pass
            raise ValueError(f"Backend prediction failed: {detail}")
        return response.json()


st.set_page_config(page_title=APP_TITLE, page_icon="📦", layout="wide")
initialize_state()

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7f5ef 0%, #f2efe6 100%);
        color: #14213d;
    }
    .block-container {
        max-width: 1100px;
        padding-top: 2.2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(20, 33, 61, 0.08);
        border-radius: 18px;
        padding: 0.8rem 1rem;
        box-shadow: 0 10px 30px rgba(20, 33, 61, 0.05);
    }
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(20, 33, 61, 0.08);
        border-radius: 24px;
        padding: 1.1rem 1.15rem 0.6rem 1.15rem;
        box-shadow: 0 16px 40px rgba(20, 33, 61, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)
st.caption(f"Frontend target: `{BACKEND_URL}`")

backend_health = None
backend_error = None
try:
    backend_health = get_backend_health(BACKEND_URL)
except Exception as exc:
    backend_error = str(exc)

backend_ready = bool(backend_health and backend_health.get("model_ready"))

if backend_error:
    st.error(
        f"Backend unavailable at {BACKEND_URL}. "
        "Start it with `python3 -m uvicorn api:app --host 127.0.0.1 --port 8000`."
    )
    st.caption(backend_error)
elif not backend_ready:
    st.error(
        "Backend is running but the model artifact is unavailable. "
        "Train it first with `python3 train_baseline_model.py`."
    )
    if backend_health and backend_health.get("error"):
        st.caption(str(backend_health["error"]))
else:
    st.success("Backend API is healthy and ready for predictions.")

st.subheader("Demo scenarios")
scenario_columns = st.columns(len(DEMO_SCENARIOS))
for column, scenario_name in zip(scenario_columns, DEMO_SCENARIOS):
    if column.button(scenario_name, use_container_width=True):
        apply_scenario(scenario_name)
        st.session_state.prediction_result = None
        st.session_state.prediction_error = None

with st.form("delay_demo_form"):
    left, right = st.columns(2)
    with left:
        st.number_input(
            "Approval delay (days)",
            min_value=0.0,
            max_value=30.0,
            step=1.0,
            key="approval_delay",
            help=FEATURE_CONTRACT["approval_delay"],
        )
        st.number_input(
            "Estimated delivery time (days)",
            min_value=1.0,
            max_value=60.0,
            step=1.0,
            key="estimated_delivery_time",
            help=FEATURE_CONTRACT["estimated_delivery_time"],
        )
        st.selectbox(
            "Purchase day of week",
            options=list(DAY_LABELS),
            format_func=lambda value: DAY_LABELS[value],
            key="purchase_day_of_week",
            help=FEATURE_CONTRACT["purchase_day_of_week"],
        )
        st.slider(
            "Purchase hour",
            min_value=0,
            max_value=23,
            step=1,
            key="purchase_hour",
            help=FEATURE_CONTRACT["purchase_hour"],
        )
    with right:
        st.number_input(
            "Total items",
            min_value=1.0,
            max_value=25.0,
            step=1.0,
            key="total_items",
            help=FEATURE_CONTRACT["total_items"],
        )
        st.number_input(
            "Total price",
            min_value=0.0,
            max_value=5000.0,
            step=10.0,
            key="total_price",
            help=FEATURE_CONTRACT["total_price"],
        )
        st.number_input(
            "Total freight value",
            min_value=0.0,
            max_value=500.0,
            step=1.0,
            key="total_freight_value",
            help=FEATURE_CONTRACT["total_freight_value"],
        )
        st.markdown("")
        st.markdown("")
        submitted = st.form_submit_button(
            "Predict",
            use_container_width=True,
            type="primary",
            disabled=not backend_ready,
        )

if submitted and backend_ready:
    try:
        payload = {
            "approval_delay": float(st.session_state.approval_delay),
            "estimated_delivery_time": float(st.session_state.estimated_delivery_time),
            "purchase_day_of_week": int(st.session_state.purchase_day_of_week),
            "purchase_hour": int(st.session_state.purchase_hour),
            "total_items": float(st.session_state.total_items),
            "total_price": float(st.session_state.total_price),
            "total_freight_value": float(st.session_state.total_freight_value),
        }
        st.session_state.prediction_result = request_prediction(payload)
        st.session_state.prediction_error = None
    except ValueError as exc:
        st.session_state.prediction_result = None
        st.session_state.prediction_error = str(exc)
    except Exception as exc:  # pragma: no cover - presentation fallback
        st.session_state.prediction_result = None
        st.session_state.prediction_error = f"Unexpected prediction error: {exc}"

if st.session_state.prediction_error:
    st.error(st.session_state.prediction_error)
elif st.session_state.prediction_result:
    render_prediction(st.session_state.prediction_result)
