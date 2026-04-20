from __future__ import annotations


APP_TITLE = "Supply Chain Delay Oracle"
APP_SUBTITLE = "Checkpoint demo wired to a local prediction API."
BACKEND_DEFAULT_URL = "http://127.0.0.1:8000"

RISK_BANDS = (
    (0.40, "Low", "Standard fulfilment and routine customer messaging."),
    (0.65, "Medium", "Monitor closely and prepare a proactive customer update."),
    (1.01, "High", "Escalate to operations, prioritize intervention, and contact the customer early."),
)

DAY_LABELS = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

DEMO_SCENARIOS = {
    "Routine order": {
        "approval_delay": 0.0,
        "estimated_delivery_time": 30.0,
        "purchase_day_of_week": 1,
        "purchase_hour": 8,
        "total_items": 4.0,
        "total_price": 220.0,
        "total_freight_value": 5.0,
    },
    "Borderline watchlist": {
        "approval_delay": 1.0,
        "estimated_delivery_time": 18.0,
        "purchase_day_of_week": 2,
        "purchase_hour": 12,
        "total_items": 2.0,
        "total_price": 90.0,
        "total_freight_value": 18.0,
    },
    "Approval bottleneck": {
        "approval_delay": 1.0,
        "estimated_delivery_time": 14.0,
        "purchase_day_of_week": 4,
        "purchase_hour": 15,
        "total_items": 1.0,
        "total_price": 65.0,
        "total_freight_value": 25.0,
    },
    "Weekend escalation": {
        "approval_delay": 3.0,
        "estimated_delivery_time": 7.0,
        "purchase_day_of_week": 5,
        "purchase_hour": 20,
        "total_items": 1.0,
        "total_price": 30.0,
        "total_freight_value": 45.0,
    },
    "Severe friction": {
        "approval_delay": 5.0,
        "estimated_delivery_time": 5.0,
        "purchase_day_of_week": 6,
        "purchase_hour": 22,
        "total_items": 1.0,
        "total_price": 20.0,
        "total_freight_value": 60.0,
    },
}


def get_risk_band(delay_probability: float) -> tuple[str, str]:
    for threshold, label, action in RISK_BANDS:
        if delay_probability < threshold:
            return label, action
    return "High", RISK_BANDS[-1][2]
