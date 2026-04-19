
DATA_PATH = r"final_dataset.csv"

TARGET = "is_delayed"

FEATURES = [
    "approval_delay",
    "estimated_delivery_time",
    "purchase_day_of_week",
    "purchase_hour",
    "total_items",
    "total_price",
    "total_freight_value",
    "distance_km",
    "is_same_city",
    "is_same_state",
    "product_volume_cm3",
    "product_weight_grams",
]

STATE_COLS    = None   # auto-detected from dataset
CATEGORY_COLS = None   # auto-detected from dataset