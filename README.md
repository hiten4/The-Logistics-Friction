# The-Logistics-Friction
## Supply Chain Delay Oracle

Predicting delivery delays in e-commerce logistics to improve customer trust and operational efficiency.

---

## Problem Statement

In large-scale e-commerce systems, inaccurate delivery estimates lead to:

- Customer dissatisfaction
- Increased support costs
- Reduced customer lifetime value

This project aims to build a predictive system that **identifies orders likely to be delayed before shipment**, enabling proactive interventions.

---

## Objective

- Predict whether an order will be delayed
- Provide actionable insights for logistics and operations teams
- Improve delivery estimation accuracy

---

## Business Impact

- Proactive customer communication
- Better logistics prioritization
- Reduced negative reviews and churn
- Improved trust in delivery timelines

---

## Current Repository State

The repository now contains both modeling work and a small local demo stack:

- Raw and processed logistics datasets under `data/`
- Experiment scripts for multiple models in `.py files/`
- A reproducible logistic-regression demo trainer at `train_baseline_model.py`
- A saved sklearn pipeline artifact at `artifacts/baseline_logreg_pipeline.joblib`
- A FastAPI backend at `api.py`
- A Streamlit frontend at `streamlit_app.py`

For clarity:

- The live demo stack uses the saved logistic regression artifact for reproducible local predictions
- Checkpoint 2 modeling work for Random Forest, XGBoost, and LightGBM also exists in the repo as experimentation code
- There is still no deployed production service in this repository

---

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

---

## Train And Save The Model

The baseline model is trained from the processed dataset already stored in the repository.

Run:

```bash
python3 train_baseline_model.py
```

This will:

- Load `data/processed/final_poc_dataset.zip`
- Train the existing 7-feature logistic regression pipeline
- Save the trained artifact to `artifacts/baseline_logreg_pipeline.joblib`

The Streamlit demo uses the saved artifact only. It does not retrain the model on startup.

---

## Run The Backend API

```bash
python3 -m uvicorn api:app --host 127.0.0.1 --port 8000
```

The backend exposes:

- `GET /health` for local runtime status
- `POST /predict` for live inference against the saved logistic regression artifact

## Run The Streamlit Frontend

In a second terminal:

```bash
python3 -m streamlit run streamlit_app.py
```

The app provides:

- 7 baseline feature inputs
- One-click canned demo scenarios
- A prediction button
- Backend health feedback
- Predicted class
- Delay probability
- Risk band
- Recommended business action

If the backend is running somewhere else, point the frontend at it with:

```bash
LOGISTICS_BACKEND_URL=http://127.0.0.1:8000 python3 -m streamlit run streamlit_app.py
```

---

## Model Input Contract

The current baseline model expects exactly these 7 numeric inputs:

### 1. `approval_delay`
Days between purchase timestamp and order approval.

### 2. `estimated_delivery_time`
Days between purchase timestamp and estimated delivery date.

### 3. `purchase_day_of_week`
Weekday encoded as:

- `0` = Monday
- `1` = Tuesday
- `2` = Wednesday
- `3` = Thursday
- `4` = Friday
- `5` = Saturday
- `6` = Sunday

### 4. `purchase_hour`
Hour of purchase in 24-hour time from `0` to `23`.

### 5. `total_items`
Total number of items in the order.

### 6. `total_price`
Total item price before freight.

### 7. `total_freight_value`
Total freight amount for the order.

---

## Prediction Output

The current app returns:

- `Predicted class`
  - `Delayed` means the model predicts the order is likely to miss the estimated delivery date.
  - `Not delayed` means the model predicts the order is likely to arrive on time.
- `Delay probability`
  - The model's estimated probability that the order will be delayed.
- `Risk band`
  - A presentation-friendly grouping of the delay probability:
    - `Low`
    - `Medium`
    - `High`

These outputs are intended for decision support in a checkpoint demo, not as a production SLA or customer-facing promise.

---

## Business Action Guidance

The app maps prediction risk into simple demo actions:

- `Low`
  - Standard fulfilment and routine customer messaging
- `Medium`
  - Monitor closely and prepare a proactive customer update
- `High`
  - Escalate to operations, prioritize intervention, and contact the customer early

This action guidance is intentionally lightweight and designed for presentation use.

---

## Dataset Context

The original business problem is based on multiple logistics tables such as:

- `orders.csv`
- `order_items.csv`
- `customers.csv`
- `sellers.csv`

The current checked-in implementation operates from the processed modeling dataset generated from that broader source structure.

### Key Challenges

- Multi-table joins
- Temporal dependencies
- Geographic variability
- Missing and inconsistent data

---
---

## 🤖 Modelling & Evaluation

### Verified Demo Baseline

The only metrics re-verified in this audit are for the live demo path:

Command run:

```bash
python3 train_baseline_model.py
```

Observed output on this machine:

| Live demo artifact | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | 0.5780 | 0.1023 |

The frontend and backend both use this saved logistic regression artifact. Those numbers are suitable for repo-truth reporting of the current local demo stack.

### Additional Experiment Scripts

The repository also contains Random Forest, XGBoost, and LightGBM experiment scripts under `.py files/`.

Those experiment paths were not re-executed in this audit, so this README does not present their metrics as freshly verified current truth.

---

### ⚙️ How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Update the data path in `settings.py` if you are running the older experiment scripts**
```python
DATA_PATH = r"path\to\your\final_dataset.csv"
```

**3. Run individual models**
```bash
python".py files/logistic_regression_baseline.py"
python ".py files/model_2_random_forest.py"
python ".py files/model_3_xgboost.py"
python ".py files/model_4_lightgbm.py"
```

**4. Run full comparison + threshold tuning**
```bash
python ".py files/model_comparison.py"
```

---

## Repository Notes

- The baseline training script, backend API, and frontend all use the same 7-feature contract.
- The saved artifact path is `artifacts/baseline_logreg_pipeline.joblib`.
- The frontend does not silently fall back to local inference; it reports backend health explicitly.
- Design notes and architecture documents remain in `docs/`.
