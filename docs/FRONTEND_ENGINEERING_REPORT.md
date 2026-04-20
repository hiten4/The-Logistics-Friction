# Frontend Engineering Report

## 1. Repository State Before My Changes

When I started, the repository was a machine learning proof-of-concept rather than a demoable application. The codebase contained:

- A project README describing the business problem and intent
- Architecture notes in `docs/HLD.md` and `docs/LLD.md`
- Three notebook-style Python scripts under `.py files/`
- A processed dataset at `data/processed/final_poc_dataset.zip`

What it did not contain was equally important:

- No saved trained model artifact
- No reproducible training entry point
- No reusable inference helper
- No frontend
- No API
- No dependency manifest

In short, the repository had the baseline ingredients for modeling, but it was not yet in a state that could be demonstrated live.

## 2. Problems That Prevented a Demo

Several concrete issues blocked a checkpoint presentation:

- The baseline model was only trained in-script and never persisted for reuse
- There was no stable inference entry point for a frontend to call
- The repository had no `requirements.txt`, so setup was not reproducible
- The existing scripts were oriented toward experimentation, not demo execution
- There was no UI for entering inputs, loading scenarios, or showing outputs
- The README did not reflect the real runnable workflow

Any live demo would have required ad hoc setup and manual code editing, which was too risky for a presentation.

## 3. What I Implemented

My contribution focused on making the repository demoable with the smallest practical change set, while keeping the original project intent and the existing logistic regression baseline intact.

I implemented:

- A reproducible baseline training script that reads the committed processed dataset
- A saved sklearn model artifact for demo reuse
- A small shared inference and feature-contract module
- A thin FastAPI backend for local prediction requests
- A thin Streamlit frontend for checkpoint presentation use
- One-click demo scenarios to make the presentation reliable
- Lightweight tests for the feature contract and API path
- A README update aligned to the actual implemented workflow
- A startup safety check so the frontend verifies that the backend and saved model artifact are both available before allowing predictions

I did not redesign the model, introduce a broad backend platform, or add fake production claims.

## 4. Files Created And Modified

### Created

- `docs/superpowers/specs/2026-04-17-demo-baseline-design.md`
- `requirements.txt`
- `model_contract.py`
- `train_baseline_model.py`
- `artifacts/baseline_logreg_pipeline.joblib`
- `api.py`
- `streamlit_app.py`
- `tests/test_api.py`
- `tests/test_model_contract.py`

### Modified

- `README.md`
- `streamlit_app.py` after initial review, to validate model loading at startup

### This Report

- `docs/FRONTEND_ENGINEERING_REPORT.md`

## 5. Why Streamlit Was Chosen

Streamlit was chosen because it was the fastest safe path to a presentation-ready frontend by the checkpoint deadline.

It fit the problem well for four reasons:

- The model already ran in Python, so the frontend could stay thin
- The UI needed only a small number of numeric controls and clear outputs
- Demo scenarios could be added quickly and safely
- The implementation cost was low enough to keep focus on reliability rather than framework overhead

For the current repo state, a thin API was then added behind the frontend so the browser flow exercises a real backend instead of bypassing it.

## 6. How The Frontend Connects To The Trained Model

The frontend does not retrain the model or load the sklearn artifact directly.

Instead, it uses the saved artifact at:

- `artifacts/baseline_logreg_pipeline.joblib`

The integration path is:

1. `streamlit_app.py` collects the 7 baseline input features
2. The app calls `POST /predict` on `api.py`
3. `api.py` validates the request payload and checks backend/model readiness
4. `model_contract.py` loads the saved sklearn pipeline artifact with caching
5. The helper builds a one-row dataframe in the exact expected feature order
6. The backend returns predicted class, delay probability, risk band, and recommended action
7. The frontend renders the response and surfaces backend health honestly

This keeps the frontend thin while still exercising a real backend boundary.

## 7. Demo Scenarios Added

To make the live presentation more reliable, I added five one-click demo scenarios:

- `Routine order`
- `Borderline watchlist`
- `Approval bottleneck`
- `Weekend escalation`
- `Severe friction`

These scenarios were chosen to produce visibly different outputs, including low-, medium-, and high-risk style examples for presentation flow.

## 8. How To Run The Demo

### Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### Train and save the baseline artifact

```bash
python3 train_baseline_model.py
```

### Run the backend

```bash
python3 -m uvicorn api:app --host 127.0.0.1 --port 8000
```

### Run the frontend

```bash
python3 -m streamlit run streamlit_app.py
```

## 9. Current Limitations

The repository is now demoable, but it is still a checkpoint build rather than a production system.

Current limitations include:

- The frontend and backend are still local-first and depend on the committed model artifact
- There is a local API boundary, but no deployment layer
- The baseline model remains the original logistic regression approach
- Input validation is structural, not deeply business-semantic
- Predictions are suitable for demo decision support, not operational automation
- The UI is intentionally minimal and does not include analytics, history, authentication, or multi-user audit features

## 10. What I Would Improve Next With More Time

If given more time, I would improve the solution in the following order:

1. Add broader automated coverage for the training path, frontend rendering, and failure-mode flows
2. Add stronger business-level validation and explanatory guidance around feature inputs
3. Persist request/response history and expose basic observability on the backend
4. Expand the API beyond a single prediction path if the product grows past checkpoint-demo needs
5. Improve demo observability with clearer input summaries, confidence context, and prediction explanation cues
6. Replace the checkpoint-style setup with a more complete packaging and environment workflow

## Summary

My contribution was to turn an ML proof-of-concept repository into a demo-ready checkpoint build with minimal engineering disruption.

I kept the original project direction intact, reused the existing baseline model, and added the smallest practical training, artifact, inference, frontend, and documentation pieces needed for a reliable live presentation.
