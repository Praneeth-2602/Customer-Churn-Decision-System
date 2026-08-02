# Customer Churn Decision System

Dataset-agnostic **churn decision pipeline**: predict risk → explain with SHAP → recommend retention actions → simulate impact — exposed as a FastAPI service.

A score alone doesn’t tell a CSM what to do. This project adds an explainable decision layer that works across datasets via YAML configs.

## Features

| Step | Module | What you get |
|---|---|---|
| Train | `ML/train.py` | XGBoost (+ preprocessor) per dataset |
| Predict | `ML/predict.py` | Churn probability + risk band |
| Explain | `ML/explain.py` | SHAP feature impacts |
| Act | `ML/actions.py` | Ranked retention recommendations |
| Simulate | `ML/simulate.py` | Before/after churn under interventions |
| Serve | `Backend/` | `POST /analyze` |

Supported datasets today: **`telco`**, **`bank`** (see `ML/configs/`). There is **no frontend** in this repo yet.

## Structure

```
ML/
  configs/          # telco.yaml, bank.yaml, …
  data/             # training CSVs
  models/           # {dataset}_model.pkl, preprocessor pkls
  train.py predict.py explain.py actions.py simulate.py
Backend/
  main.py           # FastAPI app
  __init__.py       # re-exports app for uvicorn Backend:app
```

## Setup

```bash
cd Customer-Churn-Decision-System
python3 -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn pydantic xgboost scikit-learn shap pandas pyyaml
# plus any extras used by your local ML/*.py imports
```

Train (from repo root, with `ML` on `PYTHONPATH` or run inside `ML/` as documented there):

```bash
cd ML
python train.py telco
python train.py bank
```

## Run the API

From the **repo root** (so `ML.*` and `Backend` import cleanly):

```bash
uvicorn Backend:app --reload --host 0.0.0.0 --port 8000
```

Health: `GET http://localhost:8000/`

### Analyze a customer

```bash
curl -s http://localhost:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset": "telco",
    "customer": {
      "tenure": 12,
      "MonthlyCharges": 70.0,
      "Contract": "Month-to-month"
    }
  }'
```

Response includes `churn_probability`, `risk_level`, SHAP-style `explanation`, `recommended_actions`, and `simulation`. Exact customer fields must match the dataset config / preprocessor.

## Tests

```bash
# See Backend/test_backend.py and Backend/test_simulated_customers.py
python -m pytest Backend/ -q   # if pytest + deps installed
```

## Notes

- Models are sklearn/XGBoost pickles under `ML/models/` — retrain after config or data changes.
- Keep secrets and real PII out of the tree; use synthetic or public churn CSVs.
