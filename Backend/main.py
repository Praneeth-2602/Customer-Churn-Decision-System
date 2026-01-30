# backend/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Literal

from ML.predict import predict_churn
from ML.explain import explain_customer
from ML.actions import recommend_actions
from ML.simulate import simulate_retention
from ML.config_loader import load_config


# =========================
# API App
# =========================

app = FastAPI(
    title="Churn Decision Intelligence API",
    description="Dataset-agnostic churn prediction, explanation, action recommendation, and simulation",
    version="2.0"
)


# =========================
# Request / Response Schemas
# =========================

class AnalyzeRequest(BaseModel):
    dataset: Literal["telco", "bank"]
    customer: Dict


class ExplanationItem(BaseModel):
    feature: str
    impact: float
    effect: str


class ActionItem(BaseModel):
    action: str
    reason: str
    expected_churn_reduction: float


class AnalyzeResponse(BaseModel):
    dataset: str
    churn_probability: float
    risk_level: str
    explanation: List[ExplanationItem]
    recommended_actions: List[ActionItem]
    simulation: Dict


# =========================
# Health Check
# =========================

@app.get("/")
def health():
    return {"status": "API running", "version": "2.0"}


# =========================
# Core Endpoint
# =========================

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_customer(request: AnalyzeRequest):
    dataset = request.dataset
    customer = request.customer

    # Validate dataset config exists
    try:
        load_config(dataset)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported dataset '{dataset}'"
        )

    # 1️⃣ Predict churn
    prediction = predict_churn(dataset, customer)

    # 2️⃣ Explain prediction
    explanation_result = explain_customer(dataset, customer)
    explanations = explanation_result["top_contributors"]

    # 3️⃣ Recommend actions
    action_plan = recommend_actions(
        dataset_name=dataset,
        churn_probability=prediction["churn_probability"],
        explanations=explanations
    )

    # 4️⃣ Simulate outcome
    simulation = simulate_retention(
        dataset_name=dataset,
        customer_data=customer,
        recommended_actions=action_plan["recommended_actions"]
    )

    return {
        "dataset": dataset,
        "churn_probability": prediction["churn_probability"],
        "risk_level": prediction["risk_level"],
        "explanation": explanations,
        "recommended_actions": action_plan["recommended_actions"],
        "simulation": simulation
    }
