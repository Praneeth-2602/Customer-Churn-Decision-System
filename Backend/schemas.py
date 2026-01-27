from pydantic import BaseModel
from typing import List, Dict


class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class ExplanationItem(BaseModel):
    feature: str
    impact: float
    effect: str


class ActionItem(BaseModel):
    action: str
    reason: str
    expected_churn_reduction: float


class AnalyzeResponse(BaseModel):
    churn_probability: float
    risk_level: str
    explanation: List[ExplanationItem]
    recommended_actions: List[ActionItem]
    simulation: Dict
