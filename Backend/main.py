from fastapi import FastAPI
from Backend.schemas import CustomerInput, AnalyzeResponse

from ML.predict import predict_churn
from ML.explain import explain_customer
from ML.actions import recommend_actions
from ML.simulate import simulate_retention


app = FastAPI(
    title="Churn Decision Intelligence API",
    description="Predicts churn, explains why, recommends actions, and simulates retention impact",
    version="1.0"
)


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_customer(customer: CustomerInput):
    customer_data = customer.dict()

    # 1️⃣ Predict churn
    prediction = predict_churn(customer_data)

    # 2️⃣ Explain prediction
    explanation = explain_customer(customer_data)["top_contributors"]

    # 3️⃣ Recommend actions
    action_plan = recommend_actions(
        churn_probability=prediction["churn_probability"],
        explanations=explanation,
        customer_data=customer_data
    )

    # 4️⃣ Simulate outcome
    simulation = simulate_retention(
        customer_data=customer_data,
        recommended_actions=action_plan["recommended_actions"]
    )

    return {
        "churn_probability": prediction["churn_probability"],
        "risk_level": prediction["risk_level"],
        "explanation": explanation,
        "recommended_actions": action_plan["recommended_actions"],
        "simulation": simulation
    }
