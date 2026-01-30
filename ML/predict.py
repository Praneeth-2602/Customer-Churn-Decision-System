# ml/predict.py

import os
import pandas as pd
import joblib

try:
    from ML.config_loader import load_config
except ImportError:
    from config_loader import load_config


BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

def get_risk_level(prob: float, thresholds: dict) -> str:
    """
    Assign churn risk level based on dataset-specific thresholds
    """
    if prob >= thresholds["high"]:
        return "HIGH"
    elif prob >= thresholds["medium"]:
        return "MEDIUM"
    else:
        return "LOW"


def predict_churn(dataset_name: str, customer_data: dict) -> dict:
    """
    Predict churn probability for a single customer
    """

    # Load config
    cfg = load_config(dataset_name)

    # Load artifacts
    model = joblib.load(f"{MODEL_DIR}/{dataset_name}_model.pkl")
    preprocessor = joblib.load(
        f"{MODEL_DIR}/{dataset_name}_preprocessor.pkl"
    )

    # Convert input to DataFrame
    df = pd.DataFrame([customer_data])

    # Ensure correct column order & type safety
    for col in cfg["numerical_features"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(0)

    # Transform features
    X_processed = preprocessor.transform(df)

    # Predict probability
    churn_prob = model.predict_proba(X_processed)[0][1]

    # Risk level (dataset-specific)
    risk_level = get_risk_level(
        churn_prob, cfg["risk_thresholds"]
    )

    return {
        "churn_probability": round(float(churn_prob), 4),
        "risk_level": risk_level
    }


if __name__ == "__main__":
    # Quick sanity test (example: telco dataset)
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.5,
        "TotalCharges": 950.2
    }

    result = predict_churn("telco", sample_customer)
    print(result)
