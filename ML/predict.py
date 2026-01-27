# ml/predict.py

import pandas as pd
import joblib

PIPELINE_PATH = "preprocess_pipeline.pkl"
# Prefer the trained model file present in the ML/ folder
MODEL_PATH = "model.pkl"
METADATA_PATH = "model_metadata.pkl"


# Load artifacts once
preprocessor = joblib.load(PIPELINE_PATH)
model = joblib.load(MODEL_PATH)
metadata = joblib.load(METADATA_PATH)

THRESHOLD = metadata["threshold"]


def get_risk_level(prob):
    if prob >= 0.7:
        return "HIGH"
    elif prob >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"


def predict_churn(customer_data: dict) -> dict:
    """
    customer_data: dict with same keys as training features
    """

    # Convert to DataFrame
    df = pd.DataFrame([customer_data])

    # Transform
    X_processed = preprocessor.transform(df)

    # Predict probability
    churn_prob = model.predict_proba(X_processed)[0][1]

    # Risk bucket
    risk_level = get_risk_level(churn_prob)

    return {
        "churn_probability": round(float(churn_prob), 4),
        "risk_level": risk_level
    }


if __name__ == "__main__":
    # Simple test
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

    result = predict_churn(sample_customer)
    print(result)
