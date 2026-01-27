# ml/explain.py

import os
import pandas as pd
import joblib
import shap
import numpy as np

# Resolve artifact paths relative to this file
BASE_DIR = os.path.dirname(__file__)
PIPELINE_PATH = os.path.join(BASE_DIR, "preprocess_pipeline.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")


# Load artifacts once
preprocessor = joblib.load(PIPELINE_PATH)
model = joblib.load(MODEL_PATH)

# Build SHAP explainer
explainer = shap.TreeExplainer(model)


def get_feature_names():
    """
    Extract final feature names after preprocessing
    """
    num_features = preprocessor.transformers_[0][2]

    cat_pipeline = preprocessor.transformers_[1][1]
    cat_features = preprocessor.transformers_[1][2]

    onehot = cat_pipeline.named_steps["onehot"]
    cat_feature_names = onehot.get_feature_names_out(cat_features)

    return list(num_features) + list(cat_feature_names)


FEATURE_NAMES = get_feature_names()


def explain_customer(customer_data: dict, top_k: int = 5) -> dict:
    """
    Returns top positive and negative contributors for churn
    """

    # Convert input to DataFrame
    df = pd.DataFrame([customer_data])

    # Transform features
    X_processed = preprocessor.transform(df)

    # SHAP values
    shap_values = explainer.shap_values(X_processed)[0]

    # Pair features with SHAP values
    feature_impacts = list(zip(FEATURE_NAMES, shap_values))

    # Sort by absolute impact
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

    # Top contributors
    top_features = feature_impacts[:top_k]

    explanation = []
    for feature, impact in top_features:
        explanation.append({
            "feature": feature,
            "impact": round(float(impact), 4),
            "effect": "increases churn" if impact > 0 else "reduces churn"
        })

    return {
        "top_contributors": explanation
    }


if __name__ == "__main__":
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

    result = explain_customer(sample_customer)
    for r in result["top_contributors"]:
        print(r)
