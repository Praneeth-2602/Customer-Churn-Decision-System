# ml/explain.py

import os
import pandas as pd
import joblib
import shap
# from ML.config_loader import load_config


def _paths_for_dataset(dataset_name: str):
    base = os.path.dirname(__file__)
    models_dir = os.path.join(base, "models")
    preprocessor_path = os.path.join(models_dir, f"{dataset_name}_preprocessor.pkl")
    model_path = os.path.join(models_dir, f"{dataset_name}_model.pkl")
    return preprocessor_path, model_path


def _get_feature_names(preprocessor):
    num_features = preprocessor.transformers_[0][2]
    cat_pipeline = preprocessor.transformers_[1][1]
    cat_features = preprocessor.transformers_[1][2]
    onehot = cat_pipeline.named_steps["onehot"]
    cat_feature_names = onehot.get_feature_names_out(cat_features)
    return list(num_features) + list(cat_feature_names)


def explain_customer(dataset_name: str, customer_data: dict, top_k: int = 5) -> dict:
    preprocessor_path, model_path = _paths_for_dataset(dataset_name)
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)

    explainer = shap.TreeExplainer(model)

    FEATURE_NAMES = _get_feature_names(preprocessor)

    df = pd.DataFrame([customer_data])
    X_processed = preprocessor.transform(df)

    shap_values = explainer.shap_values(X_processed)[0]

    feature_impacts = list(zip(FEATURE_NAMES, shap_values))
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

    top_features = feature_impacts[:top_k]
    explanation = []
    for feature, impact in top_features:
        explanation.append({
            "feature": feature,
            "impact": round(float(impact), 4),
            "effect": "increases churn" if impact > 0 else "reduces churn"
        })

    return {"top_contributors": explanation}


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
