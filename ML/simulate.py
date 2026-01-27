# ml/simulate.py

import copy
import pandas as pd
import joblib

from predict import predict_churn


PIPELINE_PATH = "preprocess_pipeline.pkl"
MODEL_PATH = "model.pkl"

# Load artifacts (predict.py already loads these, but kept explicit for clarity)
preprocessor = joblib.load(PIPELINE_PATH)
model = joblib.load(MODEL_PATH)


def apply_actions(customer_data: dict, actions: list) -> dict:
    """
    Modify customer features based on recommended actions
    """

    updated_customer = copy.deepcopy(customer_data)

    for action in actions:
        name = action["action"].lower()

        # Apply discount
        if "discount" in name and "MonthlyCharges" in updated_customer:
            updated_customer["MonthlyCharges"] *= 0.8  # 20% discount

        # Switch contract
        if "annual contract" in name:
            updated_customer["Contract"] = "One year"

        # Add support & security
        if "tech support" in name:
            updated_customer["TechSupport"] = "Yes"
            updated_customer["OnlineSecurity"] = "Yes"

    return updated_customer


def simulate_retention(
    customer_data: dict,
    recommended_actions: list
) -> dict:
    """
    Simulate churn probability before and after retention actions
    """

    # Original prediction
    original_result = predict_churn(customer_data)

    # Apply actions
    updated_customer = apply_actions(customer_data, recommended_actions)

    # New prediction
    new_result = predict_churn(updated_customer)

    improvement = round(
        original_result["churn_probability"] -
        new_result["churn_probability"], 4
    )

    return {
        "before": original_result,
        "after": new_result,
        "improvement": improvement,
        "updated_customer_snapshot": updated_customer
    }
