# ml/simulate.py

import copy
from typing import Dict, List

try:
    from ML.config_loader import load_config
    from ML.predict import predict_churn
except ImportError:
    from config_loader import load_config
    from predict import predict_churn


def _apply_pricing_action(customer: Dict):
    # Generic pricing lever
    if "MonthlyCharges" in customer:
        customer["MonthlyCharges"] *= 0.8  # 20% discount
    return customer


def _apply_contract_action(customer: Dict):
    # Generic contract lever
    if "Contract" in customer:
        customer["Contract"] = "One year"
    if "SubscriptionType" in customer:
        customer["SubscriptionType"] = "Premium"
    return customer


def _apply_support_action(customer: Dict):
    # Generic support lever
    for key in ["TechSupport", "OnlineSecurity", "OnlineBackup"]:
        if key in customer:
            customer[key] = "Yes"

    if "SupportTicketsPerMonth" in customer:
        customer["SupportTicketsPerMonth"] = max(
            0, customer["SupportTicketsPerMonth"] - 1
        )
    return customer


def _apply_engagement_action(customer: Dict):
    # Generic engagement lever
    if "ViewingHoursPerWeek" in customer:
        customer["ViewingHoursPerWeek"] += 2
    if "WatchlistSize" in customer:
        customer["WatchlistSize"] += 3
    if "IsActiveMember" in customer:
        customer["IsActiveMember"] = 1
    return customer


def _apply_loyalty_action(customer: Dict):
    # Generic loyalty lever
    if "Tenure" in customer:
        customer["Tenure"] += 6
    if "Card Type" in customer:
        customer["Card Type"] = "PLATINUM"
    return customer


ACTION_APPLIERS = {
    "pricing": _apply_pricing_action,
    "contract": _apply_contract_action,
    "support": _apply_support_action,
    "engagement": _apply_engagement_action,
    "loyalty": _apply_loyalty_action,
    "experience": _apply_engagement_action,
}


def simulate_retention(
    dataset_name: str,
    customer_data: Dict,
    recommended_actions: List[Dict]
) -> Dict:
    """
    Simulate churn probability before and after applying
    retention actions for a given dataset.
    """

    cfg = load_config(dataset_name)

    # 1️⃣ Original prediction
    before = predict_churn(dataset_name, customer_data)

    # 2️⃣ Apply actions safely
    updated_customer = copy.deepcopy(customer_data)

    for action in recommended_actions:
        action_name = action["action"].lower()

        for group in cfg["retention_actions"]:
            if group in action_name and group in ACTION_APPLIERS:
                updated_customer = ACTION_APPLIERS[group](updated_customer)

    # 3️⃣ New prediction
    after = predict_churn(dataset_name, updated_customer)

    improvement = round(
        before["churn_probability"] - after["churn_probability"], 4
    )

    return {
        "before": before,
        "after": after,
        "improvement": improvement,
        "updated_customer_snapshot": updated_customer
    }


if __name__ == "__main__":
    # Quick sanity test (telco example)
    sample_customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 18,
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
        "TotalCharges": 1600.0
    }

    actions = [
        {"action": "Offer 20% discount for next 3 months"},
        {"action": "Provide free tech support and security add-ons"},
    ]

    result = simulate_retention(
        "telco", sample_customer, actions
    )
    print(result)
