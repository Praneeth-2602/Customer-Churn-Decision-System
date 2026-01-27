# ml/actions.py

from typing import List, Dict


def recommend_actions(
    churn_probability: float,
    explanations: List[Dict],
    customer_data: Dict
) -> Dict:
    """
    Recommends retention actions based on churn risk and SHAP explanations.

    Parameters:
    - churn_probability (float): Predicted churn probability
    - explanations (list): Output from explain.py (top contributors)
    - customer_data (dict): Raw customer input data

    Returns:
    - dict containing recommended actions and expected impact
    """

    actions = []
    total_improvement = 0.0

    # Extract churn-increasing features
    churn_drivers = {
        e["feature"]: e["impact"]
        for e in explanations
        if e["impact"] > 0
    }

    # 🔴 HIGH RISK CUSTOMERS
    if churn_probability >= 0.7:

        # Pricing sensitivity
        if any(f in churn_drivers for f in ["MonthlyCharges", "TotalCharges"]):
            actions.append({
                "action": "Offer 20% discount for next 3 months",
                "reason": "High price sensitivity detected",
                "expected_churn_reduction": 0.15
            })
            total_improvement += 0.15

        # Contract type
        if "Contract_Month-to-month" in churn_drivers:
            actions.append({
                "action": "Offer incentive to switch to annual contract",
                "reason": "Month-to-month contracts are associated with higher churn",
                "expected_churn_reduction": 0.20
            })
            total_improvement += 0.20

        # Support & security
        if any(f in churn_drivers for f in ["TechSupport_No", "OnlineSecurity_No"]):
            actions.append({
                "action": "Provide free tech support and security add-ons",
                "reason": "Lack of support/security increases dissatisfaction",
                "expected_churn_reduction": 0.12
            })
            total_improvement += 0.12

    # 🟡 MEDIUM RISK CUSTOMERS
    elif churn_probability >= 0.4:
        actions.append({
            "action": "Send personalized engagement and usage tips email",
            "reason": "Moderate churn risk detected",
            "expected_churn_reduction": 0.08
        })
        total_improvement += 0.08

    # 🟢 LOW RISK CUSTOMERS
    else:
        actions.append({
            "action": "No immediate action required",
            "reason": "Customer shows low churn risk",
            "expected_churn_reduction": 0.0
        })

    # Safety cap
    total_improvement = min(total_improvement, churn_probability)

    return {
        "recommended_actions": actions,
        "estimated_total_improvement": round(total_improvement, 3),
        "expected_new_churn_probability": round(
            churn_probability - total_improvement, 3
        )
    }
