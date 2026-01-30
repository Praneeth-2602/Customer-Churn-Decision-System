# ml/actions.py

from typing import List, Dict

try:
    from ML.config_loader import load_config
except ImportError:
    from config_loader import load_config


def recommend_actions(
    dataset_name: str,
    churn_probability: float,
    explanations: List[Dict],
) -> Dict:
    """
    Recommend retention actions using dataset-specific configuration
    and SHAP explanations.

    Parameters:
    - dataset_name: str
    - churn_probability: float
    - explanations: list of SHAP explanation dicts

    Returns:
    - dict with recommended actions and expected impact
    """

    cfg = load_config(dataset_name)

    actionable_features = cfg["actionable_features"]
    retention_actions = cfg["retention_actions"]
    risk_thresholds = cfg["risk_thresholds"]

    actions = []
    total_improvement = 0.0

    # Extract churn-increasing features only
    churn_drivers = {
        e["feature"]: e["impact"]
        for e in explanations
        if e["impact"] > 0
    }

    # Decide risk band
    if churn_probability >= risk_thresholds["high"]:
        risk_band = "HIGH"
    elif churn_probability >= risk_thresholds["medium"]:
        risk_band = "MEDIUM"
    else:
        risk_band = "LOW"

    # LOW risk → no action
    if risk_band == "LOW":
        return {
            "recommended_actions": [
                {
                    "action": "No immediate action required",
                    "reason": "Customer shows low churn risk",
                    "expected_churn_reduction": 0.0
                }
            ],
            "estimated_total_improvement": 0.0,
            "expected_new_churn_probability": round(churn_probability, 4)
        }

    # HIGH or MEDIUM risk → generate actions
    for group, features in actionable_features.items():
        # Check if any churn driver belongs to this action group
        triggered = False

        for feature in churn_drivers:
            # SHAP feature names may be one-hot encoded
            if any(feature.startswith(f) for f in features):
                triggered = True
                break

        if triggered and group in retention_actions:
            action_cfg = retention_actions[group]

            actions.append({
                "action": action_cfg["action"],
                "reason": f"Churn drivers detected in {group} features",
                "expected_churn_reduction": action_cfg["expected_churn_reduction"]
            })

            total_improvement += action_cfg["expected_churn_reduction"]

    # Safety cap (never reduce below zero)
    total_improvement = min(total_improvement, churn_probability)

    return {
        "recommended_actions": actions,
        "estimated_total_improvement": round(total_improvement, 4),
        "expected_new_churn_probability": round(
            churn_probability - total_improvement, 4
        )
    }
