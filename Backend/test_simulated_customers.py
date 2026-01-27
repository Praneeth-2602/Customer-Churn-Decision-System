# test_simulated_customers.py

import requests
import random

API_URL = "http://127.0.0.1:8000/analyze"


def generate_customer(profile: str):
    """
    Generate synthetic customer profiles
    """

    if profile == "low":
        return {
            "gender": random.choice(["Male", "Female"]),
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": random.randint(36, 72),
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": round(random.uniform(40, 60), 2),
            "TotalCharges": round(random.uniform(2000, 5000), 2)
        }

    if profile == "medium":
        return {
            "gender": random.choice(["Male", "Female"]),
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": random.randint(12, 36),
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": round(random.uniform(60, 80), 2),
            "TotalCharges": round(random.uniform(1000, 3000), 2)
        }

    # high risk
    return {
        "gender": random.choice(["Male", "Female"]),
        "SeniorCitizen": random.choice([0, 1]),
        "Partner": "No",
        "Dependents": "No",
        "tenure": random.randint(1, 12),
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": round(random.uniform(85, 110), 2),
        "TotalCharges": round(random.uniform(100, 1000), 2)
    }


def test_simulated_profiles():
    print("\n==============================")
    print(" 🧪 SIMULATED CUSTOMER TEST")
    print("==============================\n")

    for profile in ["low", "medium", "high"]:
        customer = generate_customer(profile)

        response = requests.post(API_URL, json=customer)
        result = response.json()

        print(f"Profile: {profile.upper()}")
        print(f"  Churn Probability: {result['churn_probability']}")
        print(f"  Risk Level       : {result['risk_level']}")
        print(f"  Recommended Acts : {len(result['recommended_actions'])}")
        print(f"  Improvement      : {result['simulation']['improvement']}")
        print("-" * 40)

    print("\n✅ SIMULATED TEST COMPLETED\n")


if __name__ == "__main__":
    test_simulated_profiles()
