# test_backend.py

import requests
import json


API_URL = "http://127.0.0.1:8000/analyze"


def test_backend_pipeline():
    print("\n==============================")
    print(" 🚀 BACKEND + ML INTEGRATION TEST")
    print("==============================\n")

    customer = {
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

    # Send request (include dataset)
    payload = {
        "dataset": "telco",
        "customer": customer
    }
    response = requests.post(API_URL, json=payload)

    if response.status_code != 200:
        print("❌ API Error")
        print(response.status_code, response.text)
        return

    result = response.json()

    print("🔹 Churn Prediction")
    print(f"Probability: {result['churn_probability']}")
    print(f"Risk Level : {result['risk_level']}\n")

    print("🔹 Explanation (WHY)")
    for e in result["explanation"]:
        print(f" - {e['feature']} ({e['effect']}, impact={e['impact']})")
    print()

    print("🔹 Recommended Actions")
    for a in result["recommended_actions"]:
        print(f" - {a['action']} | Reason: {a['reason']}")
    print()

    print("🔹 Simulation Result")
    print("Before:", result["simulation"]["before"])
    print("After :", result["simulation"]["after"])
    print("Improvement:", result["simulation"]["improvement"])

    print("\n✅ BACKEND + ML TEST COMPLETED SUCCESSFULLY\n")


if __name__ == "__main__":
    test_backend_pipeline()
