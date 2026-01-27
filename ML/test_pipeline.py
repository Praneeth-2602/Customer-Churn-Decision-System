
from predict import predict_churn
from explain import explain_customer
from actions import recommend_actions
from simulate import simulate_retention


def run_test():
    print("\n==============================")
    print(" 🚀 CHURN DECISION PIPELINE TEST")
    print("==============================\n")

    # Sample high-risk customer
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

    # STEP 1: Predict
    print("🔹 Step 1: Churn Prediction")
    prediction = predict_churn(customer)
    print(prediction, "\n")

    # STEP 2: Explain
    print("🔹 Step 2: Explanation (WHY)")
    explanation = explain_customer(customer)["top_contributors"]
    for e in explanation:
        print(f" - {e['feature']} ({e['effect']}, impact={e['impact']})")
    print()

    # STEP 3: Recommend Actions
    print("🔹 Step 3: Recommended Actions (WHAT TO DO)")
    action_plan = recommend_actions(
        churn_probability=prediction["churn_probability"],
        explanations=explanation,
        customer_data=customer
    )

    for a in action_plan["recommended_actions"]:
        print(f" - {a['action']} | Reason: {a['reason']}")

    print(f"\nEstimated Improvement: {action_plan['estimated_total_improvement']}")
    print(f"Expected New Churn Probability: {action_plan['expected_new_churn_probability']}\n")

    # STEP 4: Simulate
    print("🔹 Step 4: Simulation (ACTUAL IMPROVEMENT)")
    simulation = simulate_retention(
        customer_data=customer,
        recommended_actions=action_plan["recommended_actions"]
    )

    print("Before:", simulation["before"])
    print("After :", simulation["after"])
    print("Improvement:", simulation["improvement"])

    print("\n✅ PIPELINE TEST COMPLETED SUCCESSFULLY\n")


if __name__ == "__main__":
    run_test()
