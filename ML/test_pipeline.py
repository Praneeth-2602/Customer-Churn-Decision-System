
try:
    from ML.predict import predict_churn
    from ML.explain import explain_customer
    from ML.actions import recommend_actions
    from ML.simulate import simulate_retention
except ImportError:
    from predict import predict_churn
    from explain import explain_customer
    from actions import recommend_actions
    from simulate import simulate_retention

import os
import pandas as pd



def run_test():
    print("\n==============================")
    print(" 🚀 CHURN DECISION PIPELINE TEST")
    print("==============================\n")

    datasets = ["telco", "bank"]

    for dataset in datasets:
        print(f"\n--- Testing dataset: {dataset} ---\n")

        # load a representative sample from the dataset CSV
        data_path = os.path.join(os.path.dirname(__file__), "data", f"{dataset}.csv")
        df = pd.read_csv(data_path)
        # drop target column if present and select first row as sample
        cfg_target = None
        if "Churn" in df.columns:
            cfg_target = "Churn"
        elif "Exited" in df.columns:
            cfg_target = "Exited"
        if cfg_target and cfg_target in df.columns:
            sample = df.drop(columns=[cfg_target]).iloc[0].to_dict()
        else:
            sample = df.iloc[0].to_dict()

        # STEP 1: Predict
        print("🔹 Step 1: Churn Prediction")
        prediction = predict_churn(dataset, sample)
        print(prediction, "\n")

        # STEP 2: Explain
        print("🔹 Step 2: Explanation (WHY)")
        explanation = explain_customer(dataset, sample)["top_contributors"]
        for e in explanation:
            print(f" - {e['feature']} ({e['effect']}, impact={e['impact']})")
        print()

        # STEP 3: Recommend Actions
        print("🔹 Step 3: Recommended Actions (WHAT TO DO)")
        action_plan = recommend_actions(
            dataset,
            churn_probability=prediction["churn_probability"],
            explanations=explanation,
        )

        for a in action_plan["recommended_actions"]:
            print(f" - {a['action']} | Reason: {a['reason']}")

        print(f"\nEstimated Improvement: {action_plan['estimated_total_improvement']}")
        print(f"Expected New Churn Probability: {action_plan['expected_new_churn_probability']}\n")

        # STEP 4: Simulate
        print("🔹 Step 4: Simulation (ACTUAL IMPROVEMENT)")
        simulation = simulate_retention(
            dataset,
            customer_data=sample,
            recommended_actions=action_plan["recommended_actions"]
        )

        print("Before:", simulation["before"])
        print("After :", simulation["after"])
        print("Improvement:", simulation["improvement"])

    print("\n✅ PIPELINE TEST COMPLETED SUCCESSFULLY\n")


if __name__ == "__main__":
    run_test()
