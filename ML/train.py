# ml/train.py

import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, classification_report


DATA_PATH = "data/telco.csv"
PIPELINE_PATH = "preprocess_pipeline.pkl"
MODEL_PATH = "model.pkl"

# Constants for prediction threshold and class weighting
THRESHOLD = 0.35
SCALE_POS_WEIGHT = 2.7
MODEL_METADATA_PATH = "model_metadata.pkl"


def load_data():
    df = pd.read_csv(DATA_PATH)

    # Drop customerID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Fix TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df.dropna()
    return df


def train_model():
    df = load_data()

    # Target
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])

    # Load preprocessing pipeline
    preprocessor = joblib.load(PIPELINE_PATH)

    # Transform features
    X_processed = preprocessor.transform(X)

    # Train-test split (IMPORTANT: stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Model
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=SCALE_POS_WEIGHT,
        eval_metric="logloss",
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)

    print("\n📊 Model Evaluation")
    print("------------------")
    print(f"ROC-AUC Score : {auc:.4f}")
    print(f"Recall (Churn=Yes) : {recall:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved at: {MODEL_PATH}")

    # Save metadata so other components use the same threshold and weighting
    metadata = {
        "threshold": THRESHOLD,
        "scale_pos_weight": SCALE_POS_WEIGHT
    }

    joblib.dump(metadata, MODEL_METADATA_PATH)
    print(f"✅ Model metadata saved at: {MODEL_METADATA_PATH}")


if __name__ == "__main__":
    train_model()
