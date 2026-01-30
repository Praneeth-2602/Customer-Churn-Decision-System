# ml/train.py

import os
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score

try:
    from ML.config_loader import load_config
except ImportError:
    from config_loader import load_config


DATA_DIR = "data"
# Use models directory inside the ML package so artifacts live at
# <repo-root>/ML/models rather than <repo-root>/models
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_dataset(dataset_name: str) -> pd.DataFrame:
    candidates = [
        f"{DATA_DIR}/{dataset_name}.csv",
        f"ML/{DATA_DIR}/{dataset_name}.csv",
        f"{os.path.dirname(__file__)}/{DATA_DIR}/{dataset_name}.csv",
        f"{os.path.dirname(os.path.dirname(__file__))}/{DATA_DIR}/{dataset_name}.csv",
    ]
    for path in candidates:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"Data file for dataset '{dataset_name}' not found. Tried: {candidates}")


def train_model(dataset_name: str):
    cfg = load_config(dataset_name)
    df = load_dataset(dataset_name)

    # Drop ID column
    if cfg["id_column"] in df.columns:
        df = df.drop(columns=[cfg["id_column"]])

    # Convert numeric columns safely
    for col in cfg["numerical_features"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    # Target encoding (be resilient to YAML boolean parsing of Yes/No)
    pos = cfg.get("positive_class")
    neg = cfg.get("negative_class")

    def _map_target(series):
        # if config used booleans (PyYAML converts Yes/No to True/False),
        # accept common textual representations as well
        if isinstance(pos, bool) or isinstance(neg, bool):
            mapper = {
                True: 1, False: 0,
                "Yes": 1, "No": 0, "YES": 1, "NO": 0,
                "yes": 1, "no": 0, "Y": 1, "N": 0, "y": 1, "n": 0,
                "1": 1, "0": 0, 1: 1, 0: 0
            }
            return series.map(lambda v: mapper.get(v, None))
        else:
            return series.map({pos: 1, neg: 0})

    y = _map_target(df[cfg["target_column"]])

    X = df.drop(columns=[cfg["target_column"]])

    # Drop rows with unmapped/NaN target values
    missing = y.isna()
    if missing.any():
        import warnings
        warnings.warn(f"{missing.sum()} rows have unexpected target values and will be dropped.")
        X = X.loc[~missing].reset_index(drop=True)
        y = y.loc[~missing].reset_index(drop=True)

    if len(y) == 0:
        raise ValueError(f"No valid training rows found for dataset '{dataset_name}' after cleaning.")

    # ensure integer labels
    try:
        y = y.astype(int)
    except Exception:
        pass

    # Load preprocessing pipeline
    preprocessor_path = f"{MODEL_DIR}/{dataset_name}_preprocessor.pkl"
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Model directory '{MODEL_DIR}' does not exist. Run preprocess first.")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor for '{dataset_name}' not found at {preprocessor_path}. Run preprocess first.")
    preprocessor = joblib.load(preprocessor_path)

    X_processed = preprocessor.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Model (robust defaults for churn)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluation
    probs = model.predict_proba(X_test)[:, 1]
    # Use a lower, fixed threshold for training-time evaluation only
    preds = (probs >= 0.3).astype(int)

    auc = roc_auc_score(y_test, probs)
    recall = recall_score(y_test, preds)

    print(f"\n📊 Dataset: {dataset_name}")
    print(f"ROC-AUC : {auc:.4f}")
    print(f"Recall  : {recall:.4f}")

    # Save model
    model_path = f"{MODEL_DIR}/{dataset_name}_model.pkl"
    joblib.dump(model, model_path)

    print(f"✅ Model saved: {model_path}")


if __name__ == "__main__":
    import sys
    dataset = sys.argv[1]
    train_model(dataset)
