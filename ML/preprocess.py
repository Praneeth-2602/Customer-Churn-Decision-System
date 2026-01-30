# ml/preprocess.py

import os
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

try:
    from ML.config_loader import load_config
except Exception:
    # If the ML package isn't importable (running file directly), fall back
    # to a local import that works when CWD is the ML/ folder.
    from config_loader import load_config


DATA_DIR = "data"
MODEL_DIR = "models"


def load_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Load dataset CSV based on dataset name. Try ML/data and repo-root data/ locations.
    """
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


def build_preprocessor(dataset_name: str, df: pd.DataFrame):
    """
    Build preprocessing pipeline using dataset config
    """
    cfg = load_config(dataset_name)

    categorical_cols = cfg["categorical_features"]
    numerical_cols = cfg["numerical_features"]

    numeric_pipeline = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor


def preprocess_and_save(dataset_name: str):
    """
    Fit and save preprocessing pipeline for a dataset
    """
    cfg = load_config(dataset_name)
    df = load_dataset(dataset_name)

    # Drop ID column if present
    if cfg["id_column"] in df.columns:
        df = df.drop(columns=[cfg["id_column"]])

    # Handle TotalCharges-like numeric issues safely
    for col in cfg["numerical_features"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    X = df.drop(columns=[cfg["target_column"]])

    preprocessor = build_preprocessor(dataset_name, df)
    preprocessor.fit(X)

    # Save pipeline
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = f"{MODEL_DIR}/{dataset_name}_preprocessor.pkl"
    joblib.dump(preprocessor, path)

    print(f"✅ Preprocessor saved: {path}")


if __name__ == "__main__":
    # Example usage:
    # preprocess_and_save("telco")
    # preprocess_and_save("bank")

    import sys
    dataset = sys.argv[1]
    preprocess_and_save(dataset)
