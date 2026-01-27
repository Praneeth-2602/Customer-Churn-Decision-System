# ml/preprocess.py

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


DATA_PATH = "data/telco.csv"
# Save pipeline next to this script (when run from the ML/ folder)
PIPELINE_PATH = "preprocess_pipeline.pkl"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop customerID (identifier, no predictive value)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges has spaces -> convert to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows with missing TotalCharges
    df = df.dropna()

    return df


def build_preprocessing_pipeline(df: pd.DataFrame):
    # Target
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])

    # Identify column types
    # select both object and new pandas string dtype to avoid Pandas4Warning
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Pipelines
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    # use `sparse_output=False` for newer scikit-learn compatibility
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    return X, y, preprocessor


def preprocess_and_save():
    df = load_data()
    df = clean_data(df)

    X, y, preprocessor = build_preprocessing_pipeline(df)

    # Fit preprocessor
    preprocessor.fit(X)

    # Save pipeline
    joblib.dump(preprocessor, PIPELINE_PATH)

    print("✅ Preprocessing pipeline saved at:", PIPELINE_PATH)

    return X, y


if __name__ == "__main__":
    preprocess_and_save()
