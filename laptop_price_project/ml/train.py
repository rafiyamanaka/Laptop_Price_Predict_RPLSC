from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
DATASET_PATH = WORKSPACE_DIR / "laptop_price - dataset.csv"
MODEL_PATH = PROJECT_DIR / "ml" / "model.joblib"
CHOICES_PATH = PROJECT_DIR / "ml" / "choices.json"

TARGET_COL = "Price (Euro)"

FEATURE_COLS = [
    "Company",
    "TypeName",
    "Inches",
    "CPU_Company",
    "CPU_Frequency (GHz)",
    "RAM (GB)",
    "Memory",
    "Weight (kg)",
    "OpSys",
]

NUMERIC_COLS = [
    "Inches",
    "CPU_Frequency (GHz)",
    "RAM (GB)",
    "Weight (kg)",
]

CATEGORICAL_COLS = [c for c in FEATURE_COLS if c not in NUMERIC_COLS]


def main() -> int:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    df = df[FEATURE_COLS + [TARGET_COL]].copy()

    # Ensure numeric columns are numeric
    for col in NUMERIC_COLS + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    PROJECT_DIR.joinpath("ml").mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    choices = {
        col: sorted(df[col].dropna().astype(str).unique().tolist())
        for col in CATEGORICAL_COLS
    }
    CHOICES_PATH.write_text(json.dumps(choices, indent=2), encoding="utf-8")

    print("=== TRAINING DONE ===")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved choices: {CHOICES_PATH}")
    print(f"MAE (Euro):  {mae:.2f}")
    print(f"RMSE (Euro): {rmse:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
