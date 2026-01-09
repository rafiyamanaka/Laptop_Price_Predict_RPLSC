from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
DATASET_PATH = WORKSPACE_DIR / "laptop_price - dataset.csv"
MODEL_PATH = PROJECT_DIR / "ml" / "model.joblib"
CHOICES_PATH = PROJECT_DIR / "ml" / "choices.json"

MODEL: Any | None = None
CHOICES: dict[str, list[str]] | None = None
CATALOG: dict[str, Any] | None = None


def load_artifacts() -> None:
    global MODEL, CHOICES

    if MODEL is None:
        if MODEL_PATH.exists():
            MODEL = joblib.load(MODEL_PATH)
            logger.info("Loaded model from %s", MODEL_PATH)
            print(f"[predictor] Loaded model from: {MODEL_PATH}")
        else:
            logger.warning("Model file not found at %s", MODEL_PATH)

    if CHOICES is None:
        if CHOICES_PATH.exists():
            CHOICES = json.loads(CHOICES_PATH.read_text(encoding="utf-8"))
        else:
            CHOICES = {}


def load_catalog() -> None:
    global CATALOG

    if CATALOG is not None:
        return

    if not DATASET_PATH.exists():
        CATALOG = {
            "dataset_path": str(DATASET_PATH),
            "error": "Dataset file not found.",
            "companies": [],
            "company_count": 0,
            "row_count": 0,
        }
        return

    df = pd.read_csv(DATASET_PATH, usecols=["Company", "Product", "TypeName"])
    df = df.dropna(subset=["Company"]).copy()

    companies = []
    for company, g in df.groupby("Company", sort=True):
        products = (
            g["Product"].dropna().astype(str).unique().tolist()
            if "Product" in g.columns
            else []
        )
        products = sorted(products)
        types_ = (
            g["TypeName"].dropna().astype(str).unique().tolist()
            if "TypeName" in g.columns
            else []
        )
        types_ = sorted(types_)
        companies.append(
            {
                "name": str(company),
                "product_count": len(products),
                "type_count": len(types_),
                "sample_products": products[:5],
                "sample_types": types_[:5],
            }
        )

    CATALOG = {
        "dataset_path": str(DATASET_PATH),
        "row_count": int(len(df)),
        "company_count": int(len(companies)),
        "companies": companies,
    }


def get_model():
    if MODEL is None:
        load_artifacts()
    return MODEL


def get_choices() -> dict[str, list[str]]:
    if CHOICES is None:
        load_artifacts()
    return CHOICES or {}


def get_catalog() -> dict[str, Any]:
    if CATALOG is None:
        load_catalog()
    return CATALOG or {}
