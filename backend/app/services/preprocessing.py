"""
Preprocessing service.

Builds a single reusable scikit-learn Pipeline so that IDENTICAL
transformations are applied during training, testing, and inference.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from app.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    PIPELINE_PATH,
)

logger = logging.getLogger(__name__)


# ── Pipeline builder ───────────────────────────────────────────────────────────

def build_preprocessing_pipeline() -> Pipeline:
    """
    Build and return a fresh ColumnTransformer pipeline.
    Numerical:   median imputation → StandardScaler
    Categorical: most_frequent imputation → OneHotEncoder (handle_unknown='ignore')
    """
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def save_pipeline(pipeline: Any, path: Path = PIPELINE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"Preprocessing pipeline saved to {path}")


def load_pipeline(path: Path = PIPELINE_PATH) -> Any:
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    logger.info(f"Preprocessing pipeline loaded from {path}")
    return pipeline


# ── Data validation ────────────────────────────────────────────────────────────

def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run a suite of data-quality checks.
    Returns a dict of findings. Raises ValueError on critical issues.
    """
    from app.config import REQUIRED_COLUMNS

    findings: dict[str, Any] = {}

    # Column presence
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    findings["required_columns_present"] = True

    # Shape
    findings["n_records"] = len(df)
    findings["n_columns"] = len(df.columns)

    # Missing values
    findings["missing_values_total"] = int(df.isnull().sum().sum())
    findings["missing_per_column"] = df.isnull().sum().to_dict()

    # Duplicates
    findings["duplicate_records"] = int(df.duplicated().sum())

    # Data types
    findings["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}

    # Categorical uniques
    for col in CATEGORICAL_FEATURES:
        findings[f"unique_{col}"] = sorted(df[col].dropna().unique().tolist())

    # Numerical ranges
    for col in NUMERICAL_FEATURES + [TARGET_COLUMN]:
        findings[f"range_{col}"] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
        }

    # Salary outlier detection (IQR method)
    q1 = df[TARGET_COLUMN].quantile(0.25)
    q3 = df[TARGET_COLUMN].quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (df[TARGET_COLUMN] < q1 - 1.5 * iqr) | (df[TARGET_COLUMN] > q3 + 1.5 * iqr)
    findings["salary_outliers_iqr"] = int(outlier_mask.sum())

    # Leakage check: target not in features
    feature_set = set(FEATURE_COLUMNS)
    if TARGET_COLUMN in feature_set:
        raise ValueError("DATA LEAKAGE: Salary is present in feature columns!")
    findings["leakage_check_passed"] = True

    return findings


# ── Feature record builder ─────────────────────────────────────────────────────

def request_to_dataframe(req_dict: dict[str, Any]) -> pd.DataFrame:
    """
    Convert a prediction request dict into a single-row DataFrame
    with exactly the same column order expected by the pipeline.
    """
    row = {col: [req_dict[col]] for col in FEATURE_COLUMNS}
    df = pd.DataFrame(row)
    # Ensure correct dtypes
    df["Experience"] = df["Experience"].astype(float)
    df["Age"] = df["Age"].astype(int)
    return df
