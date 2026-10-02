"""
Analytics service.

Computes all chart data directly from the CSV dataset.
NO hardcoded values.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from app.config import (
    CATEGORICAL_FEATURES,
    CSV_PATH,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from app.services.training import load_metadata

logger = logging.getLogger(__name__)


def _load_df() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


# ── Dataset summary ────────────────────────────────────────────────────────────

def get_dataset_summary() -> dict[str, Any]:
    df = _load_df()
    cat_uniques = {col: sorted(df[col].dropna().unique().tolist()) for col in CATEGORICAL_FEATURES}
    num_ranges = {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
        }
        for col in NUMERICAL_FEATURES
    }
    return {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_records": int(df.duplicated().sum()),
        "salary_mean": round(float(df[TARGET_COLUMN].mean()), 2),
        "salary_median": round(float(df[TARGET_COLUMN].median()), 2),
        "salary_min": round(float(df[TARGET_COLUMN].min()), 2),
        "salary_max": round(float(df[TARGET_COLUMN].max()), 2),
        "salary_std": round(float(df[TARGET_COLUMN].std()), 2),
        "categorical_uniques": cat_uniques,
        "numerical_ranges": num_ranges,
    }


def get_features() -> dict[str, Any]:
    df = _load_df()
    return {
        "Education": sorted(df["Education"].dropna().unique().tolist()),
        "Location": sorted(df["Location"].dropna().unique().tolist()),
        "Job_Title": sorted(df["Job_Title"].dropna().unique().tolist()),
        "Gender": sorted(df["Gender"].dropna().unique().tolist()),
        "Experience_range": {
            "min": float(df["Experience"].min()),
            "max": float(df["Experience"].max()),
        },
        "Age_range": {
            "min": float(df["Age"].min()),
            "max": float(df["Age"].max()),
        },
    }


# ── Chart data builders ────────────────────────────────────────────────────────

def _salary_distribution(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Histogram bins for salary distribution."""
    counts, bin_edges = np.histogram(df[TARGET_COLUMN], bins=20)
    return [
        {
            "bin_start": round(float(bin_edges[i]), 2),
            "bin_end": round(float(bin_edges[i + 1]), 2),
            "label": f"{bin_edges[i]:,.0f}–{bin_edges[i+1]:,.0f}",
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]


def _group_stats(df: pd.DataFrame, group_col: str) -> list[dict[str, Any]]:
    """Box-plot stats per group."""
    result = []
    for group, sub in df.groupby(group_col)[TARGET_COLUMN]:
        q1, median, q3 = sub.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        whisker_low = max(sub.min(), q1 - 1.5 * iqr)
        whisker_high = min(sub.max(), q3 + 1.5 * iqr)
        result.append({
            "group": group,
            "min": round(float(whisker_low), 2),
            "q1": round(float(q1), 2),
            "median": round(float(median), 2),
            "q3": round(float(q3), 2),
            "max": round(float(whisker_high), 2),
            "mean": round(float(sub.mean()), 2),
            "count": int(len(sub)),
        })
    return sorted(result, key=lambda x: -x["median"])


def _scatter(df: pd.DataFrame, x_col: str) -> list[dict[str, Any]]:
    """Scatter data for numeric vs salary."""
    return [
        {x_col.lower(): float(row[x_col]), "salary": round(float(row[TARGET_COLUMN]), 2)}
        for _, row in df[[x_col, TARGET_COLUMN]].iterrows()
    ]


# ── Feature importance from metadata ──────────────────────────────────────────

def _feature_importance() -> list[dict[str, Any]]:
    meta = load_metadata()
    if meta and "feature_importance" in meta:
        return meta["feature_importance"]
    return []


# ── Aggregated analytics endpoint ─────────────────────────────────────────────

def get_analytics() -> dict[str, Any]:
    df = _load_df()
    return {
        "salary_distribution": _salary_distribution(df),
        "salary_by_education": _group_stats(df, "Education"),
        "salary_by_job_title": _group_stats(df, "Job_Title"),
        "salary_by_location": _group_stats(df, "Location"),
        "salary_by_gender": _group_stats(df, "Gender"),
        "salary_vs_experience": _scatter(df, "Experience"),
        "salary_vs_age": _scatter(df, "Age"),
        "feature_importance": _feature_importance(),
    }


# ── Model performance: actual vs predicted + residuals ────────────────────────

def get_actual_vs_predicted() -> list[dict[str, Any]]:
    """
    Re-runs predictions on the full dataset using the saved model and pipeline
    to produce Actual vs Predicted data for the performance chart.
    """
    from app.services.preprocessing import load_pipeline
    from app.services.training import load_model

    pipeline = load_pipeline()
    model = load_model()
    if pipeline is None or model is None:
        return []

    df = _load_df()
    X = df[FEATURE_COLUMNS]
    y_true = df[TARGET_COLUMN].values
    X_proc = pipeline.transform(X)
    y_pred = model.predict(X_proc)

    return [
        {
            "actual": round(float(a), 2),
            "predicted": round(float(p), 2),
            "residual": round(float(a - p), 2),
        }
        for a, p in zip(y_true, y_pred)
    ]
