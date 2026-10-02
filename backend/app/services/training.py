"""
Training service.

Trains multiple regression candidates, evaluates them on held-out test data,
selects the best model by R², and persists artifacts.
"""
from __future__ import annotations

import json
import logging
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from app.config import (
    CSV_PATH,
    FEATURE_COLUMNS,
    METADATA_PATH,
    MODEL_PATH,
    PIPELINE_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    CV_FOLDS,
)
from app.services.preprocessing import (
    build_preprocessing_pipeline,
    save_pipeline,
    validate_dataset,
)

logger = logging.getLogger(__name__)

# ── Candidate models ───────────────────────────────────────────────────────────

CANDIDATE_MODELS: list[tuple[str, Any]] = [
    ("Linear Regression", LinearRegression()),
    ("Ridge Regression", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ("Random Forest", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)),
    (
        "Gradient Boosting",
        GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=RANDOM_STATE),
    ),
    (
        "Extra Trees",
        ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE),
    ),
]

# Attempt XGBoost if available
try:
    from xgboost import XGBRegressor  # type: ignore
    CANDIDATE_MODELS.append(
        ("XGBoost", XGBRegressor(n_estimators=200, learning_rate=0.05,
                                  random_state=RANDOM_STATE, verbosity=0))
    )
    logger.info("XGBoost detected and added to candidates.")
except ImportError:
    logger.info("XGBoost not installed — skipping.")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    n = len(y_true)
    adj_r2 = float(1 - (1 - r2) * (n - 1) / (n - n_features - 1)) if n > n_features + 1 else r2

    # MAPE — avoid divide-by-zero
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) \
        if nonzero.any() else 0.0

    return {
        "mae": mae, "mse": mse, "rmse": rmse,
        "r2": r2, "adjusted_r2": adj_r2, "mape": mape,
    }


def _get_feature_importance(model: Any, feature_names: list[str]) -> list[dict[str, float]]:
    """
    Extract feature importance from tree-based or linear models.
    Aggregates one-hot-encoded features back to original column names.
    """
    importances: np.ndarray | None = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)

    if importances is None or len(importances) != len(feature_names):
        return []

    importance_map: dict[str, float] = {}
    for fname, imp in zip(feature_names, importances):
        # Aggregated back: one-hot feature names look like "Education_Bachelor"
        original = fname.split("_")[0] if "_" in fname else fname
        importance_map[original] = importance_map.get(original, 0.0) + float(imp)

    # Normalise to [0, 1]
    total = sum(importance_map.values())
    if total > 0:
        importance_map = {k: round(v / total, 6) for k, v in importance_map.items()}

    return [
        {"feature": k, "importance": v}
        for k, v in sorted(importance_map.items(), key=lambda x: -x[1])
    ]


# ── Main training function ─────────────────────────────────────────────────────

def train(csv_path: Path = CSV_PATH) -> dict[str, Any]:
    """
    Full training pipeline:
      1. Load & validate data
      2. Split into train / test
      3. Build preprocessing pipeline
      4. Train all candidate models
      5. Evaluate on test set
      6. Select best by R²
      7. Cross-validate best model
      8. Persist model + pipeline + metadata
    Returns metadata dict.
    """
    # ── 1. Load & validate ─────────────────────────────────────────────────────
    logger.info(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    validation_findings = validate_dataset(df)
    logger.info(f"Validation: {validation_findings['n_records']} records, "
                f"{validation_findings['missing_values_total']} missing values")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # ── 2. Train / test split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # ── 3. Preprocessing pipeline ──────────────────────────────────────────────
    preprocessor = build_preprocessing_pipeline()
    preprocessor.fit(X_train)
    save_pipeline(preprocessor, PIPELINE_PATH)

    # Derive feature names after encoding
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_names = cat_encoder.get_feature_names_out(
        input_features=["Education", "Location", "Job_Title", "Gender"]
    ).tolist()
    all_feature_names: list[str] = ["Experience", "Age"] + encoded_cat_names

    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # ── 4. Train all candidates ────────────────────────────────────────────────
    comparison: list[dict[str, Any]] = []
    best_model = None
    best_name = ""
    best_r2 = -np.inf

    for name, model in CANDIDATE_MODELS:
        start = time.time()
        model.fit(X_train_proc, y_train.values)
        elapsed = time.time() - start
        y_pred = model.predict(X_test_proc)
        metrics = _calc_metrics(y_test.values, y_pred, len(all_feature_names))
        logger.info(
            f"{name}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.2f}  "
            f"(trained in {elapsed:.1f}s)"
        )
        comparison.append({"name": name, **metrics})

        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_model = model
            best_name = name

    # ── 5. Cross-validate best model ───────────────────────────────────────────
    logger.info(f"Best model: {best_name} (R²={best_r2:.4f}). Running {CV_FOLDS}-fold CV …")
    X_proc_full = preprocessor.transform(X)
    cv_scores = cross_val_score(
        best_model, X_proc_full, y.values,
        cv=CV_FOLDS, scoring="r2"
    )
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())
    logger.info(f"CV R²: {cv_mean:.4f} ± {cv_std:.4f}")

    # Final metrics on test set
    y_pred_best = best_model.predict(X_test_proc)
    best_metrics = _calc_metrics(y_test.values, y_pred_best, len(all_feature_names))
    best_metrics["cv_mean"] = cv_mean
    best_metrics["cv_std"] = cv_std

    # Feature importance
    feature_importance = _get_feature_importance(best_model, all_feature_names)

    # ── 6. Persist model ───────────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    logger.info(f"Model saved to {MODEL_PATH}")

    # ── 7. Persist metadata ────────────────────────────────────────────────────
    metadata: dict[str, Any] = {
        "model_version": "v1.0",
        "model_name": best_name,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "dataset_name": csv_path.name,
        "target_column": TARGET_COLUMN,
        "feature_list": FEATURE_COLUMNS,
        "train_records": len(X_train),
        "test_records": len(X_test),
        "n_features_encoded": len(all_feature_names),
        "metrics": best_metrics,
        "comparison": comparison,
        "feature_importance": feature_importance,
        "deployment_status": "local",
        "deployment_id": "",
        "all_feature_names": all_feature_names,
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {METADATA_PATH}")

    return metadata


# ── Convenience loader ─────────────────────────────────────────────────────────

def load_metadata() -> dict[str, Any] | None:
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            return json.load(f)
    return None


def load_model() -> Any | None:
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None
