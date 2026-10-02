"""
Prediction service.

Routing logic:
  IF Watson is configured and its deployment is active  → use Watson
  ELSE                                                  → use local model

Prediction intervals are computed using the std of residuals from
training metadata — a statistically defensible estimate.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from app.config import CURRENCY, CURRENCY_SYMBOL, FEATURE_COLUMNS
from app.services.preprocessing import load_pipeline, request_to_dataframe
from app.services.training import load_metadata, load_model
from app.services.watson_service import check_watson_status, predict_with_watson

logger = logging.getLogger(__name__)

# Z-score for ~90 % prediction interval
_Z90 = 1.645


def predict(request_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Main prediction entry point.

    1. Build single-row DataFrame from request.
    2. Apply the stored preprocessing pipeline.
    3. Route to Watson or local model.
    4. Compute prediction interval if possible.
    5. Return structured result.
    """
    metadata = load_metadata()
    if metadata is None:
        raise RuntimeError(
            "No trained model found. "
            "Please call POST /api/train first to train the model."
        )

    # ── Determine prediction engine ────────────────────────────────────────────
    watson_status = check_watson_status()
    use_watson = watson_status["configured"] and watson_status["deployment_active"]

    # ── Build input DataFrame ──────────────────────────────────────────────────
    input_df = request_to_dataframe(request_dict)

    # ── Preprocess ────────────────────────────────────────────────────────────
    pipeline = load_pipeline()
    X_proc = pipeline.transform(input_df)

    # ── Predict ────────────────────────────────────────────────────────────────
    if use_watson:
        try:
            predicted_salary = predict_with_watson(X_proc.tolist())
            engine = "IBM Watson"
        except Exception as exc:
            logger.warning(f"Watson prediction failed ({exc}); falling back to local model.")
            model = load_model()
            if model is None:
                raise RuntimeError("Local model not available.")
            predicted_salary = float(model.predict(X_proc)[0])
            engine = "Local Model (Watson fallback)"
    else:
        model = load_model()
        if model is None:
            raise RuntimeError("Local model not available. Please train first.")
        predicted_salary = float(model.predict(X_proc)[0])
        engine = "Local Development Model"

    # ── Prediction interval (based on RMSE from test evaluation) ──────────────
    lower_bound: float | None = None
    upper_bound: float | None = None

    metrics = metadata.get("metrics", {})
    rmse = metrics.get("rmse")
    if rmse and rmse > 0:
        margin = _Z90 * rmse
        lower_bound = round(predicted_salary - margin, 2)
        upper_bound = round(predicted_salary + margin, 2)

    return {
        "predicted_salary": round(predicted_salary, 2),
        "lower_bound": round(lower_bound, 2) if lower_bound is not None else None,
        "upper_bound": round(upper_bound, 2) if upper_bound is not None else None,
        "prediction_engine": engine,
        "model_name": metadata.get("model_name", "Unknown"),
        "model_version": metadata.get("model_version", "v1.0"),
        "currency": CURRENCY,
        "currency_symbol": CURRENCY_SYMBOL,
        "input_summary": {k: request_dict[k] for k in FEATURE_COLUMNS},
    }
