"""Routes — salary prediction."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from app.schemas import PredictionRequest, PredictionResponse
from app.services.prediction import predict

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_salary(request: PredictionRequest) -> PredictionResponse:
    """
    Accept employee features, run through preprocessing pipeline,
    and return the ML model's salary prediction.
    IBM Watson is used if configured and reachable; otherwise local model.
    """
    try:
        result = predict(request.model_dump())
        return PredictionResponse(**result)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction endpoint error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
