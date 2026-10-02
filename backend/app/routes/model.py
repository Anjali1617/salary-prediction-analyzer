"""Routes — model training, info, metrics, comparison."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.services.training import load_metadata, train
from app.services.analytics import get_actual_vs_predicted

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["model"])


@router.get("/model-info")
async def model_info() -> dict[str, Any]:
    meta = load_metadata()
    if meta is None:
        raise HTTPException(
            status_code=404,
            detail="No trained model found. Call POST /api/train first.",
        )
    return meta


@router.get("/model-metrics")
async def model_metrics() -> dict[str, Any]:
    meta = load_metadata()
    if meta is None:
        raise HTTPException(status_code=404, detail="No trained model found.")
    return meta.get("metrics", {})


@router.get("/model-comparison")
async def model_comparison() -> list[dict[str, Any]]:
    meta = load_metadata()
    if meta is None:
        raise HTTPException(status_code=404, detail="No trained model found.")
    return meta.get("comparison", [])


@router.get("/actual-vs-predicted")
async def actual_vs_predicted() -> list[dict[str, Any]]:
    try:
        return get_actual_vs_predicted()
    except Exception as exc:
        logger.exception("actual_vs_predicted failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/train")
async def train_model(push_to_watson: bool = False) -> dict[str, Any]:
    """
    Re-train all candidate models on the current dataset.
    Selects best by R² and saves artifacts.
    Optionally pushes to IBM Watson if push_to_watson=true and credentials are set.
    """
    try:
        metadata = train()
    except Exception as exc:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")

    if push_to_watson:
        try:
            from app.config import MODEL_PATH, PIPELINE_PATH
            from app.services.watson_service import deploy_model_to_watson
            deployment_id = deploy_model_to_watson(
                str(MODEL_PATH), str(PIPELINE_PATH), metadata
            )
            metadata["deployment_id"] = deployment_id
            metadata["deployment_status"] = "watson"
            # Update metadata file
            import json
            from app.config import METADATA_PATH
            with open(METADATA_PATH, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as exc:
            logger.warning(f"Watson deployment skipped: {exc}")
            metadata["watson_deploy_error"] = str(exc)

    return {
        "success": True,
        "message": f"Training complete. Best model: {metadata['model_name']}",
        "model_name": metadata["model_name"],
        "metrics": metadata["metrics"],
        "comparison": metadata["comparison"],
    }
