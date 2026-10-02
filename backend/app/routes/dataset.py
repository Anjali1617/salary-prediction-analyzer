"""Routes — dataset summary and CSV upload."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File, status

from app.config import CSV_PATH, MAX_UPLOAD_SIZE_MB, REQUIRED_COLUMNS
from app.services.analytics import get_dataset_summary, get_features
from app.services.preprocessing import validate_dataset
import pandas as pd

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["dataset"])


@router.get("/dataset-summary")
async def dataset_summary() -> dict[str, Any]:
    try:
        return get_dataset_summary()
    except Exception as exc:
        logger.exception("dataset_summary failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/features")
async def features() -> dict[str, Any]:
    try:
        return get_features()
    except Exception as exc:
        logger.exception("features failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/upload-dataset", status_code=status.HTTP_200_OK)
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Upload a replacement CSV dataset.
    Validates schema and structure — does NOT auto-retrain.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_UPLOAD_SIZE_MB} MB limit.",
        )

    # Write to a temp location for validation
    tmp_path = CSV_PATH.parent / "upload_tmp.csv"
    try:
        tmp_path.write_bytes(contents)
        df = pd.read_csv(tmp_path)
        findings = validate_dataset(df)
    except ValueError as ve:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to parse CSV: {exc}")

    # Move to production data path
    shutil.move(str(tmp_path), str(CSV_PATH))
    return {
        "message": "Dataset uploaded and validated successfully.",
        "records": findings["n_records"],
        "columns": findings["n_columns"],
        "missing_values": findings["missing_values_total"],
        "note": "Call POST /api/train to retrain with the new dataset.",
    }
