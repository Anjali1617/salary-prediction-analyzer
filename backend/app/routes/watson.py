"""Routes — IBM Watson status and deployment."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.services.watson_service import check_watson_status

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["watson"])


@router.get("/watson-status")
async def watson_status() -> dict[str, Any]:
    """
    Returns the REAL Watson connection/deployment status.
    Never fakes the connection — reflects actual credential checks.
    """
    try:
        return check_watson_status()
    except Exception as exc:
        logger.exception("watson_status failed")
        raise HTTPException(status_code=500, detail=str(exc))
