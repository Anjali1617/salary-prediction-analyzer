"""Routes — analytics charts."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.services.analytics import get_analytics

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["analytics"])


@router.get("/analytics")
async def analytics() -> dict[str, Any]:
    try:
        return get_analytics()
    except Exception as exc:
        logger.exception("Analytics endpoint error")
        raise HTTPException(status_code=500, detail=str(exc))
