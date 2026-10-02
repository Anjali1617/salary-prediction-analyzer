"""
FastAPI application entry point.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import ALLOWED_ORIGINS, CSV_PATH, MODEL_PATH, METADATA_PATH
from app.routes import analytics, dataset, model, prediction, watson

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Startup: auto-train if no saved model ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        logger.info("No saved model found — running initial training …")
        try:
            from app.services.training import train
            meta = train()
            logger.info(
                f"Initial training complete. "
                f"Best model: {meta['model_name']}, R²={meta['metrics']['r2']:.4f}"
            )
        except Exception as exc:
            logger.error(f"Auto-training failed: {exc}")
    else:
        from app.services.training import load_metadata
        meta = load_metadata()
        if meta:
            logger.info(
                f"Loaded existing model: {meta['model_name']} "
                f"(R²={meta['metrics']['r2']:.4f})"
            )
    yield


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="IBM Watson Salary Predictive Analyzer",
    description=(
        "AI-powered salary prediction API. "
        "Uses IBM Watson ML when configured; local scikit-learn model as fallback."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(prediction.router)
app.include_router(analytics.router)
app.include_router(dataset.router)
app.include_router(model.router)
app.include_router(watson.router)


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["health"])
async def health():
    model_ready = MODEL_PATH.exists() and METADATA_PATH.exists()
    dataset_ready = CSV_PATH.exists()
    return {
        "status": "ok",
        "model_ready": model_ready,
        "dataset_ready": dataset_ready,
        "dataset_path": str(CSV_PATH),
    }


# ── Root ───────────────────────────────────────────────────────────────────────

@app.get("/", tags=["root"])
async def root():
    return {
        "name": "IBM Watson Salary Predictive Analyzer",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }
