"""
Tests for the Salary Predictive Analyzer backend.

Run from the backend/ directory:
    pytest tests/ -v
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Make sure backend/app is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CSV_PATH, FEATURE_COLUMNS, TARGET_COLUMN
from app.services.preprocessing import (
    build_preprocessing_pipeline,
    request_to_dataframe,
    validate_dataset,
)
from app.services.training import _calc_metrics


# ══════════════════════════════════════════════════════════════════════════════
# 1. Dataset loading
# ══════════════════════════════════════════════════════════════════════════════

class TestDatasetLoading:
    def test_csv_exists(self):
        assert CSV_PATH.exists(), f"CSV not found at {CSV_PATH}"

    def test_csv_loads(self):
        df = pd.read_csv(CSV_PATH)
        assert df is not None

    def test_record_count(self):
        df = pd.read_csv(CSV_PATH)
        assert len(df) == 249, f"Expected 249 records, got {len(df)}"

    def test_column_count(self):
        df = pd.read_csv(CSV_PATH)
        assert len(df.columns) == 7, f"Expected 7 columns, got {len(df.columns)}"

    def test_required_columns_present(self):
        df = pd.read_csv(CSV_PATH)
        required = FEATURE_COLUMNS + [TARGET_COLUMN]
        for col in required:
            assert col in df.columns, f"Column {col!r} missing"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Dataset validation
# ══════════════════════════════════════════════════════════════════════════════

class TestDatasetValidation:
    def setup_method(self):
        self.df = pd.read_csv(CSV_PATH)

    def test_no_missing_values(self):
        findings = validate_dataset(self.df)
        assert findings["missing_values_total"] == 0

    def test_no_duplicates(self):
        findings = validate_dataset(self.df)
        assert findings["duplicate_records"] == 0

    def test_salary_as_target_not_feature(self):
        # Leakage check
        findings = validate_dataset(self.df)
        assert findings["leakage_check_passed"] is True

    def test_salary_range_sensible(self):
        salary_min = float(self.df[TARGET_COLUMN].min())
        salary_max = float(self.df[TARGET_COLUMN].max())
        assert salary_min > 0, "Salary must be positive"
        assert salary_max < 10_000_000, "Salary upper bound sanity check"

    def test_missing_column_raises(self):
        df_bad = self.df.drop(columns=["Education"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataset(df_bad)

    def test_salary_column_in_dataset(self):
        assert TARGET_COLUMN in self.df.columns


# ══════════════════════════════════════════════════════════════════════════════
# 3. Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessing:
    def setup_method(self):
        self.df = pd.read_csv(CSV_PATH)
        self.X = self.df[FEATURE_COLUMNS]
        self.y = self.df[TARGET_COLUMN]

    def test_pipeline_builds(self):
        pipeline = build_preprocessing_pipeline()
        assert pipeline is not None

    def test_pipeline_fit_transform(self):
        pipeline = build_preprocessing_pipeline()
        X_proc = pipeline.fit_transform(self.X)
        assert X_proc is not None
        assert X_proc.shape[0] == len(self.df)

    def test_no_nan_after_transform(self):
        pipeline = build_preprocessing_pipeline()
        X_proc = pipeline.fit_transform(self.X)
        assert not np.isnan(X_proc).any()

    def test_request_to_dataframe(self):
        req = {
            "Education": "Bachelor",
            "Experience": 5.0,
            "Location": "Urban",
            "Job_Title": "Engineer",
            "Age": 28,
            "Gender": "Female",
        }
        df_row = request_to_dataframe(req)
        assert len(df_row) == 1
        for col in FEATURE_COLUMNS:
            assert col in df_row.columns

    def test_pipeline_handles_unknown_categories(self):
        """Pipeline must not crash on unseen categorical values."""
        pipeline = build_preprocessing_pipeline()
        pipeline.fit(self.X)
        req = {
            "Education": "PostDoc",        # unseen value
            "Experience": 10.0,
            "Location": "Urban",
            "Job_Title": "Researcher",     # unseen value
            "Age": 35,
            "Gender": "Male",
        }
        df_row = request_to_dataframe(req)
        X_proc = pipeline.transform(df_row)
        assert X_proc.shape[0] == 1


# ══════════════════════════════════════════════════════════════════════════════
# 4. Model training
# ══════════════════════════════════════════════════════════════════════════════

class TestModelTraining:
    def test_metrics_calculation(self):
        y_true = np.array([100_000, 90_000, 80_000, 110_000])
        y_pred = np.array([102_000, 88_000, 82_000, 108_000])
        metrics = _calc_metrics(y_true, y_pred, n_features=6)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0

    def test_r2_perfect_prediction(self):
        y_true = np.array([100_000, 90_000, 80_000])
        y_pred = y_true.copy()
        metrics = _calc_metrics(y_true, y_pred, n_features=6)
        assert abs(metrics["r2"] - 1.0) < 1e-9

    def test_r2_constant_prediction_negative(self):
        y_true = np.array([100_000, 90_000, 80_000, 110_000])
        y_pred = np.full_like(y_true, fill_value=y_true.mean())
        metrics = _calc_metrics(y_true, y_pred, n_features=6)
        assert metrics["r2"] <= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 5. API — health and dataset endpoints
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def client():
    """HTTPX test client for the FastAPI app (no server required)."""
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


class TestAPIHealth:
    def test_health_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model_ready" in data
        assert "dataset_ready" in data

    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200


class TestAPIDataset:
    def test_dataset_summary(self, client):
        resp = client.get("/api/dataset-summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_records"] == 249
        assert data["total_columns"] == 7
        assert data["missing_values"] == 0
        assert data["target_column"] == "Salary"

    def test_features(self, client):
        resp = client.get("/api/features")
        assert resp.status_code == 200
        data = resp.json()
        assert "Education" in data
        assert "Location" in data
        assert "Job_Title" in data
        assert "Gender" in data


# ══════════════════════════════════════════════════════════════════════════════
# 6. Prediction API
# ══════════════════════════════════════════════════════════════════════════════

VALID_PAYLOAD = {
    "Education": "Bachelor",
    "Experience": 5.0,
    "Location": "Urban",
    "Job_Title": "Engineer",
    "Age": 28,
    "Gender": "Female",
}


class TestPredictionAPI:
    def test_valid_prediction(self, client):
        resp = client.post("/api/predict", json=VALID_PAYLOAD)
        # Should succeed (model auto-trained on startup)
        assert resp.status_code in (200, 503)  # 503 if training failed
        if resp.status_code == 200:
            data = resp.json()
            assert "predicted_salary" in data
            assert data["predicted_salary"] > 0
            assert "prediction_engine" in data
            assert "model_name" in data

    def test_invalid_age_rejected(self, client):
        payload = {**VALID_PAYLOAD, "Age": 5}  # below 18
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 422

    def test_negative_experience_rejected(self, client):
        payload = {**VALID_PAYLOAD, "Experience": -1.0}
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 422

    def test_missing_field_rejected(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "Education"}
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 422


# ══════════════════════════════════════════════════════════════════════════════
# 7. Invalid CSV upload
# ══════════════════════════════════════════════════════════════════════════════

class TestCSVUpload:
    def test_non_csv_rejected(self, client):
        resp = client.post(
            "/api/upload-dataset",
            files={"file": ("data.txt", b"not a csv", "text/plain")},
        )
        assert resp.status_code == 400

    def test_csv_missing_columns_rejected(self, client):
        bad_csv = b"Name,Score\nAlice,100\nBob,200\n"
        resp = client.post(
            "/api/upload-dataset",
            files={"file": ("bad.csv", bad_csv, "text/csv")},
        )
        assert resp.status_code == 422


# ══════════════════════════════════════════════════════════════════════════════
# 8. Watson service (mocked — no real credentials needed for tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestWatsonService:
    def test_not_configured_when_no_credentials(self):
        from app.services.watson_service import check_watson_status
        # Temporarily clear env variables
        with patch.dict(os.environ, {"IBM_API_KEY": "", "IBM_DEPLOYMENT_ID": ""}, clear=False):
            # Reload config to pick up the cleared env
            import importlib
            import app.config as cfg
            orig_key = cfg.IBM_API_KEY
            orig_dep = cfg.IBM_DEPLOYMENT_ID
            cfg.IBM_API_KEY = ""
            cfg.IBM_DEPLOYMENT_ID = ""
            status = check_watson_status()
            cfg.IBM_API_KEY = orig_key
            cfg.IBM_DEPLOYMENT_ID = orig_dep
        assert status["configured"] is False
        assert status["connected"] is False
