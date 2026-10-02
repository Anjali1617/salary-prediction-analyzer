"""
Pydantic schemas — request / response data models.
"""
from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Prediction request ─────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    Education: str = Field(..., description="Highest education level")
    Experience: float = Field(..., ge=0, le=50, description="Years of experience")
    Location: str = Field(..., description="Work location type")
    Job_Title: str = Field(..., description="Job title")
    Age: int = Field(..., ge=18, le=80, description="Age in years")
    Gender: str = Field(..., description="Gender")

    @field_validator("Experience")
    @classmethod
    def experience_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Experience must be ≥ 0")
        return v

    @field_validator("Age")
    @classmethod
    def age_range(cls, v: int) -> int:
        if not (18 <= v <= 80):
            raise ValueError("Age must be between 18 and 80")
        return v


# ── Prediction response ────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predicted_salary: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    prediction_engine: str          # "IBM Watson" | "Local Model"
    model_name: str
    model_version: str
    currency: str
    currency_symbol: str
    input_summary: dict[str, Any]


# ── Dataset summary ────────────────────────────────────────────────────────────

class DatasetSummary(BaseModel):
    total_records: int
    total_columns: int
    feature_columns: list[str]
    target_column: str
    missing_values: int
    duplicate_records: int
    salary_mean: float
    salary_median: float
    salary_min: float
    salary_max: float
    salary_std: float
    categorical_uniques: dict[str, list[str]]
    numerical_ranges: dict[str, dict[str, float]]


# ── Model metrics ──────────────────────────────────────────────────────────────

class ModelMetrics(BaseModel):
    mae: float
    mse: float
    rmse: float
    r2: float
    mape: Optional[float] = None
    adjusted_r2: Optional[float] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None


class ModelComparison(BaseModel):
    name: str
    mae: float
    mse: float
    rmse: float
    r2: float


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    model_name: str
    training_date: str
    dataset_name: str
    target_column: str
    feature_list: list[str]
    train_records: int
    test_records: int
    metrics: ModelMetrics
    comparison: list[ModelComparison]
    deployment_status: str
    deployment_id: str


# ── Analytics ──────────────────────────────────────────────────────────────────

class AnalyticsData(BaseModel):
    salary_distribution: list[dict[str, Any]]
    salary_by_education: list[dict[str, Any]]
    salary_by_job_title: list[dict[str, Any]]
    salary_by_location: list[dict[str, Any]]
    salary_by_gender: list[dict[str, Any]]
    salary_vs_experience: list[dict[str, Any]]
    salary_vs_age: list[dict[str, Any]]
    feature_importance: list[dict[str, Any]]


# ── Watson status ──────────────────────────────────────────────────────────────

class WatsonStatus(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    configured: bool
    connected: bool
    deployment_active: bool
    status_message: str
    deployment_id_masked: str
    model_name: str
    region: str


# ── Training request ───────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    dataset_path: Optional[str] = None   # use default CSV if None
    push_to_watson: bool = False


class TrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    success: bool
    message: str
    model_name: str
    metrics: Optional[ModelMetrics] = None
    comparison: Optional[list[ModelComparison]] = None


# ── Features ───────────────────────────────────────────────────────────────────

class FeaturesResponse(BaseModel):
    Education: list[str]
    Location: list[str]
    Job_Title: list[str]
    Gender: list[str]
    Experience_range: dict[str, float]
    Age_range: dict[str, float]
