"""
Application configuration using environment variables.
Credentials are NEVER hardcoded — loaded from .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the backend directory
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CSV_PATH = DATA_DIR / "salary_data.csv"

MODEL_PATH = MODELS_DIR / "salary_model.pkl"
PIPELINE_PATH = MODELS_DIR / "preprocessing_pipeline.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# ── IBM Watson / watsonx.ai credentials ───────────────────────────────────────
IBM_API_KEY: str = os.getenv("IBM_API_KEY", "")
IBM_PROJECT_ID: str = os.getenv("IBM_PROJECT_ID", "")
IBM_SPACE_ID: str = os.getenv("IBM_SPACE_ID", "")
IBM_DEPLOYMENT_ID: str = os.getenv("IBM_DEPLOYMENT_ID", "")
IBM_REGION: str = os.getenv("IBM_REGION", "us-south")
IBM_WML_URL: str = os.getenv(
    "IBM_WML_URL", f"https://{IBM_REGION}.ml.cloud.ibm.com"
)

# ── Dataset schema ─────────────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "Education", "Experience", "Location", "Job_Title", "Age", "Gender", "Salary"
]
FEATURE_COLUMNS = ["Education", "Experience", "Location", "Job_Title", "Age", "Gender"]
TARGET_COLUMN = "Salary"
NUMERICAL_FEATURES = ["Experience", "Age"]
CATEGORICAL_FEATURES = ["Education", "Location", "Job_Title", "Gender"]

# ── Currency display ───────────────────────────────────────────────────────────
# Change this to any ISO currency code (USD, INR, GBP, EUR …).
# Frontend will use this symbol for display only.
CURRENCY: str = os.getenv("CURRENCY", "USD")

CURRENCY_SYMBOLS = {
    "USD": "$", "INR": "₹", "GBP": "£", "EUR": "€", "JPY": "¥",
}
CURRENCY_SYMBOL: str = CURRENCY_SYMBOLS.get(CURRENCY, CURRENCY)

# ── Model training ─────────────────────────────────────────────────────────────
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5

# ── CORS ───────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS: list[str] = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000"
).split(",")

# ── Upload limits ──────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB: int = 10

def watson_configured() -> bool:
    """Return True only when all essential Watson credentials are present."""
    return bool(IBM_API_KEY and IBM_PROJECT_ID and IBM_DEPLOYMENT_ID)
