# IBM Watson Salary Predictive Analyzer

> **An end-to-end AI-powered salary prediction application built with IBM watsonx.ai,  
> FastAPI (Python backend), React (frontend), and scikit-learn.**

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Objective](#objective)
3. [Dataset](#dataset)
4. [Machine Learning Approach](#machine-learning-approach)
5. [Preprocessing](#preprocessing)
6. [Models Evaluated](#models-evaluated)
7. [Evaluation Metrics](#evaluation-metrics)
8. [IBM Watson Architecture](#ibm-watson-architecture)
9. [System Architecture](#system-architecture)
10. [Project Structure](#project-structure)
11. [Installation](#installation)
12. [Backend Setup](#backend-setup)
13. [Frontend Setup](#frontend-setup)
14. [IBM Watson Setup](#ibm-watson-setup)
15. [Environment Variables](#environment-variables)
16. [Training](#training)
17. [API Endpoints](#api-endpoints)
18. [Testing](#testing)
19. [How to Review This Project](#how-to-review-this-project)
20. [Limitations](#limitations)
21. [Future Improvements](#future-improvements)

---

## Problem Statement

Many companies and job seekers lack accurate, data-driven salary benchmarks.  
Manual lookup tables are static and do not account for combinations of education,  
experience, location, job title, age, and gender simultaneously.

## Objective

Build an AI-powered Salary Predictive Analyzer that:

- Trains a regression model on a real employee salary dataset
- Deploys the model to IBM Watson Machine Learning for production inference
- Exposes a REST API for salary prediction
- Provides a professional React dashboard with analytics and model metrics

The prediction must come from the trained ML model — predictions are **never hardcoded**.

---

## Dataset

| Property | Value |
|---|---|
| File | `backend/data/salary_data.csv` |
| Records | 249 |
| Columns | 7 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Target variable | **Salary** |

### Columns

| Column | Type | Role |
|---|---|---|
| Education | Categorical | Feature |
| Experience | Numerical | Feature |
| Location | Categorical | Feature |
| Job_Title | Categorical | Feature |
| Age | Numerical | Feature |
| Gender | Categorical | Feature |
| **Salary** | Numerical | **Target** |

### Categorical Values (read dynamically from CSV)

- **Education**: High School, Bachelor, Master, PhD
- **Location**: Urban, Suburban, Rural
- **Job_Title**: Manager, Director, Analyst, Engineer
- **Gender**: Male, Female

---

## Machine Learning Approach

This is a **supervised regression** problem.

```
X = [Education, Experience, Location, Job_Title, Age, Gender]
y = Salary
```

**Pipeline:**

```
salary_data.csv
↓ Data Validation (249 records, 7 columns, 0 missing, 0 duplicates, leakage check)
↓ Train/Test Split  (80 % train / 20 % test, random_state=42)
↓ Preprocessing Pipeline
↓ Multiple Model Training
↓ Evaluation on held-out test set
↓ Best model selected by R²
↓ 5-Fold Cross-Validation on best model
↓ Artifacts saved  (model.pkl, preprocessing_pipeline.pkl, model_metadata.json)
↓ IBM Watson Deployment  (if credentials configured)
↓ FastAPI Prediction API
↓ React Dashboard
```

---

## Preprocessing

A **single reusable `ColumnTransformer` pipeline** is applied identically  
during training, testing, and inference:

| Feature group | Steps |
|---|---|
| Numerical (Experience, Age) | Median imputation → `StandardScaler` |
| Categorical (Education, Location, Job_Title, Gender) | Most-frequent imputation → `OneHotEncoder(handle_unknown='ignore')` |

The pipeline is saved as `backend/models/preprocessing_pipeline.pkl`.  
The same pickle is loaded for every prediction request — no separate preprocessing logic.

---

## Models Evaluated

| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Ridge Regression | L2-regularised linear |
| Random Forest Regressor | 200 estimators |
| Gradient Boosting Regressor | 200 estimators, lr=0.05 |
| Extra Trees Regressor | 200 estimators |
| XGBoost Regressor | Optional — install `xgboost` package |

All models are trained on the **training set only**.  
Evaluated on the **unseen test set**.  
The best model is selected by **highest R²** on the test set.

---

## Evaluation Metrics

All metrics are computed dynamically from actual test-set predictions.  
**No metrics are hardcoded.**

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of Determination |
| Adjusted R² | Accounts for number of features |
| MAPE | Mean Absolute Percentage Error |
| CV R² | 5-Fold Cross-Validation R² |

---

## IBM Watson Architecture

```
Frontend (React)
     ↓  (HTTP — no API key)
Backend (FastAPI)
     ↓  (IAM token + scoring request)
IBM Watson ML Online Deployment
     ↓
Deployed Model
     ↓
Predicted Salary
     ↓
Backend
     ↓
Frontend
```

**IBM credentials are stored only in the backend `.env` file — never exposed to the browser.**

### Fallback behaviour

| Condition | Prediction Engine |
|---|---|
| Watson configured + deployment active | IBM Watson |
| Watson configured + unreachable | Local Model (automatic fallback) |
| Watson not configured | Local Development Model |

The active engine is displayed in the prediction result.

---

## System Architecture

```
salary-predictive-analyzer/
├── backend/                    Python + FastAPI
│   ├── app/
│   │   ├── main.py             FastAPI app, CORS, auto-training on startup
│   │   ├── config.py           All settings from environment variables
│   │   ├── schemas.py          Pydantic request/response models
│   │   ├── routes/             API route handlers
│   │   │   ├── prediction.py   POST /api/predict
│   │   │   ├── analytics.py    GET  /api/analytics
│   │   │   ├── dataset.py      GET  /api/dataset-summary, POST /api/upload-dataset
│   │   │   ├── model.py        GET  /api/model-info, POST /api/train
│   │   │   └── watson.py       GET  /api/watson-status
│   │   └── services/           Business logic
│   │       ├── preprocessing.py  Pipeline builder + data validation
│   │       ├── training.py       Multi-model training + evaluation
│   │       ├── prediction.py     Watson / local routing
│   │       ├── analytics.py      Chart data from CSV
│   │       └── watson_service.py IBM Watson IAM + scoring API
│   ├── data/salary_data.csv
│   ├── models/                 Saved artifacts (gitignored .pkl)
│   ├── tests/test_app.py
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/                   React + Vite
│   ├── src/
│   │   ├── App.jsx             Router + layout
│   │   ├── index.css           IBM Carbon-inspired styles
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── PredictPage.jsx
│   │   │   ├── AnalyticsPage.jsx
│   │   │   ├── ModelPerformancePage.jsx
│   │   │   ├── DatasetPage.jsx
│   │   │   ├── ModelInfoPage.jsx
│   │   │   └── AboutPage.jsx
│   │   └── services/
│   │       ├── api.js          Axios API client
│   │       └── utils.js        Formatters, chart colours
│   ├── package.json
│   ├── vite.config.js
│   └── .env.example
│
├── notebooks/                  (Jupyter exploration notebooks)
├── LICENSE
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10 or 3.11
- Node.js 18 or 20
- npm 9+

### Clone

```bash
git clone <your-repo-url>
cd salary-predictive-analyzer
```

---

## Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Copy and fill in environment variables
cp .env.example .env
# Edit .env — add IBM credentials if available (optional)

# Start the backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will **automatically train the model on first startup** if no saved model is found.  
You will see training logs in the terminal.

Interactive API docs: http://localhost:8000/docs

---

## Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Copy and fill environment (optional — defaults to localhost:8000)
cp .env.example .env

# Start the dev server
npm run dev
```

Open: http://localhost:5173

---

## IBM Watson Setup

### Step 1 — IBM Cloud Account

1. Create a free IBM Cloud account at https://cloud.ibm.com
2. Create an **IBM Watson Studio** project
3. Create a **Watson Machine Learning** service instance

### Step 2 — Get credentials

1. **API Key**: IBM Cloud → Manage → Access (IAM) → API Keys → Create
2. **Project ID**: Watson Studio → Your project → Manage → General → Project ID
3. **Space ID** (optional): Watson Studio → Deployment Spaces → Your space → Manage → Space ID

### Step 3 — Deploy your model

Option A — Use the `/api/train?push_to_watson=true` endpoint after setting credentials  
Option B — Use IBM AutoAI in Watson Studio with the salary CSV  
Option C — Manual deployment via Watson Studio UI

### Step 4 — Get Deployment ID

1. Watson Studio → Deployments → Your deployment → Deployment details → Deployment ID

### Step 5 — Configure `.env`

```env
IBM_API_KEY=your_api_key_here
IBM_PROJECT_ID=your_project_id_here
IBM_SPACE_ID=your_space_id_here
IBM_DEPLOYMENT_ID=your_deployment_id_here
IBM_REGION=us-south
```

Restart the backend. The Watson Status page will show **Connected — Deployment Active**.

---

## Environment Variables

### Backend (`backend/.env`)

| Variable | Required | Description |
|---|---|---|
| `IBM_API_KEY` | For Watson | IBM Cloud API Key |
| `IBM_PROJECT_ID` | For Watson | Watson Studio project ID |
| `IBM_SPACE_ID` | Optional | Deployment space ID |
| `IBM_DEPLOYMENT_ID` | For Watson | Online deployment ID |
| `IBM_REGION` | Optional | Cloud region (default: `us-south`) |
| `CURRENCY` | Optional | `USD`, `INR`, `GBP`, etc. (default: `USD`) |
| `ALLOWED_ORIGINS` | Optional | Comma-separated frontend URLs |

### Frontend (`frontend/.env`)

| Variable | Description |
|---|---|
| `VITE_API_BASE_URL` | Backend URL (default: `http://localhost:8000`) |

**NEVER put `IBM_API_KEY` in frontend code or frontend `.env` committed to Git.**

---

## Training

### Automatic (on startup)

The backend trains automatically if no saved model exists.

### Manual via API

```bash
# Train locally
curl -X POST http://localhost:8000/api/train

# Train and deploy to Watson
curl -X POST "http://localhost:8000/api/train?push_to_watson=true"
```

### Via Dashboard

Go to **Model Performance** → click **Retrain Model**.

### Training output

```
backend/models/
├── salary_model.pkl              Best trained model
├── preprocessing_pipeline.pkl    Fitted preprocessing pipeline
└── model_metadata.json           Metrics, comparison, feature importance
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check + model status |
| `GET` | `/api/dataset-summary` | Dataset statistics |
| `GET` | `/api/features` | Feature dropdown options |
| `GET` | `/api/analytics` | All chart data |
| `GET` | `/api/model-info` | Model metadata + metrics |
| `GET` | `/api/model-metrics` | Evaluation metrics only |
| `GET` | `/api/model-comparison` | All candidate model results |
| `GET` | `/api/actual-vs-predicted` | Actual vs predicted data |
| `GET` | `/api/watson-status` | Real IBM Watson status |
| `POST` | `/api/predict` | **Salary prediction** |
| `POST` | `/api/train` | Retrain all models |
| `POST` | `/api/upload-dataset` | Upload replacement CSV |

### Prediction request example

```json
POST /api/predict
{
  "Education": "Bachelor",
  "Experience": 5.0,
  "Location": "Urban",
  "Job_Title": "Engineer",
  "Age": 28,
  "Gender": "Female"
}
```

### Prediction response example

```json
{
  "predicted_salary": 87432.50,
  "lower_bound": 74432.50,
  "upper_bound": 100432.50,
  "prediction_engine": "Local Development Model",
  "model_name": "Gradient Boosting",
  "model_version": "v1.0",
  "currency": "USD",
  "currency_symbol": "$",
  "input_summary": { ... }
}
```

*The actual numbers come from the trained model — the above are illustrative.*

---

## Testing

```bash
cd backend
pytest tests/ -v
```

Tests cover:

- Dataset loading (249 records, 7 columns)
- Dataset validation (missing values, duplicates, leakage check)
- Preprocessing pipeline (fit/transform, no NaN, unknown categories)
- Metric calculations (MAE, RMSE, R², perfect prediction, constant prediction)
- API health endpoint
- Dataset summary endpoint
- Valid prediction request
- Invalid inputs rejected (age < 18, negative experience, missing fields)
- CSV upload validation (wrong extension, missing columns)
- Watson service (not configured when no credentials)

---

## How to Review This Project

### Demo Steps

1. Open the application (http://localhost:5173 in local development).
2. **Dashboard**: Verify dataset statistics are loaded from the actual CSV (249 records).
3. **Predict Salary**: Enter any combination of inputs and click Predict Salary.
4. Observe: prediction comes from the ML model; engine label shows "Local Development Model" or "IBM Watson".
5. **Analytics**: View all charts — each generated from the actual CSV data.
6. **Model Performance**: Review real evaluation metrics (R², RMSE, MAE, MSE, MAPE).
7. **Model Information**: Review IBM Watson status — shows "Not Configured" if `.env` is not set.
8. **Dataset**: Inspect dataset schema, salary statistics, upload a new CSV.

### Verification Checklist

- [ ] All metric values match the training logs printed on backend startup
- [ ] Dropdown options in Predict form match actual CSV categorical values
- [ ] Watson status shows "Not Configured" when no IBM credentials are provided
- [ ] Watson status shows "Connected" when valid IBM credentials are provided
- [ ] Prediction changes when inputs change
- [ ] `POST /api/predict` with invalid age returns HTTP 422

---

## Limitations

1. **Dataset size**: 249 records is small. Model accuracy may improve significantly with more data.
2. **Currency**: Salary values in the dataset have no explicit currency. The currency symbol is configurable via `CURRENCY` env var and defaults to USD. Do not assume the raw values are any specific currency.
3. **Prediction interval**: The confidence interval is based on RMSE (not a true Bayesian interval).
4. **Gender handling**: Gender is used as a statistical feature. No causal claim about gender and salary is made.
5. **IBM Watson AutoAI**: Full AutoAI integration requires the `ibm-watson-machine-learning` SDK and a paid/trial IBM Cloud account.

---

## Future Improvements

1. Add more records to the dataset to improve model accuracy.
2. Integrate IBM AutoAI pipeline directly in the training service.
3. Add SHAP explainability for per-prediction feature contributions.
4. Add authentication/authorisation to the admin endpoints.
5. Deploy the frontend to IBM Cloud Static Web Apps or Vercel.
6. Add real-time retraining notifications via WebSocket.
7. Support multi-currency display without configuration change.
8. Add A/B testing between local model and Watson deployment.
9. Add data drift detection to alert when new predictions fall outside training distribution.

---

## Author

Built as part of a B.Tech CSE project demonstrating end-to-end MLOps with IBM Watson.  
Technology stack: Python · scikit-learn · FastAPI · React · IBM watsonx.ai
