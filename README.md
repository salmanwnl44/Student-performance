<div align="center">

# EduPulse — Student Performance Prediction System

**AI-powered dropout risk assessment and academic intervention engine**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

EduPulse is a production-grade machine learning system that predicts student dropout risk using academic, behavioral, and socioeconomic data. It trains and compares three gradient-boosted and ensemble models, exposes predictions through a REST API, and presents results in an interactive analytics dashboard with personalized intervention recommendations.

The system was designed for university administrators, academic advisors, and student support teams who need early warning signals to intervene before students fall through the cracks.

### Key Capabilities

- **Multi-model training** — Random Forest, XGBoost, and LightGBM trained in a single pipeline run, with automatic best-model selection by F1 score
- **50,000-record synthetic dataset** — Realistic correlated features driven by a latent "academic strength" score, with injected missing values for robustness testing
- **REST API** — FastAPI endpoints for single and batch prediction with confidence scores
- **Interactive dashboard** — Streamlit-based analytics UI with radar charts, risk gauges, model comparison, correlation heatmaps, and personalized action plans
- **Containerized** — Docker-ready with a single `docker build` command

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models & Performance](#models--performance)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Dashboard](#dashboard)
- [Pipeline Details](#pipeline-details)
- [Configuration](#configuration)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EduPulse System                          │
├─────────────┬──────────────┬──────────────┬─────────────────────┤
│  Data Layer │  ML Pipeline │   API Layer  │   Presentation      │
├─────────────┼──────────────┼──────────────┼─────────────────────┤
│ generate_   │ DataLoader   │ FastAPI app  │ Streamlit dashboard  │
│  data.py    │ DataValidator│ /predict/*   │ Radar charts         │
│             │ DataCleaner  │ /health      │ Risk gauges          │
│ CSV storage │ DataEncoder  │ /metrics     │ Model comparison     │
│             │ ModelTrainer │ /batch       │ Correlation matrix   │
│             │ ModelEval    │              │ Action plans          │
└─────────────┴──────────────┴──────────────┴─────────────────────┘
```

**Data flows left to right:**

1. `generate_data.py` creates 50,000 synthetic student records → CSV
2. `train_pipeline.py` orchestrates ingestion → validation → cleaning → encoding → training → evaluation
3. Trained models (`.pkl`) and metadata (`.json`) are saved to `models/`
4. The FastAPI server loads models on demand and serves predictions
5. The Streamlit dashboard consumes the API and renders interactive analytics

---

## Project Structure

```
student-performance-ml/
│
├── api/                          # REST API
│   ├── __init__.py
│   └── main.py                   # FastAPI app with /predict, /batch, /health, /metrics
│
├── dashboard/
│   └── app.py                    # Streamlit analytics dashboard (3 tabs)
│
├── data/
│   ├── generate_data.py          # Synthetic data generator (50k records, 24 columns)
│   └── raw/
│       └── synthetic_student_data.csv
│
├── models/                       # Trained model artifacts
│   ├── rf_dropout_model.pkl      # Random Forest (~55 MB)
│   ├── rf_dropout_model_meta.json
│   ├── xgb_dropout_model.pkl     # XGBoost (~2 MB)
│   ├── xgb_dropout_model_meta.json
│   ├── lgbm_dropout_model.pkl    # LightGBM (~700 KB)
│   ├── lgbm_dropout_model_meta.json
│   └── training_metrics.json     # Combined metrics for all 3 models
│
├── scripts/
│   └── train_pipeline.py         # End-to-end training orchestrator
│
├── src/                          # Core ML modules
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── data_loader.py        # CSV/DB/API data loading
│   ├── validation/
│   │   ├── __init__.py
│   │   └── data_validator.py     # Vectorized schema validation
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py            # Missing values, outlier removal (chained)
│   │   └── encoder.py            # Label encoding, standard scaling
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py            # Multi-model trainer (RF, XGB, LGBM)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py          # Metrics: accuracy, precision, recall, F1, AUC, confusion matrix
│   └── prediction/
│       ├── __init__.py
│       └── predictor.py          # Inference utilities
│
├── configs/                      # Configuration files (reserved)
├── notebooks/                    # Jupyter notebooks (EDA, experiments)
├── tests/                        # Unit & integration tests
├── logs/                         # Runtime logs
│
├── Dockerfile                    # Container build
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md                     # ← You are here
```

---

## Dataset

The synthetic dataset simulates **50,000 university students** across 24 features spanning demographics, socioeconomics, academics, and LMS engagement.

### Feature Table

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `student_id` | string | UUID | Unique identifier |
| `age` | int | 17–27 | Student age with realistic distribution |
| `gender` | categorical | Male, Female, Non-Binary | 47/49/4% split |
| `department` | categorical | 6 departments | CS, Engineering, Business, Arts, Science, Medicine |
| `semester` | int | 1–8 | Current semester (weighted toward early semesters) |
| `parent_education` | categorical | 5 levels | No Formal Education → PhD |
| `family_income` | categorical | 5 levels | Low → High |
| `commute_distance` | float | 0.5–80 mi | Exponential distribution (mean ~12 mi) |
| `scholarship_status` | int | 0/1 | 28% scholarship holders |
| `has_internet_access` | int | 0/1 | 92% have home internet |
| `part_time_job` | int | 0/1 | 35% employed part-time |
| `gpa_current` | float | 0.0–4.0 | Composite of midterm, assignments, quiz, attendance |
| `attendance_rate` | float | 15–100% | Correlated with latent academic strength |
| `midterm_score` | float | 0–100 | ~2% missing values injected |
| `assignment_score` | float | 0–100 | Task completion scores |
| `quiz_score` | float | 0–100 | ~2% missing values injected |
| `login_frequency` | int | 0–80+ | Weekly LMS logins (Poisson) |
| `forum_participation_score` | int | 0–50+ | Weekly forum posts |
| `time_spent_on_materials` | float | 0–60 hrs | Time on LMS content |
| `study_hours_weekly` | float | 0–40 hrs | Self-reported study time (~2% missing) |
| `library_usage_weekly` | int | 0–30 | Weekly library visits (~2% missing) |
| `missed_deadlines_count` | int | 0–15+ | Inversely correlated with strength |
| `final_grade` | categorical | A/B/C/D/F | Mapped from GPA thresholds |
| **`dropout_risk`** | **int** | **0/1** | **Target variable** (~56% positive rate) |

### Correlation Design

All features are driven by a **latent academic strength score** that combines:
- Parent education bonus (+25 for PhD, -15 for no formal education)
- Family income effect (+15 for high, -12 for low)
- Scholarship boost (+12)
- Internet access (+8)
- Part-time job penalty (-6)
- Commute distance penalty
- Random individual variance (σ=15)

This creates realistic inter-feature correlations (e.g., high GPA ↔ high attendance ↔ low missed deadlines) rather than independent random columns.

---

## Models & Performance

Three models are trained and compared automatically. Results from the latest training run on **49,745 cleaned samples** with **21 features**:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 95.74% | 95.75% | 95.74% | 0.9574 | 0.9937 |
| XGBoost | 98.06% | 98.06% | 98.06% | 0.9806 | 0.9987 |
| **LightGBM** ◆ | **98.71%** | **98.71%** | **98.71%** | **0.9871** | **0.9994** |

◆ = Automatically selected as the best model by F1 score.

### Confusion Matrix (LightGBM)

```
              Predicted
              Safe    At-Risk
Actual Safe   4,243      62
       Risk      66   5,578
```

- **False positives (62):** Students incorrectly flagged as at-risk → over-intervention (low cost)
- **False negatives (66):** At-risk students missed → under-intervention (high cost)

---

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- ~200 MB disk space for model artifacts

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/student-performance-ml.git
cd student-performance-ml

# 2. Create a virtual environment (recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Preprocessing, Random Forest, metrics |
| `xgboost` | XGBoost classifier |
| `lightgbm` | LightGBM classifier |
| `fastapi`, `uvicorn` | REST API server |
| `streamlit` | Dashboard UI |
| `plotly` | Interactive charts |
| `pydantic` | Request validation |
| `joblib` | Model serialization |

---

## Quick Start

### 1. Generate the dataset

```bash
python data/generate_data.py
```

Output: `data/raw/synthetic_student_data.csv` (50,000 rows × 24 columns)

### 2. Train all models

```bash
python scripts/train_pipeline.py
```

This runs the full pipeline (validate → clean → encode → train RF/XGB/LGBM → evaluate) and saves:
- 3 model files (`models/*.pkl`)
- 3 metadata files (`models/*_meta.json`)
- Combined metrics (`models/training_metrics.json`)

### 3. Start the API

```bash
python -m uvicorn api.main:app --reload
```

API available at: **http://localhost:8000**
Swagger docs at: **http://localhost:8000/docs**

### 4. Launch the dashboard

```bash
python -m streamlit run dashboard/app.py
```

Dashboard available at: **http://localhost:8501**

---

## API Reference

### `GET /`

Health check.

```json
{ "status": "healthy", "service": "Student Intelligence API v2.1" }
```

### `GET /health`

Lists available model artifacts.

```json
{
  "status": "ok",
  "models_available": ["lgbm_dropout_model", "rf_dropout_model", "xgb_dropout_model"]
}
```

### `POST /predict/dropout`

Predict dropout risk for a single student.

**Request body:**

```json
{
  "age": 20,
  "gender": "Female",
  "department": "Computer Science",
  "semester": 3,
  "parent_education": "Bachelor",
  "family_income": "Middle",
  "commute_distance": 10.0,
  "scholarship_status": 0,
  "has_internet_access": 1,
  "part_time_job": 0,
  "gpa_current": 2.8,
  "attendance_rate": 78.0,
  "midterm_score": 65.0,
  "assignment_score": 70.0,
  "quiz_score": 60.0,
  "login_frequency": 8,
  "forum_participation_score": 3,
  "time_spent_on_materials": 10.0,
  "study_hours_weekly": 8.0,
  "library_usage_weekly": 2,
  "missed_deadlines_count": 2
}
```

**Response:**

```json
{
  "student_risk_assessment": "High",
  "raw_prediction": 1,
  "confidence": 0.8742,
  "dropout_probability": 0.8742
}
```

### `POST /predict/batch`

Predict for multiple students at once.

**Request body:**

```json
{
  "students": [
    { "age": 20, "gender": "Male", ... },
    { "age": 22, "gender": "Female", ... }
  ]
}
```

**Response:**

```json
{
  "predictions": [
    { "risk": "High", "dropout_probability": 0.7823 },
    { "risk": "Low", "dropout_probability": 0.1204 }
  ],
  "total": 2
}
```

### `GET /metrics`

Returns saved training metrics for all models.

```json
{
  "rf": { "name": "Random Forest", "metrics": { "accuracy": 0.9574, ... } },
  "xgb": { "name": "XGBoost", "metrics": { "accuracy": 0.9806, ... } },
  "lgbm": { "name": "LightGBM", "metrics": { "accuracy": 0.9871, ... } },
  "best_model": "lgbm",
  "dataset_size": 49745,
  "features_count": 21
}
```

---

## Dashboard

The Streamlit dashboard has three tabs:

### Tab 1 — Risk Assessment

- **Metric cards** showing risk status, dropout probability, GPA, attendance, and midterm score
- **Radar chart** comparing the student's profile against cohort averages across 6 dimensions
- **Risk gauge** with color-coded zones (green < 20%, amber 20–50%, red > 50%)
- **Personalized intervention plan** with prioritized, actionable recommendations based on the student's specific weaknesses

### Tab 2 — Model Comparison

- Training summary (best model, sample count, feature count)
- Side-by-side metric comparison table
- Grouped bar chart comparing accuracy, precision, recall, and F1 across all 3 models
- Confusion matrices rendered as heatmaps

### Tab 3 — Data Explorer

- Dataset summary metrics (record count, avg GPA, dropout rate, missing values)
- Grade distribution bar chart
- Dropout rate by department (horizontal bar)
- Full feature correlation matrix heatmap
- Raw data table browser (first 80 records)

---

## Pipeline Details

The training pipeline (`scripts/train_pipeline.py`) executes 6 stages:

### Stage 1 — Ingestion
`DataLoader` reads the CSV file into a pandas DataFrame.

### Stage 2 — Validation
`DataValidator` applies vectorized schema checks:
- Verifies required columns exist
- Enforces numeric ranges (age 15–100, GPA 0–4.5, attendance 0–100%, scores 0–100)
- NaN-safe bounds checking for columns with intentional missing values
- Drops invalid rows and logs the count

### Stage 3 — Cleaning
`DataCleaner` uses method chaining to:
- Fill missing values with median (or mean/mode/zero, configurable)
- Remove statistical outliers using Z-score > 3 threshold
- Returns cleaned DataFrame via `.get_df()`

### Stage 4 — Encoding
`DataEncoder` applies `LabelEncoder` to categorical columns:
- `gender` → 0/1/2
- `parent_education` → 0/1/2/3/4
- `department` → 0/1/2/3/4/5
- `family_income` → 0/1/2/3/4

### Stage 5 — Training
`ModelTrainer` trains three classifiers with stratified 80/20 splits:

| Model | Key Parameters |
|-------|---------------|
| Random Forest | `n_estimators=200`, `max_depth=15`, `min_samples_split=5` |
| XGBoost | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `eval_metric='logloss'` |
| LightGBM | `n_estimators=200`, `max_depth=10`, `learning_rate=0.1`, `verbose=-1` |

Each model saves:
- `.pkl` — serialized model object
- `_meta.json` — feature column order (critical for inference alignment)

### Stage 6 — Evaluation
`ModelEvaluator` computes:
- Accuracy, precision (weighted), recall (weighted), F1 (weighted)
- ROC AUC (with graceful handling of single-class edge cases)
- Confusion matrix

All metrics are aggregated into `training_metrics.json` with the best model flagged.

---

## Configuration

### Changing the dataset size

Edit `data/generate_data.py` line 6:

```python
def generate_student_data(num_records=50000):  # Change this value
```

### Switching the active model in the API

The API defaults to `rf_dropout_model`. To use a different model, edit `api/main.py`:

```python
model, feature_columns = load_model_and_meta("lgbm_dropout_model")  # or "xgb_dropout_model"
```

### Adding a new model type

1. Add the model class to `src/training/trainer.py` in the `__init__` method
2. Add the model key to the `model_types` list in `scripts/train_pipeline.py`
3. Retrain: `python scripts/train_pipeline.py`

---

## Docker Deployment

### Build

```bash
docker build -t edupulse-api .
```

### Run

```bash
docker run -p 8000:8000 edupulse-api
```

The API will be available at `http://localhost:8000`.

> **Note:** The Dockerfile only runs the API server. The Streamlit dashboard runs separately and connects to the API via `http://localhost:8000`.

---

## Troubleshooting

### `uvicorn: The term 'uvicorn' is not recognized`

Use the module syntax instead:

```bash
python -m uvicorn api.main:app --reload
```

### `ModuleNotFoundError: No module named 'src'`

Make sure you're running commands from the project root directory (`student-performance-ml/`), not from a subdirectory.

### `FileNotFoundError: Model not found`

You need to train the models first:

```bash
python data/generate_data.py
python scripts/train_pipeline.py
```

### `plotly.express has no attribute 'barh'`

This was a known bug that has been fixed. `px.bar(..., orientation='h')` is used instead of `px.barh()`.

### Dashboard shows "Cannot connect to API"

Make sure the FastAPI server is running in a separate terminal before launching the dashboard.

---

## Roadmap

- [ ] **Model Registry** — Migrate from local `.pkl` storage to MLflow for versioning and experiment tracking
- [ ] **Historical Tracking** — SQLite/PostgreSQL database to track student risk scores over time
- [ ] **Feature Importance** — SHAP values for per-student explainability
- [ ] **Automated Retraining** — Scheduled pipeline runs with drift detection (Evidently)
- [ ] **Authentication** — API key / JWT authentication for production deployment
- [ ] **Export Reports** — PDF generation of student risk reports for advisors
- [ ] **Multi-institution** — Support for multiple universities with isolated datasets

---

## License

This project is available under the [MIT License](LICENSE).

---

<div align="center">
  <strong>Built with Python, scikit-learn, FastAPI, and Streamlit</strong>
</div>
#   S t u d e n t - p e r f o r m a n c e  
 