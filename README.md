<div align="center">

# EduPulse — Explainable AI Student Retention System

**AI-powered dropout risk assessment, causal reasoning, and academic intervention engine**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

EduPulse is a production-grade machine learning and Explainable AI (XAI) system that predicts student dropout risk using academic, behavioral, and socioeconomic data. Beyond standard classification scoring, EduPulse integrates Large Language Models (LLMs) to dynamically explain *why* a student is at risk and to build causal "What-If" intervention plans.

The system features a premium dark-themed analytics dashboard paired with a persistent, data-rich analysis panel that provides real-time risk gauges, performance radars, and quantified mitigation plans. It is designed for university administrators, academic advisors, and student support teams.

### Key Capabilities

- **Explainable AI (XAI) Reasoning Engine** — Generates human-readable explanations synthesizing hundreds of features into an understandable root-cause risk diagnosis.
- **Dynamic Intervention & What-If Simulator** — Calculates quantified paths to lower risk (e.g., "Add 2+ extra study hours/week to drop risk by 12%").
- **Multi-model ML Training** — Random Forest, XGBoost, and LightGBM trained in a single pipeline run on a 50,000-record synthetic student dataset.
- **RESTful Endpoints** — Robust FastAPI serving real-time predictions, batch processing, and reasoning summaries.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EduPulse System                          │
├─────────────┬──────────────┬──────────────┬─────────────────────┤
│  Data Layer │ ML & AI Core │   API Layer  │   Presentation      │
├─────────────┼──────────────┼──────────────┼─────────────────────┤
│ generate_   │ DataLoader   │ FastAPI app  │ Streamlit dashboard │
│  data.py    │ ModelTrainer │ /predict     │ Risk Gauges         │
│             │ Explain XAI  │ /analyze     │ Radar Chart         │
│             │ LLM Servicer │ /metrics     │ What-If Simulators  │
└─────────────┴──────────────┴──────────────┴─────────────────────┘
```

**Data flows left to right:**

1. `generate_data.py` creates 50k synthetic student records.
2. `train_pipeline.py` orchestrates data ingestion → preprocessing → training → evaluation.
3. The Reasoning and Intervention engines (`src/reasoning`, `src/intervention`) interpret model predictions using LLMs.
4. FastAPI serves predictions and XAI reasonings.
5. The Streamlit dashboard renders the Risk Assessment + Analysis interface.

---

## Project Structure

```text
student-performance-ml/
├── api/                          # REST API Server
│   └── main.py                   # FastAPI with prediction, reasoning endpoints
├── dashboard/                    # Application UI
│   └── app.py                    # Streamlit risk assessment dashboard
├── data/                         # Data Generators
│   └── generate_data.py          # 50k Synthetic student dataset
├── models/                       # Trained Models (.pkl & metadata)
├── src/                          # Core Modules
│   ├── ingestion & preprocessing # Data engineering pipelines
│   ├── training & evaluation     # ML automated pipelines
│   ├── reasoning/                # Explainable AI (XAI) Engine
│   ├── intervention/             # Rule-based & LLM Intervention logic
│   └── services/                 # External service wrappers (LLM)
├── requirements.txt
├── scripts/                      # Utility runners
└── README.md
```

---

## The AI Intelligence Suite

EduPulse goes beyond standard predictions with a deep analysis and reporting layer.

### 1. Reasoning Engine (`src/reasoning`)
Takes the raw probabilities from LightGBM/XGBoost/RF models and translates the numeric feature bounds into a diagnostic summary. It identifies *why* a student was flagged as safe or at-risk.

### 2. Intervention Engine (`src/intervention`)
Using both statistical feature importance offsets (What-if modeling) and programmatic Rules (`rules.py`), the intervention engine builds step-by-step mitigation plans showing quantified risk reductions.

---

## Installation

### Prerequisites

- Python 3.11+
- API Key from Groq or another supported provider (for LLM reasoning services)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/salmanwnl44/Student-performance.git
cd Student-performance

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Environment Variables
# Create a .env file locally and add your required keys:
# GROQ_API_KEY=your_key_here
```

---

## Quick Start

### 1. Generate & Train (Initial Setup)

```bash
# Generate relational numeric data
python data/generate_data.py

# Run complete Machine Learning pipeline
python scripts/train_pipeline.py
```

### 2. Start the Backend API

```bash
python -m uvicorn api.main:app --reload
```
*API running at http://localhost:8000*  
*Swagger UI docs at http://localhost:8000/docs*

### 3. Launch the Dashboard

```bash
python -m streamlit run dashboard/app.py
```
*Application available at http://localhost:8501*

---

## Key API Endpoints

- `POST /predict/dropout` — Standard numeric inference and probability generation. Includes quantified "What-if" mathematical risk-reduction estimates.
- `POST /predict/analyze` — Hits the Reasoning and Intervention engines. Returns LLM-generated analysis and prescriptive plans.
- `GET /metrics` — Fetches the latest model evaluation metrics (F1, Accuracy, Precision, Recall).

---

## Dashboard Overview

The dashboard features a **sidebar-driven Risk Assessment** panel:

**Sidebar Profile Form**  
Fill in the student's demographic, academic, and engagement details, then click **Run Analysis**.

**Risk Assessment Panel**  
Displays an interactive risk report with:
- Risk status pill (High / Medium / Low) and dropout probability
- Performance radar chart (vs. cohort average)
- Risk gauge meter
- **Generate Detailed Report** button for LLM-powered reasoning, intervention plans, risk trend charts, and the What-If Simulator

**Developer Mode** (toggled in sidebar) exposes additional tabs: Model Comparison and Data Explorer.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

<div align="center">
  <strong>Built with Python, Scikit-learn, FastAPI, and Streamlit</strong>
</div>