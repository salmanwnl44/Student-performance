<div align="center">

# EduPulse — Explainable AI Student Retention System

**AI-powered dropout risk assessment, causal reasoning, and academic intervention engine**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-blue?logo=langchain&logoColor=white)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-orange)](https://trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

EduPulse is a state-of-production machine learning and Explainable AI (XAI) system that predicts student dropout risk using academic, behavioral, and socioeconomic data. Beyond just standard classification scoring, EduPulse deeply integrates Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to dynamically explain *why* a student is at risk and to build causal "What-If" intervention plans.

The system features an advanced, Claude-inspired dual-column dashboard. It pairs a conversational AI mentor with a persistent, data-rich analysis document panel that provides real-time adjustments and deep context. It is designed for university administrators, academic advisors, and student support teams.

### Key Capabilities

- **Explainable AI (XAI) Reasoning Engine** — Generates human-readable explanations synthesizing hundreds of features into an understandable root-cause risk diagnosis.
- **Dynamic Intervention & What-If Simulator** — Calculates quantified paths to lower risk (e.g., "Add 2+ extra study hours/week to drop risk by 12%").
- **Conversational RAG AI Mentor** — A dynamic chat assistant powered by `LangChain`, `Groq`, and `ChromaDB` fetching historic teacher notes and context to personally advise students.
- **Claude-style Dual Column UI** — A modern Streamlit dashboard featuring an interactive chat overlaid alongside a persistent detail/reporting artifact panel.
- **Multi-model ML Training** — Random Forest, XGBoost, and LightGBM trained in a single pipeline run on a 50,000-record synthetic student dataset.
- **RESTful Endpoints** — Robust FastAPI serving real-time predictions, batch processing, reasoning summaries, and chat interfacing.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EduPulse System                          │
├─────────────┬──────────────┬──────────────┬─────────────────────┤
│  Data Layer │ ML & AI Core │   API Layer  │   Presentation      │
├─────────────┼──────────────┼──────────────┼─────────────────────┤
│ generate_   │ DataLoader   │ FastAPI app  │ Streamlit dashboard │
│  data.py    │ ModelTrainer │ /predict     │ Chat Interface      │
│ Vector DB   │ Explain XAI  │ /analyze     │ Risk Gauges         │
│             │ RAG Retriever│ /chat/student│ Artifact Documents  │
│             │ LLM Servicer │ /metrics     │ What-If Simulators  │
└─────────────┴──────────────┴──────────────┴─────────────────────┘
```

**Data flows left to right:**

1. `generate_data.py` & `generate_teacher_notes.py` create student records and historical contextual notes.
2. `train_pipeline.py` orchestrates data ingestion → evaluation. Vector embeddings are stored in ChromaDB.
3. The Reasoning and Intervention engines (`src/reasoning`, `src/intervention`) interpret model predictions using LLMs.
4. FastAPI serves predictions, XAI reasonings, and chat completions.
5. The Streamlit dashboard renders the sophisticated Chat + Analysis interface.

---

## Project Structure

```text
student-performance-ml/
├── api/                          # REST API Server
│   └── main.py                   # FastAPI with prediction, reasoning, and chat endpoints
├── dashboard/                    # Application UI
│   ├── app.py                    # Dual-column Streamlit app
│   └── chat_assistant.py         # Conversational UI components
├── data/                         # Data Generators
│   ├── generate_data.py          # 50k Synthetic DB
│   └── generate_teacher_notes.py # For RAG DB Context
├── models/                       # Trained Models (.pkl & metadata)
├── src/                          # Core Modules
│   ├── ingestion & preprocessing # Data engineering pipelines
│   ├── training & evaluation     # ML automated pipelines
│   ├── reasoning/                # Explainable AI (XAI) Engine
│   ├── intervention/             # Rule-based & LLM Intervention logic
│   ├── rag/                      # Langchain & ChromaDB (retriever, vector_store)
│   └── services/                 # External service wrappers (LLM, DB)
├── chroma_db/                    # Persistent vector storage
├── requirements.txt
├── run.py                        # Utility runners
└── README.md
```

---

## The AI Intelligence Suite

EduPulse goes beyond standard predictions. The newly integrated AI packages supply deep analysis and continuous feedback.

### 1. Reasoning Engine (`src/reasoning`)
Takes the raw probabilities from LightGBM/XGBoost/RF models and translates the numeric feature bounds into a diagnostic summary. It identifies *why* a student was flagged safely or at-risk.

### 2. Intervention Engine (`src/intervention`)
Using both statistical feature importance offsets (What-if modeling) and programmatic Rules (`rules.py`), the intervention engine builds step-by-step mitigation plans. It shows quantified risk reductions dynamically.

### 3. RAG Retriever & AI Mentor (`src/rag`)
Stores institutional knowledge (e.g. historical notes from teachers on what worked for past students) in a local unstructured vector database (`ChromaDB`). When interacting with a student via the Chat Interface, the system retrieves similarity-matched historical notes to provide grounded, highly accurate feedback while ensuring hallucinations are minimized.

---

## Installation

### Prerequisites

- Python 3.11+
- API Key from Groq or another supported provider (for LLM services)

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

# Generate RAG Teacher Notes context
python data/generate_teacher_notes.py

# Run complete Machine Learning pipeline
python scripts/train_pipeline.py
```

### 2. Start the Backend API Services

```bash
python -m uvicorn api.main:app --reload
```
*API running at http://localhost:8000*
*Swagger UI docs at http://localhost:8000/docs*

### 3. Launch the Claude-Style Dashboard

```bash
python -m streamlit run dashboard/app.py
```
*Application available at http://localhost:8501*

---

## Key API Endpoints

- `POST /predict/dropout` - Standard numeric inference and probability generation. Includes quantified "What-if" mathematical risk-reduction estimates.
- `POST /predict/analyze` - Hits the Reasoning and Intervention engines. Returns LLM-generated analysis and prescriptive plans.
- `POST /chat/student` - Engages the conversational assistant. Contextualizes the student's background metrics, current risk, and similarity-retrieved historical RAG instructions into a unified Mentor response.
- `GET /metrics` - Internal dashboarding route fetching the latest model evaluation metrics (F1, Accuracy, Precision, Recall).

---

## Dashboard Overview

The newly refactored frontend uses a **Dual-Column Layout**:

**The Chat Experience (Left Column)**
Operates as a conversational AI mentor. Users input queries (e.g., "How can I improve my grades?"). The AI responds inline.

**The Document Artifact Panel (Right Column)**
Hidden by default to save space, this panel opens on-demand when deeply complex requests (e.g., Generating full intervention plans) occur. It displays rich data, radar charts, model comparison tables, data exploratory graphs, and long-form Reasoning outputs.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

<div align="center">
  <strong>Built with Python, Scikit-learn, LangChain, FastAPI, and Streamlit</strong>
</div>