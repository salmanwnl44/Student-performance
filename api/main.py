from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import json
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Student Intelligence API", version="2.1.0", description="AI-powered student risk assessment")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class StudentRecord(BaseModel):
    age: int = Field(20, ge=15, le=100)
    gender: str = "Male"
    department: str = "Computer Science"
    semester: int = Field(3, ge=1, le=8)
    parent_education: str = "Bachelor"
    family_income: str = "Middle"
    commute_distance: float = Field(10.0, ge=0)
    scholarship_status: int = Field(0, ge=0, le=1)
    has_internet_access: int = Field(1, ge=0, le=1)
    part_time_job: int = Field(0, ge=0, le=1)
    gpa_current: float = Field(2.8, ge=0.0, le=4.5)
    attendance_rate: float = Field(78.0, ge=0.0, le=100.0)
    midterm_score: float = Field(65.0, ge=0, le=100)
    assignment_score: float = Field(70.0, ge=0, le=100)
    quiz_score: float = Field(60.0, ge=0, le=100)
    login_frequency: int = Field(8, ge=0)
    forum_participation_score: int = Field(3, ge=0)
    time_spent_on_materials: float = Field(10.0, ge=0)
    study_hours_weekly: float = Field(8.0, ge=0)
    library_usage_weekly: int = Field(2, ge=0)
    missed_deadlines_count: int = Field(2, ge=0)

class BatchRequest(BaseModel):
    students: List[StudentRecord]

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

LABEL_MAPS = {
    'gender': {'Female': 0, 'Male': 1, 'Non-Binary': 2},
    'parent_education': {'No Formal Education': 0, 'Bachelor': 1, 'High School': 2, 'Master': 3, 'PhD': 4},
    'department': {'Arts': 0, 'Business': 1, 'Computer Science': 2, 'Engineering': 3, 'Medicine': 4, 'Science': 5},
    'family_income': {'High': 0, 'Low': 1, 'Lower-Middle': 2, 'Middle': 3, 'Upper-Middle': 4}
}

def load_model_and_meta(model_name: str):
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    meta_path = os.path.join(MODEL_DIR, f"{model_name}_meta.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    feature_columns = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
            feature_columns = meta.get('feature_columns')
    return model, feature_columns

def prepare_features(student: StudentRecord, feature_columns: Optional[list] = None) -> pd.DataFrame:
    df = pd.DataFrame([student.model_dump()])
    for col, mapping in LABEL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(-1).astype(int)
    df_numeric = df.select_dtypes(include=['number'])
    if feature_columns:
        for col in feature_columns:
            if col not in df_numeric.columns:
                df_numeric[col] = 0
        df_numeric = df_numeric[feature_columns]
    return df_numeric

@app.get("/")
def root():
    return {"status": "healthy", "service": "Student Intelligence API v2.1"}

@app.get("/health")
def health():
    models = [f.replace('.pkl', '') for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
    return {"status": "ok", "models_available": models}

@app.post("/predict/dropout")
def predict_dropout(student: StudentRecord):
    try:
        model, feature_columns = load_model_and_meta("rf_dropout_model")
        df = prepare_features(student, feature_columns)
        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0]
        return {
            "student_risk_assessment": "High" if prediction == 1 else "Low",
            "raw_prediction": int(prediction),
            "confidence": round(float(np.max(proba)), 4),
            "dropout_probability": round(float(proba[1]), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(batch: BatchRequest):
    try:
        model, feature_columns = load_model_and_meta("rf_dropout_model")
        results = []
        for student in batch.students:
            df = prepare_features(student, feature_columns)
            pred = model.predict(df)[0]
            prob = float(model.predict_proba(df)[0][1])
            results.append({"risk": "High" if pred == 1 else "Low", "dropout_probability": round(prob, 4)})
        return {"predictions": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    metrics_path = os.path.join(MODEL_DIR, "training_metrics.json")
    if not os.path.exists(metrics_path):
        return {"error": "No metrics found. Train first."}
    with open(metrics_path) as f:
        return json.load(f)
