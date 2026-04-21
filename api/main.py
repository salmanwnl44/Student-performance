from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import json
import pandas as pd
import numpy as np
import os

from src.reasoning.reasoning_engine import ReasoningEngine
from src.intervention.intervention_engine import InterventionEngine

reasoning_engine = ReasoningEngine()
intervention_engine = InterventionEngine()

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

class ChatRequest(BaseModel):
    student: StudentRecord
    message: str

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
        proba = float(model.predict_proba(df)[0][1])
        
        # 1. Feature Importances
        importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else []
        fi_dict = {f: float(imp) for f, imp in zip(feature_columns, importances)}
        top_features = dict(sorted(fi_dict.items(), key=lambda item: item[1], reverse=True)[:5])

        # 2. Reasoning & Interventions (Moved to /predict/analyze)
        
        # 3. What-if Quantified Interventions
        # Dynamically predict how much risk drops if key behaviors are changed
        quantified_plans = []
        base_dict = student.model_dump()
        
        if base_dict.get("attendance_rate", 100) < 85:
            mod_data = student.model_copy(update={"attendance_rate": 85.0})
            new_prob = float(model.predict_proba(prepare_features(mod_data, feature_columns))[0][1])
            reduction = proba - new_prob
            if reduction > 0:
                quantified_plans.append({"action": "Increase attendance to 85%", "reduction": round(reduction, 3)})
                
        if base_dict.get("study_hours_weekly", 10) < 10:
            mod_data = student.model_copy(update={"study_hours_weekly": 12.0})
            new_prob = float(model.predict_proba(prepare_features(mod_data, feature_columns))[0][1])
            reduction = proba - new_prob
            if reduction > 0:
                quantified_plans.append({"action": "Add 2+ extra study hours/week", "reduction": round(reduction, 3)})
                
        if base_dict.get("missed_deadlines_count", 0) > 0:
            mod_data = student.model_copy(update={"missed_deadlines_count": 0})
            new_prob = float(model.predict_proba(prepare_features(mod_data, feature_columns))[0][1])
            reduction = proba - new_prob
            if reduction > 0:
                quantified_plans.append({"action": "Clear all pending missed deadlines", "reduction": round(reduction, 3)})
                
        # 4. Mock 5-Week Risk Trend
        import random
        history = []
        curr_p = proba
        for i in range(5, 0, -1):
            if i == 5:
                history.append({"week": f"Week {i}", "risk": round(curr_p, 4)})
            else:
                curr_p = max(0.01, curr_p + random.uniform(-0.05, 0.08))
                history.append({"week": f"Week {i}", "risk": round(curr_p, 4)})
        history.reverse()

        return {
            "student_risk_assessment": "High" if proba > 0.5 else "Medium" if proba > 0.2 else "Low", 
            "raw_prediction": int(prediction),
            "confidence": round(float(np.max(model.predict_proba(df)[0])), 4),
            "dropout_probability": round(proba, 4),
            "feature_importance": top_features,
            "quantified_plans": quantified_plans,  # Raw math estimates
            "risk_history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/analyze")
def predict_analyze(student: StudentRecord):
    try:
        model, feature_columns = load_model_and_meta("rf_dropout_model")
        df = prepare_features(student, feature_columns)
        proba = float(model.predict_proba(df)[0][1])
        
        reasoning = reasoning_engine.generate_reasoning(student.model_dump(), proba)
        intervention = intervention_engine.generate_intervention(student.model_dump(), reasoning)
        
        return {
            "reasoning": reasoning,
            "intervention": intervention
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

@app.post("/predict/reasoning")
def predict_reasoning(student: StudentRecord):
    try:
        model, feature_columns = load_model_and_meta("rf_dropout_model")
        df = prepare_features(student, feature_columns)
        proba = float(model.predict_proba(df)[0][1])
        
        reasoning = reasoning_engine.generate_reasoning(student.model_dump(), proba)
        return {
            "dropout_probability": round(proba, 4),
            "reasoning": reasoning
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/intervention")
def predict_intervention(student: StudentRecord):
    try:
        model, feature_columns = load_model_and_meta("rf_dropout_model")
        df = prepare_features(student, feature_columns)
        proba = float(model.predict_proba(df)[0][1])
        
        reasoning = reasoning_engine.generate_reasoning(student.model_dump(), proba)
        intervention = intervention_engine.generate_intervention(student.model_dump(), reasoning)
        return {
            "dropout_probability": round(proba, 4),
            "intervention_plan": intervention
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/student")
def chat_student(request: ChatRequest):
    try:
        model, feature_columns = load_model_and_meta("rf_dropout_model")
        df = prepare_features(request.student, feature_columns)
        proba = float(model.predict_proba(df)[0][1])
        risk_level = "High" if proba > 0.6 else "Medium" if proba > 0.3 else "Low"
        
        reasoning = reasoning_engine.generate_reasoning(request.student.model_dump(), proba)
        from src.rag.retriever import RAGRetriever
        rag_context = RAGRetriever().get_context_for_student(request.student.department, risk_level)
        
        system_prompt = (
            "You are a friendly, personal AI Mentor for the student. "
            "Answer the student's question directly, offering actionable and encouraging advice. "
            "IMPORTANT: Use the Historical Teacher Notes purely as inspiration for advice. NEVER mention 'past students', 'historical notes', or placeholders like 'student X'. Adapt the historical advice so it sounds like a direct, personalized suggestion for this student right now."
        )
        
        prompt = (
            f"Student Message: '{request.message}'\n\n"
            f"Student Profile Metrics:\n"
            f"Attendance: {request.student.attendance_rate}%\n"
            f"GPA: {request.student.gpa_current}\n"
            f"AI Risk Reasoning: {reasoning}\n"
            f"Historical Teacher Notes (FOR INTERNAL SYSTEM INSPIRATION ONLY): {rag_context}\n\n"
            "Respond directly to the student:"
        )
        
        response = reasoning_engine.llm.generate_response(prompt, system_prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
