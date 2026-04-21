import sys
import os
import time
import traceback
import tempfile
import anyio
import httpx
import pandas as pd
import numpy as np

# Force UTF-8 for Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.text import Text

from src.ingestion.data_loader import DataLoader
from src.validation.data_validator import DataValidator
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.encoder import DataEncoder
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from api.main import app

console = Console()
results = []

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'synthetic_student_data.csv')

# ─── Async HTTP helpers ───────────────────────────────────────────────────────

async def _get(url):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as ac:
        return await ac.get(url)

async def _post(url, payload):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as ac:
        return await ac.post(url, json=payload)

def get(url):      return anyio.run(_get, url)
def post(url, p):  return anyio.run(_post, url, p)

# ─── Shared student payload ───────────────────────────────────────────────────

STUDENT = {
    "age": 20, "gender": "Male", "department": "Computer Science",
    "semester": 3, "parent_education": "Bachelor", "family_income": "Middle",
    "commute_distance": 10.0, "scholarship_status": 0, "has_internet_access": 1,
    "part_time_job": 1, "gpa_current": 2.1, "attendance_rate": 62.0,
    "midterm_score": 55.0, "assignment_score": 58.0, "quiz_score": 50.0,
    "login_frequency": 6, "forum_participation_score": 2,
    "time_spent_on_materials": 8.0, "study_hours_weekly": 5.0,
    "library_usage_weekly": 1, "missed_deadlines_count": 4
}

# ─── Test Runner ──────────────────────────────────────────────────────────────

def run_test(name, fn):
    start = time.time()
    try:
        fn()
        elapsed = time.time() - start
        results.append(("PASS", name, f"{elapsed:.3f}s", ""))
        console.print(f"  [bold green]PASS[/bold green]  {name} [dim]({elapsed:.3f}s)[/dim]")
    except Exception:
        elapsed = time.time() - start
        tb = traceback.format_exc().strip().splitlines()[-1]
        results.append(("FAIL", name, f"{elapsed:.3f}s", tb))
        console.print(f"  [bold red]FAIL[/bold red]  {name} [dim]({elapsed:.3f}s)[/dim]")
        console.print(f"         [red]{tb}[/red]")

# ─── INTEGRATION 1: Full Data Pipeline ───────────────────────────────────────
# Loader → Validator → Cleaner → Encoder — all chained on real CSV data

def test_pipeline_load_to_encode():
    """Full data pipeline produces a clean, encoded numeric DataFrame."""
    df = DataLoader(DATA_PATH).load_csv()
    df = DataValidator(df).validate_schema()
    df = (DataCleaner(df)
          .handle_missing_values(strategy='median')
          .remove_outliers('gpa_current')
          .remove_outliers('attendance_rate')
          .get_df())
    df = DataEncoder(df).encode_categorical(['gender', 'parent_education', 'department', 'family_income'])

    assert len(df) > 40000,               "Pipeline should retain most records"
    assert df.isnull().sum().sum() == 0,  "No nulls should remain after cleaning"
    assert df['gender'].dtype != object,  "gender should be encoded to numeric"

def test_pipeline_row_count_consistency():
    """Records after cleaning are fewer than raw, but within expected bounds."""
    raw_df = DataLoader(DATA_PATH).load_csv()
    clean_df = (DataCleaner(raw_df.copy())
                .handle_missing_values()
                .remove_outliers('gpa_current')
                .get_df())
    loss_pct = (len(raw_df) - len(clean_df)) / len(raw_df)
    assert loss_pct < 0.05, f"Unexpectedly large data loss: {loss_pct:.2%}"

def test_pipeline_target_column_preserved():
    """Target column dropout_risk survives the full pipeline."""
    df = DataLoader(DATA_PATH).load_csv()
    df = DataValidator(df).validate_schema()
    df = DataCleaner(df).handle_missing_values().get_df()
    assert 'dropout_risk' in df.columns
    assert df['dropout_risk'].nunique() == 2  # Binary: 0 or 1

# ─── INTEGRATION 2: Training → Evaluation ────────────────────────────────────
# Small synthetic data, end-to-end train + evaluate

def _build_train_df(n=500):
    np.random.seed(42)
    gpa           = np.random.uniform(1.0, 4.0, n)
    attendance    = np.random.uniform(40, 100, n)
    missed        = np.random.randint(0, 10, n)
    part_time_job = np.random.randint(0, 2, n)
    # Correlated dropout label (mirrors real dataset logic)
    risk_score = (4.0 - gpa) * 12 + (100 - attendance) * 0.45 + missed * 3.5 + part_time_job * 5
    dropout_risk = (risk_score > risk_score.mean()).astype(int)

    df = pd.DataFrame({
        'age':                     np.random.randint(18, 26, n),
        'gender':                  np.random.randint(0, 2, n),
        'department':              np.random.randint(0, 5, n),
        'semester':                np.random.randint(1, 8, n),
        'gpa_current':             gpa,
        'attendance_rate':         attendance,
        'midterm_score':           np.clip(55 + (gpa - 2.5) * 15 + np.random.normal(0, 8, n), 0, 100),
        'assignment_score':        np.clip(60 + (gpa - 2.5) * 10 + np.random.normal(0, 8, n), 0, 100),
        'quiz_score':              np.clip(50 + (gpa - 2.5) * 12 + np.random.normal(0, 8, n), 0, 100),
        'login_frequency':         np.random.randint(1, 30, n),
        'study_hours_weekly':      np.clip(8 + (gpa - 2.5) * 3 + np.random.normal(0, 3, n), 0, 40),
        'missed_deadlines_count':  missed,
        'dropout_risk':            dropout_risk,
    })
    return df


def test_trainer_rf_trains_and_evaluates():
    """Random Forest trains and evaluates with F1 > 0."""
    df = _build_train_df()
    trainer = ModelTrainer(df, target_col='dropout_risk', model_type='rf')
    results_obj = trainer.train(test_size=0.2)
    metrics = ModelEvaluator(
        results_obj['model'], results_obj['X_test'], results_obj['y_test']
    ).evaluate()
    assert metrics['accuracy'] > 0.5, "RF accuracy should be above 50%"
    assert metrics['f1_score'] > 0.0, "RF F1 score should be positive"

def test_trainer_xgb_trains_and_evaluates():
    """XGBoost trains and evaluates with Accuracy > 0."""
    df = _build_train_df()
    trainer = ModelTrainer(df, target_col='dropout_risk', model_type='xgb')
    results_obj = trainer.train(test_size=0.2)
    metrics = ModelEvaluator(
        results_obj['model'], results_obj['X_test'], results_obj['y_test']
    ).evaluate()
    assert metrics['accuracy'] > 0.5

def test_trainer_lgbm_trains_and_evaluates():
    """LightGBM trains and evaluates with Accuracy > 0."""
    df = _build_train_df()
    trainer = ModelTrainer(df, target_col='dropout_risk', model_type='lgbm')
    results_obj = trainer.train(test_size=0.2)
    metrics = ModelEvaluator(
        results_obj['model'], results_obj['X_test'], results_obj['y_test']
    ).evaluate()
    assert metrics['accuracy'] > 0.5

def test_trainer_saves_and_loads_model():
    """Model saves to disk and loads back producing the same predictions."""
    import joblib
    df = _build_train_df()
    trainer = ModelTrainer(df, target_col='dropout_risk', model_type='rf')
    r = trainer.train()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test_model.pkl')
        trainer.save_model(path)
        assert os.path.exists(path), "Saved model file not found"
        loaded = joblib.load(path)
        preds_original = r['model'].predict(r['X_test'])
        preds_loaded   = loaded.predict(r['X_test'])
        assert list(preds_original) == list(preds_loaded), "Loaded model gives different predictions"

def test_evaluator_metrics_keys():
    """Evaluator returns all required metric keys."""
    df = _build_train_df()
    r = ModelTrainer(df, target_col='dropout_risk', model_type='rf').train()
    metrics = ModelEvaluator(r['model'], r['X_test'], r['y_test']).evaluate()
    for key in ['accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix']:
        assert key in metrics, f"Missing metric: {key}"

# ─── INTEGRATION 3: API → Model → Response ───────────────────────────────────
# Verifies the full API-to-model-to-structured-response flow

def test_full_dropout_prediction_flow():
    """POST /predict/dropout returns a valid structured prediction using real model."""
    r = post("/predict/dropout", STUDENT)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["dropout_probability"] <= 1.0, "Probability must be in [0, 1]"
    assert body["student_risk_assessment"] in ["Low", "Medium", "High"]
    assert isinstance(body["feature_importance"], dict)
    assert len(body["feature_importance"]) > 0, "Feature importance must not be empty"

def test_full_batch_pipeline_flow():
    """POST /predict/batch runs full pipeline for multiple students."""
    students = [STUDENT, {**STUDENT, "gpa_current": 3.9, "attendance_rate": 97.0}]
    r = post("/predict/batch", {"students": students})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2
    probs = [p["dropout_probability"] for p in preds]
    assert probs[0] > probs[1], "Struggling student should have higher dropout prob than high achiever"

def test_risk_history_has_5_weeks():
    """Risk history in /predict/dropout should always contain 5 weekly data points."""
    r = post("/predict/dropout", STUDENT)
    history = r.json()["risk_history"]
    assert len(history) == 5, f"Expected 5 weeks, got {len(history)}"
    for entry in history:
        assert "week" in entry and "risk" in entry

def test_metrics_f1_above_threshold():
    """Trained models in /metrics should all have F1 score above 0.90."""
    r = get("/metrics")
    body = r.json()
    for model_key in ["rf", "xgb", "lgbm"]:
        f1 = body[model_key]["metrics"]["f1_score"]
        assert f1 > 0.90, f"{model_key} F1 {f1:.4f} is below acceptable threshold of 0.90"

def test_api_health_matches_model_files():
    """Models listed in /health should match actual .pkl files in models/ dir."""
    r = get("/health")
    models_available = set(r.json()["models_available"])
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    files_on_disk = {f.replace('.pkl', '') for f in os.listdir(model_dir) if f.endswith('.pkl')}
    assert models_available == files_on_disk, \
        f"Health endpoint mismatch. API: {models_available} vs Disk: {files_on_disk}"

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    console.print()
    console.print(Panel("[bold white]EduPulse Integration Test Suite[/bold white]", expand=False))
    console.print()

    console.print("[bold cyan][ DATA PIPELINE: Loader -> Validator -> Cleaner -> Encoder ][/bold cyan]")
    run_test("Full pipeline load-to-encode on real CSV",    test_pipeline_load_to_encode)
    run_test("Row count loss < 5% after cleaning",         test_pipeline_row_count_consistency)
    run_test("Target column dropout_risk stays intact",    test_pipeline_target_column_preserved)

    console.print("\n[bold cyan][ TRAINING -> EVALUATION: All 3 Models ][/bold cyan]")
    run_test("Random Forest trains and evaluates",         test_trainer_rf_trains_and_evaluates)
    run_test("XGBoost trains and evaluates",               test_trainer_xgb_trains_and_evaluates)
    run_test("LightGBM trains and evaluates",              test_trainer_lgbm_trains_and_evaluates)
    run_test("Model saves to disk and reloads correctly",  test_trainer_saves_and_loads_model)
    run_test("Evaluator returns all metric keys",          test_evaluator_metrics_keys)

    console.print("\n[bold cyan][ API -> MODEL -> RESPONSE: End-to-End ][/bold cyan]")
    run_test("Full dropout prediction flow (API + model)", test_full_dropout_prediction_flow)
    run_test("Batch: struggling student has higher risk",  test_full_batch_pipeline_flow)
    run_test("Risk history contains 5 weekly entries",     test_risk_history_has_5_weeks)
    run_test("All trained models have F1 > 0.90",          test_metrics_f1_above_threshold)
    run_test("Health endpoint matches .pkl files on disk", test_api_health_matches_model_files)

    # ─── Summary Table ────────────────────────────────────────────────────────

    console.print()
    table = Table(title="Integration Test Results Summary", box=box.ASCII,
                  show_lines=True, header_style="bold white")
    table.add_column("Status",    justify="center", width=8)
    table.add_column("Test Name", justify="left")
    table.add_column("Time",      justify="center", width=10)
    table.add_column("Error",     justify="left", style="dim red")

    passed = sum(1 for r in results if r[0] == "PASS")
    failed = sum(1 for r in results if r[0] == "FAIL")

    for status, name, elapsed, err in results:
        style = "bold green" if status == "PASS" else "bold red"
        table.add_row(Text(status, style=style), name, elapsed, err)

    console.print(table)
    console.print()
    console.print(f"  [bold green]{passed} passed[/bold green]  "
                  f"[bold red]{failed} failed[/bold red]  "
                  f"out of {len(results)} tests")
    console.print()

    sys.exit(0 if failed == 0 else 1)
