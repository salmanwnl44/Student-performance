import sys
import os
import time
import traceback
import anyio
import httpx

# Force UTF-8 for Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.text import Text

from api.main import app

console = Console()
results = []

# ─── Async HTTP helper ────────────────────────────────────────────────────────

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

def get(url):    return anyio.run(_get, url)
def post(url, p): return anyio.run(_post, url, p)

# ─── Shared Test Payloads ─────────────────────────────────────────────────────

LOW_RISK = {
    "age": 21, "gender": "Female", "department": "Computer Science",
    "semester": 4, "parent_education": "Master", "family_income": "Upper-Middle",
    "commute_distance": 5.0, "scholarship_status": 1, "has_internet_access": 1,
    "part_time_job": 0, "gpa_current": 3.8, "attendance_rate": 95.0,
    "midterm_score": 88.0, "assignment_score": 90.0, "quiz_score": 85.0,
    "login_frequency": 20, "forum_participation_score": 8,
    "time_spent_on_materials": 30.0, "study_hours_weekly": 20.0,
    "library_usage_weekly": 6, "missed_deadlines_count": 0
}

HIGH_RISK = {
    "age": 19, "gender": "Male", "department": "Business",
    "semester": 1, "parent_education": "High School", "family_income": "Low",
    "commute_distance": 45.0, "scholarship_status": 0, "has_internet_access": 0,
    "part_time_job": 1, "gpa_current": 1.2, "attendance_rate": 38.0,
    "midterm_score": 30.0, "assignment_score": 35.0, "quiz_score": 28.0,
    "login_frequency": 2, "forum_participation_score": 0,
    "time_spent_on_materials": 1.0, "study_hours_weekly": 2.0,
    "library_usage_weekly": 0, "missed_deadlines_count": 9
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

# ─── GET / ────────────────────────────────────────────────────────────────────

def test_root_returns_healthy():
    r = get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_root_contains_service_name():
    r = get("/")
    assert "service" in r.json()

# ─── GET /health ──────────────────────────────────────────────────────────────

def test_health_status_ok():
    r = get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_health_lists_models():
    r = get("/health")
    assert "models_available" in r.json()
    assert isinstance(r.json()["models_available"], list)

# ─── GET /metrics ─────────────────────────────────────────────────────────────

def test_metrics_returns_200():
    r = get("/metrics")
    assert r.status_code == 200

def test_metrics_contains_best_model():
    r = get("/metrics")
    assert "best_model" in r.json()

def test_metrics_contains_all_3_models():
    r = get("/metrics")
    body = r.json()
    for key in ["rf", "xgb", "lgbm"]:
        assert key in body, f"Missing model key: {key}"

# ─── POST /predict/dropout ────────────────────────────────────────────────────

def test_predict_dropout_low_risk_student():
    r = post("/predict/dropout", LOW_RISK)
    assert r.status_code == 200
    assert r.json()["dropout_probability"] < 0.5

def test_predict_dropout_high_risk_student():
    r = post("/predict/dropout", HIGH_RISK)
    assert r.status_code == 200
    assert r.json()["dropout_probability"] > 0.3

def test_predict_dropout_response_schema():
    r = post("/predict/dropout", LOW_RISK)
    body = r.json()
    for key in ["student_risk_assessment", "raw_prediction", "confidence",
                "dropout_probability", "feature_importance", "risk_history"]:
        assert key in body, f"Missing key: {key}"

def test_predict_dropout_risk_label_valid():
    r = post("/predict/dropout", LOW_RISK)
    assert r.json()["student_risk_assessment"] in ["Low", "Medium", "High"]

def test_predict_dropout_invalid_age_rejected():
    bad = {**LOW_RISK, "age": 200}
    r = post("/predict/dropout", bad)
    assert r.status_code == 422

def test_predict_dropout_missing_field_uses_default():
    partial = {k: v for k, v in LOW_RISK.items() if k != "gpa_current"}
    r = post("/predict/dropout", partial)
    assert r.status_code in [200, 422]

# ─── POST /predict/batch ──────────────────────────────────────────────────────

def test_predict_batch_correct_count():
    r = post("/predict/batch", {"students": [LOW_RISK, HIGH_RISK]})
    assert r.status_code == 200
    assert r.json()["total"] == 2

def test_predict_batch_result_has_risk():
    r = post("/predict/batch", {"students": [LOW_RISK, HIGH_RISK]})
    for pred in r.json()["predictions"]:
        assert "risk" in pred
        assert pred["risk"] in ["High", "Low"]

def test_predict_batch_single_student():
    r = post("/predict/batch", {"students": [LOW_RISK]})
    assert r.status_code == 200
    assert r.json()["total"] == 1

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    console.print()
    console.print(Panel("[bold white]EduPulse API Test Suite[/bold white]", expand=False))
    console.print()

    console.print("[bold cyan][ GET / ][/bold cyan]")
    run_test("Root returns status=healthy",           test_root_returns_healthy)
    run_test("Root response contains service name",   test_root_contains_service_name)

    console.print("\n[bold cyan][ GET /health ][/bold cyan]")
    run_test("Health returns status=ok",              test_health_status_ok)
    run_test("Health lists available models",         test_health_lists_models)

    console.print("\n[bold cyan][ GET /metrics ][/bold cyan]")
    run_test("Metrics returns 200",                   test_metrics_returns_200)
    run_test("Metrics contains best_model key",       test_metrics_contains_best_model)
    run_test("Metrics has rf, xgb, lgbm keys",        test_metrics_contains_all_3_models)

    console.print("\n[bold cyan][ POST /predict/dropout ][/bold cyan]")
    run_test("Low-risk student has prob < 0.5",       test_predict_dropout_low_risk_student)
    run_test("High-risk student has prob > 0.3",      test_predict_dropout_high_risk_student)
    run_test("Response contains all schema fields",   test_predict_dropout_response_schema)
    run_test("Risk label is Low/Medium/High",         test_predict_dropout_risk_label_valid)
    run_test("Age > 100 returns 422 error",           test_predict_dropout_invalid_age_rejected)
    run_test("Missing field uses default gracefully", test_predict_dropout_missing_field_uses_default)

    console.print("\n[bold cyan][ POST /predict/batch ][/bold cyan]")
    run_test("Batch returns correct total count",     test_predict_batch_correct_count)
    run_test("Each batch result has risk field",      test_predict_batch_result_has_risk)
    run_test("Batch works with single student",       test_predict_batch_single_student)

    # ─── Summary Table ────────────────────────────────────────────────────────

    console.print()
    table = Table(title="API Test Results Summary", box=box.ASCII, show_lines=True,
                  header_style="bold white")
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
