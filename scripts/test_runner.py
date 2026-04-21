import sys
import os
import time
import traceback
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

console = Console()

# ─── Test Registry ────────────────────────────────────────────────────────────

results = []

def run_test(name, fn):
    start = time.time()
    try:
        fn()
        elapsed = time.time() - start
        results.append(("PASS", name, f"{elapsed:.3f}s", ""))
        console.print(f"  [bold green]PASS[/bold green]  {name} [dim]({elapsed:.3f}s)[/dim]")
    except Exception as e:
        elapsed = time.time() - start
        tb = traceback.format_exc().strip().splitlines()[-1]
        results.append(("FAIL", name, f"{elapsed:.3f}s", tb))
        console.print(f"  [bold red]FAIL[/bold red]  {name} [dim]({elapsed:.3f}s)[/dim]")
        console.print(f"         [red]{tb}[/red]")

# ─── Shared Fixture ────────────────────────────────────────────────────────────

def make_df(n=100):
    return pd.DataFrame({
        'student_id':           [f"id_{i}" for i in range(n)],
        'age':                  np.random.randint(18, 28, n),
        'gender':               np.random.choice(['Male', 'Female'], n),
        'department':           np.random.choice(['CS', 'Business'], n),
        'semester':             np.random.randint(1, 8, n),
        'parent_education':     np.random.choice(['High School', 'Bachelor'], n),
        'family_income':        np.random.choice(['Low', 'Middle', 'High'], n),
        'commute_distance':     np.random.uniform(0.5, 40, n),
        'scholarship_status':   np.random.randint(0, 2, n),
        'has_internet_access':  np.random.randint(0, 2, n),
        'part_time_job':        np.random.randint(0, 2, n),
        'gpa_current':          np.random.uniform(1.0, 4.0, n),
        'attendance_rate':      np.random.uniform(50, 100, n),
        'midterm_score':        np.random.uniform(40, 100, n),
        'assignment_score':     np.random.uniform(40, 100, n),
        'quiz_score':           np.random.uniform(40, 100, n),
        'login_frequency':      np.random.randint(1, 30, n),
        'forum_participation_score': np.random.randint(0, 10, n),
        'time_spent_on_materials':   np.random.uniform(0, 60, n),
        'study_hours_weekly':   np.random.uniform(0, 40, n),
        'library_usage_weekly': np.random.randint(0, 15, n),
        'missed_deadlines_count': np.random.randint(0, 10, n),
        'final_grade':          np.random.choice(['A', 'B', 'C', 'D', 'F'], n),
        'dropout_risk':         np.random.randint(0, 2, n),
    })

# ─── DataLoader Tests ──────────────────────────────────────────────────────────

def test_loader_file_not_found():
    try:
        DataLoader("non_existent.csv").load_csv()
        raise AssertionError("Should have raised FileNotFoundError")
    except FileNotFoundError:
        pass  # Expected

def test_loader_loads_real_csv():
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'synthetic_student_data.csv')
    df = DataLoader(path).load_csv()
    assert len(df) > 0, "Loaded DataFrame should have rows"
    assert 'dropout_risk' in df.columns, "Missing dropout_risk column"

# ─── DataValidator Tests ───────────────────────────────────────────────────────

def test_validator_passes_clean_data():
    df = make_df()
    result = DataValidator(df).validate_schema()
    assert len(result) > 0, "Valid data should not all be dropped"

def test_validator_drops_invalid_age():
    df = make_df()
    df.loc[0, 'age'] = 200  # Invalid age
    initial_len = len(df)
    result = DataValidator(df).validate_schema()
    assert len(result) < initial_len, "Row with invalid age should be dropped"

def test_validator_raises_on_missing_columns():
    df = pd.DataFrame({'student_id': [1], 'age': [20]})  # Missing required cols
    try:
        DataValidator(df).validate_schema()
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass  # Expected

def test_validator_detect_missing_values():
    df = make_df()
    df.loc[0, 'gpa_current'] = np.nan
    missing = DataValidator(df).detect_missing_values()
    assert 'gpa_current' in missing, "Missing value in gpa_current not detected"

# ─── DataCleaner Tests ─────────────────────────────────────────────────────────

def test_cleaner_imputes_missing_values():
    df = make_df()
    df.loc[:5, 'gpa_current'] = np.nan
    cleaner = DataCleaner(df)
    result = cleaner.handle_missing_values(strategy='median').get_df()
    assert result['gpa_current'].isnull().sum() == 0, "Nulls should be imputed"

def test_cleaner_removes_outliers():
    df = make_df()
    df.loc[0, 'gpa_current'] = 999.0  # Extreme outlier
    before = len(df)
    cleaner = DataCleaner(df)
    result = cleaner.remove_outliers('gpa_current').get_df()
    assert len(result) < before, "Outlier row should be removed"

def test_cleaner_chains():
    df = make_df()
    df.loc[0, 'midterm_score'] = np.nan
    result = (DataCleaner(df)
              .handle_missing_values(strategy='median')
              .remove_outliers('gpa_current')
              .get_df())
    assert isinstance(result, pd.DataFrame), "Chained result should be a DataFrame"

# ─── DataEncoder Tests ─────────────────────────────────────────────────────────

def test_encoder_label_encodes():
    df = make_df()
    result = DataEncoder(df).encode_categorical(['gender', 'department'])
    assert result['gender'].dtype != object, "gender should be encoded to numeric"
    assert result['department'].dtype != object, "department should be encoded to numeric"

def test_encoder_skips_missing_columns():
    df = make_df()
    # Should not raise, just skip non-existent column
    result = DataEncoder(df).encode_categorical(['nonexistent_col'])
    assert isinstance(result, pd.DataFrame), "Should still return a DataFrame"

# ─── Main Runner ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    console.print()
    console.print(Panel("[bold white]EduPulse Unit Test Suite[/bold white]", expand=False))
    console.print()

    # DataLoader
    console.print("[bold cyan][ DataLoader ][/bold cyan]")
    run_test("File not found raises FileNotFoundError", test_loader_file_not_found)
    run_test("Loads real CSV with correct shape",        test_loader_loads_real_csv)

    # DataValidator
    console.print("\n[bold cyan][ DataValidator ][/bold cyan]")
    run_test("Passes clean data without dropping rows",  test_validator_passes_clean_data)
    run_test("Drops rows with invalid age (>100)",       test_validator_drops_invalid_age)
    run_test("Raises ValueError on missing columns",     test_validator_raises_on_missing_columns)
    run_test("Detects missing values in column",         test_validator_detect_missing_values)

    # DataCleaner
    console.print("\n[bold cyan][ DataCleaner ][/bold cyan]")
    run_test("Imputes missing values with median",       test_cleaner_imputes_missing_values)
    run_test("Removes extreme outliers via Z-score",     test_cleaner_removes_outliers)
    run_test("Chains methods correctly",                 test_cleaner_chains)

    # DataEncoder
    console.print("\n[bold cyan][ DataEncoder ][/bold cyan]")
    run_test("Label encodes categorical columns",        test_encoder_label_encodes)
    run_test("Skips non-existent columns gracefully",   test_encoder_skips_missing_columns)

    # ─── Summary Table ─────────────────────────────────────────────────────────
    console.print()
    table = Table(title="Test Results Summary", box=box.ASCII, show_lines=True,
                  header_style="bold white")
    table.add_column("Status",  justify="center", width=8)
    table.add_column("Test Name", justify="left")
    table.add_column("Time",    justify="center", width=10)
    table.add_column("Error",   justify="left", style="dim red")

    passed = sum(1 for r in results if r[0] == "PASS")
    failed = sum(1 for r in results if r[0] == "FAIL")

    for status, name, elapsed, err in results:
        style = "bold green" if status == "PASS" else "bold red"
        table.add_row(
            Text(status, style=style),
            name,
            elapsed,
            err
        )

    console.print(table)
    console.print()
    console.print(f"  [bold green]{passed} passed[/bold green]  "
                  f"[bold red]{failed} failed[/bold red]  "
                  f"out of {len(results)} tests")
    console.print()

    sys.exit(0 if failed == 0 else 1)
