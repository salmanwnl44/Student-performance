import pandas as pd
import numpy as np
import uuid
import os

def generate_student_data(num_records=50000):
    """
    Generates a large, realistic synthetic student dataset with:
    - Correlated features mimicking real-world academic patterns
    - Semester-level time dimension
    - Realistic noise and edge cases
    - Balanced-enough dropout rate (~18-22%)
    """
    np.random.seed(42)
    print(f"Generating {num_records:,} student records...")

    # === DEMOGRAPHICS ===
    student_ids = [str(uuid.uuid4()) for _ in range(num_records)]
    ages = np.random.choice(range(17, 28), size=num_records, p=[
        0.02, 0.15, 0.22, 0.20, 0.15, 0.10, 0.06, 0.04, 0.03, 0.02, 0.01
    ])
    genders = np.random.choice(['Male', 'Female', 'Non-Binary'], size=num_records, p=[0.47, 0.49, 0.04])
    departments = np.random.choice(
        ['Computer Science', 'Engineering', 'Business', 'Arts', 'Science', 'Medicine'],
        size=num_records, p=[0.25, 0.20, 0.18, 0.12, 0.15, 0.10]
    )
    semesters = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], size=num_records, p=[
        0.18, 0.16, 0.14, 0.13, 0.12, 0.10, 0.09, 0.08
    ])

    # === SOCIOECONOMIC ===
    parent_edu = np.random.choice(
        ['No Formal Education', 'High School', 'Bachelor', 'Master', 'PhD'],
        size=num_records, p=[0.05, 0.30, 0.40, 0.18, 0.07]
    )
    family_income = np.random.choice(
        ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High'],
        size=num_records, p=[0.10, 0.20, 0.35, 0.25, 0.10]
    )
    commute_distance = np.clip(np.random.exponential(scale=12, size=num_records), 0.5, 80)
    scholarship = np.random.choice([0, 1], size=num_records, p=[0.72, 0.28])
    has_internet = np.random.choice([0, 1], size=num_records, p=[0.08, 0.92])
    part_time_job = np.random.choice([0, 1], size=num_records, p=[0.65, 0.35])

    # === BUILD LATENT "ACADEMIC STRENGTH" SCORE ===
    # This drives correlated features realistically
    edu_score = np.zeros(num_records)
    edu_score[parent_edu == 'No Formal Education'] = -15
    edu_score[parent_edu == 'High School'] = 0
    edu_score[parent_edu == 'Bachelor'] = 10
    edu_score[parent_edu == 'Master'] = 18
    edu_score[parent_edu == 'PhD'] = 25

    income_score = np.zeros(num_records)
    income_score[family_income == 'Low'] = -12
    income_score[family_income == 'Lower-Middle'] = -5
    income_score[family_income == 'Middle'] = 0
    income_score[family_income == 'Upper-Middle'] = 8
    income_score[family_income == 'High'] = 15

    # Latent score combines background + randomness (talent/motivation)
    latent_strength = (
        edu_score * 0.3 +
        income_score * 0.25 +
        scholarship * 12 +
        has_internet * 8 -
        part_time_job * 6 -
        (commute_distance - 10) * 0.3 +
        np.random.normal(0, 15, size=num_records)  # individual variance
    )

    # === ACADEMIC FEATURES (driven by latent strength) ===
    attendance_rate = np.clip(70 + latent_strength * 0.4 + np.random.normal(0, 8, size=num_records), 15, 100)

    # Midterm & assignment scores
    midterm_score = np.clip(55 + latent_strength * 0.5 + attendance_rate * 0.15 + np.random.normal(0, 10, size=num_records), 0, 100)
    assignment_score = np.clip(60 + latent_strength * 0.3 + np.random.normal(0, 12, size=num_records), 0, 100)
    quiz_score = np.clip(50 + latent_strength * 0.35 + np.random.normal(0, 14, size=num_records), 0, 100)

    # === BEHAVIORAL LMS DATA ===
    login_lambda = np.clip(attendance_rate / 10 * 2.5 + has_internet * 5, 1, 80)
    login_frequency = np.random.poisson(login_lambda)
    forum_lambda = np.clip(login_frequency.astype(float) / 3, 0.1, 50)
    forum_participation = np.random.poisson(forum_lambda)
    time_spent = np.clip(login_frequency * np.random.uniform(0.3, 2.5, size=num_records), 0, 60)

    # Study hours (partially independent, partially correlated)
    study_hours = np.clip(
        5 + latent_strength * 0.15 + np.random.normal(0, 5, size=num_records) - part_time_job * 4,
        0, 40
    )

    # Library usage
    library_usage = np.clip(np.random.poisson(study_hours / 3), 0, 30)

    # Missed deadlines — inversely correlated with strength
    deadline_lambda = np.clip(5 - latent_strength * 0.08 + part_time_job * 2, 0.1, 15)
    missed_deadlines = np.random.poisson(deadline_lambda)

    # === GPA CALCULATION (composite) ===
    raw_gpa = (
        midterm_score * 0.012 +
        assignment_score * 0.008 +
        quiz_score * 0.006 +
        attendance_rate * 0.005 +
        study_hours * 0.015 +
        np.random.normal(0, 0.3, size=num_records)
    )
    gpa_current = np.clip(raw_gpa, 0.0, 4.0)

    # === TARGET: Final Grade ===
    conditions = [gpa_current >= 3.5, gpa_current >= 3.0, gpa_current >= 2.0, gpa_current >= 1.5]
    choices = ['A', 'B', 'C', 'D']
    final_grade = np.select(conditions, choices, default='F')

    # === TARGET: Dropout Risk (multi-factor) ===
    risk_score = (
        (100 - attendance_rate) * 0.45 +
        missed_deadlines * 3.5 +
        (4.0 - gpa_current) * 12 +
        (100 - midterm_score) * 0.15 +
        part_time_job * 5 -
        scholarship * 10 -
        has_internet * 5 +
        np.where(family_income == 'Low', 8, 0) +
        np.where(semesters <= 2, 5, 0)  # first-year students have higher risk
    )
    dropout_risk = np.where(risk_score > 48, 1, 0)

    # === ASSEMBLE DATAFRAME ===
    df = pd.DataFrame({
        'student_id': student_ids,
        'age': ages,
        'gender': genders,
        'department': departments,
        'semester': semesters,
        'parent_education': parent_edu,
        'family_income': family_income,
        'commute_distance': np.round(commute_distance, 1),
        'scholarship_status': scholarship,
        'has_internet_access': has_internet,
        'part_time_job': part_time_job,
        'gpa_current': np.round(gpa_current, 2),
        'attendance_rate': np.round(attendance_rate, 1),
        'midterm_score': np.round(midterm_score, 1),
        'assignment_score': np.round(assignment_score, 1),
        'quiz_score': np.round(quiz_score, 1),
        'login_frequency': login_frequency,
        'forum_participation_score': forum_participation,
        'time_spent_on_materials': np.round(time_spent, 1),
        'study_hours_weekly': np.round(study_hours, 1),
        'library_usage_weekly': library_usage,
        'missed_deadlines_count': missed_deadlines,
        'final_grade': final_grade,
        'dropout_risk': dropout_risk
    })

    # Inject ~2% missing values in non-critical columns for realism
    for col in ['midterm_score', 'quiz_score', 'library_usage_weekly', 'study_hours_weekly']:
        mask = np.random.random(num_records) < 0.02
        df.loc[mask, col] = np.nan

    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/synthetic_student_data.csv', index=False)

    print(f"\n{'='*50}")
    print(f"Generated {num_records:,} records")
    print(f"Columns: {len(df.columns)}")
    print(f"Dropout Rate: {df['dropout_risk'].mean():.2%}")
    print(f"\nGrade Distribution:")
    print(df['final_grade'].value_counts().sort_index().to_string())
    print(f"\nMissing Values:")
    print(df.isnull().sum()[df.isnull().sum() > 0].to_string())
    print(f"{'='*50}")

if __name__ == "__main__":
    generate_student_data()
