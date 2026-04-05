import pandas as pd
import numpy as np
from typing import Optional
from logging import getLogger

logger = getLogger(__name__)

class DataValidator:
    """Vectorized schema validation for large datasets."""

    SCHEMA = {
        'age': {'min': 15, 'max': 100},
        'gpa_current': {'min': 0.0, 'max': 4.5},
        'attendance_rate': {'min': 0.0, 'max': 100.0},
        'midterm_score': {'min': 0.0, 'max': 100.0},
        'assignment_score': {'min': 0.0, 'max': 100.0},
        'quiz_score': {'min': 0.0, 'max': 100.0},
        'login_frequency': {'min': 0},
    }

    REQUIRED_COLUMNS = [
        'student_id', 'age', 'gender', 'gpa_current', 'attendance_rate',
        'login_frequency', 'missed_deadlines_count', 'dropout_risk'
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def validate_schema(self) -> pd.DataFrame:
        df = self.df.copy()
        initial = len(df)

        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        for col, bounds in self.SCHEMA.items():
            if col in df.columns:
                # Skip NaN for optional columns
                mask = df[col].notna()
                if 'min' in bounds:
                    df = df[~mask | (df[col] >= bounds['min'])]
                if 'max' in bounds:
                    df = df[~mask | (df[col] <= bounds['max'])]

        dropped = initial - len(df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} invalid records.")
        else:
            logger.info(f"All {initial:,} records passed validation.")
        return df.reset_index(drop=True)

    def detect_missing_values(self) -> dict:
        missing = self.df.isnull().sum()
        return missing[missing > 0].to_dict()

    def get_summary(self) -> dict:
        return {
            'total_records': len(self.df),
            'columns': len(self.df.columns),
            'missing_values': self.detect_missing_values(),
            'dtypes': self.df.dtypes.astype(str).to_dict()
        }
