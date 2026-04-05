import pandas as pd
import joblib
import os
import json
from typing import Dict, Any, List
from logging import getLogger
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

logger = getLogger(__name__)

class ModelTrainer:
    """
    Multi-model trainer with column order persistence.
    BUG 9 FIX: Removed deprecated use_label_encoder from XGBoost.
    IMPROVEMENT 1: Saves feature column order alongside model.
    """
    SUPPORTED_MODELS = {
        'rf': 'Random Forest',
        'xgb': 'XGBoost',
        'lgbm': 'LightGBM'
    }

    def __init__(self, data: pd.DataFrame, target_col: str, model_type: str = 'rf'):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type '{model_type}'. Choose from {list(self.SUPPORTED_MODELS.keys())}")
        self.data = data.copy()
        self.target_col = target_col
        self.model_type = model_type
        self.model = None
        self.feature_columns = None

    def _create_model(self, random_state: int = 42):
        if self.model_type == 'rf':
            return RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=random_state, n_jobs=-1)
        elif self.model_type == 'xgb':
            return XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, eval_metric='logloss', random_state=random_state)
        elif self.model_type == 'lgbm':
            return LGBMClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=random_state, verbose=-1)

    def train(self, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        logger.info(f"Training {self.SUPPORTED_MODELS[self.model_type]} for target: {self.target_col}")
        self.data = self.data.dropna(subset=[self.target_col])
        y = self.data[self.target_col]
        X = self.data.drop(columns=[self.target_col])
        X = X.select_dtypes(include=['number'])

        # IMPROVEMENT 1: Save feature column order
        self.feature_columns = X.columns.tolist()

        # Use stratified split for imbalanced data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.model = self._create_model(random_state)
        self.model.fit(X_train, y_train)
        logger.info(f"{self.SUPPORTED_MODELS[self.model_type]} trained successfully on {len(X_train)} samples.")

        return {
            "model": self.model,
            "model_type": self.model_type,
            "model_name": self.SUPPORTED_MODELS[self.model_type],
            "feature_columns": self.feature_columns,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test
        }

    def save_model(self, path: str):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

        # Save feature columns alongside model
        meta_path = path.replace('.pkl', '_meta.json')
        with open(meta_path, 'w') as f:
            json.dump({'feature_columns': self.feature_columns, 'model_type': self.model_type}, f)

        logger.info(f"Model + metadata saved to {path}")
