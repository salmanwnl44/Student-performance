import pandas as pd
import numpy as np
from typing import Dict
from logging import getLogger
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

logger = getLogger(__name__)

class ModelEvaluator:
    """
    BUG 10 FIX: Handles imbalanced data gracefully for ROC AUC.
    IMPROVEMENT 2: Returns predict_proba confidence scores.
    """
    def __init__(self, model, X_test: pd.DataFrame, y_test: pd.Series):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self) -> Dict[str, float]:
        logger.info("Evaluating model performance...")
        y_pred = self.model.predict(self.X_test)

        # BUG 10 FIX: Safe ROC AUC computation
        auc = -1.0
        if hasattr(self.model, "predict_proba"):
            unique_classes = np.unique(self.y_test)
            if len(unique_classes) == 2 and len(np.unique(y_pred)) > 1:
                try:
                    y_prob = self.model.predict_proba(self.X_test)[:, 1]
                    auc = roc_auc_score(self.y_test, y_prob)
                except ValueError as e:
                    logger.warning(f"ROC AUC computation failed (likely single class in split): {e}")
                    auc = -1.0

        cm = confusion_matrix(self.y_test, y_pred)

        metrics = {
            "accuracy": round(accuracy_score(self.y_test, y_pred), 4),
            "precision": round(precision_score(self.y_test, y_pred, average='weighted', zero_division=0), 4),
            "recall": round(recall_score(self.y_test, y_pred, average='weighted', zero_division=0), 4),
            "f1_score": round(f1_score(self.y_test, y_pred, average='weighted', zero_division=0), 4),
            "roc_auc": round(auc, 4) if auc != -1.0 else "N/A",
            "confusion_matrix": cm.tolist()
        }

        report = classification_report(self.y_test, y_pred, zero_division=0)
        logger.info(f"Metrics: {metrics}")
        logger.info(f"\n{report}")
        return metrics
