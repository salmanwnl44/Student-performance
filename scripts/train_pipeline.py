import os
import sys
import json
import logging
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ingestion.data_loader import DataLoader
from src.validation.data_validator import DataValidator
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.encoder import DataEncoder
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Pipeline')

def run_pipeline():
    start = time.time()
    logger.info("=" * 60)
    logger.info("STUDENT PERFORMANCE ML — TRAINING PIPELINE v2.1")
    logger.info("=" * 60)

    # 1. Ingest
    loader = DataLoader('data/raw/synthetic_student_data.csv')
    raw_df = loader.load_csv()

    # 2. Validate
    validator = DataValidator(raw_df)
    valid_df = validator.validate_schema()
    logger.info(f"Validation: {len(valid_df):,} records passed.")

    # 3. Clean (chained)
    cleaner = DataCleaner(valid_df)
    clean_df = (cleaner
                .handle_missing_values(strategy='median')
                .remove_outliers('gpa_current')
                .remove_outliers('attendance_rate')
                .remove_outliers('midterm_score')
                .get_df())
    logger.info(f"After cleaning: {len(clean_df):,} records.")

    # 4. Encode all categoricals
    encoder = DataEncoder(clean_df)
    categorical_cols = ['gender', 'parent_education', 'department', 'family_income']
    # Only encode columns that exist
    categorical_cols = [c for c in categorical_cols if c in clean_df.columns]
    processed_df = encoder.encode_categorical(categorical_cols)

    # Drop non-feature columns
    drop_cols = ['student_id', 'final_grade']
    drop_cols = [c for c in drop_cols if c in processed_df.columns]
    model_df = processed_df.drop(columns=drop_cols)

    # 5. Multi-Model Training
    model_types = ['rf', 'xgb', 'lgbm']
    all_metrics = {}
    best_model_name = None
    best_f1 = -1

    for mt in model_types:
        logger.info(f"\n{'—' * 40}")
        logger.info(f"Training: {mt.upper()} on {len(model_df):,} samples")
        trainer = ModelTrainer(model_df, target_col='dropout_risk', model_type=mt)
        results = trainer.train(test_size=0.2)

        evaluator = ModelEvaluator(results['model'], results['X_test'], results['y_test'])
        metrics = evaluator.evaluate()
        all_metrics[mt] = {'name': results['model_name'], 'metrics': metrics}

        model_path = os.path.join('models', f"{mt}_dropout_model.pkl")
        trainer.save_model(model_path)

        f1 = metrics['f1_score']
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = mt

    # 6. Save metrics
    all_metrics['best_model'] = best_model_name
    all_metrics['dataset_size'] = len(model_df)
    all_metrics['features_count'] = len(model_df.columns) - 1
    metrics_path = os.path.join('models', 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)

    elapsed = time.time() - start
    logger.info(f"\n{'=' * 60}")
    logger.info(f"BEST MODEL: {best_model_name.upper()} (F1: {best_f1:.4f})")
    logger.info(f"Trained on {len(model_df):,} records with {len(model_df.columns)-1} features")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"{'=' * 60}")

if __name__ == '__main__':
    run_pipeline()
