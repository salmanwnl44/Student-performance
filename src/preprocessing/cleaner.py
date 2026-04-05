import pandas as pd
from typing import List, Optional
from logging import getLogger
from sklearn.impute import SimpleImputer
import numpy as np

logger = getLogger(__name__)

class DataCleaner:
    """
    BUG 11 FIX: Methods now update self.df so they chain properly.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def handle_missing_values(self, strategy: str = 'median', columns: Optional[List[str]] = None) -> 'DataCleaner':
        logger.info(f"Handling missing values using {strategy} strategy.")
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(columns) > 0:
            imputer = SimpleImputer(strategy=strategy)
            self.df[columns] = imputer.fit_transform(self.df[columns])
        return self

    def remove_outliers(self, column: str, z_thresh: float = 3.0) -> 'DataCleaner':
        logger.info(f"Removing outliers in {column} with Z > {z_thresh}")
        mean = self.df[column].mean()
        std = self.df[column].std()
        if std == 0:
            return self
        z_scores = np.abs((self.df[column] - mean) / std)
        before = len(self.df)
        self.df = self.df[z_scores < z_thresh]
        logger.info(f"Dropped {before - len(self.df)} outlier records from {column}.")
        return self

    def get_df(self) -> pd.DataFrame:
        return self.df.copy()
