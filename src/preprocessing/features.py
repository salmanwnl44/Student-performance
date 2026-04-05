import pandas as pd
from typing import List, Tuple
from logging import getLogger
from sklearn.ensemble import RandomForestClassifier

logger = getLogger(__name__)

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def find_highly_correlated(self, threshold: float = 0.85) -> List[Tuple[str, str]]:
        """
        Identifies pairs of features that are highly correlated.
        This helps in taking decisions for dimensionality reduction.
        """
        logger.info(f"Finding correlated features with threshold {threshold}")
        corr_matrix = self.df.corr().abs()
        
        correlated_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > threshold:
                    colname = corr_matrix.columns[i]
                    correlated_pairs.append((corr_matrix.columns[j], colname))
                    
        return correlated_pairs

    def extract_feature_importance(self, target_col: str) -> pd.DataFrame:
        """
        Quickly computes feature importance using a RandomForest logic.
        """
        logger.info(f"Extracting Feature Importance for target: {target_col}")
        
        df_feat = self.df.copy().dropna()
        y = df_feat[target_col]
        X = df_feat.drop(columns=[target_col])
        
        # Ensure categorical dropping or encoding if running generically
        # For safety in utility function, drop non-numerics
        X = X.select_dtypes(include=['number'])

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        return importance_df
