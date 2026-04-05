import pandas as pd
from typing import List, Tuple
from logging import getLogger
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = getLogger(__name__)

class DataEncoder:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.scalers = {}
        self.label_encoders = {}
        
    def encode_categorical(self, categorical_cols: List[str]) -> pd.DataFrame:
        """
        Applies Label Encoding to categorical columns.
        Saves the encoders in self.label_encoders for future inverse transforms.
        """
        logger.info(f"Label encoding columns: {categorical_cols}")
        df_encoded = self.df.copy()
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
            else:
                logger.warning(f"Column {col} not found for encoding. Skipping.")
                
        return df_encoded
        
    def scale_features(self, numeric_cols: List[str]) -> pd.DataFrame:
        """
        Scales numeric features using StandardScaler.
        """
        logger.info(f"Standard scaling columns: {numeric_cols}")
        df_scaled = self.df.copy()
        
        if numeric_cols:
            scaler = StandardScaler()
            df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
            self.scalers['standard_scaler'] = scaler
            
        return df_scaled
