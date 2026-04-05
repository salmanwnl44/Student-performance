import pandas as pd
import os
from logging import getLogger, INFO, StreamHandler

logger = getLogger(__name__)
logger.setLevel(INFO)
handler = StreamHandler()
logger.addHandler(handler)

class DataLoader:
    """
    Handles robust data ingestion for the Student Performance Pipeline.
    Supports CSV loading currently, but architected to scale to DBs and APIs.
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_csv(self) -> pd.DataFrame:
        """Loads data from a local CSV file into a pandas DataFrame."""
        if not os.path.exists(self.data_path):
            logger.error(f"Data file not found at: {self.data_path}")
            raise FileNotFoundError(f"Data file not found at: {self.data_path}")
        
        logger.info(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        logger.info(f"Successfully loaded {len(df)} records with {len(df.columns)} columns.")
        return df

    # Scalability hooks
    def load_database(self, query: str):
        raise NotImplementedError("Database ingestion module coming in v1.1")

    def load_api_stream(self, endpoint: str):
        raise NotImplementedError("Streaming API ingestion coming in v1.2")
