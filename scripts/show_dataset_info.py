import os
import logging
import pandas as pd
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DatasetInfo')

def show_dataset_info():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'synthetic_student_data.csv')
    
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}")
        return

    try:
        logger.info(f"Loading dataset from: {data_path}")
        df = pd.read_csv(data_path)
        
        logger.info("=" * 50)
        logger.info("DATASET ATTRIBUTES")
        logger.info("=" * 50)
        
        # Capture dataframe info to log
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        
        for line in info_str.split('\n'):
            if line.strip():
                logger.info(line)
                
        logger.info("=" * 50)
        logger.info("DATASET HEAD (FIRST 5 ROWS)")
        logger.info("=" * 50)
        
        head_str = df[['age', 'department', 'gpa_current', 'attendance_rate', 'final_grade', 'dropout_risk']].head(5).to_string()
        for line in head_str.split('\n'):
            logger.info(line)
            
    except Exception as e:
        logger.error(f"An error occurred while processing the dataset: {e}")

if __name__ == '__main__':
    show_dataset_info()
