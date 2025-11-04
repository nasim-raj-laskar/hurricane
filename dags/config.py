"""Configuration for Hurricane Damage Training Pipeline"""
from datetime import datetime, timedelta
import os

# S3 Configuration
S3_BUCKET = 'hurricane-damage-data'

# Data Configuration
DATA_DIR = '/tmp/data'
LOCAL_ZIP = os.path.join(DATA_DIR, 'dataset.zip')

# Model Configuration
IMG_SIZE = (128, 128)
BATCH = 16
EPOCHS = 2
LEARNING_RATE = 0.001
OPTIMIZER = "Adam"

# DAG Configuration
DEFAULT_ARGS = {
    'owner': 'hurricane-team',
    'start_date': datetime(2025, 10, 26),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# MLflow Configuration
MLFLOW_EXPERIMENT_NAME = "hurricane_damage_training_v5"
