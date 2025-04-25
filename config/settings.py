# config/settings.py
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Kafka settings
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_RAW = "xraygpt-raw"
KAFKA_TOPIC_PROCESSED = "xraygpt-processed"

# Spark settings
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
SPARK_APP_NAME = "XrayGPT-ETL"

# MIMIC-CXR dataset settings
MIMIC_DATA_PATH = os.getenv("MIMIC_DATA_PATH", os.path.join(DATA_DIR, "mimic-cxr"))
MIMIC_CSV_PATH = os.path.join(MIMIC_DATA_PATH, "mimic-cxr-2.0.0-metadata.csv")
MIMIC_IMAGES_PATH = os.path.join(MIMIC_DATA_PATH, "files")

# Model settings
MODEL_TYPE = "bio-gpt"
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

# MLflow settings
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:6000")
EXPERIMENT_NAME = "XrayGPT"