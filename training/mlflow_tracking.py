# training/experiment_tracking.py
import mlflow
import os
import logging
from config.settings import MLFLOW_TRACKING_URI, EXPERIMENT_NAME

logger = logging.getLogger(__name__)

class MLflowTracker:
    """Handles MLflow experiment tracking"""
    
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.experiment = mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info(f"MLflow tracking at {MLFLOW_TRACKING_URI}, experiment: {EXPERIMENT_NAME}")
    
    def start_run(self, run_name=None):
        """Start a new MLflow run"""
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params):
        """Log parameters to MLflow"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics, step=None):
        """Log metrics to MLflow"""
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model, artifact_path="model"):
        """Log model to MLflow"""
        mlflow.pytorch.log_model(model, artifact_path)
    
    def log_artifact(self, local_path):
        """Log artifact to MLflow"""
        mlflow.log_artifact(local_path)
    
    def get_best_run(self, metric_name="val_f1", ascending=False):
        """Get the best run based on a metric"""
        client = mlflow.tracking.MlflowClient()
        
        # Get all runs for the experiment
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
        )
        
        if runs:
            return runs[0]
        return None
    
    def load_model(self, run_id, model_path="model"):
        """Load model from MLflow run"""
        return mlflow.pytorch.load_model(f"runs:/{run_id}/{model_path}")