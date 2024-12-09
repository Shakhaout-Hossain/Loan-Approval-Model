# mlflow ui --backend-store-uri sqlite:///experiment.db
import mlflow
import mlflow.sklearn
import joblib
import logging
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix


class MLFlowTracker:
    def __init__(self, experiment_name: str = "loan_approval_model_experiment", 
                 sqlite_db_path: str = './mlflow.db', artifact_location: str = './artifacts'):
        """
        Initializes the MLFlowTracker class for managing experiment tracking with MLFlow using SQLite.

        Args:
            experiment_name (str): The name of the MLFlow experiment.
            sqlite_db_path (str): Path to the SQLite database to store experiment data.
            artifact_location (str): The location to store artifacts like models and metrics.
        """
        self.experiment_name = experiment_name
        self.sqlite_db_path = sqlite_db_path
        self.artifact_location = artifact_location
        
        # Set up the MLFlow experiment and SQLite backend URI
        mlflow.set_tracking_uri(f"sqlite:///{self.sqlite_db_path}")
        mlflow.set_experiment(self.experiment_name)
        
        # Ensure artifact directory exists
        os.makedirs(self.artifact_location, exist_ok=True)
    
    def start_run(self):
        """Start a new MLFlow run"""
        mlflow.start_run()
    
    def log_metrics(self, model_name: str, y_test: np.ndarray, y_pred: np.ndarray):
        """
        Log evaluation metrics such as accuracy, precision, recall, F1-score, MCC, and confusion matrix.

        Args:
            model_name (str): Name of the model.
            y_test (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        """
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred).ravel()

        # Log metrics to MLFlow
        mlflow.log_metric(f"{model_name}_accuracy", accuracy)
        mlflow.log_metric(f"{model_name}_precision", precision)
        mlflow.log_metric(f"{model_name}_recall", recall)
        mlflow.log_metric(f"{model_name}_f1_score", f1)
        mlflow.log_metric(f"{model_name}_mcc", mcc)

        # Log confusion matrix as an artifact
        np.save(os.path.join(self.artifact_location, f"{model_name}_confusion_matrix.npy"), conf_matrix)
        mlflow.log_artifact(os.path.join(self.artifact_location, f"{model_name}_confusion_matrix.npy"))
        
    
    def log_model(self, model, model_name: str):
        """Log the trained model to MLFlow"""
        mlflow.sklearn.log_model(model, model_name)
    
    def log_artifact(self, artifact_path: str):
        """Log an artifact (e.g., preprocessing pipeline) to MLFlow"""
        mlflow.log_artifact(artifact_path)
    
    def end_run(self):
        """End the current MLFlow run"""
        mlflow.end_run()

    def register_model(self, model, model_name: str):
        """Register the model in MLFlow Model Registry"""
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, model_name)
    
    # def transition_to_production(self, model_name: str, version: int):
    #     """Transition a model to 'Production' stage in MLFlow Model Registry"""
    #     client = mlflow.client.MlflowClient()
    #     client.transition_model_version_stage(
    #         name=model_name,
    #         version=version,
    #         stage="Production"
    #     )

    def transition_to_production(self, model_name: str):
        """Transition a model to 'Production' stage in MLFlow Model Registry"""
        client = mlflow.client.MlflowClient()

        # Get the latest registered model version
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
        
        # Transition to 'Production' stage
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Production"
        )
        print(f"Model {model_name} version {latest_version} moved to Production.")
        
    def load_model(self, model_name: str):
        """
        Load a registered model from MLFlow Model Registry.

        Args:
            model_name (str): The name of the model to load.

        Returns:
            The loaded model.
        """
        try:
            # Construct the model URI using the model name and stage (e.g., 'Production')
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.pyfunc.load_model(model_uri)
            logging.info(f"Model {model_name} successfully loaded from {model_uri}.")
            return model
        except Exception as e:
            logging.error(f"Error loading model {model_name}: {e}")
            raise

