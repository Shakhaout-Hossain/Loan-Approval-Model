import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, matthews_corrcoef, confusion_matrix)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              ExtraTreesClassifier)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from mlflow_tracker import MLFlowTracker  # Import the new MLFlowTracker class


class ModelTrainer:
    def __init__(self, mlflow_tracker: MLFlowTracker):
        """
        Initialize the ModelTrainer class with MLFlowTracker integration.
        
        Args:
            mlflow_tracker (MLFlowTracker): An instance of the MLFlowTracker class.
        """
        self.mlflow_tracker = mlflow_tracker
        self.transformed_data_dir = "artifacts/transformed_data"
        self.model_dir = "artifacts/models"
        
        # List of models to train
        self.models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(),
            "LightGBM": LGBMClassifier(),
            "CatBoost": CatBoostClassifier(verbose=0),
            "SVC": SVC(probability=True),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "MLP": MLPClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "Extra Trees": ExtraTreesClassifier()
        }

    def create_directories(self):
        """Create necessary directories for storing models and logs"""
        os.makedirs(self.model_dir, exist_ok=True)

    def load_transformed_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load transformed data for model training"""
        try:
            X = pd.read_csv(os.path.join(self.transformed_data_dir, "X_transformed.csv"))
            y = pd.read_csv(os.path.join(self.transformed_data_dir, "y_transformed.csv")).iloc[:, 0]
            return X, y
        except Exception as e:
            print(f"Error loading transformed data: {str(e)}")
            raise e

    def evaluate_model(self, model: Any, X_train: pd.DataFrame, 
                       y_train: pd.Series, X_test: pd.DataFrame, 
                       y_test: pd.Series) -> Dict[str, float]:
        """Train and evaluate a single model, log metrics to MLFlow"""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Log metrics and model
        self.mlflow_tracker.start_run()
        self.mlflow_tracker.log_metrics(model.__class__.__name__, y_test, y_pred)
        self.mlflow_tracker.log_model(model, model.__class__.__name__)
        
        # Register the model to the MLflow Model Registry
        self.mlflow_tracker.register_model(model, model.__class__.__name__)

        # Transition the model to 'Production' stage
        self.mlflow_tracker.transition_to_production(model.__class__.__name__)
        
        self.mlflow_tracker.end_run()

        return {
            'Test Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'MCC': matthews_corrcoef(y_test, y_pred),
            'Confusion Matrix': confusion_matrix(y_test, y_pred).ravel()
        }

    def save_models(self, results: Dict[str, Dict]):
        """Save all models and their metrics locally"""
        try:
            for model_name, model in self.models.items():
                model_path = os.path.join(self.model_dir, f"{model_name.lower().replace(' ', '_')}.joblib")
                joblib.dump(model, model_path)
            
            results_df = pd.DataFrame(results).T
            results_df.to_csv(os.path.join(self.model_dir, "model_metrics.csv"))
            
            print(f"Models and metrics saved to {self.model_dir}")
            
        except Exception as e:
            print(f"Error saving models: {str(e)}")
            raise e
    
    def initiate_model_training(self) -> Dict[str, Dict]:
        """Train and evaluate all models, log and save them"""
        try:
            # Create directories
            self.create_directories()

            # Load transformed data
            X, y = self.load_transformed_data()

            # Split data into train/test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            results = {}
            for model_name, model in self.models.items():
                print(f"Training and evaluating {model_name}...")
                metrics = self.evaluate_model(model, X_train, y_train, X_test, y_test)
                results[model_name] = metrics

            # Save models and their metrics
            self.save_models(results)
            
            print("Model training completed successfully")
            return results
            
        except Exception as e:
            print(f"Error in model training: {str(e)}")
            raise e


if __name__ == "__main__":
    # Initialize MLFlowTracker with SQLite
    mlflow_tracker = MLFlowTracker(experiment_name="loan_approval_model_experiment", 
                                   sqlite_db_path='./mlflow.db', artifact_location="./artifacts")
    
    # Initialize ModelTrainer with MLFlowTracker
    model_trainer = ModelTrainer(mlflow_tracker=mlflow_tracker)
    results = model_trainer.initiate_model_training()
    
    # Print results
    results_df = pd.DataFrame(results).T
    print("\nModel Evaluation Results:")
    print(results_df)
