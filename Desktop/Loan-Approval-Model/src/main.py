from data_ingestion import DataIngestion
from data_transformation import DataTransformation
from model_trainer import ModelTrainer
import pandas as pd
import os
import logging
from datetime import datetime

from mlflow_tracker import MLFlowTracker  # Import the new MLFlowTracker class


def setup_logging():
    """Setup logging configuration"""
    log_filename = f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_directories():
    """Create necessary directories for the project"""
    directories = [
        'artifacts',
        'artifacts/ingested_data',
        'artifacts/transformed_data',
        'artifacts/models',
        'logs'
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Created directory: {dir_path}")

def save_pipeline_summary(ingestion_summary: dict, transformation_summary: dict, 
                        training_summary: pd.DataFrame):
    """Save pipeline execution summary"""
    summary = {
        'data_ingestion': ingestion_summary,
        'data_transformation': transformation_summary,
        'model_training': training_summary.to_dict()
    }
    
    summary_df = pd.DataFrame(summary)
    summary_path = f"artifacts/pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(summary_path)
    logging.info(f"Pipeline summary saved to {summary_path}")

def main():
    try:
        # Setup logging
        logger = setup_logging()
        logger.info("Starting ML pipeline...")
        
        # Create directories
        create_directories()
        
        # Initialize pipeline components
        data_ingestion = DataIngestion('/Users/flash/Desktop/Loan-Approval-Model/src/dataset/loan_approval_dataset.csv')
        data_transformation = DataTransformation()
        # model_trainer = ModelTrainer()
        # Initialize MLFlowTracker with SQLite
        mlflow_tracker = MLFlowTracker(experiment_name="loan_approval_model_experiment", 
                                    sqlite_db_path='./mlflow.db', artifact_location="./artifacts")
        
        # Initialize ModelTrainer with MLFlowTracker
        model_trainer = ModelTrainer(mlflow_tracker=mlflow_tracker)
        
        # 1. Data Ingestion
        logger.info("Step 1: Data Ingestion")
        X, y = data_ingestion.initiate_data_ingestion()
        ingestion_summary = {
            'features_shape': X.shape,
            'target_shape': y.shape,
            'ingested_data_path': data_ingestion.ingested_data_dir
        }
        logger.info(f"Data ingestion completed. Data saved to {data_ingestion.ingested_data_dir}")
        
        # 2. Data Transformation
        logger.info("Step 2: Data Transformation")
        X_transformed, y = data_transformation.initiate_data_transformation()
        transformation_summary = {
            'transformed_features_shape': X_transformed.shape,
            'transformed_data_path': data_transformation.transformed_data_dir
        }
        logger.info(f"Data transformation completed. Data saved to {data_transformation.transformed_data_dir}")
        
        # 3. Model Training and Evaluation
        logger.info("Step 3: Model Training and Evaluation")
        results = model_trainer.initiate_model_training()
        results_df = pd.DataFrame(results).T
        
        # Save results
        results_path = os.path.join(model_trainer.model_dir, 'model_evaluation_results.csv')
        results_df.to_csv(results_path)
        logger.info(f"Model evaluation results saved to {results_path}")
        
        # Find and log best model
        print(results_df.dtypes)  # Check data types

        # Convert 'F1 Score' to numeric
        results_df['F1 Score'] = pd.to_numeric(results_df['F1 Score'], errors='coerce')

        # Drop rows with NaN values in 'F1 Score'
        results_df = results_df.dropna(subset=['F1 Score'])

        # Now you can safely find the best model
        best_model = results_df['F1 Score'].idxmax()
        logger.info(f"Best performing model: {best_model} (F1 Score: {results_df.loc[best_model, 'F1 Score']:.4f})")
        
        # Save pipeline summary
        save_pipeline_summary(ingestion_summary, transformation_summary, results_df)
        
        logger.info("ML pipeline completed successfully!")
        
        # Print final summary
        print("\nPipeline Summary:")
        print("-" * 50)
        print(f"Data Ingestion:")
        print(f"- Features shape: {X.shape}")
        print(f"- Target shape: {y.shape}")
        print(f"\nData Transformation:")
        print(f"- Transformed features shape: {X_transformed.shape}")
        print(f"\nModel Training:")
        print(f"- Best model: {best_model}")
        print(f"- Best F1 Score: {results_df.loc[best_model, 'F1 Score']:.4f}")
        print(f"\nArtifacts saved in:")
        print(f"- Ingested data: {data_ingestion.ingested_data_dir}")
        print(f"- Transformed data: {data_transformation.transformed_data_dir}")
        print(f"- Models: {model_trainer.model_dir}")
        print("-" * 50)
        
    except Exception as e:
        logger.error(f"Error in ML pipeline: {str(e)}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()