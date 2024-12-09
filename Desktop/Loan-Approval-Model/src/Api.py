from flask import Flask, request, jsonify
import logging
import pandas as pd
from datetime import datetime
from mlflow_tracker import MLFlowTracker  # Assuming MLFlowTracker is in mlflow_tracker.py

# Enable logging
logging.basicConfig(level=logging.INFO)

class PredictionService:
    def __init__(self):
        self.mlflow_tracker = MLFlowTracker()  # Initialize MLFlowTracker
        self.loaded_models = {}
        self.app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self):
        """Define all routes for the Flask app."""
        self.app.add_url_rule('/predict', 'predict_single', self.predict_single, methods=['POST'])
        self.app.add_url_rule('/predict/batch', 'predict_batch', self.predict_batch, methods=['POST'])
        self.app.add_url_rule('/health', 'health_check', self.health_check, methods=['GET'])

    def load_model(self, model_name):
        """Dynamically load a model by name using MLFlowTracker."""
        try:
            if model_name not in self.loaded_models:
                model = self.mlflow_tracker.load_model(model_name)
                self.loaded_models[model_name] = model
                logging.info(f"Model {model_name} loaded successfully.")
            return self.loaded_models[model_name]
        except Exception as e:
            logging.error(f"Error loading model {model_name}: {e}")
            raise Exception(f"Model {model_name} could not be loaded")

    def validate_input(self, data, required_fields):
        """Validate input JSON data for required fields."""
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return {"error": f"Missing required fields: {', '.join(missing_fields)}"}, 400
        return None

    def predict_single(self):
        """Handle single prediction requests."""
        try:
            data = request.get_json()
            required_fields = [
                "income_annum", "loan_amount", "loan_term", "cibil_score",
                "residential_assets_value", "commercial_assets_value",
                "luxury_assets_value", "bank_asset_value", "education",
                "self_employed", "no_of_dependents"
            ]

            validation_error = self.validate_input(data, required_fields)
            if validation_error:
                return jsonify(validation_error[0]), validation_error[1]

            model_name = data.get("model_name", "loan_approval_model")
            model = self.load_model(model_name)
            df = pd.DataFrame([data])

            prediction = model.predict(None, df)[0]  # Assuming the model returns a single prediction
            self.mlflow_tracker.start_run()  # Start MLFlow tracking
            self.mlflow_tracker.log_metrics(model_name, [data], [prediction])  # Log metrics
            self.mlflow_tracker.end_run()  # End MLFlow tracking

            response = {
                "prediction": int(prediction),
                "probability": 0.85,  # Placeholder probability
                "model_name": model_name,
                "model_version": "latest",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

            return jsonify(response)

        except Exception as e:
            logging.error(f"Error during single prediction: {e}")
            return jsonify({"error": "An error occurred during prediction"}), 500

    def predict_batch(self):
        """Handle batch prediction requests."""
        try:
            data = request.get_json()
            if 'instances' not in data:
                return jsonify({"error": "'instances' key missing"}), 400

            instances = data['instances']
            required_fields = [
                "income_annum", "loan_amount", "loan_term", "cibil_score",
                "residential_assets_value", "commercial_assets_value",
                "luxury_assets_value", "bank_asset_value", "education",
                "self_employed", "no_of_dependents"
            ]

            for instance in instances:
                validation_error = self.validate_input(instance, required_fields)
                if validation_error:
                    return jsonify(validation_error[0]), validation_error[1]

            model_name = data.get("model_name", "loan_approval_model")
            model = self.load_model(model_name)
            df = pd.DataFrame(instances)

            predictions = model.predict(None, df)
            self.mlflow_tracker.start_run()  # Start MLFlow tracking
            self.mlflow_tracker.log_metrics(model_name, instances, predictions)  # Log metrics
            self.mlflow_tracker.end_run()  # End MLFlow tracking

            response = {
                "predictions": [{"prediction": int(pred), "probability": 0.85, "row_id": idx} for idx, pred in enumerate(predictions)],
                "model_name": model_name,
                "model_version": "latest",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "batch_size": len(instances),
                "processing_time": "0.15s"
            }

            return jsonify(response)

        except Exception as e:
            logging.error(f"Error during batch prediction: {e}")
            return jsonify({"error": "An error occurred during batch prediction"}), 500

    def health_check(self):
        """Perform a health check for the API."""
        try:
            health = {
                "status": "healthy",
                "checks": {
                    "model_loaded": all(model is not None for model in self.loaded_models.values()),
                    "api_status": "ok",
                    "model_versions": {model_name: "latest" for model_name in self.loaded_models},
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            }
            return jsonify(health)
        except Exception as e:
            logging.error(f"Error during health check: {e}")
            return jsonify({"status": "unhealthy", "error": str(e)}), 500

    def run(self, host="0.0.0.0", port=5000, debug=True):
        """Run the Flask application."""
        self.app.run(host=host, port=port, debug=debug)


# Run the application
if __name__ == "__main__":
    service = PredictionService()
    service.run()
