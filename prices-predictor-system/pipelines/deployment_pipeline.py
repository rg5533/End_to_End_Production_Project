import os

from pipelines.training_pipeline import ml_pipeline
from steps.dynamic_importer import dynamic_importer
from steps.model_loader import model_loader
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor import predictor
from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")

