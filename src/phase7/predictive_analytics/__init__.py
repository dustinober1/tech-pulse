"""
Predictive Analytics Module

This module provides predictive analytics capabilities for the Tech Pulse dashboard,
including:
- Trend prediction using machine learning models
- Anomaly detection for unusual patterns
- Feature engineering for time series data
- Model training and evaluation
"""

from .predictor import PredictiveEngine, PredictionResult, AnomalyResult
from .features import (
    extract_time_features,
    extract_technology_features,
    extract_growth_features,
    extract_popularity_features,
    extract_temporal_features,
    engineer_features,
    select_features
)
from .training_data import TrainingDataCollector, TrainingDataset, DataSourceConfig
from .train_model import ModelTrainer
from .dashboard import PredictiveDashboard

__version__ = "1.0.0"
__all__ = [
    "PredictiveEngine",
    "PredictionResult",
    "AnomalyResult",
    "extract_time_features",
    "extract_technology_features",
    "extract_growth_features",
    "extract_popularity_features",
    "extract_temporal_features",
    "engineer_features",
    "select_features",
    "TrainingDataCollector",
    "TrainingDataset",
    "DataSourceConfig",
    "ModelTrainer",
    "PredictiveDashboard"
]