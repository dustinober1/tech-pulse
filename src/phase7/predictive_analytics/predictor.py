"""
Predictive Analytics - Prediction Engine

Implements the main prediction logic for tech trends,
growth patterns, and outlier detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pickle
import os
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from pathlib import Path

# Import local modules
from .features import (
    extract_time_features,
    extract_technology_features,
    extract_growth_features,
    extract_popularity_features,
    extract_temporal_features
)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from cache_manager import CacheManager
cache_manager = CacheManager()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Container for prediction results"""
    timestamp: datetime
    technology: str
    metric_type: str
    predicted_value: float
    confidence: float
    time_horizon: int  # days
    feature_importance: Optional[Dict[str, float]] = None
    trend_direction: Optional[str] = None  # 'up', 'down', 'stable'
    anomaly_score: Optional[float] = None

@dataclass
class AnomalyResult:
    """Container for anomaly detection results"""
    timestamp: datetime
    technology: str
    anomaly_score: float
    is_outlier: bool
    outlier_type: Optional[str] = None  # 'spike', 'drop', 'pattern'
    contributing_features: Optional[Dict[str, float]] = None

class PredictiveEngine:
    """
    Main prediction engine for tech trends and anomalies.

    Supports:
    - Time series forecasting for technology metrics
    - Anomaly detection for unusual patterns
    - Feature importance analysis
    - Confidence scoring
    """

    def __init__(self, models_dir: str = "src/phase7/models"):
        """
        Initialize the prediction engine.

        Args:
            models_dir: Directory to store trained models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize models
        self.trend_models = {}  # Technology-specific trend models
        self.anomaly_models = {}  # Technology-specific anomaly models
        self.scalers = {}  # Feature scalers for each technology

        # Model parameters
        self.trend_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'min_samples_split': 5
        }

        self.anomaly_params = {
            'contamination': 0.1,
            'random_state': 42,
            'n_estimators': 100
        }

        # Cache configuration
        self.cache_ttl = 3600  # 1 hour

    def train_trend_model(
        self,
        technology: str,
        features_df: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """
        Train a trend prediction model for a specific technology.

        Args:
            technology: Name of the technology
            features_df: DataFrame with features and target
            target_column: Name of the target column to predict

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training trend model for {technology}")

        # Prepare features - only use numeric columns
        feature_columns = [col for col in features_df.columns
                          if col != target_column
                          and not col.endswith('_date')
                          and features_df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        X = features_df[feature_columns].fillna(0)
        y = features_df[target_column].fillna(0)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model
        model = RandomForestRegressor(**self.trend_params)
        model.fit(X_scaled, y)

        # Calculate metrics
        predictions = model.predict(X_scaled)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))

        # Feature importance
        feature_importance = dict(zip(feature_columns, model.feature_importances_))

        # Store model and scaler
        self.trend_models[technology] = model
        self.scalers[technology] = scaler

        # Save to disk
        model_path = self.models_dir / f"{technology}_trend_model.pkl"
        scaler_path = self.models_dir / f"{technology}_scaler.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        logger.info(f"Model trained. MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        return {
            'mae': mae,
            'rmse': rmse,
            'feature_importance': feature_importance,
            'training_samples': len(X)
        }

    def train_anomaly_model(
        self,
        technology: str,
        features_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Train an anomaly detection model for a specific technology.

        Args:
            technology: Name of the technology
            features_df: DataFrame with features

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training anomaly model for {technology}")

        # Prepare features - only use numeric columns
        feature_columns = [col for col in features_df.columns
                          if not col.endswith('_date')
                          and features_df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        X = features_df[feature_columns].fillna(0)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model
        model = IsolationForest(**self.anomaly_params)
        model.fit(X_scaled)

        # Get anomaly scores
        scores = model.decision_function(X_scaled)
        predictions = model.predict(X_scaled)
        outlier_count = np.sum(predictions == -1)

        # Store model and scaler
        self.anomaly_models[technology] = model
        if technology not in self.scalers:
            self.scalers[technology] = scaler

        # Save to disk
        model_path = self.models_dir / f"{technology}_anomaly_model.pkl"
        if not os.path.exists(model_path):
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        logger.info(f"Anomaly model trained. Outliers detected: {outlier_count}")

        return {
            'outlier_count': outlier_count,
            'outlier_percentage': outlier_count / len(X) * 100,
            'average_score': np.mean(scores),
            'training_samples': len(X)
        }

    # TODO: Implement proper caching decorator
    # @cache_manager.cached(ttl=cache_ttl)
    def predict_trend(
        self,
        technology: str,
        current_features: Dict[str, float],
        time_horizon: int = 30
    ) -> PredictionResult:
        """
        Predict future trend for a technology.

        Args:
            technology: Name of the technology
            current_features: Current feature values
            time_horizon: Number of days to predict ahead

        Returns:
            PredictionResult with forecast and confidence
        """
        # Load model if not in memory
        if technology not in self.trend_models:
            self._load_trend_model(technology)

        if technology not in self.trend_models:
            raise ValueError(f"No trained model found for {technology}")

        model = self.trend_models[technology]
        scaler = self.scalers[technology]

        # Prepare features
        feature_columns = [col for col in scaler.feature_names_in_
                          if col in current_features]

        if not feature_columns:
            raise ValueError("No matching features found for prediction")

        # Create feature vector
        X = np.array([[current_features.get(col, 0) for col in feature_columns]])
        X_scaled = scaler.transform(X)

        # Make prediction
        prediction = model.predict(X_scaled)[0]

        # Calculate confidence (based on tree variance)
        tree_predictions = np.array([tree.predict(X_scaled)[0] for tree in model.estimators_])
        confidence = 1.0 - (np.std(tree_predictions) / np.abs(prediction + 1e-6))
        confidence = max(0, min(1, confidence))

        # Determine trend direction
        trend_direction = self._determine_trend_direction(
            current_features.get('current_value', 0),
            prediction
        )

        # Get feature importance
        feature_importance = dict(zip(
            feature_columns,
            model.feature_importances_[:len(feature_columns)]
        ))

        return PredictionResult(
            timestamp=datetime.now(),
            technology=technology,
            metric_type='trend',
            predicted_value=prediction,
            confidence=confidence,
            time_horizon=time_horizon,
            feature_importance=feature_importance,
            trend_direction=trend_direction
        )

    # TODO: Implement proper caching decorator
    # @cache_manager.cached(ttl=cache_ttl//2)
    def detect_anomaly(
        self,
        technology: str,
        current_features: Dict[str, float]
    ) -> AnomalyResult:
        """
        Detect if current features represent an anomaly.

        Args:
            technology: Name of the technology
            current_features: Current feature values

        Returns:
            AnomalyResult with detection results
        """
        # Load model if not in memory
        if technology not in self.anomaly_models:
            self._load_anomaly_model(technology)

        if technology not in self.anomaly_models:
            raise ValueError(f"No trained anomaly model found for {technology}")

        model = self.anomaly_models[technology]
        scaler = self.scalers[technology]

        # Prepare features
        feature_columns = [col for col in scaler.feature_names_in_
                          if col in current_features]

        if not feature_columns:
            raise ValueError("No matching features found for anomaly detection")

        # Create feature vector
        X = np.array([[current_features.get(col, 0) for col in feature_columns]])
        X_scaled = scaler.transform(X)

        # Detect anomaly
        prediction = model.predict(X_scaled)[0]
        anomaly_score = model.decision_function(X_scaled)[0]
        is_outlier = prediction == -1

        # Determine outlier type if applicable
        outlier_type = None
        contributing_features = None

        if is_outlier:
            outlier_type, contributing_features = self._analyze_outlier_type(
                current_features,
                feature_columns,
                anomaly_score
            )

        return AnomalyResult(
            timestamp=datetime.now(),
            technology=technology,
            anomaly_score=anomaly_score,
            is_outlier=is_outlier,
            outlier_type=outlier_type,
            contributing_features=contributing_features
        )

    def predict_batch(
        self,
        technology: str,
        features_batch: List[Dict[str, float]],
        time_horizon: int = 30
    ) -> List[PredictionResult]:
        """
        Make predictions for a batch of feature sets.

        Args:
            technology: Name of the technology
            features_batch: List of feature dictionaries
            time_horizon: Number of days to predict ahead

        Returns:
            List of PredictionResult objects
        """
        results = []

        for features in features_batch:
            try:
                result = self.predict_trend(technology, features, time_horizon)
                results.append(result)
            except Exception as e:
                logger.error(f"Prediction failed for {technology}: {str(e)}")
                # Add a failed prediction result
                results.append(PredictionResult(
                    timestamp=datetime.now(),
                    technology=technology,
                    metric_type='trend',
                    predicted_value=0,
                    confidence=0,
                    time_horizon=time_horizon
                ))

        return results

    def detect_anomalies_batch(
        self,
        technology: str,
        features_batch: List[Dict[str, float]]
    ) -> List[AnomalyResult]:
        """
        Detect anomalies for a batch of feature sets.

        Args:
            technology: Name of the technology
            features_batch: List of feature dictionaries

        Returns:
            List of AnomalyResult objects
        """
        results = []

        for features in features_batch:
            try:
                result = self.detect_anomaly(technology, features)
                results.append(result)
            except Exception as e:
                logger.error(f"Anomaly detection failed for {technology}: {str(e)}")
                # Add a failed detection result
                results.append(AnomalyResult(
                    timestamp=datetime.now(),
                    technology=technology,
                    anomaly_score=0,
                    is_outlier=False
                ))

        return results

    def _load_trend_model(self, technology: str) -> None:
        """Load trend model and scaler from disk."""
        model_path = self.models_dir / f"{technology}_trend_model.pkl"
        scaler_path = self.models_dir / f"{technology}_scaler.pkl"

        if model_path.exists():
            with open(model_path, 'rb') as f:
                self.trend_models[technology] = pickle.load(f)

        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scalers[technology] = pickle.load(f)

    def _load_anomaly_model(self, technology: str) -> None:
        """Load anomaly model from disk."""
        model_path = self.models_dir / f"{technology}_anomaly_model.pkl"

        if model_path.exists():
            with open(model_path, 'rb') as f:
                self.anomaly_models[technology] = pickle.load(f)

    def _determine_trend_direction(
        self,
        current_value: float,
        predicted_value: float
    ) -> str:
        """Determine if trend is up, down, or stable."""
        if current_value == 0:
            return 'stable'

        change_percent = (predicted_value - current_value) / abs(current_value)

        if change_percent > 0.05:
            return 'up'
        elif change_percent < -0.05:
            return 'down'
        else:
            return 'stable'

    def _analyze_outlier_type(
        self,
        features: Dict[str, float],
        feature_columns: List[str],
        anomaly_score: float
    ) -> Tuple[str, Dict[str, float]]:
        """Analyze the type of outlier and contributing features."""
        contributing_features = {}

        # Calculate feature deviations
        for col in feature_columns:
            if col in features:
                value = features[col]
                # Simple heuristic: features with extreme values contribute more
                if 'growth' in col and abs(value) > 0.5:
                    contributing_features[col] = abs(value)
                elif 'popularity' in col and (value > 0.9 or value < 0.1):
                    contributing_features[col] = 1 - value if value > 0.5 else value

        # Determine outlier type
        if 'growth_rate' in contributing_features and contributing_features['growth_rate'] > 0.5:
            outlier_type = 'spike'
        elif 'growth_rate' in contributing_features and contributing_features['growth_rate'] < -0.5:
            outlier_type = 'drop'
        else:
            outlier_type = 'pattern'

        return outlier_type, contributing_features

    def get_model_info(self, technology: str) -> Dict[str, Any]:
        """Get information about trained models for a technology."""
        info = {
            'technology': technology,
            'trend_model_trained': technology in self.trend_models,
            'anomaly_model_trained': technology in self.anomaly_models,
            'model_files': []
        }

        # Check for model files
        for model_type in ['trend', 'anomaly']:
            model_path = self.models_dir / f"{technology}_{model_type}_model.pkl"
            if model_path.exists():
                info['model_files'].append({
                    'type': model_type,
                    'path': str(model_path),
                    'size': os.path.getsize(model_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(model_path))
                })

        return info

    def cleanup_cache(self) -> None:
        """Clear prediction cache."""
        cache_manager.delete_pattern("predict_trend:*")
        cache_manager.delete_pattern("detect_anomaly:*")
        logger.info("Predictive analytics cache cleared")