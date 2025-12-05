"""
Unit tests for the predictive analytics predictor module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from phase7.predictive_analytics.predictor import (
    PredictiveEngine,
    PredictionResult,
    AnomalyResult
)

class TestPredictiveEngine(unittest.TestCase):
    """Test cases for PredictiveEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for models
        self.temp_dir = tempfile.mkdtemp()
        self.engine = PredictiveEngine(models_dir=self.temp_dir)

        # Create sample data
        self.sample_features_df = pd.DataFrame({
            'technology': ['Python'] * 100,
            'date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
            'commit_count': np.random.randint(10, 100, 100),
            'stars': np.random.randint(100, 1000, 100),
            'forks': np.random.randint(50, 500, 100),
            'issues': np.random.randint(5, 50, 100),
            'contributors': np.random.randint(1, 20, 100)
        })

        self.sample_current_features = {
            'current_value': 50,
            'growth_rate_7d': 0.05,
            'growth_rate_30d': 0.10,
            'stars': 500,
            'forks': 250,
            'issues': 25,
            'volatility_30d': 0.15,
            'ma_7d': 45,
            'ma_30d': 40
        }

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_train_trend_model(self):
        """Test training a trend model."""
        # Add necessary features for training
        features_df = self.sample_features_df.copy()
        features_df['target'] = features_df['commit_count'].shift(-7)  # 7-day ahead target
        features_df = features_df.dropna()

        # Train model
        result = self.engine.train_trend_model(
            'Python',
            features_df,
            'target'
        )

        # Check results
        self.assertIn('mae', result)
        self.assertIn('rmse', result)
        self.assertIn('feature_importance', result)
        self.assertIn('training_samples', result)
        self.assertGreater(result['training_samples'], 0)

        # Check model was saved
        self.assertIn('Python', self.engine.trend_models)
        self.assertIn('Python', self.engine.scalers)

    def test_train_anomaly_model(self):
        """Test training an anomaly detection model."""
        # Train model
        result = self.engine.train_anomaly_model(
            'Python',
            self.sample_features_df
        )

        # Check results
        self.assertIn('outlier_count', result)
        self.assertIn('outlier_percentage', result)
        self.assertIn('average_score', result)
        self.assertIn('training_samples', result)
        self.assertGreater(result['training_samples'], 0)

        # Check model was saved
        self.assertIn('Python', self.engine.anomaly_models)

    def test_predict_trend(self):
        """Test making a trend prediction."""
        # First train a model
        features_df = self.sample_features_df.copy()
        features_df['target'] = features_df['commit_count'].shift(-7)
        features_df = features_df.dropna()

        self.engine.train_trend_model('Python', features_df, 'target')

        # Make prediction
        prediction = self.engine.predict_trend(
            'Python',
            self.sample_current_features,
            time_horizon=30
        )

        # Check result structure
        self.assertIsInstance(prediction, PredictionResult)
        self.assertEqual(prediction.technology, 'Python')
        self.assertEqual(prediction.metric_type, 'trend')
        self.assertEqual(prediction.time_horizon, 30)
        self.assertGreater(prediction.predicted_value, 0)
        self.assertGreaterEqual(prediction.confidence, 0)
        self.assertLessEqual(prediction.confidence, 1)
        self.assertIn(prediction.trend_direction, ['up', 'down', 'stable'])

    def test_detect_anomaly(self):
        """Test anomaly detection."""
        # First train an anomaly model
        self.engine.train_anomaly_model('Python', self.sample_features_df)

        # Detect anomaly
        anomaly = self.engine.detect_anomaly(
            'Python',
            self.sample_current_features
        )

        # Check result structure
        self.assertIsInstance(anomaly, AnomalyResult)
        self.assertEqual(anomaly.technology, 'Python')
        self.assertIsInstance(anomaly.is_outlier, bool)
        self.assertIsInstance(anomaly.anomaly_score, float)

    def test_predict_batch(self):
        """Test batch predictions."""
        # Train model first
        features_df = self.sample_features_df.copy()
        features_df['target'] = features_df['commit_count'].shift(-7)
        features_df = features_df.dropna()

        self.engine.train_trend_model('Python', features_df, 'target')

        # Create batch features
        batch_features = [self.sample_current_features] * 5

        # Make batch predictions
        results = self.engine.predict_batch(
            'Python',
            batch_features,
            time_horizon=30
        )

        # Check results
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIsInstance(result, PredictionResult)

    def test_detect_anomalies_batch(self):
        """Test batch anomaly detection."""
        # Train anomaly model first
        self.engine.train_anomaly_model('Python', self.sample_features_df)

        # Create batch features
        batch_features = [self.sample_current_features] * 5

        # Detect batch anomalies
        results = self.engine.detect_anomalies_batch(
            'Python',
            batch_features
        )

        # Check results
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIsInstance(result, AnomalyResult)

    def test_get_model_info(self):
        """Test getting model information."""
        # Train models
        features_df = self.sample_features_df.copy()
        features_df['target'] = features_df['commit_count'].shift(-7)
        features_df = features_df.dropna()

        self.engine.train_trend_model('Python', features_df, 'target')
        self.engine.train_anomaly_model('Python', self.sample_features_df)

        # Get model info
        info = self.engine.get_model_info('Python')

        # Check info structure
        self.assertEqual(info['technology'], 'Python')
        self.assertTrue(info['trend_model_trained'])
        self.assertTrue(info['anomaly_model_trained'])
        self.assertGreater(len(info['model_files']), 0)

    def test_determine_trend_direction(self):
        """Test trend direction determination."""
        # Test upward trend
        direction = self.engine._determine_trend_direction(100, 110)
        self.assertEqual(direction, 'up')

        # Test downward trend
        direction = self.engine._determine_trend_direction(100, 90)
        self.assertEqual(direction, 'down')

        # Test stable trend
        direction = self.engine._determine_trend_direction(100, 102)
        self.assertEqual(direction, 'stable')

    def test_analyze_outlier_type(self):
        """Test outlier type analysis."""
        features = {
            'growth_rate': 0.8,
            'popularity': 0.95
        }

        outlier_type, contributing = self.engine._analyze_outlier_type(
            features,
            ['growth_rate', 'popularity'],
            -0.5
        )

        self.assertEqual(outlier_type, 'spike')
        self.assertIn('growth_rate', contributing)

    def test_cleanup_cache(self):
        """Test cache cleanup."""
        # Mock cache manager
        with patch('src.phase7.predictive_analytics.predictor.cache_manager') as mock_cache:
            self.engine.cleanup_cache()
            mock_cache.delete_pattern.assert_any_call("predict_trend:*")
            mock_cache.delete_pattern.assert_any_call("detect_anomaly:*")

    def test_predict_without_model(self):
        """Test prediction without trained model raises error."""
        with self.assertRaises(ValueError):
            self.engine.predict_trend(
                'NonExistent',
                self.sample_current_features
            )

    def test_detect_anomaly_without_model(self):
        """Test anomaly detection without trained model raises error."""
        with self.assertRaises(ValueError):
            self.engine.detect_anomaly(
                'NonExistent',
                self.sample_current_features
            )

class TestPredictionResult(unittest.TestCase):
    """Test cases for PredictionResult dataclass."""

    def test_creation(self):
        """Test PredictionResult creation."""
        timestamp = datetime.now()
        result = PredictionResult(
            timestamp=timestamp,
            technology='Python',
            metric_type='trend',
            predicted_value=100.5,
            confidence=0.85,
            time_horizon=30
        )

        self.assertEqual(result.timestamp, timestamp)
        self.assertEqual(result.technology, 'Python')
        self.assertEqual(result.metric_type, 'trend')
        self.assertEqual(result.predicted_value, 100.5)
        self.assertEqual(result.confidence, 0.85)
        self.assertEqual(result.time_horizon, 30)

class TestAnomalyResult(unittest.TestCase):
    """Test cases for AnomalyResult dataclass."""

    def test_creation(self):
        """Test AnomalyResult creation."""
        timestamp = datetime.now()
        result = AnomalyResult(
            timestamp=timestamp,
            technology='Python',
            anomaly_score=-0.2,
            is_outlier=True,
            outlier_type='spike'
        )

        self.assertEqual(result.timestamp, timestamp)
        self.assertEqual(result.technology, 'Python')
        self.assertEqual(result.anomaly_score, -0.2)
        self.assertTrue(result.is_outlier)
        self.assertEqual(result.outlier_type, 'spike')

class TestPredictiveEngineIntegration(unittest.TestCase):
    """Integration tests for PredictiveEngine."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.engine = PredictiveEngine(models_dir=self.temp_dir)

    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch('src.phase7.predictive_analytics.predictor.cache_manager')
    def test_prediction_workflow(self, mock_cache):
        """Test complete prediction workflow."""
        # Mock cache decorator
        mock_cache.cached.return_value = lambda func: func

        # Create comprehensive test data
        dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
        data = {
            'date': dates,
            'technology': ['Python'] * 200,
            'commit_count': np.random.randint(10, 100, 200),
            'stars': np.random.randint(100, 1000, 200),
            'forks': np.random.randint(50, 500, 200),
            'issues': np.random.randint(5, 50, 200),
            'contributors': np.random.randint(1, 20, 200),
            'growth_rate_7d': np.random.normal(0.05, 0.1, 200),
            'growth_rate_30d': np.random.normal(0.1, 0.15, 200),
            'volatility_30d': np.random.uniform(0.1, 0.3, 200),
            'ma_7d': np.random.uniform(40, 60, 200),
            'ma_30d': np.random.uniform(35, 55, 200)
        }
        df = pd.DataFrame(data)

        # Create target
        df['target'] = df['commit_count'].shift(-30)
        df = df.dropna()

        # Train model
        train_result = self.engine.train_trend_model(
            'Python',
            df,
            'target'
        )
        self.assertGreater(train_result['training_samples'], 0)

        # Make prediction
        current_features = {
            'current_value': 50,
            'growth_rate_7d': 0.05,
            'growth_rate_30d': 0.10,
            'stars': 500,
            'forks': 250,
            'issues': 25,
            'volatility_30d': 0.15,
            'ma_7d': 45,
            'ma_30d': 40
        }

        prediction = self.engine.predict_trend(
            'Python',
            current_features,
            time_horizon=30
        )

        self.assertIsInstance(prediction, PredictionResult)
        self.assertGreater(prediction.predicted_value, 0)

    def test_anomaly_detection_workflow(self):
        """Test complete anomaly detection workflow."""
        # Create test data with some anomalies
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        data = {
            'date': dates,
            'technology': ['Python'] * 100,
            'commit_count': np.random.normal(50, 10, 100),
            'stars': np.random.normal(500, 100, 100),
            'forks': np.random.normal(250, 50, 100),
            'issues': np.random.normal(25, 5, 100)
        }

        # Add some anomalies
        data['commit_count'][10] = 200  # Spike
        data['commit_count'][50] = 5    # Drop
        df = pd.DataFrame(data)

        # Train anomaly model
        result = self.engine.train_anomaly_model('Python', df)
        self.assertGreater(result['training_samples'], 0)

        # Detect anomalies
        anomaly_features = {
            'current_value': 200,  # Anomalous value
            'growth_rate_7d': 3.0,  # High growth
            'growth_rate_30d': 1.5,
            'volatility_30d': 0.5
        }

        anomaly = self.engine.detect_anomaly('Python', anomaly_features)

        self.assertIsInstance(anomaly, AnomalyResult)
        self.assertEqual(anomaly.technology, 'Python')
        self.assertIsInstance(anomaly.is_outlier, bool)

if __name__ == '__main__':
    unittest.main()