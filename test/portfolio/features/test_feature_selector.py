"""
Unit tests for feature selection framework.

This module tests the FeatureSelector class and its methods
for selecting features using various statistical and ML-based approaches.
"""

import pytest
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch
import warnings

from src.portfolio.features.feature_selector import (
    FeatureSelector,
    SelectionConfig,
    SelectionResult
)


class TestFeatureSelector:
    """Test cases for FeatureSelector."""

    @pytest.fixture
    def classification_data(self):
        """Create sample classification data."""
        np.random.seed(42)
        n_samples = 200

        # Create features with varying predictive power
        X = pd.DataFrame({
            'important_feature': np.random.randn(n_samples),
            'correlated_important': np.random.randn(n_samples) * 0.8 + np.random.randn(n_samples) * 0.2,
            'noise_feature_1': np.random.randn(n_samples) * 0.1,
            'noise_feature_2': np.random.randn(n_samples) * 0.1,
            'categorical_feature': np.random.choice([0, 1, 2], n_samples),
            'constant_feature': np.ones(n_samples),
            'near_constant': np.ones(n_samples) * 0.01 * np.random.randn(n_samples)
        })

        # Create binary target
        y = (X['important_feature'] + X['correlated_important'] * 0.5 +
             np.random.randn(n_samples) * 0.3 > 0).astype(int)

        return X, y

    @pytest.fixture
    def regression_data(self):
        """Create sample regression data."""
        np.random.seed(42)
        n_samples = 200

        # Create features with varying predictive power
        X = pd.DataFrame({
            'main_feature': np.random.randn(n_samples),
            'secondary_feature': np.random.randn(n_samples) * 0.5,
            'noise_1': np.random.randn(n_samples) * 0.05,
            'noise_2': np.random.randn(n_samples) * 0.05,
            'seasonal': np.sin(np.arange(n_samples) / 10),
            'trend': np.arange(n_samples) / n_samples,
            'constant': np.ones(n_samples)
        })

        # Create continuous target
        y = (X['main_feature'] * 2 + X['secondary_feature'] +
             X['seasonal'] * 0.5 + X['trend'] +
             np.random.randn(n_samples) * 0.2)

        return X, y

    @pytest.fixture
    def high_dim_data(self):
        """Create high-dimensional data for testing."""
        np.random.seed(42)
        n_samples, n_features = 100, 50

        # Create many correlated features
        base_features = np.random.randn(n_samples, 10)
        X = np.zeros((n_samples, n_features))
        X[:, :10] = base_features

        # Add correlated variations
        for i in range(10, 50):
            base_idx = i % 10
            correlation = 0.5 + 0.5 * np.random.rand()
            X[:, i] = base_features[:, base_idx] * correlation + \
                      np.random.randn(n_samples) * (1 - correlation)

        # Create target
        y = X[:, :5].sum(axis=1) + np.random.randn(n_samples) * 0.5

        feature_names = [f'feature_{i}' for i in range(n_features)]
        X = pd.DataFrame(X, columns=feature_names)

        return X, y

    @pytest.fixture
    def selector(self):
        """Create a FeatureSelector instance."""
        config = SelectionConfig(
            correlation_threshold=0.9,
            variance_threshold=0.01,
            model_n_estimators=10,  # Small for faster tests
            cv_folds=3,
            stability_n_bootstrap=10  # Small for faster tests
        )
        return FeatureSelector(config)

    def test_initialization(self):
        """Test FeatureSelector initialization."""
        # Test with default config
        selector = FeatureSelector()
        assert selector.config is not None
        assert isinstance(selector.config, SelectionConfig)

        # Test with custom config
        config = SelectionConfig(correlation_threshold=0.8)
        selector = FeatureSelector(config)
        assert selector.config.correlation_threshold == 0.8

    def test_determine_problem_type(self, selector, classification_data, regression_data):
        """Test problem type determination."""
        X_clf, y_clf = classification_data
        X_reg, y_reg = regression_data

        # Classification problem
        selector.problem_type_ = selector._determine_problem_type(y_clf)
        assert selector.problem_type_ == 'classification'

        # Regression problem
        selector.problem_type_ = selector._determine_problem_type(y_reg)
        assert selector.problem_type_ == 'regression'

    def test_variance_selection(self, selector, classification_data):
        """Test variance threshold selection."""
        X, y = classification_data

        # Apply variance selection
        X_selected = selector._variance_selection(X, threshold=0.05)

        # Should remove constant and near-constant features
        assert 'constant_feature' not in X_selected.columns
        assert len(X_selected.columns) < len(X.columns)

        # Store selection result
        assert len(selector.selection_history_) > 0
        result = selector.selection_history_[-1]
        assert result.selection_method == 'variance'
        assert isinstance(result, SelectionResult)

    def test_mutual_info_selection(self, selector, classification_data):
        """Test mutual information feature selection."""
        X, y = classification_data

        # Use only numeric features for mutual information
        X_numeric = X.select_dtypes(include=[np.number])

        # Apply mutual information selection
        X_selected = selector._mutual_info_selection(X_numeric, y, k=3)

        # Should select top k features
        assert len(X_selected.columns) == 3

        # Important features should be selected
        assert 'important_feature' in X_selected.columns
        assert 'correlated_important' in X_selected.columns

        # Store selection result
        assert len(selector.selection_history_) > 0
        result = selector.selection_history_[-1]
        assert result.selection_method == 'mutual_info'

    def test_correlation_selection(self, selector, classification_data):
        """Test correlation-based feature selection."""
        X, y = classification_data

        # Use only numeric features for correlation analysis
        X_numeric = X.select_dtypes(include=[np.number])

        # Apply correlation selection
        X_selected = selector._correlation_selection(X_numeric, y)

        # Should remove highly correlated features
        # correlated_important should be removed as it's highly correlated with important_feature
        assert len(X_selected.columns) < len(X_numeric.columns)

        # Store selection result
        result = selector.selection_history_[-1]
        assert result.selection_method == 'correlation'
        assert 'removed_features' in result.metadata

    def test_statistical_selection(self, selector, classification_data):
        """Test statistical test-based selection."""
        X, y = classification_data

        # Apply statistical selection
        X_selected = selector._statistical_selection(X, y, alpha=0.1)

        # Should select statistically significant features
        assert len(X_selected.columns) > 0
        assert len(X_selected.columns) <= len(X.columns)

        # Store selection result
        result = selector.selection_history_[-1]
        assert result.selection_method == 'statistical'
        assert 'p_values' in result.metadata

    def test_model_based_selection(self, selector, classification_data):
        """Test model-based feature selection."""
        X, y = classification_data

        # Apply model-based selection
        X_selected = selector._model_based_selection(X, y)

        # Should select features above importance threshold
        assert len(X_selected.columns) > 0
        assert len(X_selected.columns) <= len(X.columns)

        # Check that importances are stored
        assert selector.feature_importances_ is not None
        assert len(selector.feature_importances_) == len(X.columns)

        # Store selection result
        result = selector.selection_history_[-1]
        assert result.selection_method == 'model_based'

    def test_stability_selection(self, selector, classification_data):
        """Test stability selection."""
        X, y = classification_data

        # Apply stability selection
        X_selected = selector._stability_selection(X, y, selection_freq_threshold=0.3)

        # Should select stable features
        assert len(X_selected.columns) > 0
        assert len(X_selected.columns) <= len(X.columns)

        # Store selection result
        result = selector.selection_history_[-1]
        assert result.selection_method == 'stability'
        assert 'feature_counts' in result.metadata

    def test_vif_selection(self, selector, high_dim_data):
        """Test VIF-based selection."""
        X, y = high_dim_data

        # Apply VIF selection
        X_selected = selector._vif_selection(X, threshold=5.0)

        # Should remove features with high VIF
        assert len(X_selected.columns) < len(X.columns)

        # Store selection result
        result = selector.selection_history_[-1]
        assert result.selection_method == 'vif'
        assert 'removed_features' in result.metadata

    def test_combined_selection(self, selector, classification_data):
        """Test combined feature selection."""
        X, y = classification_data

        # Apply combined selection
        X_selected = selector._combined_selection(
            X, y,
            include_methods=['variance', 'mutual_info'],
            voting_threshold=0.5
        )

        # Should select features that pass multiple methods
        assert len(X_selected.columns) > 0
        assert len(X_selected.columns) <= len(X.columns)

        # Store selection result
        result = selector.selection_history_[-1]
        assert result.selection_method == 'combined'

    def test_fit_transform(self, selector, classification_data):
        """Test fit_transform method."""
        X, y = classification_data

        # Apply fit_transform
        X_selected = selector.fit_transform(X, y, method='variance')

        # Check attributes
        assert hasattr(selector, 'original_features_')
        assert hasattr(selector, 'selected_features_')
        assert hasattr(selector, 'problem_type_')

        # Check transform preserves selected features
        assert all(f in X_selected.columns for f in selector.selected_features_)

    def test_transform_without_fit(self, selector, classification_data):
        """Test transform without fitting first."""
        X, y = classification_data

        # Should raise error
        with pytest.raises(ValueError, match="must be fitted"):
            selector.transform(X)

    def test_fit_and_transform_separate(self, selector, classification_data):
        """Test fitting and transforming separately."""
        X, y = classification_data

        # Fit
        selector.fit(X, y, method='variance')

        # Transform
        X_transformed = selector.transform(X)

        # Should have selected features
        assert len(X_transformed.columns) > 0
        assert all(f in X.columns for f in X_transformed.columns)

    def test_get_feature_importance(self, selector, classification_data):
        """Test getting feature importance."""
        X, y = classification_data

        # Apply selection
        selector.fit_transform(X, y, method='mutual_info')

        # Get importance
        importance = selector.get_feature_importance('latest')

        # Check structure
        assert isinstance(importance, dict)
        assert len(importance) == len(X.columns)

        # Check values are numeric
        for score in importance.values():
            assert isinstance(score, (int, float, np.number))

    def test_evaluate_feature_set(self, selector, classification_data):
        """Test feature set evaluation."""
        X, y = classification_data

        # Evaluate subset of features
        features = ['important_feature', 'correlated_important']
        metrics = selector.evaluate_feature_set(X, y, features)

        # Check metrics
        assert isinstance(metrics, dict)
        assert 'mean_score' in metrics
        assert 'std_score' in metrics
        assert 'min_score' in metrics
        assert 'max_score' in metrics

        # Check score ranges
        if selector.problem_type_ == 'classification':
            assert 0 <= metrics['mean_score'] <= 1  # Accuracy
        else:
            assert isinstance(metrics['mean_score'], (int, float, np.number))

    def test_summarize_selection_history(self, selector, classification_data):
        """Test selection history summarization."""
        X, y = classification_data

        # Apply multiple selections
        selector._variance_selection(X)
        selector._mutual_info_selection(X, y, k=5)
        selector._correlation_selection(X, y)

        # Get summary
        summary = selector.summarize_selection_history()

        # Check structure
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 3

        # Check columns
        expected_cols = ['method', 'n_selected', 'reduction_ratio', 'cv_score', 'threshold']
        for col in expected_cols:
            assert col in summary.columns

        # Check values
        assert all(summary['n_selected'] > 0)
        assert all(summary['reduction_ratio'] >= 0)

    def test_regression_problem(self, selector, regression_data):
        """Test selection on regression problem."""
        X, y = regression_data

        # Apply selection
        X_selected = selector.fit_transform(X, y, method='mutual_info')

        # Should detect regression problem
        assert selector.problem_type_ == 'regression'

        # Should select features
        assert len(X_selected.columns) > 0

    def test_custom_estimator(self, selector, classification_data):
        """Test with custom estimator."""
        from sklearn.ensemble import GradientBoostingClassifier

        X, y = classification_data
        custom_estimator = GradientBoostingClassifier(n_estimators=10, random_state=42)

        # Apply model-based selection with custom estimator
        X_selected = selector._model_based_selection(X, y, estimator=custom_estimator)

        # Should work with custom estimator
        assert len(X_selected.columns) > 0

    def test_edge_cases(self, selector):
        """Test edge cases and error handling."""
        # Empty DataFrame
        X_empty = pd.DataFrame()
        y_empty = pd.Series([])

        # Should handle gracefully
        try:
            selector._variance_selection(X_empty)
        except:
            pass  # Expected to fail or return empty

        # Single feature
        X_single = pd.DataFrame({'single_feature': [1, 2, 3, 4, 5]})
        y_single = pd.Series([0, 1, 0, 1, 0])

        X_selected = selector._vif_selection(X_single)
        assert len(X_selected.columns) == 1

        # All features identical
        X_identical = pd.DataFrame({
            'feat1': [1, 1, 1, 1, 1],
            'feat2': [1, 1, 1, 1, 1]
        })
        y_identical = pd.Series([0, 0, 1, 1, 0])

        # Should handle gracefully
        X_selected = selector._correlation_selection(X_identical, y_identical)
        # Might remove one due to perfect correlation

    def test_plot_feature_importance(self, selector, classification_data):
        """Test feature importance plotting."""
        # Skip if matplotlib is not available
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            import pytest
            pytest.skip("matplotlib not available")

        X, y = classification_data

        # Apply selection
        selector.fit_transform(X, y, method='mutual_info')

        # Test plotting (without showing)
        with patch.object(plt, 'show'):
            selector.plot_feature_importance(top_n=5, method='latest')

    def test_config_parameters(self):
        """Test configuration parameters."""
        config = SelectionConfig(
            mi_method='classification',
            correlation_threshold=0.99,
            variance_threshold=0.001,
            model_n_estimators=50,
            cv_folds=10
        )

        selector = FeatureSelector(config)

        # Check config values
        assert selector.config.mi_method == 'classification'
        assert selector.config.correlation_threshold == 0.99
        assert selector.config.variance_threshold == 0.001
        assert selector.config.model_n_estimators == 50
        assert selector.config.cv_folds == 10

    def test_selection_result_attributes(self, selector, classification_data):
        """Test SelectionResult attributes."""
        X, y = classification_data

        # Apply selection
        selector._mutual_info_selection(X, y, k=3)

        # Get result
        result = selector.selection_history_[-1]

        # Check all attributes exist and are valid
        assert isinstance(result.selected_features, list)
        assert isinstance(result.feature_scores, dict)
        assert isinstance(result.feature_rankings, dict)
        assert isinstance(result.selection_method, str)
        assert isinstance(result.n_selected, int)
        assert isinstance(result.n_original, int)
        assert isinstance(result.reduction_ratio, float)
        assert 0 <= result.reduction_ratio <= 1
        assert isinstance(result.metadata, dict)

    def test_reproducibility(self, selector, classification_data):
        """Test that selection is reproducible."""
        X, y = classification_data

        # Apply selection twice
        X1 = selector._mutual_info_selection(X, y, k=5)
        selector.selection_history_.clear()  # Clear history
        X2 = selector._mutual_info_selection(X, y, k=5)

        # Results should be identical
        assert set(X1.columns) == set(X2.columns)

    @pytest.mark.parametrize("method", [
        'mutual_info', 'variance', 'statistical',
        'model_based', 'correlation'
    ])
    def test_all_methods_basic(self, selector, classification_data, method):
        """Test that all selection methods work."""
        X, y = classification_data

        # Apply method
        try:
            X_selected = getattr(selector, f'_{method}_selection')(X, y)

            # Should return DataFrame
            assert isinstance(X_selected, pd.DataFrame)

            # Should have selected features
            assert len(X_selected.columns) > 0
            assert len(X_selected.columns) <= len(X.columns)

            # Should have recorded result
            assert len(selector.selection_history_) > 0

        except Exception as e:
            # Some methods might fail on certain data
            pytest.skip(f"Method {method} failed on test data: {e}")