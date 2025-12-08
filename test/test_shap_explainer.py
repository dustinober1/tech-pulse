"""Tests for SHAP explainer."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_classification, make_regression

from src.interpretability.shap_explainer import SHAPExplainer


@pytest.fixture
def classification_data():
    """Create sample classification data."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, y


@pytest.fixture
def regression_data():
    """Create sample regression data."""
    X, y = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=5,
        noise=0.1,
        random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, y


@pytest.fixture
def classification_model(classification_data):
    """Create a trained classification model."""
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def regression_model(regression_data):
    """Create a trained regression model."""
    X, y = regression_data
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def linear_model(regression_data):
    """Create a trained linear model."""
    X, y = regression_data
    model = LinearRegression()
    model.fit(X, y)
    return model


class TestSHAPExplainer:
    """Test cases for SHAPExplainer class."""

    def test_initialization_with_tree_model(self, classification_model, classification_data):
        """Test initialization with a tree-based model."""
        X, _ = classification_data
        explainer = SHAPExplainer(classification_model, background_data=X[:10])

        assert explainer.model == classification_model
        assert explainer.background_data is not None
        assert explainer.explainer is not None

    def test_initialization_with_linear_model(self, linear_model, regression_data):
        """Test initialization with a linear model."""
        X, _ = regression_data
        explainer = SHAPExplainer(linear_model, background_data=X[:10])

        assert explainer.model == linear_model
        assert explainer.explainer is not None

    def test_initialization_without_background_data(self, classification_model):
        """Test initialization without background data."""
        explainer = SHAPExplainer(classification_model)

        assert explainer.model == classification_model
        assert explainer.background_data is None
        assert explainer.explainer is not None

    @patch('src.interpretability.shap_explainer.shap')
    def test_explain_with_dataframe(self, mock_shap, classification_model, classification_data):
        """Test explain method with DataFrame input."""
        X, _ = classification_data
        mock_explainer = Mock()
        mock_shap.TreeExplainer.return_value = mock_explainer
        mock_shap.LinearExplainer.return_value = mock_explainer
        mock_shap.Explainer.return_value = mock_explainer

        # Mock SHAP values
        mock_shap_values = np.random.randn(10, 10)
        mock_explainer.shap_values.return_value = mock_shap_values
        mock_explainer.expected_value = 0.5

        explainer = SHAPExplainer(classification_model)
        result = explainer.explain(X[:10])

        assert 'shap_values' in result
        assert 'expected_value' in result
        assert 'feature_names' in result
        assert result['shap_values'].shape == (10, 10)

    def test_explain_with_numpy_array(self, classification_model, classification_data):
        """Test explain method with numpy array input."""
        X, _ = classification_data
        explainer = SHAPExplainer(classification_model, background_data=X[:10])

        result = explainer.explain(X.values[:10])

        assert 'shap_values' in result
        assert 'expected_value' in result
        assert isinstance(result['shap_values'], (np.ndarray, list))

    def test_explain_instance_with_series(self, classification_model, classification_data):
        """Test explain_instance method with Series input."""
        X, _ = classification_data
        explainer = SHAPExplainer(classification_model, background_data=X[:10])

        result = explainer.explain_instance(X.iloc[0])

        assert 'shap_values' in result
        assert 'expected_value' in result
        assert 'feature_names' in result
        assert 'prediction' in result
        assert 'features' in result
        assert len(result['shap_values']) == X.shape[1]

    def test_explain_instance_with_numpy_array(self, classification_model, classification_data):
        """Test explain_instance method with numpy array input."""
        X, _ = classification_data
        explainer = SHAPExplainer(classification_model, background_data=X[:10])

        result = explainer.explain_instance(X.values[0])

        assert 'shap_values' in result
        assert 'prediction' in result
        assert len(result['shap_values']) == X.shape[1]

    def test_get_global_summary(self, classification_model, classification_data):
        """Test get_global_summary method."""
        X, _ = classification_data
        explainer = SHAPExplainer(classification_model, background_data=X[:10])

        # First calculate SHAP values
        explainer.explain(X[:10])

        summary = explainer.get_global_summary()

        assert 'feature_importance' in summary
        assert 'feature_ranking' in summary
        assert 'mean_shap_values' in summary
        assert 'std_shap_values' in summary

        # Check that feature ranking is sorted by importance
        importance_values = [item[1] for item in summary['feature_ranking']]
        assert all(importance_values[i] >= importance_values[i+1]
                  for i in range(len(importance_values)-1))

    @patch('src.interpretability.shap_explainer.shap')
    def test_create_waterfall_plot(self, mock_shap, classification_model, classification_data):
        """Test create_waterfall_plot method."""
        X, _ = classification_data

        # Mock shap.plots.waterfall
        mock_shap.plots.waterfall = Mock()

        explainer = SHAPExplainer(classification_model, background_data=X[:10])

        result = explainer.create_waterfall_plot(X.iloc[0])

        assert 'plot_base64' in result
        assert 'explanation' in result
        assert result['plot_type'] == 'waterfall'

    @patch('src.interpretability.shap_explainer.shap')
    def test_create_summary_plot(self, mock_shap, classification_model, classification_data):
        """Test create_summary_plot method."""
        X, _ = classification_data

        # Mock shap.plots.beeswarm
        mock_shap.plots.beeswarm = Mock()

        explainer = SHAPExplainer(classification_model, background_data=X[:10])

        result = explainer.create_summary_plot(X[:10])

        assert 'plot_base64' in result
        assert result['plot_type'] == 'summary'
        assert result['n_features'] == 10
        assert result['n_samples'] == 10

    @patch('src.interpretability.shap_explainer.shap')
    def test_create_force_plot(self, mock_shap, classification_model, classification_data):
        """Test create_force_plot method."""
        X, _ = classification_data

        # Mock shap.plots.force
        mock_shap.plots.force = Mock()

        explainer = SHAPExplainer(classification_model, background_data=X[:10])

        result = explainer.create_force_plot(X.iloc[0])

        assert 'plot_base64' in result
        assert 'explanation' in result
        assert result['plot_type'] == 'force'

    def test_create_feature_importance_plot(self, classification_model, classification_data):
        """Test create_feature_importance_plot method."""
        X, _ = classification_data
        explainer = SHAPExplainer(classification_model, background_data=X[:10])

        # First calculate SHAP values
        explainer.explain(X[:10])

        result = explainer.create_feature_importance_plot()

        assert 'plot_base64' in result
        assert 'feature_importance' in result
        assert result['plot_type'] == 'feature_importance'
        assert result['n_features_shown'] <= 20  # Should show top 20 features

    def test_error_without_shap_library(self, classification_model):
        """Test error when SHAP library is not installed."""
        with patch.dict('sys.modules', {'shap': None}):
            with patch('src.interpretability.shap_explainer.shap', side_effect=ImportError):
                with pytest.raises(ImportError, match="SHAP library is not installed"):
                    SHAPExplainer(classification_model)

    def test_error_explaining_without_initialization(self, classification_data):
        """Test error when trying to explain without proper initialization."""
        # Create a mock explainer that is None
        explainer = SHAPExplainer.__new__(SHAPExplainer)
        explainer.model = Mock()
        explainer.background_data = None
        explainer.explainer = None
        explainer.shap_values = None

        X, _ = classification_data

        with pytest.raises(ValueError, match="Explainer not initialized"):
            explainer.explain(X[:10])

    def test_error_global_summary_without_shap_values(self, classification_model):
        """Test error when getting global summary without SHAP values."""
        explainer = SHAPExplainer(classification_model)
        explainer.shap_values = None

        with pytest.raises(ValueError, match="No SHAP values computed"):
            explainer.get_global_summary()

    def test_multi_class_shap_values(self, classification_data):
        """Test handling of multi-class SHAP values."""
        X, y = classification_data

        # Create a multi-class model
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Convert to 3-class problem
        y_multi = y % 3
        model.fit(X, y_multi)

        explainer = SHAPExplainer(model, background_data=X[:10])
        result = explainer.explain(X[:10])

        # Should handle multi-class SHAP values (list of arrays)
        assert 'shap_values' in result
        assert isinstance(result['shap_values'], (np.ndarray, list))

    def test_feature_names_preservation(self, classification_model, classification_data):
        """Test that feature names are preserved throughout the process."""
        X, _ = classification_data
        explainer = SHAPExplainer(classification_model, background_data=X[:10])

        result = explainer.explain(X[:10])

        assert result['feature_names'] == list(X.columns)

        instance_result = explainer.explain_instance(X.iloc[0])
        assert instance_result['feature_names'] == list(X.columns)

        summary = explainer.get_global_summary()
        assert set(summary['feature_importance'].keys()) == set(X.columns)

    def test_explainer_with_logistic_regression(self, classification_data):
        """Test explainer with logistic regression model."""
        X, y = classification_data
        model = LogisticRegression(max_iter=100, random_state=42)
        model.fit(X, y)

        explainer = SHAPExplainer(model, background_data=X[:10])
        result = explainer.explain(X[:10])

        assert 'shap_values' in result
        assert 'expected_value' in result

    def test_expected_value_handling(self, classification_model, classification_data):
        """Test proper handling of expected values (base values)."""
        X, _ = classification_data
        explainer = SHAPExplainer(classification_model, background_data=X[:10])

        result = explainer.explain(X[:10])

        # Expected value should be available
        assert result['expected_value'] is not None

    def test_shap_value_consistency(self, classification_model, classification_data):
        """Test that SHAP values are consistent across explanations."""
        X, _ = classification_data
        explainer = SHAPExplainer(classification_model, background_data=X[:10])

        # Calculate SHAP values twice
        result1 = explainer.explain(X[:10])
        result2 = explainer.explain(X[:10])

        # Results should be consistent
        np.testing.assert_array_almost_equal(
            np.array(result1['shap_values']),
            np.array(result2['shap_values']),
            decimal=5
        )