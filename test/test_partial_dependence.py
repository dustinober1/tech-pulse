"""Tests for partial dependence plot generator."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_classification, make_regression

from src.interpretability.partial_dependence import PartialDependencePlotter


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
def mixed_data():
    """Create data with mixed feature types."""
    n_samples = 100
    data = {
        'numerical_1': np.random.randn(n_samples),
        'numerical_2': np.random.randn(n_samples) * 2 + 1,
        'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples),
        'categorical_2': np.random.choice([0, 1, 2], n_samples),
        'low_cardinality': np.random.choice([0, 1], n_samples)  # Binary numeric
    }
    X_df = pd.DataFrame(data)
    y = np.random.randn(n_samples)
    return X_df, y


class TestPartialDependencePlotter:
    """Test cases for PartialDependencePlotter class."""

    def test_initialization_classification(self, classification_model, classification_data):
        """Test initialization with classification model."""
        X, _ = classification_data
        plotter = PartialDependencePlotter(classification_model, X)

        assert plotter.model == classification_model
        assert plotter.X.equals(X)
        assert plotter.n_features == 10
        assert plotter.is_classification is True
        assert plotter.is_regression is False
        assert len(plotter.feature_names) == 10

    def test_initialization_regression(self, regression_model, regression_data):
        """Test initialization with regression model."""
        X, _ = regression_data
        plotter = PartialDependencePlotter(regression_model, X)

        assert plotter.model == regression_model
        assert plotter.is_classification is False
        assert plotter.is_regression is True

    def test_feature_type_identification(self, mixed_data):
        """Test identification of categorical and numerical features."""
        X, y = mixed_data
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        plotter = PartialDependencePlotter(model, X)

        # Check that categorical features are identified
        # categorical_1 (object) and categorical_2 (int with low cardinality) should be categorical
        # low_cardinality (binary) should also be treated as categorical
        assert len(plotter.categorical_features) >= 2  # At least categorical_1 and categorical_2
        assert len(plotter.numerical_features) >= 2    # At least numerical_1 and numerical_2

    def test_plot_1d_partial_dependence_classification(self, classification_model, classification_data):
        """Test 1D partial dependence plot for classification."""
        X, _ = classification_data
        plotter = PartialDependencePlotter(classification_model, X)

        result = plotter.plot_1d_partial_dependence(
            feature='feature_0',
            target_class=0,
            n_points=10
        )

        assert 'plot_base64' in result
        assert result['feature'] == 'feature_0'
        assert result['plot_type'] == '1d_partial_dependence'
        assert result['target_class'] == 0
        assert len(result['feature_values']) > 0
        assert len(result['pdp_values']) > 0
        assert isinstance(result['plot_base64'], str)
        assert len(result['plot_base64']) > 0

    def test_plot_1d_partial_dependence_regression(self, regression_model, regression_data):
        """Test 1D partial dependence plot for regression."""
        X, _ = regression_data
        plotter = PartialDependencePlotter(regression_model, X)

        result = plotter.plot_1d_partial_dependence(
            feature=0,  # Use index instead of name
            n_points=10
        )

        assert 'plot_base64' in result
        assert result['feature'] == 'feature_0'
        assert result['plot_type'] == '1d_partial_dependence'
        assert result['target_class'] is None

    def test_plot_1d_partial_dependence_with_ice(self, classification_model, classification_data):
        """Test 1D partial dependence plot with ICE curves."""
        X, _ = classification_data
        plotter = PartialDependencePlotter(classification_model, X)

        result = plotter.plot_1d_partial_dependence(
            feature='feature_1',
            include_ice=True,
            n_ice_samples=10
        )

        assert 'plot_base64' in result
        assert result['include_ice'] is True
        assert result['feature'] == 'feature_1'

    @patch('sklearn.inspection.partial_dependence')
    def test_plot_2d_partial_dependence_contour(self, mock_pdp, classification_model, classification_data):
        """Test 2D partial dependence plot with contour."""
        X, _ = classification_data

        # Mock partial dependence result
        mock_result = {
            'values': [np.linspace(0, 1, 10), np.linspace(0, 1, 10)],
            'average': np.random.randn(10, 10)
        }
        mock_pdp.return_value = mock_result

        plotter = PartialDependencePlotter(classification_model, X)

        result = plotter.plot_2d_partial_dependence(
            features=['feature_0', 'feature_1'],
            plot_type='contour'
        )

        assert 'plot_base64' in result
        assert result['features'] == ['feature_0', 'feature_1']
        assert result['plot_type'] == '2d_partial_dependence'
        assert result['plot_subtype'] == 'contour'
        mock_pdp.assert_called_once()

    @patch('sklearn.inspection.partial_dependence')
    def test_plot_2d_partial_dependence_heatmap(self, mock_pdp, regression_model, regression_data):
        """Test 2D partial dependence plot with heatmap."""
        X, _ = regression_data

        # Mock partial dependence result
        mock_result = {
            'values': [np.linspace(0, 1, 10), np.linspace(0, 1, 10)],
            'average': np.random.randn(10, 10)
        }
        mock_pdp.return_value = mock_result

        plotter = PartialDependencePlotter(regression_model, X)

        result = plotter.plot_2d_partial_dependence(
            features=[0, 1],  # Use indices
            plot_type='heatmap',
            n_points=(10, 10)
        )

        assert 'plot_base64' in result
        assert result['features'] == ['feature_0', 'feature_1']
        assert result['plot_subtype'] == 'heatmap'

    def test_plot_2d_partial_dependence_invalid_features(self, classification_model, classification_data):
        """Test error for invalid feature input in 2D plot."""
        X, _ = classification_data
        plotter = PartialDependencePlotter(classification_model, X)

        with pytest.raises(ValueError, match="Exactly two features must be specified"):
            plotter.plot_2d_partial_dependence(features=['feature_0'])

    def test_plot_2d_partial_dependence_invalid_plot_type(self, classification_model, classification_data):
        """Test error for invalid plot type in 2D plot."""
        X, _ = classification_data
        plotter = PartialDependencePlotter(classification_model, X)

        with pytest.raises(ValueError, match="Invalid plot_type"):
            plotter.plot_2d_partial_dependence(
                features=['feature_0', 'feature_1'],
                plot_type='invalid'
            )

    def test_plot_ice_curves(self, classification_model, classification_data):
        """Test ICE curves plotting."""
        X, _ = classification_data
        plotter = PartialDependencePlotter(classification_model, X)

        result = plotter.plot_ice_curves(
            feature='feature_0',
            n_samples=10,
            centered=False
        )

        assert 'plot_base64' in result
        assert result['feature'] == 'feature_0'
        assert result['plot_type'] == 'ice_curves'
        assert result['centered'] is False
        assert result['n_curves_plotted'] == 10

    def test_plot_ice_curves_centered(self, regression_model, regression_data):
        """Test centered ICE curves plotting."""
        X, _ = regression_data
        plotter = PartialDependencePlotter(regression_model, X)

        result = plotter.plot_ice_curves(
            feature=0,
            centered=True
        )

        assert result['centered'] is True
        assert 'plot_base64' in result

    def test_plot_multiple_features(self, classification_model, classification_data):
        """Test plotting multiple features."""
        X, _ = classification_data
        plotter = PartialDependencePlotter(classification_model, X)

        result = plotter.plot_multiple_features(
            features=['feature_0', 'feature_1', 'feature_2'],
            n_cols=2,
            n_points=10
        )

        assert 'plot_base64' in result
        assert result['plot_type'] == 'multiple_partial_dependence'
        assert result['n_features'] == 3
        assert len(result['features']) == 3

    def test_plot_multiple_features_single(self, classification_model, classification_data):
        """Test plotting single feature with multiple features method."""
        X, _ = classification_data
        plotter = PartialDependencePlotter(classification_model, X)

        result = plotter.plot_multiple_features(
            features=['feature_0'],
            figsize=(8, 6)
        )

        assert 'plot_base64' in result
        assert result['n_features'] == 1

    @patch('sklearn.inspection.partial_dependence')
    def test_get_feature_importance_from_pdp(self, mock_pdp, classification_model, classification_data):
        """Test feature importance calculation from PDP."""
        X, _ = classification_data

        # Mock partial dependence results for all features
        def mock_pdp_side_effect(*args, **kwargs):
            feature_idx = kwargs.get('features', [[0]])[0][0] if 'features' in kwargs else 0
            n_points = kwargs.get('grid_resolution', 100)
            values = np.linspace(0, 1, n_points)
            # Create varying importance ranges
            avg_range = np.random.randn(n_points) * (feature_idx + 1)
            return {
                'values': [values],
                'average': [avg_range]
            }

        mock_pdp.side_effect = mock_pdp_side_effect

        plotter = PartialDependencePlotter(classification_model, X)

        result = plotter.get_feature_importance_from_pdp(n_points=10)

        assert 'feature_importance' in result
        assert 'feature_ranges' in result
        assert 'feature_ranking' in result
        assert len(result['feature_importance']) == 10

        # Check that ranking is sorted by importance
        ranking_values = [imp for _, imp in result['feature_ranking']]
        assert all(ranking_values[i] >= ranking_values[i+1]
                  for i in range(len(ranking_values)-1))

    def test_generate_interaction_summary(self, classification_model, classification_data):
        """Test interaction summary generation."""
        X, _ = classification_data
        plotter = PartialDependencePlotter(classification_model, X)

        result = plotter.generate_interaction_summary(
            n_interactions=5,
            n_points=(5, 5)
        )

        assert 'interaction_strengths' in result
        assert 'interaction_details' in result
        assert 'top_interactions' in result
        assert 'n_interactions_analyzed' in result
        assert len(result['interaction_strengths']) <= 5

    def test_generate_interaction_summary_insufficient_features(self, classification_data):
        """Test interaction summary with insufficient features."""
        X, y = classification_data
        X = X.iloc[:, :1]  # Keep only one feature
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        plotter = PartialDependencePlotter(model, X)

        result = plotter.generate_interaction_summary()

        assert 'error' in result
        assert 'Need at least 2 features' in result['error']

    def test_feature_conversion_string_to_index(self, classification_model, classification_data):
        """Test conversion of feature names to indices."""
        X, _ = classification_data
        plotter = PartialDependencePlotter(classification_model, X)

        # Test with feature name
        assert plotter.feature_names[0] == 'feature_0'
        assert len(plotter.feature_names) == 10

    def test_percentile_range_parameter(self, classification_model, classification_data):
        """Test percentile range parameter in plots."""
        X, _ = classification_data
        plotter = PartialDependencePlotter(classification_model, X)

        result = plotter.plot_1d_partial_dependence(
            feature='feature_0',
            percentile_range=(0.1, 0.9)
        )

        assert 'plot_base64' in result
        # The plot should be generated with the specified percentile range

    def test_custom_figure_size(self, regression_model, regression_data):
        """Test custom figure size in plots."""
        X, _ = regression_data
        plotter = PartialDependencePlotter(regression_model, X)

        result = plotter.plot_1d_partial_dependence(
            feature='feature_0',
            figsize=(12, 8)
        )

        assert 'plot_base64' in result
        # The plot should be generated with the specified figure size

    def test_grid_resolution_parameter(self, classification_model, classification_data):
        """Test grid resolution parameter in plots."""
        X, _ = classification_data
        plotter = PartialDependencePlotter(classification_model, X)

        result = plotter.plot_1d_partial_dependence(
            feature='feature_0',
            n_points=50
        )

        assert 'plot_base64' in result
        # The plot should be generated with the specified number of points

    @patch('sklearn.inspection.partial_dependence')
    def test_2d_grid_resolution(self, mock_pdp, classification_model, classification_data):
        """Test custom grid resolution in 2D plots."""
        X, _ = classification_data

        # Mock partial dependence result
        mock_result = {
            'values': [np.linspace(0, 1, 20), np.linspace(0, 1, 30)],  # Different resolutions
            'average': np.random.randn(20, 30)
        }
        mock_pdp.return_value = mock_result

        plotter = PartialDependencePlotter(classification_model, X)

        result = plotter.plot_2d_partial_dependence(
            features=['feature_0', 'feature_1'],
            n_points=(20, 30)
        )

        assert 'plot_base64' in result
        mock_pdp.assert_called_with(
            plotter.model,
            plotter.X,
            features=[0, 1],
            grid_resolution=(20, 30),
            percentiles=(0.05, 0.95),
            method='brute'
        )

    def test_base64_encoding(self, classification_model, classification_data):
        """Test that plots are properly base64 encoded."""
        X, _ = classification_data
        plotter = PartialDependencePlotter(classification_model, X)

        result = plotter.plot_1d_partial_dependence(feature='feature_0')

        # Verify base64 encoding
        assert isinstance(result['plot_base64'], str)
        assert len(result['plot_base64']) > 0

        # Try to decode to verify it's valid base64
        import base64
        try:
            decoded = base64.b64decode(result['plot_base64'])
            assert len(decoded) > 0, "Decoded image is empty"
        except Exception as e:
            pytest.fail(f"Invalid base64 encoding: {e}")

    def test_plot_data_returned(self, classification_model, classification_data):
        """Test that plot data is returned along with images."""
        X, _ = classification_data
        plotter = PartialDependencePlotter(classification_model, X)

        # Test 1D PDP data
        result_1d = plotter.plot_1d_partial_dependence(feature='feature_0')
        assert 'feature_values' in result_1d
        assert 'pdp_values' in result_1d
        assert isinstance(result_1d['feature_values'], list)
        assert isinstance(result_1d['pdp_values'], list)

        # Test 2D PDP data
        with patch('sklearn.inspection.partial_dependence') as mock_pdp:
            mock_pdp.return_value = {
                'values': [np.linspace(0, 1, 10), np.linspace(0, 1, 10)],
                'average': np.random.randn(10, 10)
            }

            result_2d = plotter.plot_2d_partial_dependence(
                features=['feature_0', 'feature_1']
            )
            assert 'grid_x' in result_2d
            assert 'grid_y' in result_2d
            assert 'grid_z' in result_2d

    def test_target_class_handling(self, classification_model, classification_data):
        """Test target class parameter handling."""
        X, _ = classification_data
        plotter = PartialDependencePlotter(classification_model, X)

        # Test with specific target class
        result_with_class = plotter.plot_1d_partial_dependence(
            feature='feature_0',
            target_class=1
        )
        assert result_with_class['target_class'] == 1

        # Test without target class (should default to 0)
        result_without_class = plotter.plot_1d_partial_dependence(
            feature='feature_0'
        )
        assert result_without_class['target_class'] == 0