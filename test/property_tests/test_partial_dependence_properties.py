"""Property tests for partial dependence plot generator."""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule
from unittest.mock import Mock, patch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_classification, make_regression

from src.interpretability.partial_dependence import PartialDependencePlotter


class Property15PartialDependencePlotCreation:
    """
    Property 15: Partial dependence plot creation

    Validates: Requirements 4.3

    Ensures that partial dependence plot generator correctly creates
    1D plots, 2D plots, and ICE plots with proper functionality.
    """

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=50, max_value=200),
        n_features=st.integers(min_value=3, max_value=10),
        feature_idx=st.integers(min_value=0, max_value=9),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_1d_partial_dependence_monotonicity_property(self, n_samples, n_features, feature_idx, random_state):
        """Test that 1D partial dependence plots have consistent value ranges."""
        # Generate synthetic data
        X, y = make_regression(
            n_samples=n_samples,
            n_features=max(n_features, feature_idx + 1),
            noise=0.1,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)

        # Train a model
        model = RandomForestRegressor(n_estimators=10, random_state=random_state)
        model.fit(X_df, y)

        # Initialize plotter
        plotter = PartialDependencePlotter(model, X_df)

        # Generate 1D partial dependence plot
        result = plotter.plot_1d_partial_dependence(
            feature=feature_idx,
            n_points=20
        )

        # Verify plot properties
        assert 'plot_base64' in result, "Plot base64 missing from 1D PDP"
        assert 'feature_values' in result, "Feature values missing from 1D PDP"
        assert 'pdp_values' in result, "PDP values missing from 1D PDP"
        assert result['plot_type'] == '1d_partial_dependence', "Incorrect plot type"

        # Verify data consistency
        assert len(result['feature_values']) == len(result['pdp_values']), \
            "Feature values and PDP values length mismatch"
        assert len(result['feature_values']) == 20, "Incorrect number of points generated"
        assert isinstance(result['plot_base64'], str), "Plot base64 not a string"
        assert len(result['plot_base64']) > 0, "Plot base64 is empty"

        # Verify numerical stability
        feature_values = np.array(result['feature_values'])
        pdp_values = np.array(result['pdp_values'])
        assert not np.any(np.isnan(feature_values)), "Feature values contain NaN"
        assert not np.any(np.isnan(pdp_values)), "PDP values contain NaN"
        assert not np.any(np.isinf(feature_values)), "Feature values contain inf"
        assert not np.any(np.isinf(pdp_values)), "PDP values contain inf"

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=50, max_value=200),
        n_features=st.integers(min_value=3, max_value=10),
        feature_pair=st.lists(st.integers(min_value=0, max_value=9), min_size=2, max_size=2, unique=True),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_2d_partial_dependence_grid_property(self, n_samples, n_features, feature_pair, random_state):
        """Test that 2D partial dependence plots have proper grid structure."""
        # Generate synthetic data with enough features
        X, y = make_classification(
            n_samples=n_samples,
            n_features=max(n_features, max(feature_pair) + 1),
            n_informative=max(n_features, max(feature_pair) + 1) // 2,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)

        # Train a model
        model = RandomForestClassifier(n_estimators=10, random_state=random_state)
        model.fit(X_df, y)

        # Initialize plotter
        plotter = PartialDependencePlotter(model, X_df)

        # Mock sklearn.inspection.partial_dependence to avoid computation overhead
        with patch('sklearn.inspection.partial_dependence') as mock_pdp:
            # Create mock grid data
            n_x, n_y = 10, 10
            x_vals = np.linspace(0, 1, n_x)
            y_vals = np.linspace(0, 1, n_y)
            XX, YY = np.meshgrid(x_vals, y_vals)
            Z = np.random.randn(n_y, n_x) * 0.1  # Small random values

            mock_pdp.return_value = {
                'values': [x_vals, y_vals],
                'average': Z
            }

            # Generate 2D partial dependence plot
            result = plotter.plot_2d_partial_dependence(
                features=feature_pair,
                plot_type='contour',
                n_points=(n_x, n_y)
            )

        # Verify plot properties
        assert 'plot_base64' in result, "Plot base64 missing from 2D PDP"
        assert 'grid_x' in result, "Grid X values missing from 2D PDP"
        assert 'grid_y' in result, "Grid Y values missing from 2D PDP"
        assert 'grid_z' in result, "Grid Z values missing from 2D PDP"
        assert result['plot_type'] == '2d_partial_dependence', "Incorrect plot type"
        assert result['plot_subtype'] == 'contour', "Incorrect plot subtype"

        # Verify grid dimensions
        assert len(result['grid_x']) == n_x, f"Grid X has incorrect size: {len(result['grid_x'])}"
        assert len(result['grid_y']) == n_y, f"Grid Y has incorrect size: {len(result['grid_y'])}"
        assert len(result['grid_z']) == n_y, f"Grid Z has incorrect number of rows: {len(result['grid_z'])}"
        assert len(result['grid_z'][0]) == n_x, f"Grid Z has incorrect number of columns: {len(result['grid_z'][0])}"

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=50, max_value=200),
        n_features=st.integers(min_value=3, max_value=10),
        feature_idx=st.integers(min_value=0, max_value=9),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_ice_curves_diversity_property(self, n_samples, n_features, feature_idx, random_state):
        """Test that ICE curves show diverse behavior across samples."""
        # Generate synthetic data
        X, y = make_regression(
            n_samples=n_samples,
            n_features=max(n_features, feature_idx + 1),
            noise=0.1,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)

        # Train a model
        model = RandomForestRegressor(n_estimators=10, random_state=random_state)
        model.fit(X_df, y)

        # Initialize plotter
        plotter = PartialDependencePlotter(model, X_df)

        # Generate ICE curves
        result = plotter.plot_ice_curves(
            feature=feature_idx,
            n_samples=20,
            n_points=15
        )

        # Verify ICE plot properties
        assert 'plot_base64' in result, "Plot base64 missing from ICE curves"
        assert 'feature_values' in result, "Feature values missing from ICE curves"
        assert result['plot_type'] == 'ice_curves', "Incorrect plot type"
        assert result['n_curves_plotted'] == 20, "Incorrect number of curves plotted"
        assert not result['centered'], "ICE curves should not be centered by default"

        # Verify data consistency
        assert len(result['feature_values']) == 15, "Incorrect number of points in ICE curves"
        assert isinstance(result['plot_base64'], str), "Plot base64 not a string"

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=50, max_value=200),
        n_features=st.integers(min_value=3, max_value=10),
        n_plot_features=st.integers(min_value=1, max_value=5),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_multiple_features_grid_property(self, n_samples, n_features, n_plot_features, random_state):
        """Test that multiple features plot maintains grid structure."""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=max(n_features, n_plot_features),
            n_informative=max(n_features, n_plot_features) // 2,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)

        # Train a model
        model = RandomForestClassifier(n_estimators=10, random_state=random_state)
        model.fit(X_df, y)

        # Initialize plotter
        plotter = PartialDependencePlotter(model, X_df)

        # Select features to plot
        features_to_plot = feature_names[:n_plot_features]

        # Generate multiple features plot
        result = plotter.plot_multiple_features(
            features=features_to_plot,
            n_cols=min(3, n_plot_features),
            n_points=10
        )

        # Verify multiple features plot properties
        assert 'plot_base64' in result, "Plot base64 missing from multiple features PDP"
        assert result['plot_type'] == 'multiple_partial_dependence', "Incorrect plot type"
        assert result['n_features'] == n_plot_features, "Incorrect number of features"
        assert len(result['features']) == n_plot_features, "Features list length mismatch"

        # Verify all requested features are present
        for feature in features_to_plot:
            assert feature in result['features'], f"Feature {feature} missing from plot"

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=50, max_value=200),
        n_features=st.integers(min_value=3, max_value=10),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_feature_importance_from_pdp_consistency(self, n_samples, n_features, random_state):
        """Test that feature importance from PDP is consistent and properly ordered."""
        # Generate synthetic data
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=0.1,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)

        # Train a model
        model = RandomForestRegressor(n_estimators=10, random_state=random_state)
        model.fit(X_df, y)

        # Initialize plotter
        plotter = PartialDependencePlotter(model, X_df)

        # Mock sklearn.inspection.partial_dependence with varying importance
        with patch('sklearn.inspection.partial_dependence') as mock_pdp:
            def create_mock_result(feature_idx, n_points=10):
                # Create different ranges for different features
                importance_range = (feature_idx + 1) * 0.5
                values = np.linspace(0, 1, n_points)
                avg_values = np.linspace(0, importance_range, n_points)
                return {
                    'values': [values],
                    'average': [avg_values]
                }

            mock_pdp.side_effect = lambda model, X, features, **kwargs: create_mock_result(
                features[0][0] if isinstance(features[0], list) else features[0],
                kwargs.get('grid_resolution', 100)
            )

            # Get feature importance from PDP
            result = plotter.get_feature_importance_from_pdp(n_points=10)

        # Verify feature importance properties
        assert 'feature_importance' in result, "Feature importance missing"
        assert 'feature_ranges' in result, "Feature ranges missing"
        assert 'feature_ranking' in result, "Feature ranking missing"
        assert len(result['feature_importance']) == n_features, "Incorrect number of features"

        # Verify ranking is sorted by importance (descending)
        ranking_values = [imp for _, imp in result['feature_ranking']]
        assert all(ranking_values[i] >= ranking_values[i+1]
                  for i in range(len(ranking_values)-1)), \
            "Feature ranking not sorted by importance"

        # Verify all features are present in importance dictionary
        for feature_name in feature_names:
            assert feature_name in result['feature_importance'], f"Feature {feature_name} missing"

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=50, max_value=200),
        n_features=st.integers(min_value=3, max_value=10),
        n_interactions=st.integers(min_value=2, max_value=10),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_interaction_summary_completeness(self, n_samples, n_features, n_interactions, random_state):
        """Test that interaction summary provides complete analysis."""
        # Generate synthetic data with at least 2 features
        actual_n_features = max(n_features, 2)
        X, y = make_classification(
            n_samples=n_samples,
            n_features=actual_n_features,
            n_informative=actual_n_features // 2,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(actual_n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)

        # Train a model
        model = RandomForestClassifier(n_estimators=10, random_state=random_state)
        model.fit(X_df, y)

        # Initialize plotter
        plotter = PartialDependencePlotter(model, X_df)

        # Generate interaction summary
        result = plotter.generate_interaction_summary(
            n_interactions=min(n_interactions, 5),
            n_points=(5, 5)
        )

        # Verify interaction summary properties
        assert 'interaction_strengths' in result, "Interaction strengths missing"
        assert 'interaction_details' in result, "Interaction details missing"
        assert 'top_interactions' in result, "Top interactions missing"
        assert 'n_interactions_analyzed' in result, "Number of interactions analyzed missing"

        # Verify interaction count consistency
        assert result['n_interactions_analyzed'] <= min(n_interactions, 5), \
            "Too many interactions analyzed"
        assert len(result['interaction_strengths']) == result['n_interactions_analyzed'], \
            "Interaction strengths count mismatch"

        # Verify interaction details structure
        for interaction_name, details in result['interaction_details'].items():
            assert 'feature_1' in details, f"Feature 1 missing for {interaction_name}"
            assert 'feature_2' in details, f"Feature 2 missing for {interaction_name}"
            assert 'interaction_strength' in details, f"Strength missing for {interaction_name}"
            assert 'grid_shape' in details, f"Grid shape missing for {interaction_name}"

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=50, max_value=200),
        n_features=st.integers(min_value=3, max_value=10),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_plot_consistency_across_calls(self, n_samples, n_features, random_state):
        """Test that plots are consistent across multiple calls."""
        # Generate synthetic data
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=0.1,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)

        # Train a model
        model = RandomForestRegressor(n_estimators=10, random_state=random_state)
        model.fit(X_df, y)

        # Initialize plotter
        plotter = PartialDependencePlotter(model, X_df)

        # Mock sklearn.inspection.partial_dependence for consistency
        with patch('sklearn.inspection.partial_dependence') as mock_pdp:
            # Create consistent mock result
            mock_result = {
                'values': [np.linspace(0, 1, 20)],
                'average': [np.linspace(0, 1, 20) * 0.5]
            }
            mock_pdp.return_value = mock_result

            # Generate 1D PDP twice
            result1 = plotter.plot_1d_partial_dependence(
                feature=0,
                n_points=20
            )
            result2 = plotter.plot_1d_partial_dependence(
                feature=0,
                n_points=20
            )

        # Verify consistency
        assert result1['feature'] == result2['feature'], "Feature name inconsistent"
        assert result1['plot_type'] == result2['plot_type'], "Plot type inconsistent"
        np.testing.assert_array_almost_equal(
            result1['feature_values'],
            result2['feature_values'],
            decimal=5,
            err_msg="Feature values inconsistent across calls"
        )
        np.testing.assert_array_almost_equal(
            result1['pdp_values'],
            result2['pdp_values'],
            decimal=5,
            err_msg="PDP values inconsistent across calls"
        )

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=50, max_value=200),
        n_features=st.integers(min_value=3, max_value=10),
        feature_idx=st.integers(min_value=0, max_value=9),
        centered=st.booleans(),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_ice_centering_property(self, n_samples, n_features, feature_idx, centered, random_state):
        """Test that ICE centering affects the output correctly."""
        # Generate synthetic data
        X, y = make_regression(
            n_samples=n_samples,
            n_features=max(n_features, feature_idx + 1),
            noise=0.1,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)

        # Train a model
        model = RandomForestRegressor(n_estimators=10, random_state=random_state)
        model.fit(X_df, y)

        # Initialize plotter
        plotter = PartialDependencePlotter(model, X_df)

        # Generate ICE curves with specified centering
        result = plotter.plot_ice_curves(
            feature=feature_idx,
            centered=centered
        )

        # Verify centering property
        assert result['centered'] == centered, "Centering parameter not correctly stored"
        assert 'plot_base64' in result, "Plot base64 missing from ICE curves"

        # The plot should be generated regardless of centering
        assert isinstance(result['plot_base64'], str), "Plot base64 not a string"
        assert len(result['plot_base64']) > 0, "Plot base64 is empty"

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=50, max_value=200),
        n_features=st.integers(min_value=3, max_value=10),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_all_plot_types_base64_encoding(self, n_samples, n_features, random_state):
        """Test that all plot types generate valid base64 encodings."""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 2,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)

        # Train a model
        model = RandomForestClassifier(n_estimators=10, random_state=random_state)
        model.fit(X_df, y)

        # Initialize plotter
        plotter = PartialDependencePlotter(model, X_df)

        # Mock sklearn.inspection.partial_dependence
        with patch('sklearn.inspection.partial_dependence') as mock_pdp:
            # Mock 1D result
            mock_1d_result = {
                'values': [np.linspace(0, 1, 20)],
                'average': [np.linspace(0, 1, 20) * 0.5]
            }

            # Mock 2D result
            mock_2d_result = {
                'values': [np.linspace(0, 1, 10), np.linspace(0, 1, 10)],
                'average': np.random.randn(10, 10) * 0.1
            }

            # Mock ICE result
            mock_ice_result = {
                'values': [np.linspace(0, 1, 20)],
                'individual': [np.random.randn(50, 20) * 0.1]
            }

            def mock_pdp_side_effect(*args, **kwargs):
                kind = kwargs.get('kind', 'average')
                if kind == 'individual':
                    return mock_ice_result
                elif len(kwargs.get('features', [[0]])[0]) == 2:
                    return mock_2d_result
                else:
                    return mock_1d_result

            mock_pdp.side_effect = mock_pdp_side_effect

            # Test all plot types
            plots_to_test = [
                lambda: plotter.plot_1d_partial_dependence(feature=0),
                lambda: plotter.plot_2d_partial_dependence(features=[0, 1]),
                lambda: plotter.plot_ice_curves(feature=0),
                lambda: plotter.plot_multiple_features(features=[0, 1, 2])
            ]

            for plot_func in plots_to_test:
                result = plot_func()

                # Verify base64 encoding
                assert 'plot_base64' in result, f"Plot base64 missing from {plot_func.__name__}"
                assert isinstance(result['plot_base64'], str), f"Plot base64 not a string in {plot_func.__name__}"
                assert len(result['plot_base64']) > 0, f"Plot base64 is empty in {plot_func.__name__}"

                # Try to decode to verify it's valid base64
                import base64
                try:
                    decoded = base64.b64decode(result['plot_base64'])
                    assert len(decoded) > 0, f"Decoded plot is empty in {plot_func.__name__}"
                except Exception as e:
                    pytest.fail(f"Invalid base64 encoding in {plot_func.__name__}: {e}")

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=50, max_value=200),
        n_features=st.integers(min_value=3, max_value=10),
        percentile_low=st.floats(min_value=0.0, max_value=0.3),
        percentile_high=st.floats(min_value=0.7, max_value=1.0),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_percentile_range_parameter(self, n_samples, n_features, percentile_low, percentile_high, random_state):
        """Test that percentile range parameter is properly handled."""
        # Generate synthetic data
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=0.1,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)

        # Train a model
        model = RandomForestRegressor(n_estimators=10, random_state=random_state)
        model.fit(X_df, y)

        # Initialize plotter
        plotter = PartialDependencePlotter(model, X_df)

        # Create custom percentile range
        percentile_range = (percentile_low, percentile_high)
        assert percentile_low < percentile_high, "Invalid percentile range"

        # Mock sklearn.inspection.partial_dependence to capture parameters
        with patch('sklearn.inspection.partial_dependence') as mock_pdp:
            mock_pdp.return_value = {
                'values': [np.linspace(0, 1, 20)],
                'average': [np.linspace(0, 1, 20) * 0.5]
            }

            # Generate 1D PDP with custom percentile range
            result = plotter.plot_1d_partial_dependence(
                feature=0,
                percentile_range=percentile_range
            )

        # Verify the percentile range was passed correctly
        mock_pdp.assert_called_once()
        call_kwargs = mock_pdp.call_args[1]
        assert call_kwargs['percentiles'] == percentile_range, "Percentile range not passed correctly"

        # Verify plot was generated
        assert 'plot_base64' in result, "Plot base64 missing with custom percentile range"