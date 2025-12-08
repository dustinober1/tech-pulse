"""Property tests for SHAP explainer."""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_classification, make_regression

from src.interpretability.shap_explainer import SHAPExplainer


class Property13SHAPValueGeneration:
    """
    Property 13: SHAP value generation

    Validates: Requirements 4.1

    Ensures that SHAP explainer correctly generates SHAP values with
    proper additive property and consistency.
    """

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=10, max_value=50),
        n_features=st.integers(min_value=3, max_value=8),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_shap_additive_property(self, n_samples, n_features, random_state):
        """Test that SHAP values satisfy the additive property."""
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

        # Initialize SHAP explainer
        explainer = SHAPExplainer(model, background_data=X_df[:5])

        # Explain a single instance
        instance = X_df.iloc[0:1]
        result = explainer.explain(instance)

        # Check additive property: prediction = expected_value + sum(SHAP_values)
        prediction = model.predict(instance)[0]
        expected_value = result['expected_value']
        shap_values = result['shap_values'][0]  # First instance

        # For tree models, expected_value might be an array
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[0]

        # The sum should approximately equal the prediction - expected_value
        sum_shap = np.sum(shap_values)

        # Allow for some numerical tolerance
        np.testing.assert_allclose(
            prediction,
            expected_value + sum_shap,
            rtol=1e-2,
            err_msg="SHAP additive property not satisfied: prediction ≠ expected_value + ΣSHAP"
        )

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=10, max_value=50),
        n_features=st.integers(min_value=3, max_value=8),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_shap_consistency_property(self, n_samples, n_features, random_state):
        """Test that SHAP values satisfy the consistency property."""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
            n_redundant=0,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)

        # Train a model
        model = RandomForestClassifier(n_estimators=10, random_state=random_state)
        model.fit(X_df, y)

        # Initialize SHAP explainer
        explainer = SHAPExplainer(model, background_data=X_df[:5])

        # Explain the same instance multiple times
        instance = X_df.iloc[0:1]
        result1 = explainer.explain(instance)
        result2 = explainer.explain(instance)

        # Results should be consistent
        np.testing.assert_array_almost_equal(
            result1['shap_values'],
            result2['shap_values'],
            decimal=5,
            err_msg="SHAP values are not consistent across multiple explanations"
        )

        assert result1['expected_value'] == result2['expected_value'], \
            "Expected values are not consistent"

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=20, max_value=50),
        n_features=st.integers(min_value=3, max_value=8),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_shap_global_feature_importance(self, n_samples, n_features, random_state):
        """Test that global SHAP feature importance is properly calculated."""
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

        # Initialize SHAP explainer
        explainer = SHAPExplainer(model, background_data=X_df[:10])

        # Explain multiple instances
        result = explainer.explain(X_df[:15])

        # Get global summary
        summary = explainer.get_global_summary()

        # Check that all features have importance values
        assert len(summary['feature_importance']) == n_features, \
            f"Expected {n_features} features in importance dict, got {len(summary['feature_importance'])}"

        # Check that feature ranking is properly sorted
        ranking_values = [imp for _, imp in summary['feature_ranking']]
        assert all(ranking_values[i] >= ranking_values[i+1]
                  for i in range(len(ranking_values)-1)), \
            "Feature ranking is not sorted by importance"

        # Check that all feature names are present
        for feature_name in feature_names:
            assert feature_name in summary['feature_importance'], \
                f"Feature {feature_name} missing from importance dictionary"

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=20, max_value=50),
        n_features=st.integers(min_value=3, max_value=8),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_shap_explain_completeness(self, n_samples, n_features, random_state):
        """Test that SHAP explanation contains all required components."""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features//2,
            n_redundant=0,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)

        # Train a model
        model = RandomForestClassifier(n_estimators=10, random_state=random_state)
        model.fit(X_df, y)

        # Initialize SHAP explainer
        explainer = SHAPExplainer(model, background_data=X_df[:5])

        # Test batch explanation
        batch_result = explainer.explain(X_df[:10])

        assert 'shap_values' in batch_result, "SHAP values missing from batch explanation"
        assert 'expected_value' in batch_result, "Expected value missing from batch explanation"
        assert 'feature_names' in batch_result, "Feature names missing from batch explanation"
        assert len(batch_result['feature_names']) == n_features, \
            f"Expected {n_features} feature names, got {len(batch_result['feature_names'])}"

        # Test single instance explanation
        instance_result = explainer.explain_instance(X_df.iloc[0])

        assert 'shap_values' in instance_result, "SHAP values missing from instance explanation"
        assert 'expected_value' in instance_result, "Expected value missing from instance explanation"
        assert 'feature_names' in instance_result, "Feature names missing from instance explanation"
        assert 'prediction' in instance_result, "Prediction missing from instance explanation"
        assert 'features' in instance_result, "Feature values missing from instance explanation"

        # Check SHAP values shape
        assert len(instance_result['shap_values']) == n_features, \
            f"Expected {n_features} SHAP values, got {len(instance_result['shap_values'])}"

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=10, max_value=30),
        n_features=st.integers(min_value=3, max_value=6),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_shap_visualization_generation(self, n_samples, n_features, random_state):
        """Test that SHAP visualizations are properly generated."""
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

        # Initialize SHAP explainer
        explainer = SHAPExplainer(model, background_data=X_df[:5])

        # Test waterfall plot
        waterfall_result = explainer.create_waterfall_plot(X_df.iloc[0])

        assert 'plot_base64' in waterfall_result, "Waterfall plot base64 missing"
        assert 'explanation' in waterfall_result, "Waterfall explanation missing"
        assert waterfall_result['plot_type'] == 'waterfall', "Incorrect plot type for waterfall"
        assert isinstance(waterfall_result['plot_base64'], str), "Plot base64 should be string"
        assert len(waterfall_result['plot_base64']) > 0, "Plot base64 should not be empty"

        # Test summary plot
        summary_result = explainer.create_summary_plot(X_df[:10])

        assert 'plot_base64' in summary_result, "Summary plot base64 missing"
        assert summary_result['plot_type'] == 'summary', "Incorrect plot type for summary"
        assert summary_result['n_features'] == n_features, "Incorrect feature count in summary"
        assert summary_result['n_samples'] == 10, "Incorrect sample count in summary"

        # Test feature importance plot
        importance_result = explainer.create_feature_importance_plot()

        assert 'plot_base64' in importance_result, "Feature importance plot base64 missing"
        assert 'feature_importance' in importance_result, "Feature importance dictionary missing"
        assert importance_result['plot_type'] == 'feature_importance', \
            "Incorrect plot type for feature importance"
        assert importance_result['n_features_shown'] <= n_features, \
            "Too many features shown in importance plot"

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=10, max_value=30),
        n_features=st.integers(min_value=3, max_value=6),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_shap_model_type_handling(self, n_samples, n_features, random_state):
        """Test that SHAP handles different model types correctly."""
        # Generate synthetic data
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=0.1,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)

        # Test with RandomForest
        rf_model = RandomForestRegressor(n_estimators=10, random_state=random_state)
        rf_model.fit(X_df, y)
        rf_explainer = SHAPExplainer(rf_model, background_data=X_df[:5])
        rf_result = rf_explainer.explain(X_df[:5])
        assert rf_result['shap_values'] is not None, "RandomForest SHAP values missing"

        # Test with LinearRegression
        lr_model = LinearRegression()
        lr_model.fit(X_df, y)
        lr_explainer = SHAPExplainer(lr_model, background_data=X_df[:5])
        lr_result = lr_explainer.explain(X_df[:5])
        assert lr_result['shap_values'] is not None, "LinearRegression SHAP values missing"

        # Both should have SHAP values for the same number of features
        assert len(rf_result['feature_names']) == len(lr_result['feature_names']) == n_features, \
            "Feature name mismatch between model types"

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=10, max_value=30),
        n_features=st.integers(min_value=3, max_value=6),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_shap_data_format_handling(self, n_samples, n_features, random_state):
        """Test that SHAP handles different data formats correctly."""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)

        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=random_state)
        model.fit(X_df, y)

        # Initialize explainer
        explainer = SHAPExplainer(model, background_data=X_df[:5])

        # Test with DataFrame
        df_result = explainer.explain(X_df[:5])
        assert df_result['feature_names'] == feature_names, "DataFrame feature names incorrect"

        # Test with numpy array
        np_result = explainer.explain(X[:5])
        assert np_result['feature_names'] == feature_names, "Numpy array feature names incorrect"

        # Test instance explanation with Series
        series_result = explainer.explain_instance(X_df.iloc[0])
        assert series_result['feature_names'] == feature_names, "Series feature names incorrect"

        # Test instance explanation with numpy array
        np_instance_result = explainer.explain_instance(X[0])
        assert len(np_instance_result['feature_names']) == n_features, \
            "Numpy instance feature names incorrect"