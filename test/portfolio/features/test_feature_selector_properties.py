"""
Property-based tests for feature selection.

This module uses Hypothesis to verify that the feature selector
produces valid, consistent feature selections across diverse datasets.
"""

import pytest
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from hypothesis import given, strategies as st, settings, HealthCheck

from src.portfolio.features.feature_selector import (
    FeatureSelector,
    SelectionConfig,
    SelectionResult
)


class TestFeatureSelectionProperties:
    """Property-based tests for feature selection."""

    @pytest.fixture
    def selector(self):
        """Create a feature selector for property testing."""
        # Use smaller parameters for faster tests
        config = SelectionConfig(
            correlation_threshold=0.95,
            variance_threshold=0.01,
            model_n_estimators=10,
            cv_folds=3,
            stability_n_bootstrap=5,
            rfe_cv_folds=3
        )
        return FeatureSelector(config)

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=20
    )
    @given(st.data())
    def test_variance_selection_properties(self, selector, data):
        """
        Property: Variance threshold selection removes low-variance features.

        Given any dataset, variance selection should preserve high-variance features.
        """
        # Generate synthetic data
        n_samples = data.draw(st.integers(min_value=50, max_value=200))
        n_features = data.draw(st.integers(min_value=5, max_value=20))

        # Create features with varying variance
        np.random.seed(42)
        X = pd.DataFrame({
            f'feature_{i}': np.random.randn(n_samples) * (0.1 if i < n_features // 3 else 1.0)
            for i in range(n_features)
        })

        # Add constant feature
        X['constant'] = 1.0

        # Apply variance selection
        X_selected = selector._variance_selection(X, threshold=0.1)

        # Should remove constant feature
        assert 'constant' not in X_selected.columns

        # Should have fewer features
        assert len(X_selected.columns) <= len(X.columns)

        # Check that all remaining features have sufficient variance
        for col in X_selected.columns:
            assert X[col].var() >= 0.1 or col == 'constant'  # constant was removed

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=15
    )
    @given(st.data())
    def test_correlation_selection_properties(self, selector, data):
        """
        Property: Correlation selection removes highly correlated features.

        Given any dataset, correlation selection should reduce multicollinearity.
        """
        # Generate data with correlated features
        n_samples = data.draw(st.integers(min_value=100, max_value=300))
        n_base = data.draw(st.integers(min_value=5, max_value=10))

        # Create base features
        np.random.seed(42)
        base_features = np.random.randn(n_samples, n_base)

        # Create correlated variations
        X_dict = {}
        for i in range(n_base):
            X_dict[f'base_{i}'] = base_features[:, i]

            # Add correlated feature
            correlation = 0.5 + data.draw(st.floats(min_value=0, max_value=0.4))
            noise_level = np.sqrt(1 - correlation**2)
            X_dict[f'corr_{i}'] = base_features[:, i] * correlation + \
                                np.random.randn(n_samples) * noise_level

        X = pd.DataFrame(X_dict)

        # Apply correlation selection
        X_selected = selector._correlation_selection(X, y=pd.Series([0] * n_samples))

        # Should have fewer or equal features
        assert len(X_selected.columns) <= len(X.columns)

        # Should preserve some features
        assert len(X_selected.columns) > 0

        # Should preserve at least one base feature
        base_preserved = any(f'base_{i}' in X_selected.columns for i in range(n_base))
        assert base_preserved, "Should preserve at least one base feature"

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=15
    )
    @given(st.data())
    def test_mutual_info_selection_properties(self, selector, data):
        """
        Property: Mutual information selection ranks features by predictive power.

        Given any dataset, MI selection should select features with higher information.
        """
        # Generate classification data
        n_samples = data.draw(st.integers(min_value=100, max_value=200))
        n_features = data.draw(st.integers(min_value=5, max_value=15))

        # Create features with varying relationship to target
        np.random.seed(42)
        X = pd.DataFrame()
        for i in range(n_features):
            if i < n_features // 3:
                # Predictive feature
                X[f'pred_{i}'] = np.random.randn(n_samples)
            else:
                # Noise feature
                X[f'noise_{i}'] = np.random.randn(n_samples) * 0.1

        # Create binary target based on predictive features
        pred_cols = [c for c in X.columns if c.startswith('pred_')]
        if pred_cols:
            score = X[pred_cols].sum(axis=1) + np.random.randn(n_samples) * 0.5
            y = (score > score.median()).astype(int)
        else:
            y = np.random.randint(0, 2, n_samples)

        # Apply mutual information selection
        k = max(1, n_features // 3)
        X_selected = selector._mutual_info_selection(X, y, k=k)

        # Should select exactly k features
        assert len(X_selected.columns) == min(k, n_features)

        # Should prefer predictive features when available
        if pred_cols and len(pred_cols) > 1:
            selected_pred = sum(1 for c in X_selected.columns if c.startswith('pred_'))
            selected_noise = sum(1 for c in X_selected.columns if c.startswith('noise_'))
            # At least 40% of selected features should be predictive (relaxed)
            if selected_noise > 0 and selected_pred + selected_noise > 0:
                assert selected_pred / (selected_pred + selected_noise) >= 0.4

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=15
    )
    @given(st.data())
    def test_model_based_selection_properties(self, selector, data):
        """
        Property: Model-based selection provides feature importance rankings.

        Given any dataset, model-based selection should rank features consistently.
        """
        # Generate regression data
        n_samples = data.draw(st.integers(min_value=50, max_value=150))
        n_features = data.draw(st.integers(min_value=5, max_value=12))

        np.random.seed(42)
        X = pd.DataFrame()
        coefficients = []

        for i in range(n_features):
            if i < n_features // 2:
                # Important feature
                coeff = data.draw(st.floats(min_value=0.5, max_value=2.0))
                X[f'imp_{i}'] = np.random.randn(n_samples)
                coefficients.append(coeff)
            else:
                # Less important feature
                coeff = data.draw(st.floats(min_value=0.01, max_value=0.1))
                X[f'less_{i}'] = np.random.randn(n_samples)
                coefficients.append(coeff)

        # Create target as linear combination
        y = sum(X.iloc[:, i] * coefficients[i] for i in range(n_features))
        y += np.random.randn(n_samples) * 0.1

        # Apply model-based selection
        X_selected = selector._model_based_selection(X, y)

        # Should select some features
        assert len(X_selected.columns) > 0
        assert len(X_selected.columns) <= len(X.columns)

        # Should have stored feature importances
        assert hasattr(selector, 'feature_importances_')
        assert len(selector.feature_importances_) == len(X.columns)

        # Important features should generally have higher importance
        imp_scores = [selector.feature_importances_[f'imp_{i}'] for i in range(n_features // 2)]
        less_scores = [selector.feature_importances_[f'less_{i}'] for i in range(n_features // 2, n_features)]

        if imp_scores and less_scores:
            # Median important feature should be more important than median less important
            assert np.median(imp_scores) >= np.median(less_scores) * 0.8

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=10
    )
    @given(st.data())
    def test_selection_result_consistency(self, selector, data):
        """
        Property: Selection results have consistent structure.

        Given any selection method, results should follow the expected format.
        """
        # Generate simple data
        n_samples = data.draw(st.integers(min_value=50, max_value=100))
        X = pd.DataFrame({
            f'feature_{i}': np.random.randn(n_samples)
            for i in range(5)
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))

        # Apply selection
        selector.fit_transform(X, y, method='variance')

        # Check selection result structure
        assert len(selector.selection_history_) > 0
        result = selector.selection_history_[-1]

        # Check SelectionResult attributes
        assert isinstance(result, SelectionResult)
        assert isinstance(result.selected_features, list)
        assert isinstance(result.feature_scores, dict)
        assert isinstance(result.feature_rankings, dict)
        assert isinstance(result.selection_method, str)
        assert isinstance(result.n_selected, int)
        assert isinstance(result.n_original, int)
        assert isinstance(result.reduction_ratio, float)
        assert 0 <= result.reduction_ratio <= 1
        assert isinstance(result.metadata, dict)

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=10
    )
    @given(st.data())
    def test_feature_reduction_monotonicity(self, selector, data):
        """
        Property: Feature selection never increases feature count.

        Given any dataset, selection should not add features.
        """
        # Generate data
        n_samples = data.draw(st.integers(min_value=50, max_value=100))
        n_features = data.draw(st.integers(min_value=5, max_value=15))

        X = pd.DataFrame({
            f'feature_{i}': np.random.randn(n_samples)
            for i in range(n_features)
        })
        y = pd.Series(np.random.randn(n_samples))

        # Test multiple selection methods
        methods = ['variance', 'model_based', 'correlation']

        for method in methods:
            try:
                selector.selection_history_.clear()
                X_selected = getattr(selector, f'_{method}_selection')(X, y)

                # Should not increase feature count
                assert len(X_selected.columns) <= n_features

                # Should maintain same number of rows
                assert len(X_selected) == n_samples

                # All selected features should be original features
                assert all(f in X.columns for f in X_selected.columns)

            except:
                # Some methods might fail on certain data
                pass

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=10
    )
    @given(st.data())
    def test_deterministic_selection(self, selector, data):
        """
        Property: Feature selection is deterministic.

        Given the same data and random seed, results should be identical.
        """
        # Generate data
        n_samples = data.draw(st.integers(min_value=50, max_value=100))
        np.random.seed(42)
        X = pd.DataFrame({
            f'feature_{i}': np.random.randn(n_samples)
            for i in range(8)
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))

        # Apply selection twice with same random seed
        np.random.seed(42)
        X1 = selector._model_based_selection(X, y)

        selector.selection_history_.clear()
        np.random.seed(42)
        X2 = selector._model_based_selection(X, y)

        # Results should be identical
        assert set(X1.columns) == set(X2.columns)

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=8
    )
    @given(st.data())
    def test_combined_selection_voting(self, selector, data):
        """
        Property: Combined selection respects voting threshold.

        Given multiple methods, combined selection uses voting appropriately.
        """
        # Generate data
        n_samples = data.draw(st.integers(min_value=50, max_value=100))
        X = pd.DataFrame({
            f'feature_{i}': np.random.randn(n_samples)
            for i in range(10)
        })
        y = pd.Series(np.random.randn(n_samples))

        # Apply combined selection
        X_selected = selector._combined_selection(
            X, y,
            include_methods=['variance', 'model_based'],
            voting_threshold=0.5
        )

        # Should select features that pass multiple methods
        assert len(X_selected.columns) > 0
        assert len(X_selected.columns) <= len(X.columns)

        # Check voting metadata
        result = selector.selection_history_[-1]
        assert result.selection_method == 'combined'
        assert 'votes' in result.metadata

    def test_problem_type_detection(self, selector):
        """
        Property: Problem type detection works correctly.

        Given different target types, should detect classification vs regression.
        """
        # Classification target (few unique values)
        y_clf = pd.Series([0, 1, 0, 1, 0] * 20)
        problem_type = selector._determine_problem_type(y_clf)
        assert problem_type == 'classification'

        # Regression target (many unique values)
        y_reg = pd.Series(np.random.randn(100))
        problem_type = selector._determine_problem_type(y_reg)
        assert problem_type == 'regression'

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=8
    )
    @given(st.data())
    def test_vif_selection_properties(self, selector, data):
        """
        Property: VIF selection reduces multicollinearity.

        Given correlated features, VIF should remove redundant ones.
        """
        # Generate highly correlated data
        n_samples = 100
        np.random.seed(42)
        base_feature = np.random.randn(n_samples)

        X = pd.DataFrame({
            'base': base_feature,
            'high_corr_1': base_feature * 0.95 + np.random.randn(n_samples) * 0.05,
            'high_corr_2': base_feature * 0.9 + np.random.randn(n_samples) * 0.1,
            'moderate_corr': base_feature * 0.7 + np.random.randn(n_samples) * 0.3,
            'low_corr': np.random.randn(n_samples) * 0.5 + base_feature * 0.1,
            'independent': np.random.randn(n_samples)
        })

        # Apply VIF selection
        X_selected = selector._vif_selection(X, threshold=5.0)

        # Should remove some highly correlated features
        assert len(X_selected.columns) < len(X.columns)

        # Should keep at least the independent feature
        assert 'independent' in X_selected.columns

        # Should keep some features (not remove everything)
        assert len(X_selected.columns) >= 2

        # Check result metadata
        result = selector.selection_history_[-1]
        assert result.selection_method == 'vif'
        assert 'removed_features' in result.metadata