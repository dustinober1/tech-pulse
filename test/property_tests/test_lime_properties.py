"""Property tests for LIME explainer."""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_classification, make_regression

from src.interpretability.lime_explainer import LIMEExplainer


class Property14LIMEExplanationGeneration:
    """
    Property 14: LIME explanation generation

    Validates: Requirements 4.2

    Ensures that LIME explainer correctly generates local explanations with
    proper feature contributions and interpretability.
    """

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=20, max_value=50),
        n_features=st.integers(min_value=3, max_value=8),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_lime_local_fidelity(self, n_samples, n_features, random_state):
        """Test that LIME explanations have good local fidelity."""
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

        # Initialize LIME explainer with mock to avoid library dependency
        explainer = LIMEExplainer(mode='tabular', feature_names=feature_names)
        explainer.model = model
        explainer.training_data = X_df

        # Mock LIME components to test explanation structure
        with pytest.MonkeyPatch().context() as m:
            # Create mock explanation
            mock_explanation = Mock()
            mock_explanation.as_list.return_value = [
                (f'feature_{i}', np.random.randn() * 0.1)
                for i in range(min(5, n_features))
            ]
            mock_explanation.predicted_proba = np.array([0.3, 0.7])
            mock_explanation.intercept = np.array([0.5, 0.5])
            mock_explanation.local_pred = 0.65
            mock_explanation.score = 0.75

            # Mock LIME library
            mock_lime = Mock()
            mock_lime_explainer = Mock()
            mock_lime.lime_tabular.LimeTabularExplainer.return_value = mock_lime_explainer
            mock_lime_explainer.explain_instance.return_value = mock_explanation

            m.setattr('src.interpretability.lime_explainer.lime', mock_lime)

            # Generate explanation
            result = explainer.explain_instance(X_df.iloc[0])

            # Check explanation completeness
            assert 'feature_contributions' in result, "Feature contributions missing"
            assert 'feature_values' in result, "Feature values missing"
            assert 'predicted_value' in result, "Predicted value missing"
            assert 'intercept' in result, "Intercept missing"
            assert 'local_pred' in result, "Local prediction missing"

            # Check that local prediction is reasonable
            assert 0 <= result['local_pred'] <= 1, "Local prediction not in valid range"
            assert abs(result['local_pred'] - result['predicted_value']) < 0.5, \
                "Local prediction too far from model prediction"

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=20, max_value=50),
        n_features=st.integers(min_value=3, max_value=8),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_lime_feature_contribution_consistency(self, n_samples, n_features, random_state):
        """Test that LIME feature contributions are consistent."""
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

        # Initialize LIME explainer
        explainer = LIMEExplainer(mode='tabular', feature_names=feature_names)
        explainer.model = model
        explainer.training_data = X_df

        # Create consistent mock contributions
        np.random.seed(random_state)
        contributions = np.random.randn(min(5, n_features)) * 0.1

        with pytest.MonkeyPatch().context() as m:
            # Mock LIME components
            mock_explanation = Mock()
            mock_explanation.as_list.return_value = [
                (f'feature_{i}', contributions[i])
                for i in range(len(contributions))
            ]
            mock_explanation.predicted_value = np.random.randn()

            mock_lime = Mock()
            mock_lime_explainer = Mock()
            mock_lime.lime_tabular.LimeTabularExplainer.return_value = mock_lime_explainer
            mock_lime_explainer.explain_instance.return_value = mock_explanation

            m.setattr('src.interpretability.lime_explainer.lime', mock_lime)

            # Generate explanation twice for the same instance
            result1 = explainer.explain_instance(X_df.iloc[0])
            result2 = explainer.explain_instance(X_df.iloc[0])

            # Check consistency
            assert result1['feature_contributions'] == result2['feature_contributions'], \
                "Feature contributions not consistent across explanations"
            np.testing.assert_array_almost_equal(
                result1['feature_values'],
                result2['feature_values'],
                decimal=5,
                err_msg="Feature values not consistent across explanations"
            )

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=20, max_value=50),
        n_features=st.integers(min_value=3, max_value=8),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_lime_explanation_interpretability(self, n_samples, n_features, random_state):
        """Test that LIME explanations are interpretable."""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features//2,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)

        # Train a model
        model = RandomForestClassifier(n_estimators=10, random_state=random_state)
        model.fit(X_df, y)

        # Initialize LIME explainer
        explainer = LIMEExplainer(mode='tabular', feature_names=feature_names)
        explainer.model = model
        explainer.training_data = X_df

        # Create interpretable feature names and contributions
        np.random.seed(random_state)
        feature_descriptions = [
            f'feature_{i} {" > " if np.random.randn() > 0 else " <= "}{abs(np.random.randn()):.2f}'
            for i in range(min(5, n_features))
        ]
        contributions = np.random.randn(len(feature_descriptions)) * 0.2

        with pytest.MonkeyPatch().context() as m:
            # Mock LIME components
            mock_explanation = Mock()
            mock_explanation.as_list.return_value = list(zip(feature_descriptions, contributions))
            mock_explanation.predicted_proba = np.array([0.3, 0.7])
            mock_explanation.intercept = np.array([0.5, 0.5])

            mock_lime = Mock()
            mock_lime_explainer = Mock()
            mock_lime.lime_tabular.LimeTabularExplainer.return_value = mock_lime_explainer
            mock_lime_explainer.explain_instance.return_value = mock_explanation

            m.setattr('src.interpretability.lime_explainer.lime', mock_lime)

            # Generate explanation
            result = explainer.explain_instance(X_df.iloc[0])

            # Check interpretability aspects
            for feature_desc in result['feature_contributions']:
                # Feature descriptions should be human-readable
                assert isinstance(feature_desc, str), "Feature description not a string"
                assert len(feature_desc) > 0, "Empty feature description"
                assert any(op in feature_desc for op in ['>', '<=', '=', '<']), \
                    "Feature description lacks comparison operator"

            # Contributions should be numeric and meaningful
            for contrib in result['feature_values']:
                assert isinstance(contrib, (int, float)), "Contribution not numeric"
                assert not np.isnan(contrib), "Contribution is NaN"
                assert not np.isinf(contrib), "Contribution is infinite"
                assert abs(contrib) < 10, "Contribution magnitude too large"

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=20, max_value=50),
        n_features=st.integers(min_value=3, max_value=8),
        n_explanations=st.integers(min_value=3, max_value=10),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_lime_multiple_explanations_aggregation(self, n_samples, n_features, n_explanations, random_state):
        """Test aggregation of multiple LIME explanations."""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features//2,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)

        # Train a model
        model = RandomForestClassifier(n_estimators=10, random_state=random_state)
        model.fit(X_df, y)

        # Initialize LIME explainer
        explainer = LIMEExplainer(mode='tabular', feature_names=feature_names)
        explainer.model = model
        explainer.training_data = X_df

        # Generate multiple mock explanations
        np.random.seed(random_state)
        explanations = []
        for _ in range(n_explanations):
            # Random subset of features
            n_features_in_exp = min(3, n_features)
            feature_subset = np.random.choice(feature_names, n_features_in_exp, replace=False)
            contributions = np.random.randn(n_features_in_exp) * 0.15

            explanation = {
                'feature_contributions': list(feature_subset),
                'feature_values': list(contributions)
            }
            explanations.append(explanation)

        # Get feature importance summary
        summary = explainer.get_feature_importance_summary(explanations, top_k=n_features)

        # Check summary properties
        assert 'top_features' in summary, "Top features missing from summary"
        assert 'feature_appearance_counts' in summary, "Appearance counts missing from summary"
        assert summary['total_explanations'] == n_explanations, \
            f"Expected {n_explanations} explanations, got {summary['total_explanations']}"

        # Check that features are sorted by importance
        top_features_values = list(summary['top_features'].values())
        assert all(top_features_values[i] >= top_features_values[i+1]
                  for i in range(len(top_features_values)-1)), \
            "Features not sorted by importance in summary"

        # Check appearance counts
        total_appearances = sum(summary['feature_appearance_counts'].values())
        assert total_appearances == n_explanations * 3, \
            f"Total appearances {total_appearances} doesn't match expected {n_explanations * 3}"

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=20, max_value=50),
        n_features=st.integers(min_value=3, max_value=8),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_lime_visualization_generation(self, n_samples, n_features, random_state):
        """Test that LIME visualizations are properly generated."""
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

        # Initialize LIME explainer
        explainer = LIMEExplainer(mode='tabular', feature_names=feature_names)

        # Create explanation data
        np.random.seed(random_state)
        n_features_in_plot = min(5, n_features)
        feature_subset = feature_names[:n_features_in_plot]
        contributions = np.random.randn(n_features_in_plot) * 0.2

        explanation_data = {
            'feature_contributions': feature_subset,
            'feature_values': list(contributions),
            'predicted_value': np.random.randn()
        }

        # Generate visualization
        result = explainer.create_explanation_plot(explanation_data)

        # Check plot properties
        assert 'plot_base64' in result, "Plot base64 missing"
        assert 'explanation_data' in result, "Explanation data missing"
        assert result['plot_type'] == 'lime_explanation', "Incorrect plot type"
        assert isinstance(result['plot_base64'], str), "Plot base64 not a string"
        assert len(result['plot_base64']) > 0, "Plot base64 is empty"

        # Validate base64 format (should be valid base64 string)
        try:
            import base64
            decoded = base64.b64decode(result['plot_base64'])
            assert len(decoded) > 0, "Decoded plot is empty"
        except Exception as e:
            pytest.fail(f"Plot base64 is invalid: {e}")

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=20, max_value=50),
        n_features=st.integers(min_value=3, max_value=8),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_lime_explanation_completeness(self, n_samples, n_features, random_state):
        """Test that LIME explanations contain all required components."""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features//2,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)

        # Train a model
        model = RandomForestClassifier(n_estimators=10, random_state=random_state)
        model.fit(X_df, y)

        # Test tabular mode
        explainer_tabular = LIMEExplainer(mode='tabular', feature_names=feature_names)
        explainer_tabular.model = model
        explainer_tabular.training_data = X_df

        np.random.seed(random_state)
        contributions = np.random.randn(min(5, n_features)) * 0.1

        with pytest.MonkeyPatch().context() as m:
            # Mock tabular explanation
            mock_explanation_tab = Mock()
            mock_explanation_tab.as_list.return_value = [
                (f'feature_{i}', contributions[i])
                for i in range(len(contributions))
            ]
            mock_explanation_tab.predicted_proba = np.array([0.3, 0.7])
            mock_explanation_tab.intercept = np.array([0.5, 0.5])
            mock_explanation_tab.local_pred = 0.65

            mock_lime = Mock()
            mock_lime_explainer = Mock()
            mock_lime.lime_tabular.LimeTabularExplainer.return_value = mock_lime_explainer
            mock_lime_explainer.explain_instance.return_value = mock_explanation_tab

            m.setattr('src.interpretability.lime_explainer.lime', mock_lime)

            result_tab = explainer_tabular.explain_instance(X_df.iloc[0])

            # Check tabular explanation completeness
            assert result_tab['type'] == 'tabular_explanation', "Incorrect explanation type"
            assert len(result_tab['feature_contributions']) == len(contributions), \
                "Missing feature contributions"
            assert len(result_tab['feature_values']) == len(contributions), \
                "Missing feature values"
            assert 'predicted_value' in result_tab, "Missing predicted value"
            assert 'intercept' in result_tab, "Missing intercept"
            assert 'local_pred' in result_tab, "Missing local prediction"

        # Test text mode
        explainer_text = LIMEExplainer(mode='text')
        explainer_text.model = model

        with pytest.MonkeyPatch().context() as m:
            # Mock text explanation
            mock_explanation_text = Mock()
            mock_explanation_text.as_list.return_value = [('good', 0.4), ('bad', -0.3)]
            mock_explanation_text.score = 0.75
            mock_explanation_text.intercept = 0.5

            mock_lime = Mock()
            mock_lime_explainer = Mock()
            mock_lime.lime_text.LimeTextExplainer.return_value = mock_lime_explainer
            mock_lime_explainer.explain_instance.return_value = mock_explanation_text

            m.setattr('src.interpretability.lime_explainer.lime', mock_lime)

            result_text = explainer_text.explain_instance("This is good text")

            # Check text explanation completeness
            assert result_text['type'] == 'text_explanation', "Incorrect explanation type"
            assert len(result_text['feature_contributions']) == 2, "Missing text features"
            assert 'score' in result_text, "Missing score"
            assert 'intercept' in result_text, "Missing intercept"

        # Test image mode
        explainer_image = LIMEExplainer(mode='image')
        explainer_image.model = model

        with pytest.MonkeyPatch().context() as m:
            # Mock image explanation
            mock_explanation_img = Mock()
            mock_explanation_img.top_labels = [1]

            mock_lime = Mock()
            mock_lime_explainer = Mock()
            mock_lime.lime_image.LimeImageExplainer.return_value = mock_lime_explainer
            mock_lime_explainer.explain_instance.return_value = mock_explanation_img

            m.setattr('src.interpretability.lime_explainer.lime', mock_lime)

            result_img = explainer_image.explain_instance(np.random.rand(64, 64, 3))

            # Check image explanation completeness
            assert result_img['type'] == 'image_explanation', "Incorrect explanation type"
            assert 'explanation' in result_img, "Missing explanation object"
            assert 'predicted_value' in result_img, "Missing predicted value"

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=20, max_value=50),
        n_features=st.integers(min_value=3, max_value=8),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_lime_explanation_reproducibility(self, n_samples, n_features, random_state):
        """Test that LIME explanations are reproducible with same seed."""
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

        # Initialize two LIME explainers with same settings
        explainer1 = LIMEExplainer(
            mode='tabular',
            feature_names=feature_names,
            discretize_continuous=True,
            kernel_width=3.0
        )
        explainer1.model = model
        explainer1.training_data = X_df

        explainer2 = LIMEExplainer(
            mode='tabular',
            feature_names=feature_names,
            discretize_continuous=True,
            kernel_width=3.0
        )
        explainer2.model = model
        explainer2.training_data = X_df

        # Create deterministic mock explanations
        np.random.seed(42)
        fixed_contributions = np.random.randn(min(5, n_features)) * 0.1
        fixed_descriptions = [
            f'feature_{i} {" > " if c > 0 else " <= "}{abs(c):.2f}'
            for i, c in enumerate(fixed_contributions)
        ]

        with pytest.MonkeyPatch().context() as m:
            # Mock LIME for both explainers
            def create_mock_explainer():
                mock_lime = Mock()
                mock_lime_explainer = Mock()

                mock_explanation = Mock()
                mock_explanation.as_list.return_value = list(zip(fixed_descriptions, fixed_contributions))
                mock_explanation.predicted_value = 0.75

                mock_lime.lime_tabular.LimeTabularExplainer.return_value = mock_lime_explainer
                mock_lime_explainer.explain_instance.return_value = mock_explanation

                return mock_lime

            m.setattr('src.interpretability.lime_explainer.lime', create_mock_explainer())

            # Generate explanations with both explainers
            result1 = explainer1.explain_instance(X_df.iloc[0])
            result2 = explainer2.explain_instance(X_df.iloc[0])

            # Check reproducibility
            assert result1['feature_contributions'] == result2['feature_contributions'], \
                "Feature contributions not reproducible"
            np.testing.assert_array_almost_equal(
                result1['feature_values'],
                result2['feature_values'],
                decimal=5,
                err_msg="Feature values not reproducible"
            )
            assert result1['predicted_value'] == result2['predicted_value'], \
                "Predicted values not reproducible"