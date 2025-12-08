"""Property tests for interpretability report generator."""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st
from unittest.mock import Mock, patch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_classification, make_regression
import tempfile
import os

from src.interpretability.interpretability_report import InterpretabilityReport


class Property16GlobalInterpretability:
    """
    Property 16: Global interpretability for complex models

    Validates: Requirements 4.4

    Ensures that interpretability report generator correctly provides
    comprehensive global interpretability summaries for complex models.
    """

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=100, max_value=500),
        n_features=st.integers(min_value=5, max_value=20),
        n_informative=st.integers(min_value=3, max_value=10),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_global_interpretability_completeness(self, n_samples, n_features, n_informative, random_state):
        """Test that global interpretability analysis provides complete information."""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=0,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y)

        # Train a model
        model = RandomForestClassifier(n_estimators=20, random_state=random_state)
        model.fit(X_df, y_series)

        # Initialize report
        report = InterpretabilityReport(
            model=model,
            X=X_df,
            y=y_series,
            model_name="TestClassifier",
            description="Test classification model"
        )

        # Mock explainers with comprehensive data
        mock_shap = Mock()
        mock_shap.explain.return_value = {
            'values': np.random.randn(min(50, n_samples), n_features)
        }

        # Create meaningful feature importance
        feature_importance = {}
        feature_ranking = []
        mean_shap_values = {}
        std_shap_values = {}

        for i in range(n_features):
            importance = (i + 1) * 0.1 * (1 if i < n_informative else 0.1)
            feature_importance[f'feature_{i}'] = importance
            feature_ranking.append((f'feature_{i}', importance))
            mean_shap_values[f'feature_{i}'] = np.random.randn() * 0.1
            std_shap_values[f'feature_{i}'] = abs(np.random.randn()) * 0.05

        mock_shap.get_global_summary.return_value = {
            'feature_importance': feature_importance,
            'feature_ranking': sorted(feature_ranking, key=lambda x: x[1], reverse=True),
            'mean_shap_values': mean_shap_values,
            'std_shap_values': std_shap_values
        }

        mock_pdp = Mock()
        pdp_importance = {}
        pdp_ranking = []

        for i in range(n_features):
            pdp_imp = (n_features - i) * 0.05  # Reverse order for variety
            pdp_importance[f'feature_{i}'] = pdp_imp
            pdp_ranking.append((f'feature_{i}', pdp_imp))

        mock_pdp.get_feature_importance_from_pdp.return_value = {
            'feature_importance': pdp_importance,
            'feature_ranking': sorted(pdp_ranking, key=lambda x: x[1], reverse=True)
        }

        # Mock interactions
        mock_pdp.generate_interaction_summary.return_value = {
            'interaction_strengths': {
                f'feature_{i} x feature_{j}': np.random.rand() * 0.1
                for i in range(min(3, n_features))
                for j in range(i+1, min(3, n_features))
            },
            'interaction_details': {},
            'top_interactions': [
                (f'feature_{i} x feature_{j}', np.random.rand() * 0.1)
                for i in range(min(2, n_features))
                for j in range(i+1, min(2, n_features))
            ],
            'n_interactions_analyzed': min(5, n_features * (n_features - 1) // 2)
        }

        report.shap_explainer = mock_shap
        report.pdp_plotter = mock_pdp

        # Analyze global interpretability
        report.analyze_global_interpretability(n_samples=min(50, n_samples), top_k_features=min(10, n_features))

        # Verify completeness of global interpretability
        global_interp = report.report_data['global_interpretability']

        # Check SHAP analysis
        assert 'shap' in global_interp, "SHAP analysis missing from global interpretability"
        shap_data = global_interp['shap']
        assert 'feature_importance' in shap_data, "SHAP feature importance missing"
        assert 'feature_ranking' in shap_data, "SHAP feature ranking missing"
        assert 'mean_shap_values' in shap_data, "SHAP mean values missing"
        assert 'std_shap_values' in shap_data, "SHAP std values missing"

        # Check PDP analysis
        assert 'pdp_importance' in global_interp, "PDP importance analysis missing"
        pdp_data = global_interp['pdp_importance']
        assert 'feature_importance' in pdp_data, "PDP feature importance missing"
        assert 'feature_ranking' in pdp_data, "PDP feature ranking missing"

        # Check interaction analysis
        assert 'feature_interactions' in global_interp, "Feature interactions analysis missing"
        interactions = global_interp['feature_interactions']
        assert 'interaction_strengths' in interactions, "Interaction strengths missing"
        assert 'top_interactions' in interactions, "Top interactions missing"
        assert 'n_interactions_analyzed' in interactions, "Number of interactions analyzed missing"

        # Check target statistics
        assert 'target_stats' in global_interp, "Target statistics missing"
        target_stats = global_interp['target_stats']
        required_stats = ['mean', 'std', 'min', 'max', 'unique_values']
        for stat in required_stats:
            assert stat in target_stats, f"Target statistic {stat} missing"

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=100, max_value=500),
        n_features=st.integers(min_value=5, max_value=20),
        n_informative=st.integers(min_value=3, max_value=10),
        top_k=st.integers(min_value=3, max_value=10),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_feature_importance_consistency(self, n_samples, n_features, n_informative, top_k, random_state):
        """Test that feature importance rankings are consistent and meaningful."""
        # Generate synthetic data
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=0.1,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y)

        # Train a model
        model = RandomForestRegressor(n_estimators=20, random_state=random_state)
        model.fit(X_df, y_series)

        # Initialize report
        report = InterpretabilityReport(model=model, X=X_df, y=y_series)

        # Create consistent feature importance across methods
        base_importance = np.random.rand(n_features)
        base_importance[0:n_informative] *= 3  # Make informative features more important

        # Mock SHAP with consistent importance
        mock_shap = Mock()
        shap_importance = {f'feature_{i}': base_importance[i] for i in range(n_features)}
        shap_ranking = sorted(
            [(f'feature_{i}', base_importance[i]) for i in range(n_features)],
            key=lambda x: x[1],
            reverse=True
        )

        mock_shap.get_global_summary.return_value = {
            'feature_importance': shap_importance,
            'feature_ranking': shap_ranking,
            'mean_shap_values': {f'feature_{i}': np.random.randn() * 0.1 for i in range(n_features)},
            'std_shap_values': {f'feature_{i}': abs(np.random.randn()) * 0.05 for i in range(n_features)}
        }

        # Mock PDP with slightly different but correlated importance
        pdp_noise = np.random.randn(n_features) * 0.1
        pdp_importance_values = base_importance + pdp_noise
        pdp_importance = {f'feature_{i}': max(0, pdp_importance_values[i]) for i in range(n_features)}
        pdp_ranking = sorted(
            [(f'feature_{i}', pdp_importance_values[i]) for i in range(n_features)],
            key=lambda x: x[1],
            reverse=True
        )

        mock_pdp = Mock()
        mock_pdp.get_feature_importance_from_pdp.return_value = {
            'feature_importance': pdp_importance,
            'feature_ranking': pdp_ranking
        }

        report.shap_explainer = mock_shap
        report.pdp_plotter = mock_pdp

        # Analyze global interpretability
        report.analyze_global_interpretability(top_k_features=min(top_k, n_features))

        # Verify feature importance consistency
        global_interp = report.report_data['global_interpretability']
        shap_data = global_interp['shap']
        pdp_data = global_interp['pdp_importance']

        # Check that rankings are properly sorted
        shap_rankings = shap_data['feature_ranking']
        pdp_rankings = pdp_data['feature_ranking']

        assert all(
            shap_rankings[i][1] >= shap_rankings[i+1][1]
            for i in range(len(shap_rankings)-1)
        ), "SHAP ranking not sorted properly"

        assert all(
            pdp_rankings[i][1] >= pdp_rankings[i+1][1]
            for i in range(len(pdp_rankings)-1)
        ), "PDP ranking not sorted properly"

        # Check that informative features tend to rank higher
        top_shap_features = [f[0] for f in shap_rankings[:min(5, n_features)]]
        top_pdp_features = [f[0] for f in pdp_rankings[:min(5, n_features)]]

        # At least some informative features should be in top rankings
        informative_features = [f'feature_{i}' for i in range(n_informative)]
        informative_in_shap_top = sum(1 for f in top_shap_features if f in informative_features)
        informative_in_pdp_top = sum(1 for f in top_pdp_features if f in informative_features)

        assert informative_in_shap_top > 0, "No informative features in SHAP top ranking"
        assert informative_in_pdp_top > 0, "No informative features in PDP top ranking"

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=100, max_value=500),
        n_features=st.integers(min_value=3, max_value=15),
        n_interactions=st.integers(min_value=2, max_value=10),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_interaction_analysis_depth(self, n_samples, n_features, n_interactions, random_state):
        """Test that interaction analysis provides comprehensive insights."""
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
        y_series = pd.Series(y)

        # Train a model
        model = RandomForestClassifier(n_estimators=20, random_state=random_state)
        model.fit(X_df, y_series)

        # Initialize report
        report = InterpretabilityReport(model=model, X=X_df, y=y_series)

        # Mock interaction analysis
        mock_pdp = Mock()

        # Create realistic interaction strengths
        n_pairs = min(n_interactions, actual_n_features * (actual_n_features - 1) // 2)
        interaction_pairs = []
        interaction_strengths = {}

        pair_count = 0
        for i in range(actual_n_features):
            for j in range(i + 1, actual_n_features):
                if pair_count >= n_pairs:
                    break

                # Stronger interactions between informative features
                strength = np.random.rand() * 0.2 + (0.3 if i < actual_n_features // 2 and j < actual_n_features // 2 else 0)
                pair_name = f'feature_{i} x feature_{j}'
                interaction_pairs.append((pair_name, strength))
                interaction_strengths[pair_name] = strength
                pair_count += 1

        mock_pdp.generate_interaction_summary.return_value = {
            'interaction_strengths': interaction_strengths,
            'interaction_details': {
                pair_name: {
                    'feature_1': f'feature_{i}',
                    'feature_2': f'feature_{j}',
                    'interaction_strength': strength,
                    'grid_shape': (10, 10)
                }
                for pair_name, strength in interaction_pairs.items()
                for i, j in [(0, 1), (2, 3)]  # Mock details
            },
            'top_interactions': sorted(interaction_pairs, key=lambda x: x[1], reverse=True),
            'n_interactions_analyzed': n_pairs
        }

        report.pdp_plotter = mock_pdp

        # Analyze global interpretability
        report.analyze_global_interpretability()

        # Verify interaction analysis depth
        global_interp = report.report_data['global_interpretability']
        interactions = global_interp['feature_interactions']

        # Check structure completeness
        assert 'interaction_strengths' in interactions, "Interaction strengths missing"
        assert 'interaction_details' in interactions, "Interaction details missing"
        assert 'top_interactions' in interactions, "Top interactions missing"
        assert 'n_interactions_analyzed' in interactions, "Interaction count missing"

        # Check interaction data consistency
        assert len(interactions['interaction_strengths']) == n_pairs, "Incorrect number of interactions"
        assert len(interactions['top_interactions']) == n_pairs, "Mismatch in top interactions"

        # Check that interactions are properly sorted
        top_interactions = interactions['top_interactions']
        assert all(
            top_interactions[i][1] >= top_interactions[i+1][1]
            for i in range(len(top_interactions)-1)
        ), "Top interactions not sorted by strength"

        # Check interaction details structure
        for interaction_name, details in interactions['interaction_details'].items():
            assert 'feature_1' in details, f"Feature 1 missing for {interaction_name}"
            assert 'feature_2' in details, f"Feature 2 missing for {interaction_name}"
            assert 'interaction_strength' in details, f"Strength missing for {interaction_name}"
            assert 'grid_shape' in details, f"Grid shape missing for {interaction_name}"

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=100, max_value=500),
        n_features=st.integers(min_value=5, max_value=20),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_executive_summary_quality(self, n_samples, n_features, random_state):
        """Test that executive summary provides high-level insights."""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 2,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y)

        # Train a model
        model = RandomForestClassifier(n_estimators=20, random_state=random_state)
        model.fit(X_df, y_series)

        # Initialize report
        report = InterpretabilityReport(
            model=model,
            X=X_df,
            y=y_series,
            model_name="ExecutiveSummaryTestModel",
            description="Model for testing executive summary generation"
        )

        # Set up comprehensive global interpretability data
        report.report_data['global_interpretability'] = {
            'shap': {
                'feature_ranking': [
                    (f'feature_{i}', (n_features - i) * 0.1)
                    for i in range(min(5, n_features))
                ],
                'feature_importance': {
                    f'feature_{i}': (n_features - i) * 0.1
                    for i in range(min(5, n_features))
                }
            },
            'pdp_importance': {
                'feature_ranking': [
                    (f'feature_{i}', (n_features - i) * 0.05)
                    for i in range(min(3, n_features))
                ]
            },
            'feature_interactions': {
                'top_interactions': [
                    (f'feature_{i} x feature_{j}', np.random.rand() * 0.2)
                    for i in range(min(2, n_features))
                    for j in range(i+1, min(2, n_features))
                ]
            }
        }

        report.report_data['metadata']['shap_available'] = True
        report.report_data['metadata']['lime_available'] = True

        # Generate executive summary
        summary = report.generate_executive_summary()

        # Verify executive summary structure
        required_sections = ['model_overview', 'key_findings', 'top_features', 'recommendations']
        for section in required_sections:
            assert section in summary, f"Executive summary section '{section}' missing"

        # Verify model overview
        overview = summary['model_overview']
        assert overview['name'] == "ExecutiveSummaryTestModel"
        assert overview['type'] == "classification"
        assert overview['n_features'] == n_features
        assert overview['n_samples'] == n_samples

        # Verify key findings quality
        findings = summary['key_findings']
        assert len(findings) > 0, "No key findings generated"
        for finding in findings:
            assert isinstance(finding, str), "Key finding not a string"
            assert len(finding) > 10, "Key finding too short"

        # Verify top features
        top_features = summary['top_features']
        assert len(top_features) > 0, "No top features identified"
        assert all(isinstance(f, str) for f in top_features), "Top features not strings"

        # Verify recommendations
        recommendations = summary['recommendations']
        assert len(recommendations) > 0, "No recommendations generated"
        for rec in recommendations:
            assert isinstance(rec, str), "Recommendation not a string"
            assert len(rec) > 10, "Recommendation too short"

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=100, max_value=500),
        n_features=st.integers(min_value=5, max_value=20),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_report_generation_completeness(self, n_samples, n_features, random_state):
        """Test that report generation includes all components."""
        # Generate synthetic data
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=0.1,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y)

        # Train a model
        model = RandomForestRegressor(n_estimators=20, random_state=random_state)
        model.fit(X_df, y_series)

        # Initialize report
        report = InterpretabilityReport(
            model=model,
            X=X_df,
            y=y_series,
            model_name="CompleteReportTestModel",
            description="Model for testing complete report generation"
        )

        # Add comprehensive data
        report.report_data['global_interpretability'] = {
            'shap': {
                'feature_ranking': [(f'feature_{i}', np.random.rand()) for i in range(min(5, n_features))],
                'feature_importance': {f'feature_{i}': np.random.rand() for i in range(min(5, n_features))}
            }
        }

        report.report_data['local_interpretability'] = {
            'instances': {
                0: {
                    'index': 0,
                    'features': {f'feature_{i}': np.random.randn() for i in range(min(3, n_features))},
                    'shap': {
                        'feature_contributions': {f'feature_{i}': np.random.randn() for i in range(min(3, n_features))},
                        'prediction': np.random.rand()
                    }
                }
            }
        }

        report.report_data['visualizations'] = {
            'shap_summary': {'plot_base64': 'fake_base64_data'},
            'pdp_1d_feature_0': {'plot_base64': 'fake_base64_data'}
        }

        report.analyze_features()

        # Generate HTML report
        html_content = report.generate_html_report()
        assert len(html_content) > 1000, "HTML report too short"
        assert 'CompleteReportTestModel' in html_content, "Model name missing from HTML"
        assert 'Executive Summary' in html_content, "Executive summary missing from HTML"
        assert 'Global Interpretability' in html_content, "Global interpretability missing from HTML"
        assert 'Local Interpretability' in html_content, "Local interpretability missing from HTML"
        assert 'Visualizations' in html_content, "Visualizations missing from HTML"
        assert 'Feature Analysis' in html_content, "Feature analysis missing from HTML"

        # Generate JSON report
        json_content = report.generate_json_report()
        import json
        parsed_json = json.loads(json_content)

        assert 'metadata' in parsed_json, "Metadata missing from JSON"
        assert 'global_interpretability' in parsed_json, "Global interpretability missing from JSON"
        assert 'local_interpretability' in parsed_json, "Local interpretability missing from JSON"
        assert 'feature_analysis' in parsed_json, "Feature analysis missing from JSON"
        assert 'visualizations' in parsed_json, "Visualizations missing from JSON"
        assert 'executive_summary' in parsed_json, "Executive summary missing from JSON"

    @pytest.mark.property
    @given(
        n_samples=st.integers(min_value=100, max_value=500),
        n_features=st.integers(min_value=5, max_value=20),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_report_file_generation(self, n_samples, n_features, random_state):
        """Test that report files are properly generated and saved."""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 2,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y)

        # Train a model
        model = RandomForestClassifier(n_estimators=20, random_state=random_state)
        model.fit(X_df, y_series)

        # Initialize report
        report = InterpretabilityReport(model=model, X=X_df, y=y_series)

        # Test HTML file generation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_html:
            html_path = tmp_html.name

        try:
            html_content = report.generate_html_report(output_path=html_path)
            assert os.path.exists(html_path), "HTML file not created"

            with open(html_path, 'r') as f:
                saved_html = f.read
            assert os.path.getsize(html_path) > 0, "HTML file is empty"

        finally:
            if os.path.exists(html_path):
                os.unlink(html_path)

        # Test JSON file generation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_json:
            json_path = tmp_json.name

        try:
            json_content = report.generate_json_report(output_path=json_path)
            assert os.path.exists(json_path), "JSON file not created"

            with open(json_path, 'r') as f:
                saved_json = f.read
            assert os.path.getsize(json_path) > 0, "JSON file is empty"

            # Verify JSON is valid
            import json
            parsed = json.loads(saved_json)
            assert 'metadata' in parsed, "JSON file content invalid"

        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)