"""Tests for interpretability report generator."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_classification, make_regression
import tempfile
import os

from src.interpretability.interpretability_report import InterpretabilityReport


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
    y_series = pd.Series(y, name='target')
    return X_df, y_series


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
    y_series = pd.Series(y, name='target')
    return X_df, y_series


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


class TestInterpretabilityReport:
    """Test cases for InterpretabilityReport class."""

    def test_initialization_classification(self, classification_model, classification_data):
        """Test initialization with classification model."""
        X, y = classification_data
        report = InterpretabilityReport(
            model=classification_model,
            X=X,
            y=y,
            model_name="TestClassifier",
            description="Test classification model"
        )

        assert report.model == classification_model
        assert report.X.equals(X)
        assert report.y.equals(y)
        assert report.model_name == "TestClassifier"
        assert report.model_type == "classification"
        assert report.description == "Test classification model"

    def test_initialization_regression(self, regression_model, regression_data):
        """Test initialization with regression model."""
        X, y = regression_data
        report = InterpretabilityReport(model=regression_model, X=X, y=y)

        assert report.model == regression_model
        assert report.model_type == "regression"

    def test_initialization_without_target(self, classification_model, classification_data):
        """Test initialization without target values."""
        X, y = classification_data
        report = InterpretabilityReport(model=classification_model, X=X)

        assert report.y is None
        assert report.model_type == "classification"  # Detected from model

    def test_detect_model_name(self, classification_model, classification_data):
        """Test automatic model name detection."""
        X, y = classification_data
        report = InterpretabilityReport(model=classification_model, X=X)

        assert report.model_name == "RandomForestClassifier"

    def test_detect_model_type_from_target(self, classification_data):
        """Test model type detection from target when model type is unknown."""
        X, y = classification_data
        mock_model = Mock()  # Model without sklearn base class

        # Test with discrete target (classification)
        report = InterpretabilityReport(model=mock_model, X=X, y=y)
        assert report.model_type == "classification"

        # Test with continuous target (regression)
        y_continuous = pd.Series(np.random.randn(len(y)))
        report_reg = InterpretabilityReport(model=mock_model, X=X, y=y_continuous)
        assert report_reg.model_type == "regression"

    def test_initialize_explainers_success(self, classification_model, classification_data):
        """Test successful initialization of all explainers."""
        X, y = classification_data
        report = InterpretabilityReport(model=classification_model, X=X, y=y)

        # Mock the explainers to avoid actual library dependencies
        with patch('src.interpretability.interpretability_report.SHAPExplainer') as mock_shap, \
             patch('src.interpretability.interpretability_report.LIMEExplainer') as mock_lime, \
             patch('src.interpretability.interpretability_report.PartialDependencePlotter') as mock_pdp:

            # Configure mocks
            mock_shap.return_value = Mock()
            mock_lime.return_value = Mock()
            mock_pdp.return_value = Mock()

            report.initialize_explainers(background_data_size=50)

            assert report.shap_explainer is not None
            assert report.lime_explainer is not None
            assert report.pdp_plotter is not None
            assert report.report_data['metadata']['shap_available'] is True
            assert report.report_data['metadata']['lime_available'] is True
            assert report.report_data['metadata']['pdp_available'] is True

    def test_initialize_explainers_with_warnings(self, classification_model, classification_data):
        """Test explainer initialization with warnings."""
        X, y = classification_data
        report = InterpretabilityReport(model=classification_model, X=X, y=y)

        with patch('src.interpretability.interpretability_report.SHAPExplainer', side_effect=ImportError("SHAP not available")), \
             patch('src.interpretability.interpretability_report.LIMEExplainer', side_effect=ImportError("LIME not available")), \
             patch('src.interpretability.interpretability_report.PartialDependencePlotter', side_effect=ImportError("PDP not available")):

            with pytest.warns(UserWarning, match="Could not initialize SHAP explainer"):
                with pytest.warns(UserWarning, match="Could not initialize LIME explainer"):
                    with pytest.warns(UserWarning, match="Could not initialize PDP plotter"):
                        report.initialize_explainers()

            assert report.shap_explainer is None
            assert report.lime_explainer is None
            assert report.pdp_plotter is None
            assert report.report_data['metadata']['shap_available'] is False
            assert report.report_data['metadata']['lime_available'] is False
            assert report.report_data['metadata']['pdp_available'] is False

    def test_analyze_global_interpretability(self, classification_model, classification_data):
        """Test global interpretability analysis."""
        X, y = classification_data
        report = InterpretabilityReport(model=classification_model, X=X, y=y)

        # Mock explainers
        mock_shap = Mock()
        mock_shap.explain.return_value = {'values': np.random.randn(10, 10)}
        mock_shap.get_global_summary.return_value = {
            'feature_importance': {f'feature_{i}': np.random.rand() for i in range(10)},
            'feature_ranking': [(f'feature_{i}', np.random.rand()) for i in range(10)],
            'mean_shap_values': {f'feature_{i}': np.random.randn() for i in range(10)},
            'std_shap_values': {f'feature_{i}': np.random.rand() for i in range(10)}
        }

        mock_pdp = Mock()
        mock_pdp.get_feature_importance_from_pdp.return_value = {
            'feature_importance': {f'feature_{i}': np.random.rand() for i in range(10)},
            'feature_ranking': [(f'feature_{i}', np.random.rand()) for i in range(10)]
        }
        mock_pdp.generate_interaction_summary.return_value = {
            'top_interactions': [('feature_0 x feature_1', 0.5), ('feature_2 x feature_3', 0.3)]
        }

        report.shap_explainer = mock_shap
        report.pdp_plotter = mock_pdp

        report.analyze_global_interpretability(n_samples=20, top_k_features=5)

        # Check results
        assert 'shap' in report.report_data['global_interpretability']
        assert 'pdp_importance' in report.report_data['global_interpretability']
        assert 'feature_interactions' in report.report_data['global_interpretability']
        assert 'target_stats' in report.report_data['global_interpretability']

        # Check SHAP results
        shap_data = report.report_data['global_interpretability']['shap']
        assert len(shap_data['feature_importance']) == 5  # top_k_features
        assert len(shap_data['feature_ranking']) == 5

    def test_analyze_local_interpretability(self, classification_model, classification_data):
        """Test local interpretability analysis."""
        X, y = classification_data
        report = InterpretabilityReport(model=classification_model, X=X, y=y)

        # Mock explainers
        mock_shap = Mock()
        mock_shap.explain_instance.return_value = {
            'feature_names': [f'feature_{i}' for i in range(10)],
            'shap_values': np.random.randn(10),
            'prediction': 0.75,
            'expected_value': 0.5
        }

        mock_lime = Mock()
        mock_lime.explain_instance.return_value = {
            'feature_contributions': [f'feature_{i}' for i in range(5)],
            'feature_values': np.random.randn(5),
            'predicted_value': 0.8,
            'score': 0.9
        }

        report.shap_explainer = mock_shap
        report.lime_explainer = mock_lime

        report.analyze_local_interpretability(n_instances=3)

        # Check results
        instances = report.report_data['local_interpretability']['instances']
        assert len(instances) == 3

        for idx, instance_data in instances.items():
            assert 'index' in instance_data
            assert 'features' in instance_data
            assert 'shap' in instance_data
            assert 'lime' in instance_data

            # Check SHAP structure
            assert 'feature_contributions' in instance_data['shap']
            assert 'prediction' in instance_data['shap']

            # Check LIME structure
            assert 'feature_contributions' in instance_data['lime']
            assert 'predicted_value' in instance_data['lime']

    def test_analyze_local_interpretability_specific_indices(self, classification_model, classification_data):
        """Test local interpretability analysis with specific indices."""
        X, y = classification_data
        report = InterpretabilityReport(model=classification_model, X=X, y=y)

        # Mock explainers
        mock_shap = Mock()
        mock_shap.explain_instance.return_value = {
            'feature_names': [f'feature_{i}' for i in range(10)],
            'shap_values': np.random.randn(10),
            'prediction': 0.75,
            'expected_value': 0.5
        }

        report.shap_explainer = mock_shap

        specific_indices = [0, 5, 10]
        report.analyze_local_interpretability(instance_indices=specific_indices)

        # Check that specific indices were analyzed
        instances = report.report_data['local_interpretability']['instances']
        assert len(instances) == 3
        for idx in specific_indices:
            assert idx in instances

    def test_analyze_features(self, classification_data):
        """Test feature analysis."""
        X, y = classification_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        report = InterpretabilityReport(model=model, X=X, y=y)
        report.analyze_features()

        # Check results
        feature_analysis = report.report_data['feature_analysis']
        assert len(feature_analysis) == 10  # Number of features

        for feature, info in feature_analysis.items():
            assert 'dtype' in info
            assert 'missing_count' in info
            assert 'missing_percentage' in info
            assert 'unique_count' in info
            assert 'cardinality' in info

            # Check numerical features (all features in this case are numerical)
            assert 'mean' in info
            assert 'std' in info
            assert 'min' in info
            assert 'max' in info
            assert 'skewness' in info
            assert 'kurtosis' in info

    def test_analyze_features_with_categorical(self):
        """Test feature analysis with categorical features."""
        # Create mixed data
        data = {
            'numerical_1': np.random.randn(100),
            'numerical_2': np.random.randn(100) * 2 + 1,
            'categorical_1': np.random.choice(['A', 'B', 'C'], 100),
            'categorical_2': np.random.choice([0, 1, 2], 100)
        }
        X_df = pd.DataFrame(data)
        y = np.random.choice([0, 1], 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_df, y)

        report = InterpretabilityReport(model=model, X=X_df, y=y)
        report.analyze_features()

        feature_analysis = report.report_data['feature_analysis']

        # Check numerical features
        for feature in ['numerical_1', 'numerical_2']:
            info = feature_analysis[feature]
            assert 'mean' in info
            assert 'std' in info

        # Check categorical features
        for feature in ['categorical_1', 'categorical_2']:
            info = feature_analysis[feature]
            assert 'most_frequent' in info
            assert 'frequency_distribution' in info

    def test_generate_visualizations(self, classification_model, classification_data):
        """Test visualization generation."""
        X, y = classification_data
        report = InterpretabilityReport(model=classification_model, X=X, y=y)

        # Mock explainers and their plot methods
        mock_shap = Mock()
        mock_shap.create_summary_plot.return_value = {'plot_base64': 'fake_base64_data'}
        mock_shap.create_waterfall_plot.return_value = {'plot_base64': 'fake_base64_data'}
        mock_shap.create_feature_importance_plot.return_value = {'plot_base64': 'fake_base64_data'}

        mock_lime = Mock()
        mock_lime.explain_instance.return_value = {
            'feature_contributions': ['feature_1', 'feature_2'],
            'feature_values': [0.1, 0.2]
        }
        mock_lime.create_explanation_plot.return_value = {'plot_base64': 'fake_base64_data'}

        mock_pdp = Mock()
        mock_pdp.plot_1d_partial_dependence.return_value = {'plot_base64': 'fake_base64_data'}
        mock_pdp.plot_2d_partial_dependence.return_value = {'plot_base64': 'fake_base64_data'}

        # Mock data sampling
        X_sample = X.sample(20, random_state=42)
        X_sample_instance = X.sample(1, random_state=42)

        with patch.object(X, 'sample', side_effect=[X_sample, X_sample_instance]):
            report.shap_explainer = mock_shap
            report.lime_explainer = mock_lime
            report.pdp_plotter = mock_pdp

            report.generate_visualizations(n_plots=5)

            # Check results
            visualizations = report.report_data['visualizations']
            assert len(visualizations) > 0

            expected_plots = ['shap_summary', 'shap_waterfall', 'shap_feature_importance']
            for plot_name in expected_plots:
                if plot_name in visualizations:
                    assert 'plot_base64' in visualizations[plot_name]

    def test_generate_executive_summary(self, classification_model, classification_data):
        """Test executive summary generation."""
        X, y = classification_data
        report = InterpretabilityReport(model=classification_model, X=X, y=y)

        # Add some mock global interpretability data
        report.report_data['global_interpretability'] = {
            'shap': {
                'feature_ranking': [('feature_1', 0.5), ('feature_2', 0.3)],
                'feature_importance': {'feature_1': 0.5, 'feature_2': 0.3}
            },
            'pdp_importance': {
                'feature_ranking': [('feature_1', 0.4), ('feature_3', 0.2)]
            },
            'feature_interactions': {
                'top_interactions': [('feature_1 x feature_2', 0.1)]
            }
        }

        report.report_data['metadata']['shap_available'] = True

        summary = report.generate_executive_summary()

        # Check structure
        assert 'model_overview' in summary
        assert 'key_findings' in summary
        assert 'top_features' in summary
        assert 'recommendations' in summary

        # Check content
        assert summary['model_overview']['name'] == 'RandomForestClassifier'
        assert summary['model_overview']['type'] == 'classification'
        assert len(summary['key_findings']) > 0
        assert 'feature_1' in summary['top_features']
        assert len(summary['recommendations']) > 0

    def test_generate_html_report(self, classification_model, classification_data):
        """Test HTML report generation."""
        X, y = classification_data
        report = InterpretabilityReport(
            model=classification_model,
            X=X,
            y=y,
            model_name="TestModel",
            description="Test model for HTML report"
        )

        # Add some mock data
        report.report_data['global_interpretability'] = {
            'shap': {
                'feature_ranking': [('feature_1', 0.5), ('feature_2', 0.3)]
            }
        }

        report.report_data['visualizations'] = {
            'test_plot': {'plot_base64': 'fake_base64_data'}
        }

        report.report_data['feature_analysis'] = {
            'feature_1': {
                'dtype': 'float64',
                'missing_percentage': 0.0,
                'unique_count': 100
            }
        }

        # Generate HTML
        html_content = report.generate_html_report()

        # Check structure
        assert '<!DOCTYPE html>' in html_content
        assert 'TestModel' in html_content
        assert 'Test model for HTML report' in html_content
        assert 'feature_1' in html_content

    def test_generate_html_report_with_file(self, classification_model, classification_data):
        """Test HTML report generation to file."""
        X, y = classification_data
        report = InterpretabilityReport(model=classification_model, X=X, y=y)

        # Generate HTML to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            html_content = report.generate_html_report(output_path=tmp_path)

            # Check file was created
            assert os.path.exists(tmp_path)

            # Check file content
            with open(tmp_path, 'r') as f:
                file_content = f.read()
            assert file_content == html_content

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_generate_json_report(self, classification_model, classification_data):
        """Test JSON report generation."""
        X, y = classification_data
        report = InterpretabilityReport(model=classification_model, X=X, y=y)

        # Add executive summary
        report.report_data['executive_summary'] = {
            'model_overview': {'name': 'TestModel'},
            'key_findings': ['Test finding']
        }

        # Generate JSON
        json_content = report.generate_json_report()

        # Check structure
        import json
        parsed_content = json.loads(json_content)
        assert 'metadata' in parsed_content
        assert 'executive_summary' in parsed_content
        assert parsed_content['executive_summary']['model_overview']['name'] == 'TestModel'

    def test_generate_json_report_with_file(self, classification_model, classification_data):
        """Test JSON report generation to file."""
        X, y = classification_data
        report = InterpretabilityReport(model=classification_model, X=X, y=y)

        # Generate JSON to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            json_content = report.generate_json_report(output_path=tmp_path)

            # Check file was created
            assert os.path.exists(tmp_path)

            # Check file content
            with open(tmp_path, 'r') as f:
                file_content = f.read()
            assert file_content == json_content

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_build_html_template_includes_css(self, classification_model, classification_data):
        """Test HTML template generation with CSS."""
        X, y = classification_data
        report = InterpretabilityReport(model=classification_model, X=X, y=y)

        executive_summary = report.generate_executive_summary()

        # Build HTML with CSS
        html_with_css = report._build_html_template(executive_summary, include_css=True)
        assert '<style>' in html_with_css
        assert 'body {' in html_with_css

        # Build HTML without CSS
        html_without_css = report._build_html_template(executive_summary, include_css=False)
        assert '<style>' not in html_without_css

    def test_report_data_structure(self, classification_model, classification_data):
        """Test that report data has correct structure."""
        X, y = classification_data
        report = InterpretabilityReport(model=classification_model, X=X, y=y)

        # Check initial structure
        assert 'metadata' in report.report_data
        assert 'global_interpretability' in report.report_data
        assert 'local_interpretability' in report.report_data
        assert 'feature_analysis' in report.report_data
        assert 'visualizations' in report.report_data

        # Check metadata structure
        metadata = report.report_data['metadata']
        assert 'model_name' in metadata
        assert 'model_type' in metadata
        assert 'description' in metadata
        assert 'generated_at' in metadata
        assert 'n_features' in metadata
        assert 'n_samples' in metadata