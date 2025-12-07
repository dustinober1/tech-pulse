"""
Tests for performance metrics dashboard.

This module tests the dashboard visualization capabilities including
model comparisons, trends, and comprehensive dashboards.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Import the classes we're testing
from src.portfolio.visualization.performance_dashboard import (
    PerformanceDashboard,
    DashboardConfig,
    MetricConfig
)

# Mock the optional dependencies
plotly_mock = MagicMock()
matplotlib_mock = MagicMock()

# Create mock modules
plotly_mock.graph_objects = MagicMock()
plotly_mock.express = MagicMock()
plotly_mock.subplots = MagicMock()
plotly_mock.io = MagicMock()
plotly_mock.io.templates = MagicMock()
plotly_mock.io.templates.default = "plotly_white"

matplotlib_mock.pyplot = MagicMock()
matplotlib_mock.dates = MagicMock()
matplotlib_mock.backends = MagicMock()
matplotlib_mock.backends.backend_agg = MagicMock()


class TestPerformanceDashboard:
    """Test cases for PerformanceDashboard"""

    @pytest.fixture
    def mock_model_registry(self):
        """Create a mock model registry"""
        registry = Mock()

        # Mock list_models response
        models_df = pd.DataFrame({
            'model_name': ['test_model_1', 'test_model_2', 'test_model_3'],
            'latest_version': ['1.0.0', '2.1.0', '1.5.2'],
            'model_type': ['RandomForest', 'LogisticRegression', 'XGBoost'],
            'task_type': ['classification', 'classification', 'classification'],
            'created_at': [datetime.now() - timedelta(days=i) for i in range(3)]
        })
        registry.list_models.return_value = models_df

        # Mock get_model_metadata response
        def mock_get_metadata(model_name, version=None):
            if model_name == 'test_model_1':
                metadata = Mock()
                metadata.model_name = model_name
                metadata.model_version = version or '1.0.0'
                metadata.created_at = datetime.now() - timedelta(days=2)
                metadata.model_type = 'RandomForest'
                metadata.task_type = 'classification'
                metadata.environment = 'development'
                metadata.performance_metrics = {
                    'accuracy': 0.85,
                    'precision': 0.82,
                    'recall': 0.88,
                    'f1': 0.85,
                    'roc_auc': 0.92
                }

                # Mock version history
                version1 = Mock()
                version1.version = '1.0.0'
                version1.created_at = datetime.now() - timedelta(days=10)
                version1.performance_metrics = {
                    'accuracy': 0.82,
                    'precision': 0.80,
                    'recall': 0.85,
                    'f1': 0.82,
                    'roc_auc': 0.90
                }

                version2 = Mock()
                version2.version = '1.1.0'
                version2.created_at = datetime.now() - timedelta(days=5)
                version2.performance_metrics = {
                    'accuracy': 0.84,
                    'precision': 0.81,
                    'recall': 0.87,
                    'f1': 0.84,
                    'roc_auc': 0.91
                }

                metadata.version_history = [version1, version2]
                return metadata

            elif model_name == 'test_model_2':
                metadata = Mock()
                metadata.model_name = model_name
                metadata.model_version = version or '2.1.0'
                metadata.created_at = datetime.now() - timedelta(days=1)
                metadata.model_type = 'LogisticRegression'
                metadata.task_type = 'classification'
                metadata.environment = 'production'
                metadata.performance_metrics = {
                    'accuracy': 0.78,
                    'precision': 0.75,
                    'recall': 0.80,
                    'f1': 0.77,
                    'roc_auc': 0.85
                }
                metadata.version_history = []
                return metadata

            elif model_name == 'test_model_3':
                metadata = Mock()
                metadata.model_name = model_name
                metadata.model_version = version or '1.5.2'
                metadata.created_at = datetime.now()
                metadata.model_type = 'XGBoost'
                metadata.task_type = 'classification'
                metadata.environment = 'staging'
                metadata.performance_metrics = {
                    'accuracy': 0.89,
                    'precision': 0.87,
                    'recall': 0.91,
                    'f1': 0.89,
                    'roc_auc': 0.95
                }
                metadata.version_history = []
                return metadata

            return None

        registry.get_model_metadata.side_effect = mock_get_metadata

        # Mock get_model_rankings response
        registry.get_model_rankings.return_value = [
            {
                'model_name': 'test_model_3',
                'version': '1.5.2',
                'metrics': {
                    'accuracy': 0.89,
                    'precision': 0.87,
                    'recall': 0.91,
                    'f1': 0.89,
                    'roc_auc': 0.95
                }
            },
            {
                'model_name': 'test_model_1',
                'version': '1.0.0',
                'metrics': {
                    'accuracy': 0.85,
                    'precision': 0.82,
                    'recall': 0.88,
                    'f1': 0.85,
                    'roc_auc': 0.92
                }
            },
            {
                'model_name': 'test_model_2',
                'version': '2.1.0',
                'metrics': {
                    'accuracy': 0.78,
                    'precision': 0.75,
                    'recall': 0.80,
                    'f1': 0.77,
                    'roc_auc': 0.85
                }
            }
        ]

        return registry

    @pytest.fixture
    def dashboard(self, mock_model_registry):
        """Create a PerformanceDashboard instance"""
        config = DashboardConfig(
            theme="plotly_white",
            height=600,
            width=800
        )
        return PerformanceDashboard(mock_model_registry, config)

    @pytest.fixture
    def mock_cv_results(self):
        """Create mock cross-validation results"""
        from src.portfolio.experimentation.cross_validator import CVResult

        cv_results = []
        for i in range(3):
            fold_scores = [0.78 + i * 0.05, 0.82 + i * 0.05, 0.79 + i * 0.05, 0.81 + i * 0.05, 0.80 + i * 0.05]
            scores_by_fold = {j: {'score': score} for j, score in enumerate(fold_scores)}
            fold_indices = {j: (list(range(0, 80)), list(range(80, 100))) for j in range(5)}

            cv_result = CVResult(
                cv_method='stratified',
                n_folds=5,
                mean_score=0.80 + i * 0.05,
                std_score=0.02 + i * 0.01,
                fold_scores=fold_scores,
                confidence_interval=(0.78 + i * 0.05, 0.82 + i * 0.05),
                scores_by_fold=scores_by_fold,
                fold_indices=fold_indices
            )
            cv_results.append(cv_result)

        return cv_results

    def test_dashboard_initialization(self, mock_model_registry):
        """Test dashboard initialization"""
        dashboard = PerformanceDashboard(mock_model_registry)

        assert dashboard.model_registry == mock_model_registry
        assert dashboard.config is not None
        assert isinstance(dashboard.metric_configs, dict)
        assert 'accuracy' in dashboard.metric_configs
        assert dashboard.metric_configs['accuracy'].higher_is_better == True

    def test_custom_metric_config(self, dashboard):
        """Test adding custom metric configuration"""
        custom_config = MetricConfig(
            name='custom_metric',
            display_name='Custom Metric',
            higher_is_better=False,
            format_spec='.4f',
            range=(0, 100)
        )

        dashboard.add_custom_metric_config('custom_metric', custom_config)

        assert 'custom_metric' in dashboard.metric_configs
        assert dashboard.metric_configs['custom_metric'].display_name == 'Custom Metric'
        assert dashboard.metric_configs['custom_metric'].higher_is_better == False

    @patch('src.portfolio.visualization.performance_dashboard.PLOTLY_AVAILABLE', True)
    def test_create_model_comparison_bar_chart(self, dashboard):
        """Test creating model comparison bar chart"""
        with patch('src.portfolio.visualization.performance_dashboard.px') as mock_px:
            mock_fig = Mock()
            mock_px.bar.return_value = mock_fig

            result = dashboard.create_model_comparison_chart('accuracy', chart_type='bar')

            # Verify plotly was called
            mock_px.bar.assert_called_once()

            # Check the call arguments
            call_args = mock_px.bar.call_args
            assert call_args[1]['x'] == 'model_name'
            assert call_args[1]['y'] == 'accuracy'
            assert 'Accuracy' in call_args[1]['title']

            assert result == mock_fig

    @patch('src.portfolio.visualization.performance_dashboard.PLOTLY_AVAILABLE', True)
    def test_create_model_comparison_box_chart(self, dashboard):
        """Test creating model comparison box chart"""
        with patch('src.portfolio.visualization.performance_dashboard.px') as mock_px:
            mock_fig = Mock()
            mock_px.box.return_value = mock_fig

            result = dashboard.create_model_comparison_chart('accuracy', chart_type='box')

            # Verify plotly was called
            mock_px.box.assert_called_once()
            assert result == mock_fig

    @patch('src.portfolio.visualization.performance_dashboard.PLOTLY_AVAILABLE', True)
    def test_create_cross_validation_plot(self, dashboard, mock_cv_results):
        """Test creating cross-validation visualization"""
        with patch('src.portfolio.visualization.performance_dashboard.make_subplots') as mock_subplots:
            mock_fig = Mock()
            mock_subplots.return_value = mock_fig

            result = dashboard.create_cross_validation_plot(mock_cv_results)

            # Verify subplots was called
            mock_subplots.assert_called_once_with(
                rows=2, cols=1,
                subplot_titles=('Fold Scores by Model', 'Score Distribution'),
                vertical_spacing=0.1
            )

            assert result == mock_fig

    @patch('src.portfolio.visualization.performance_dashboard.PLOTLY_AVAILABLE', True)
    def test_create_performance_trends(self, dashboard):
        """Test creating performance trends plot"""
        with patch('src.portfolio.visualization.performance_dashboard.go') as mock_go:
            mock_fig = Mock()
            mock_go.Figure.return_value = mock_fig

            result = dashboard.create_performance_trends('test_model_1')

            # Verify Figure was created
            mock_go.Figure.assert_called_once()

            # Verify traces were added
            assert mock_go.Scatter.call_count >= 1  # At least one metric trace

            assert result == mock_fig

    @patch('src.portfolio.visualization.performance_dashboard.PLOTLY_AVAILABLE', True)
    def test_create_model_leaderboard(self, dashboard):
        """Test creating model leaderboard"""
        with patch('src.portfolio.visualization.performance_dashboard.go') as mock_go:
            mock_fig = Mock()
            mock_go.Figure.return_value = mock_fig

            result = dashboard.create_model_leaderboard('accuracy', ['precision', 'recall'])

            # Verify Figure was created
            mock_go.Figure.assert_called_once()

            # Verify table was added
            mock_go.Table.assert_called_once()

            assert result == mock_fig

    def test_create_model_comparison_invalid_metric(self, dashboard):
        """Test comparison chart with invalid metric"""
        with pytest.raises(ValueError, match="No models have metric"):
            dashboard.create_model_comparison_chart('invalid_metric')

    def test_create_model_comparison_no_models(self, dashboard):
        """Test comparison chart with no matching models"""
        with pytest.raises(ValueError, match="No models found"):
            dashboard.create_model_comparison_chart('accuracy', model_names=['nonexistent_model'])

    def test_get_available_metrics(self, dashboard):
        """Test getting list of available metrics"""
        metrics = dashboard.get_available_metrics()

        assert isinstance(metrics, list)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics

    def test_export_metrics_summary_json(self, dashboard):
        """Test exporting metrics summary as JSON"""
        summary = dashboard.export_metrics_summary(format='json')

        assert isinstance(summary, list)
        assert len(summary) == 3  # Three mock models

        # Check first model entry
        model_entry = summary[0]
        assert 'model_name' in model_entry
        assert 'accuracy' in model_entry
        assert 'precision' in model_entry

    def test_export_metrics_summary_invalid_format(self, dashboard):
        """Test exporting metrics with invalid format"""
        with pytest.raises(ValueError, match="Unsupported export format"):
            dashboard.export_metrics_summary(format='invalid')

    @patch('src.portfolio.visualization.performance_dashboard.PLOTLY_AVAILABLE', True)
    def test_create_comprehensive_dashboard(self, dashboard):
        """Test creating comprehensive dashboard"""
        with patch('src.portfolio.visualization.performance_dashboard.px') as mock_px:
            mock_fig = Mock()
            mock_px.bar.return_value = mock_fig

            dashboard._save_dashboard = Mock()

            result = dashboard.create_comprehensive_dashboard()

            # Should create charts for available metrics
            assert isinstance(result, dict)
            assert 'accuracy_comparison' in result
            assert 'leaderboard' in result

            # Should create trend plots for models with version history
            assert 'test_model_1_trends' in result
            # Note: test_model_2 and test_model_3 don't have version history in the mock

    def test_no_visualization_libraries(self):
        """Test behavior when no visualization libraries are available"""
        with patch('src.portfolio.visualization.performance_dashboard.PLOTLY_AVAILABLE', False), \
             patch('src.portfolio.visualization.performance_dashboard.MATPLOTLIB_AVAILABLE', False):

            with pytest.raises(ImportError, match="Neither plotly nor matplotlib available"):
                dashboard = PerformanceDashboard(Mock())
                dashboard.create_model_comparison_chart('accuracy')


class TestDashboardConfigs:
    """Test cases for dashboard configuration classes"""

    def test_metric_config_creation(self):
        """Test MetricConfig creation"""
        config = MetricConfig(
            name='test_metric',
            display_name='Test Metric',
            higher_is_better=False,
            format_spec='.4f',
            range=(0, 100)
        )

        assert config.name == 'test_metric'
        assert config.display_name == 'Test Metric'
        assert config.higher_is_better == False
        assert config.format_spec == '.4f'
        assert config.range == (0, 100)

    def test_dashboard_config_default_values(self):
        """Test DashboardConfig default values"""
        config = DashboardConfig()

        assert config.theme == "plotly_white"
        assert config.height == 600
        assert config.width == 800
        assert config.show_grid == True
        assert config.title_font_size == 20
        assert len(config.color_palette) == 10  # Default palette

    def test_dashboard_config_custom_values(self):
        """Test DashboardConfig with custom values"""
        custom_palette = ['#FF0000', '#00FF00', '#0000FF']
        config = DashboardConfig(
            theme="plotly_dark",
            height=800,
            width=1200,
            color_palette=custom_palette
        )

        assert config.theme == "plotly_dark"
        assert config.height == 800
        assert config.width == 1200
        assert config.color_palette == custom_palette


class TestDashboardErrorHandling:
    """Test error handling scenarios"""

    @pytest.fixture
    def empty_model_registry(self):
        """Create a mock model registry with no models"""
        registry = Mock()
        registry.list_models.return_value = pd.DataFrame({
            'model_name': [],
            'latest_version': [],
            'model_type': [],
            'task_type': [],
            'created_at': []
        })
        registry.get_model_metadata.return_value = None
        registry.get_model_rankings.return_value = []
        return registry

    @pytest.fixture
    def dashboard(self, empty_model_registry):
        """Create a dashboard with empty registry"""
        return PerformanceDashboard(empty_model_registry)

    def test_empty_registry_leaderboard(self, empty_model_registry):
        """Test leaderboard with empty registry"""
        dashboard = PerformanceDashboard(empty_model_registry)

        with pytest.raises(ValueError, match="No models found with metric"):
            dashboard.create_model_leaderboard('accuracy')

    def test_empty_registry_trends(self, empty_model_registry):
        """Test trends with empty registry"""
        dashboard = PerformanceDashboard(empty_model_registry)

        with pytest.raises(ValueError, match="Model.*not found"):
            dashboard.create_performance_trends('nonexistent_model')

    def test_empty_cv_results(self, dashboard):
        """Test cross-validation plot with no results"""
        with pytest.raises(ValueError, match="No cross-validation results provided"):
            dashboard.create_cross_validation_plot([])

    def test_model_without_metrics(self):
        """Test handling model without performance metrics"""
        registry = Mock()
        models_df = pd.DataFrame({
            'model_name': ['no_metrics_model'],
            'latest_version': ['1.0.0'],
            'model_type': ['TestModel'],
            'task_type': ['test'],
            'created_at': [datetime.now()]
        })
        registry.list_models.return_value = models_df

        # Mock metadata without performance metrics
        metadata = Mock()
        metadata.performance_metrics = {}
        registry.get_model_metadata.return_value = metadata

        dashboard = PerformanceDashboard(registry)

        # Should handle gracefully when no metrics available
        with pytest.raises(ValueError, match="No models have metric"):
            dashboard.create_model_comparison_chart('accuracy')


class TestDashboardIntegration:
    """Integration tests for dashboard functionality"""

    @pytest.fixture
    def mock_model_registry(self):
        """Create a mock model registry"""
        registry = Mock()

        # Mock list_models response
        models_df = pd.DataFrame({
            'model_name': ['integration_model'],
            'latest_version': ['1.0.0'],
            'model_type': ['TestModel'],
            'task_type': ['test'],
            'created_at': [datetime.now()]
        })
        registry.list_models.return_value = models_df

        # Mock get_model_metadata response
        metadata = Mock()
        metadata.performance_metrics = {'accuracy': 0.85}
        registry.get_model_metadata.return_value = metadata
        registry.get_model_rankings.return_value = []

        return registry

    @pytest.fixture
    def dashboard(self, mock_model_registry):
        """Create a PerformanceDashboard instance"""
        return PerformanceDashboard(mock_model_registry)

    def test_full_workflow(self, dashboard):
        """Test complete dashboard workflow"""
        # Get available metrics
        metrics = dashboard.get_available_metrics()
        assert len(metrics) > 0

        # Create comparison chart
        with patch('src.portfolio.visualization.performance_dashboard.PLOTLY_AVAILABLE', True), \
             patch('src.portfolio.visualization.performance_dashboard.px') as mock_px:
            mock_fig = Mock()
            mock_px.bar.return_value = mock_fig

            chart = dashboard.create_model_comparison_chart('accuracy')
            assert chart is not None

        # Export summary
        summary = dashboard.export_metrics_summary()
        assert isinstance(summary, list)
        assert len(summary) > 0

        # Verify summary contains expected fields
        for model_data in summary:
            assert 'model_name' in model_data
            assert 'latest_version' in model_data
            assert 'model_type' in model_data

    def test_custom_metric_workflow(self, dashboard):
        """Test workflow with custom metrics"""
        # Add custom metric
        custom_config = MetricConfig(
            name='business_impact',
            display_name='Business Impact Score',
            higher_is_better=True,
            format_spec='.2f',
            range=(0, 1000)
        )
        dashboard.add_custom_metric_config('business_impact', custom_config)

        # Verify metric is available
        assert 'business_impact' in dashboard.metric_configs

        # Test config properties
        metric_config = dashboard.metric_configs['business_impact']
        assert metric_config.display_name == 'Business Impact Score'
        assert metric_config.range == (0, 1000)