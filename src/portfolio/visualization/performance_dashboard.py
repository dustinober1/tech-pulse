"""
Performance metrics dashboard for comprehensive model visualization.

This module provides interactive visualizations for model performance metrics,
including comparisons, trends, and detailed analysis using Plotly.
"""

import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from src.portfolio.experimentation.model_registry import ModelRegistry
from src.portfolio.experimentation.cross_validator import CVResult


@dataclass
class MetricConfig:
    """Configuration for metric visualization."""
    name: str
    display_name: str
    higher_is_better: bool = True
    format_spec: str = ".3f"
    color_scale: Optional[List[str]] = None
    range: Optional[Tuple[float, float]] = None


@dataclass
class DashboardConfig:
    """Configuration for dashboard layout and styling."""
    theme: str = "plotly_white"
    color_palette: List[str] = None
    height: int = 600
    width: int = 800
    show_grid: bool = True
    title_font_size: int = 20

    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
            ]


class PerformanceDashboard:
    """
    Interactive performance metrics dashboard.

    Provides comprehensive visualization capabilities for model performance
    metrics, including comparisons, time series, and detailed analysis.
    """

    def __init__(self,
                 model_registry: ModelRegistry,
                 config: Optional[DashboardConfig] = None):
        """
        Initialize performance dashboard.

        Args:
            model_registry: ModelRegistry instance containing model data
            config: Dashboard configuration options
        """
        self.model_registry = model_registry
        self.config = config or DashboardConfig()

        # Standard metric configurations
        self.metric_configs = {
            'accuracy': MetricConfig(
                'accuracy', 'Accuracy', True, '.3f', None, (0, 1)
            ),
            'precision': MetricConfig(
                'precision', 'Precision', True, '.3f', None, (0, 1)
            ),
            'recall': MetricConfig(
                'recall', 'Recall', True, '.3f', None, (0, 1)
            ),
            'f1': MetricConfig(
                'f1', 'F1-Score', True, '.3f', None, (0, 1)
            ),
            'roc_auc': MetricConfig(
                'roc_auc', 'ROC AUC', True, '.3f', None, (0, 1)
            ),
            'mse': MetricConfig(
                'mse', 'Mean Squared Error', False, '.3f', None, (0, None)
            ),
            'rmse': MetricConfig(
                'rmse', 'Root MSE', False, '.3f', None, (0, None)
            ),
            'mae': MetricConfig(
                'mae', 'Mean Absolute Error', False, '.3f', None, (0, None)
            ),
            'r2': MetricConfig(
                'r2', 'R²', True, '.3f', None, (0, 1)
            )
        }

        if PLOTLY_AVAILABLE:
            pio.templates.default = self.config.theme

    def create_model_comparison_chart(self,
                                    metric_name: str,
                                    model_names: Optional[List[str]] = None,
                                    chart_type: str = 'bar') -> Union[go.Figure, Any]:
        """
        Create a model comparison chart for a specific metric.

        Args:
            metric_name: Name of the metric to compare
            model_names: List of model names to include (all if None)
            chart_type: Type of chart ('bar', 'box', 'violin')

        Returns:
            Plotly figure or matplotlib chart
        """
        if not PLOTLY_AVAILABLE and not MATPLOTLIB_AVAILABLE:
            raise ImportError("Neither plotly nor matplotlib available")

        # Get model data
        models_df = self.model_registry.list_models()
        if model_names:
            models_df = models_df[models_df['model_name'].isin(model_names)]

        if models_df.empty:
            raise ValueError("No models found for comparison")

        # Extract metric data
        model_metrics = []
        for _, model_row in models_df.iterrows():
            model_name = model_row['model_name']
            latest_version = model_row['latest_version']

            metadata = self.model_registry.get_model_metadata(model_name, latest_version)
            if metadata and metric_name in metadata.performance_metrics:
                model_metrics.append({
                    'model_name': model_name,
                    'model_version': latest_version,
                    metric_name: metadata.performance_metrics[metric_name]
                })

        if not model_metrics:
            raise ValueError(f"No models have metric '{metric_name}'")

        metrics_df = pd.DataFrame(model_metrics)

        if PLOTLY_AVAILABLE:
            return self._create_plotly_comparison_chart(
                metrics_df, metric_name, chart_type
            )
        else:
            return self._create_matplotlib_comparison_chart(
                metrics_df, metric_name, chart_type
            )

    def _create_plotly_comparison_chart(self,
                                      metrics_df: pd.DataFrame,
                                      metric_name: str,
                                      chart_type: str) -> go.Figure:
        """Create comparison chart using Plotly."""
        metric_config = self.metric_configs.get(metric_name)
        display_name = metric_config.display_name if metric_config else metric_name

        if chart_type == 'bar':
            fig = px.bar(
                metrics_df,
                x='model_name',
                y=metric_name,
                title=f'Model Comparison: {display_name}',
                labels={
                    'model_name': 'Model',
                    metric_name: display_name
                },
                color='model_name',
                color_discrete_sequence=self.config.color_palette
            )

            # Add value labels on bars
            fig.update_traces(texttemplate=f'%{{y:{metric_config.format_spec if metric_config else ".3f"}}}',
                            textposition='outside')

        elif chart_type == 'box':
            fig = px.box(
                metrics_df,
                x='model_name',
                y=metric_name,
                title=f'Model Performance Distribution: {display_name}',
                labels={
                    'model_name': 'Model',
                    metric_name: display_name
                },
                color='model_name',
                color_discrete_sequence=self.config.color_palette
            )

        elif chart_type == 'violin':
            fig = px.violin(
                metrics_df,
                x='model_name',
                y=metric_name,
                title=f'Model Performance Distribution: {display_name}',
                labels={
                    'model_name': 'Model',
                    metric_name: display_name
                },
                color='model_name',
                color_discrete_sequence=self.config.color_palette
            )
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        # Apply styling
        fig.update_layout(
            height=self.config.height,
            width=self.config.width,
            showlegend=len(metrics_df['model_name'].unique()) > 1,
            title_font_size=self.config.title_font_size
        )

        if metric_config and metric_config.range:
            fig.update_yaxes(range=metric_config.range)

        return fig

    def create_cross_validation_plot(self,
                                   cv_results: List[CVResult],
                                   metric_name: str = 'score') -> Union[go.Figure, Any]:
        """
        Create visualization of cross-validation results.

        Args:
            cv_results: List of CVResult objects
            metric_name: Name of metric to visualize from fold scores

        Returns:
            Plotly figure or matplotlib chart
        """
        if not PLOTLY_AVAILABLE and not MATPLOTLIB_AVAILABLE:
            raise ImportError("Neither plotly nor matplotlib available")

        if not cv_results:
            raise ValueError("No cross-validation results provided")

        # Prepare data
        cv_data = []
        for i, cv_result in enumerate(cv_results):
            for fold_idx, fold_score in enumerate(cv_result.fold_scores):
                cv_data.append({
                    'model': f'Model {i+1}',
                    'fold': f'Fold {fold_idx+1}',
                    'score': fold_score,
                    'cv_method': cv_result.cv_method
                })

        df = pd.DataFrame(cv_data)

        if PLOTLY_AVAILABLE:
            return self._create_plotly_cv_plot(df, metric_name)
        else:
            return self._create_matplotlib_cv_plot(df, metric_name)

    def _create_plotly_cv_plot(self, df: pd.DataFrame, metric_name: str) -> go.Figure:
        """Create cross-validation plot using Plotly."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Fold Scores by Model', 'Score Distribution'),
            vertical_spacing=0.1
        )

        # Box plot for distribution
        for i, model in enumerate(df['model'].unique()):
            model_data = df[df['model'] == model]
            fig.add_trace(
                go.Box(
                    y=model_data['score'],
                    name=model,
                    marker_color=self.config.color_palette[i % len(self.config.color_palette)]
                ),
                row=1, col=1
            )

        # Bar plot for individual fold scores
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            fig.add_trace(
                go.Bar(
                    x=model_data['fold'],
                    y=model_data['score'],
                    name=f'{model} (folds)',
                    marker_color=self.config.color_palette[df['model'].unique().tolist().index(model) % len(self.config.color_palette)],
                    showlegend=False
                ),
                row=2, col=1
            )

        fig.update_layout(
            height=self.config.height * 1.5,
            width=self.config.width,
            title_text='Cross-Validation Results Analysis',
            title_font_size=self.config.title_font_size
        )

        return fig

    def create_performance_trends(self,
                                model_name: str,
                                metric_names: Optional[List[str]] = None) -> Union[go.Figure, Any]:
        """
        Create performance trends across model versions.

        Args:
            model_name: Name of the model
            metric_names: List of metrics to track (all if None)

        Returns:
            Plotly figure or matplotlib chart
        """
        if not PLOTLY_AVAILABLE and not MATPLOTLIB_AVAILABLE:
            raise ImportError("Neither plotly nor matplotlib available")

        # Get all versions of the model
        models_df = self.model_registry.list_models()
        model_row = models_df[models_df['model_name'] == model_name]

        if model_row.empty:
            raise ValueError(f"Model '{model_name}' not found")

        # Collect version data
        version_data = []
        metadata = self.model_registry.get_model_metadata(model_name)

        if metadata:
            for version in metadata.version_history:
                version_metrics = {}
                if metric_names is None:
                    version_metrics = version.performance_metrics
                else:
                    version_metrics = {
                        k: v for k, v in version.performance_metrics.items()
                        if k in metric_names
                    }

                if version_metrics:
                    version_data.append({
                        'version': version.version,
                        'created_at': version.created_at,
                        **version_metrics
                    })

        if not version_data:
            raise ValueError(f"No performance data found for model '{model_name}'")

        df = pd.DataFrame(version_data)
        df['created_at'] = pd.to_datetime(df['created_at'])

        if PLOTLY_AVAILABLE:
            return self._create_plotly_trends(df, model_name, metric_names)
        else:
            return self._create_matplotlib_trends(df, model_name, metric_names)

    def _create_plotly_trends(self,
                            df: pd.DataFrame,
                            model_name: str,
                            metric_names: Optional[List[str]]) -> go.Figure:
        """Create performance trends plot using Plotly."""
        fig = go.Figure()

        # Determine which metrics to plot
        if metric_names is None:
            metrics_to_plot = [col for col in df.columns if col not in ['version', 'created_at']]
        else:
            metrics_to_plot = [col for col in df.columns if col in metric_names]

        for i, metric in enumerate(metrics_to_plot):
            metric_config = self.metric_configs.get(metric)
            display_name = metric_config.display_name if metric_config else metric

            fig.add_trace(
                go.Scatter(
                    x=df['created_at'],
                    y=df[metric],
                    mode='lines+markers',
                    name=display_name,
                    line=dict(color=self.config.color_palette[i % len(self.config.color_palette)]),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Value: %{y:.3f}<extra></extra>'
                )
            )

        fig.update_layout(
            title=f'Performance Trends: {model_name}',
            xaxis_title='Date',
            yaxis_title='Metric Value',
            height=self.config.height,
            width=self.config.width,
            title_font_size=self.config.title_font_size,
            hovermode='x unified'
        )

        return fig

    def create_model_leaderboard(self,
                               primary_metric: str,
                               secondary_metrics: Optional[List[str]] = None,
                               top_n: int = 10) -> Union[go.Figure, Any]:
        """
        Create a leaderboard visualization of top models.

        Args:
            primary_metric: Primary metric for ranking
            secondary_metrics: Additional metrics to display
            top_n: Number of top models to show

        Returns:
            Plotly figure or matplotlib chart
        """
        if not PLOTLY_AVAILABLE and not MATPLOTLIB_AVAILABLE:
            raise ImportError("Neither plotly nor matplotlib available")

        # Get model rankings
        rankings = self.model_registry.get_model_rankings(primary_metric)

        if not rankings:
            raise ValueError(f"No models found with metric '{primary_metric}'")

        # Take top N models
        top_models = rankings[:top_n]

        # Prepare data for visualization
        leaderboard_data = []
        for rank, model_info in enumerate(top_models, 1):
            model_name = model_info['model_name']
            model_version = model_info['version']
            primary_score = model_info['metrics'][primary_metric]

            entry = {
                'rank': rank,
                'model_name': model_name,
                'model_version': model_version,
                primary_metric: primary_score
            }

            # Add secondary metrics if available
            if secondary_metrics:
                metadata = self.model_registry.get_model_metadata(model_name, model_version)
                if metadata:
                    for metric in secondary_metrics:
                        if metric in metadata.performance_metrics:
                            entry[metric] = metadata.performance_metrics[metric]

            leaderboard_data.append(entry)

        df = pd.DataFrame(leaderboard_data)

        if PLOTLY_AVAILABLE:
            return self._create_plotly_leaderboard(df, primary_metric, secondary_metrics)
        else:
            return self._create_matplotlib_leaderboard(df, primary_metric, secondary_metrics)

    def _create_plotly_leaderboard(self,
                                 df: pd.DataFrame,
                                 primary_metric: str,
                                 secondary_metrics: Optional[List[str]]) -> go.Figure:
        """Create leaderboard using Plotly."""
        # Create table for detailed view
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Rank', 'Model', 'Version', primary_metric] + (secondary_metrics or []),
                fill_color='paleturquoise',
                align='left',
                font=dict(size=14)
            ),
            cells=dict(
                values=[df[col] for col in ['rank', 'model_name', 'model_version', primary_metric] + (secondary_metrics or [])],
                fill_color='lavender',
                align='left',
                font=dict(size=12)
            )
        )])

        # Add bar chart overlay for primary metric
        metric_config = self.metric_configs.get(primary_metric)
        display_name = metric_config.display_name if metric_config else primary_metric

        fig.add_trace(go.Bar(
            x=df['model_name'],
            y=df[primary_metric],
            name=display_name,
            yaxis='y2',
            marker_color=self.config.color_palette[0],
            opacity=0.7
        ))

        # Create dual-axis layout
        fig.update_layout(
            title='Model Leaderboard',
            height=self.config.height * 1.5,
            width=self.config.width,
            title_font_size=self.config.title_font_size,
            yaxis2=dict(
                title=display_name,
                overlaying='y',
                side='right'
            )
        )

        return fig

    def create_comprehensive_dashboard(self,
                                     model_names: Optional[List[str]] = None,
                                     save_path: Optional[str] = None) -> Union[Dict[str, go.Figure], Any]:
        """
        Create a comprehensive dashboard with multiple visualizations.

        Args:
            model_names: List of model names to include (all if None)
            save_path: Path to save dashboard HTML (optional)

        Returns:
            Dictionary of figures or dashboard object
        """
        if not PLOTLY_AVAILABLE and not MATPLOTLIB_AVAILABLE:
            raise ImportError("Neither plotly nor matplotlib available")

        dashboard = {}

        # Get available metrics
        models_df = self.model_registry.list_models()
        if model_names:
            models_df = models_df[models_df['model_name'].isin(model_names)]

        # Determine common metrics across models
        all_metrics = set()
        for _, model_row in models_df.iterrows():
            model_name = model_row['model_name']
            latest_version = model_row['latest_version']
            metadata = self.model_registry.get_model_metadata(model_name, latest_version)
            if metadata:
                all_metrics.update(metadata.performance_metrics.keys())

        # Create comparison charts for key metrics
        key_metrics = ['accuracy', 'f1', 'roc_auc', 'mse', 'r2']
        for metric in key_metrics:
            if metric in all_metrics:
                try:
                    dashboard[f'{metric}_comparison'] = self.create_model_comparison_chart(
                        metric, model_names
                    )
                except ValueError:
                    pass  # Skip if no data for metric

        # Create performance trends for each model
        for model_name in models_df['model_name'].unique():
            try:
                dashboard[f'{model_name}_trends'] = self.create_performance_trends(model_name)
            except ValueError:
                pass  # Skip if no trend data

        # Create leaderboard
        if all_metrics:
            primary_metric = max(['accuracy', 'f1', 'roc_auc', 'r2'],
                               key=lambda x: x in all_metrics)
            try:
                dashboard['leaderboard'] = self.create_model_leaderboard(primary_metric)
            except ValueError:
                pass

        # Save dashboard if requested
        if save_path and PLOTLY_AVAILABLE:
            self._save_dashboard(dashboard, save_path)

        return dashboard

    def _save_dashboard(self, dashboard: Dict[str, go.Figure], save_path: str):
        """Save dashboard to HTML file."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Performance Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .chart { margin-bottom: 30px; }
                h1 { color: #333; text-align: center; }
            </style>
        </head>
        <body>
            <h1>Model Performance Dashboard</h1>
        """

        for title, fig in dashboard.items():
            html_content += f'<div class="chart">{fig.to_html(include_plotlyjs="cdn")}</div>'

        html_content += "</body></html>"

        with open(save_path, 'w') as f:
            f.write(html_content)

    def export_metrics_summary(self,
                             model_names: Optional[List[str]] = None,
                             format: str = 'json') -> Dict[str, Any]:
        """
        Export summary of all model metrics.

        Args:
            model_names: List of model names to include (all if None)
            format: Export format ('json', 'csv', 'excel')

        Returns:
            Summary data
        """
        models_df = self.model_registry.list_models()
        if model_names:
            models_df = models_df[models_df['model_name'].isin(model_names)]

        summary_data = []

        for _, model_row in models_df.iterrows():
            model_name = model_row['model_name']
            latest_version = model_row['latest_version']
            metadata = self.model_registry.get_model_metadata(model_name, latest_version)

            if metadata:
                summary_entry = {
                    'model_name': model_name,
                    'latest_version': latest_version,
                    'created_at': metadata.created_at.isoformat() if metadata.created_at else None,
                    'model_type': metadata.model_type,
                    'task_type': metadata.task_type,
                    'environment': metadata.environment
                }
                summary_entry.update(metadata.performance_metrics)
                summary_data.append(summary_entry)

        summary_df = pd.DataFrame(summary_data)

        if format.lower() == 'json':
            return summary_df.to_dict('records')
        elif format.lower() == 'csv':
            return summary_df.to_csv(index=False)
        elif format.lower() == 'excel':
            from io import BytesIO
            output = BytesIO()
            summary_df.to_excel(output, index=False)
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def add_custom_metric_config(self,
                               metric_name: str,
                               config: MetricConfig):
        """
        Add custom metric configuration.

        Args:
            metric_name: Name of the metric
            config: MetricConfig object
        """
        self.metric_configs[metric_name] = config

    def get_available_metrics(self) -> List[str]:
        """
        Get list of available metrics across all models.

        Returns:
            List of metric names
        """
        models_df = self.model_registry.list_models()
        all_metrics = set()

        for _, model_row in models_df.iterrows():
            model_name = model_row['model_name']
            latest_version = model_row['latest_version']
            metadata = self.model_registry.get_model_metadata(model_name, latest_version)

            if metadata:
                all_metrics.update(metadata.performance_metrics.keys())

        return sorted(list(all_metrics))