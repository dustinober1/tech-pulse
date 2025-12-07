"""
Visualization utilities for experiment tracking.

This module provides functions to create visualizations for comparing
and analyzing experiments.
"""

from typing import List, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.portfolio.experimentation.tracker import ExperimentTracker, Experiment


def create_metric_comparison_plot(
    tracker: ExperimentTracker,
    experiment_ids: List[str],
    metric_name: str,
    title: Optional[str] = None
) -> go.Figure:
    """
    Create bar chart comparing a metric across experiments.
    
    Args:
        tracker: ExperimentTracker instance
        experiment_ids: List of experiment IDs to compare
        metric_name: Name of metric to compare
        title: Optional custom title
        
    Returns:
        Plotly figure
    """
    experiments = [tracker.get_experiment(exp_id) for exp_id in experiment_ids]
    
    names = [exp.name for exp in experiments]
    values = [exp.metrics.get(metric_name, 0) for exp in experiments]
    
    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=values,
            text=[f"{v:.4f}" for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=title or f"Comparison of {metric_name}",
        xaxis_title="Experiment",
        yaxis_title=metric_name,
        template="plotly_white"
    )
    
    return fig


def create_multi_metric_comparison(
    tracker: ExperimentTracker,
    experiment_ids: List[str],
    metrics: List[str]
) -> go.Figure:
    """
    Create grouped bar chart comparing multiple metrics.
    
    Args:
        tracker: ExperimentTracker instance
        experiment_ids: List of experiment IDs
        metrics: List of metric names to compare
        
    Returns:
        Plotly figure
    """
    experiments = [tracker.get_experiment(exp_id) for exp_id in experiment_ids]
    
    fig = go.Figure()
    
    for metric in metrics:
        values = [exp.metrics.get(metric, 0) for exp in experiments]
        names = [exp.name for exp in experiments]
        
        fig.add_trace(go.Bar(
            name=metric,
            x=names,
            y=values,
            text=[f"{v:.4f}" for v in values],
            textposition='auto'
        ))
    
    fig.update_layout(
        title="Multi-Metric Comparison",
        xaxis_title="Experiment",
        yaxis_title="Metric Value",
        barmode='group',
        template="plotly_white"
    )
    
    return fig


def create_parameter_impact_plot(
    tracker: ExperimentTracker,
    experiment_ids: List[str],
    parameter_name: str,
    metric_name: str
) -> go.Figure:
    """
    Create scatter plot showing parameter impact on metric.
    
    Args:
        tracker: ExperimentTracker instance
        experiment_ids: List of experiment IDs
        parameter_name: Parameter to analyze
        metric_name: Metric to plot against parameter
        
    Returns:
        Plotly figure
    """
    experiments = [tracker.get_experiment(exp_id) for exp_id in experiment_ids]
    
    param_values = []
    metric_values = []
    names = []
    
    for exp in experiments:
        if parameter_name in exp.parameters and metric_name in exp.metrics:
            param_values.append(exp.parameters[parameter_name])
            metric_values.append(exp.metrics[metric_name])
            names.append(exp.name)
    
    fig = go.Figure(data=[
        go.Scatter(
            x=param_values,
            y=metric_values,
            mode='markers+text',
            text=names,
            textposition='top center',
            marker=dict(size=10, color=metric_values, colorscale='Viridis', showscale=True)
        )
    ])
    
    fig.update_layout(
        title=f"Impact of {parameter_name} on {metric_name}",
        xaxis_title=parameter_name,
        yaxis_title=metric_name,
        template="plotly_white"
    )
    
    return fig


def create_experiment_timeline(
    tracker: ExperimentTracker,
    name_filter: Optional[str] = None,
    metric_name: Optional[str] = None
) -> go.Figure:
    """
    Create timeline of experiments with optional metric overlay.
    
    Args:
        tracker: ExperimentTracker instance
        name_filter: Optional filter for experiment names
        metric_name: Optional metric to show on timeline
        
    Returns:
        Plotly figure
    """
    experiments = tracker.list_experiments(name_filter=name_filter)
    
    if not experiments:
        return go.Figure()
    
    timestamps = [exp.timestamp for exp in experiments]
    names = [exp.name for exp in experiments]
    
    if metric_name:
        metric_values = [exp.metrics.get(metric_name, 0) for exp in experiments]
        
        fig = go.Figure(data=[
            go.Scatter(
                x=timestamps,
                y=metric_values,
                mode='markers+lines+text',
                text=names,
                textposition='top center',
                marker=dict(size=10)
            )
        ])
        
        fig.update_layout(
            title=f"Experiment Timeline - {metric_name}",
            xaxis_title="Time",
            yaxis_title=metric_name,
            template="plotly_white"
        )
    else:
        fig = go.Figure(data=[
            go.Scatter(
                x=timestamps,
                y=list(range(len(timestamps))),
                mode='markers+text',
                text=names,
                textposition='middle right',
                marker=dict(size=10)
            )
        ])
        
        fig.update_layout(
            title="Experiment Timeline",
            xaxis_title="Time",
            yaxis_title="Experiment Index",
            template="plotly_white"
        )
    
    return fig


def create_experiment_heatmap(
    tracker: ExperimentTracker,
    experiment_ids: List[str],
    metrics: Optional[List[str]] = None
) -> go.Figure:
    """
    Create heatmap of metrics across experiments.
    
    Args:
        tracker: ExperimentTracker instance
        experiment_ids: List of experiment IDs
        metrics: Optional list of metrics to include
        
    Returns:
        Plotly figure
    """
    df = tracker.compare_experiments(experiment_ids, metrics)
    
    # Extract metric columns
    metric_cols = [col for col in df.columns if col.startswith('metric_')]
    
    if not metric_cols:
        return go.Figure()
    
    # Create matrix
    metric_names = [col.replace('metric_', '') for col in metric_cols]
    experiment_names = df['name'].tolist()
    values = df[metric_cols].values
    
    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=metric_names,
        y=experiment_names,
        colorscale='Viridis',
        text=values,
        texttemplate='%{text:.4f}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Experiment Metrics Heatmap",
        xaxis_title="Metric",
        yaxis_title="Experiment",
        template="plotly_white"
    )
    
    return fig


def create_parameter_distribution(
    tracker: ExperimentTracker,
    experiment_ids: List[str],
    parameter_name: str
) -> go.Figure:
    """
    Create histogram of parameter values across experiments.
    
    Args:
        tracker: ExperimentTracker instance
        experiment_ids: List of experiment IDs
        parameter_name: Parameter to analyze
        
    Returns:
        Plotly figure
    """
    experiments = [tracker.get_experiment(exp_id) for exp_id in experiment_ids]
    
    param_values = [
        exp.parameters.get(parameter_name) 
        for exp in experiments 
        if parameter_name in exp.parameters
    ]
    
    fig = go.Figure(data=[
        go.Histogram(x=param_values, nbinsx=20)
    ])
    
    fig.update_layout(
        title=f"Distribution of {parameter_name}",
        xaxis_title=parameter_name,
        yaxis_title="Count",
        template="plotly_white"
    )
    
    return fig
