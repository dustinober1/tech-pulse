"""
Predictive Analytics - Dashboard Integration

Provides dashboard components and API endpoints for
visualizing and interacting with predictive analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from pathlib import Path

# Import local modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from .predictor import (
    PredictiveEngine,
    PredictionResult,
    AnomalyResult
)
from .training_data import TrainingDataCollector
from .train_model import ModelTrainer
# Note: DataLoader class doesn't exist, using data_loader functions directly
# from data_loader import DataLoader

logger = logging.getLogger(__name__)

class PredictiveDashboard:
    """
    Dashboard components for predictive analytics.

    Features:
    - Prediction visualizations
    - Anomaly detection displays
    - Model performance metrics
    - Training status monitoring
    """

    def __init__(self):
        """Initialize the predictive dashboard."""
        self.predictive_engine = PredictiveEngine()
        self.data_collector = TrainingDataCollector()
        self.model_trainer = ModelTrainer()
        # self.data_loader = DataLoader()  # DataLoader class doesn't exist, using data_loader functions directly

    def render_prediction_tab(self):
        """Render the prediction tab in the dashboard."""
        st.header("📈 Technology Trend Predictions")

        # Technology selection
        technologies = self._get_available_technologies()
        selected_tech = st.selectbox(
            "Select Technology",
            technologies,
            key="prediction_tech_select"
        )

        if selected_tech:
            # Prediction configuration
            col1, col2 = st.columns(2)
            with col1:
                time_horizon = st.slider(
                    "Prediction Horizon (days)",
                    min_value=1,
                    max_value=90,
                    value=30,
                    key="pred_horizon"
                )
            with col2:
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.05,
                    key="conf_threshold"
                )

            # Make prediction
            if st.button("Generate Prediction", key="generate_pred"):
                with st.spinner("Generating prediction..."):
                    try:
                        # Get current features
                        current_features = self._get_current_features(selected_tech)

                        if current_features:
                            # Make prediction
                            prediction = self.predictive_engine.predict_trend(
                                selected_tech,
                                current_features,
                                time_horizon
                            )

                            # Display prediction
                            self._display_prediction(prediction, confidence_threshold)
                        else:
                            st.error(f"No data available for {selected_tech}")

                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")

            # Show historical predictions
            self._show_prediction_history(selected_tech)

    def render_anomaly_tab(self):
        """Render the anomaly detection tab."""
        st.header("🔍 Anomaly Detection")

        # Technology selection
        technologies = self._get_available_technologies()
        selected_tech = st.selectbox(
            "Select Technology",
            technologies,
            key="anomaly_tech_select"
        )

        if selected_tech:
            # Anomaly configuration
            anomaly_threshold = st.slider(
                "Anomaly Sensitivity",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Lower values detect more anomalies",
                key="anomaly_sensitivity"
            )

            # Check for anomalies
            if st.button("Detect Anomalies", key="detect_anomalies"):
                with st.spinner("Analyzing patterns..."):
                    try:
                        # Get current features
                        current_features = self._get_current_features(selected_tech)

                        if current_features:
                            # Detect anomalies
                            anomaly_result = self.predictive_engine.detect_anomaly(
                                selected_tech,
                                current_features
                            )

                            # Display anomaly result
                            self._display_anomaly_result(anomaly_result)
                        else:
                            st.error(f"No data available for {selected_tech}")

                    except Exception as e:
                        st.error(f"Anomaly detection failed: {str(e)}")

            # Show recent anomalies
            self._show_recent_anomalies(selected_tech)

    def render_model_training_tab(self):
        """Render the model training tab."""
        st.header("🤖 Model Training Management")

        # Technology selection
        technologies = self._get_available_technologies()
        selected_tech = st.selectbox(
            "Select Technology",
            ["All Technologies"] + technologies,
            key="training_tech_select"
        )

        # Training configuration
        with st.expander("Training Configuration", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                lookback_days = st.number_input(
                    "Historical Data (days)",
                    min_value=30,
                    max_value=1095,
                    value=365,
                    key="lookback_days"
                )
                optimize_hyperparams = st.checkbox(
                    "Optimize Hyperparameters",
                    value=True,
                    key="optimize_params"
                )
            with col2:
                prediction_horizon = st.number_input(
                    "Prediction Horizon (days)",
                    min_value=1,
                    max_value=90,
                    value=30,
                    key="pred_horizon_train"
                )
                model_types = st.multiselect(
                    "Model Types",
                    ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm'],
                    default=['random_forest', 'xgboost'],
                    key="model_types"
                )

        # Train models
        if st.button("Start Training", key="start_training", type="primary"):
            with st.spinner("Training models..."):
                try:
                    if selected_tech == "All Technologies":
                        # Train all models
                        results = self.model_trainer.train_all_models(
                            technologies,
                            lookback_days,
                            prediction_horizon,
                            optimize_hyperparams=optimize_hyperparams
                        )
                        self._display_batch_training_results(results)
                    else:
                        # Train single technology
                        results = self.model_trainer.train_technology_models(
                            selected_tech,
                            lookback_days,
                            prediction_horizon,
                            optimize_hyperparams=optimize_hyperparams,
                            model_types=model_types
                        )
                        self._display_training_results(results)

                except Exception as e:
                    st.error(f"Training failed: {str(e)}")

        # Show model info
        if selected_tech != "All Technologies":
            self._show_model_info(selected_tech)

        # Show training history
        self._show_training_history()

    def render_performance_tab(self):
        """Render the model performance tab."""
        st.header("📊 Model Performance Analytics")

        # Technology selection
        technologies = self._get_available_technologies()
        selected_tech = st.selectbox(
            "Select Technology",
            technologies,
            key="perf_tech_select"
        )

        if selected_tech:
            # Performance metrics
            if st.button("Evaluate Model", key="evaluate_model"):
                with st.spinner("Evaluating model performance..."):
                    try:
                        results = self.model_trainer.evaluate_model(selected_tech)
                        self._display_evaluation_results(results)
                    except Exception as e:
                        st.error(f"Evaluation failed: {str(e)}")

            # Feature importance
            self._show_feature_importance(selected_tech)

            # Performance comparison
            self._show_model_comparison(selected_tech)

    def _get_available_technologies(self) -> List[str]:
        """Get list of technologies with data."""
        # Return fallback technologies since data_loader doesn't exist
        # Could be enhanced later to query actual database
        return ['Python', 'JavaScript', 'Java', 'TypeScript', 'Go', 'Rust', 'React', 'Node.js', 'Docker', 'Kubernetes']

    def _get_current_features(self, technology: str) -> Optional[Dict[str, float]]:
        """Get current features for a technology."""
        try:
            # Get latest data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            data = self.data_collector.collect_historical_data(
                technology,
                start_date,
                end_date
            )

            if not data.empty:
                # Get the latest row
                latest = data.iloc[-1]

                # Extract features
                features = {
                    'current_value': latest.get('commit_count', 0),
                    'growth_rate_7d': latest.get('growth_rate_7d', 0),
                    'growth_rate_30d': latest.get('growth_rate_30d', 0),
                    'stars': latest.get('stars', 0),
                    'forks': latest.get('forks', 0),
                    'issues': latest.get('issues', 0),
                    'volatility_30d': latest.get('volatility_30d', 0),
                    'ma_7d': latest.get('ma_7d', 0),
                    'ma_30d': latest.get('ma_30d', 0)
                }

                return features

        except Exception as e:
            logger.error(f"Failed to get features for {technology}: {str(e)}")

        return None

    def _display_prediction(self, prediction: PredictionResult, threshold: float):
        """Display prediction results."""
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Predicted Value",
                f"{prediction.predicted_value:.2f}",
                delta=f"Trend: {prediction.trend_direction}"
            )

        with col2:
            confidence_color = "green" if prediction.confidence >= threshold else "orange"
            st.metric(
                "Confidence",
                f"{prediction.confidence:.1%}",
                delta=None,
                delta_color=confidence_color
            )

        with col3:
            st.metric(
                "Time Horizon",
                f"{prediction.time_horizon} days",
                delta=None
            )

        # Feature importance
        if prediction.feature_importance:
            st.subheader("Feature Importance")
            top_features = sorted(
                prediction.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            fig = px.bar(
                x=[v for v in dict(top_features).values()],
                y=[k for k in dict(top_features).keys()],
                orientation='h',
                title="Top Contributing Features"
            )
            fig.update_xaxis(title="Importance")
            fig.update_yaxis(title="Feature")
            st.plotly_chart(fig, use_container_width=True)

        # Prediction visualization
        st.subheader("Prediction Visualization")
        self._create_prediction_chart(prediction)

    def _display_anomaly_result(self, result: AnomalyResult):
        """Display anomaly detection results."""
        if result.is_outlier:
            st.error(
                f"⚠️ Anomaly Detected! (Score: {result.anomaly_score:.3f})\n"
                f"Type: {result.outlier_type}"
            )
        else:
            st.success(
                f"✅ No Anomaly Detected (Score: {result.anomaly_score:.3f})"
            )

        # Contributing features
        if result.contributing_features:
            st.subheader("Contributing Features")
            for feature, contribution in result.contributing_features.items():
                st.write(f"- {feature}: {contribution:.3f}")

        # Anomaly visualization
        st.subheader("Anomaly Analysis")
        self._create_anomaly_chart(result)

    def _display_training_results(self, results: Dict[str, Any]):
        """Display training results for a single technology."""
        st.success(f"✅ Training completed for {results['technology']}")

        # Best model info
        if 'best_model' in results:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best Model", results['best_model'])
            with col2:
                st.metric("CV Score", f"{results['best_score']:.4f}")

        # Model comparison
        if 'models' in results:
            st.subheader("Model Performance Comparison")
            model_data = []
            for model_name, model_info in results['models'].items():
                if 'error' not in model_info:
                    model_data.append({
                        'Model': model_name,
                        'CV Score': model_info.get('cv_score', 0),
                        'MAE': model_info.get('mae', 0),
                        'RMSE': model_info.get('rmse', 0),
                        'R²': model_info.get('r2', 0)
                    })

            if model_data:
                df = pd.DataFrame(model_data)
                st.dataframe(df, use_container_width=True)

    def _display_batch_training_results(self, results: Dict[str, Dict[str, Any]]):
        """Display batch training results."""
        st.success("✅ Batch training completed")

        # Summary table
        summary_data = []
        for tech, tech_results in results.items():
            if 'error' not in tech_results:
                summary_data.append({
                    'Technology': tech,
                    'Best Model': tech_results.get('best_model', 'N/A'),
                    'Best Score': tech_results.get('best_score', 'N/A'),
                    'Models Trained': tech_results.get('models_trained', 0)
                })

        if summary_data:
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)

        # Detailed results
        if st.checkbox("Show Detailed Results"):
            for tech, tech_results in results.items():
                if st.checkbox(f"{tech}", key=f"detail_{tech}"):
                    self._display_training_results(tech_results)

    def _display_evaluation_results(self, results: Dict[str, Any]):
        """Display model evaluation results."""
        st.subheader(f"Evaluation Results for {results['technology']}")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE", f"{results['mae']:.4f}")
        with col2:
            st.metric("RMSE", f"{results['rmse']:.4f}")
        with col3:
            st.metric("R²", f"{results['r2']:.4f}")
        with col4:
            st.metric("MAPE", f"{results['mape']:.2f}%")

        # Actual vs Predicted chart
        st.subheader("Actual vs Predicted")
        self._create_evaluation_chart(results)

    def _create_prediction_chart(self, prediction: PredictionResult):
        """Create a prediction visualization chart."""
        # Generate historical data points
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now() + timedelta(days=prediction.time_horizon),
            freq='D'
        )

        # Create chart
        fig = go.Figure()

        # Historical trend (simulated)
        historical_values = np.random.normal(
            loc=prediction.predicted_value * 0.8,
            scale=prediction.predicted_value * 0.1,
            size=30
        )

        fig.add_trace(go.Scatter(
            x=dates[:30],
            y=historical_values,
            name='Historical',
            line=dict(color='blue')
        ))

        # Prediction with confidence interval
        confidence_range = prediction.predicted_value * (1 - prediction.confidence)

        fig.add_trace(go.Scatter(
            x=dates[30:],
            y=[prediction.predicted_value] * prediction.time_horizon,
            name='Prediction',
            line=dict(color='red', dash='dash')
        ))

        # Confidence interval
        fig.add_trace(go.Scatter(
            x=dates[30:].tolist() + dates[30:].tolist()[::-1],
            y=([prediction.predicted_value + confidence_range] * prediction.time_horizon +
               [prediction.predicted_value - confidence_range] * prediction.time_horizon),
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))

        fig.update_layout(
            title=f"{prediction.technology} Trend Prediction",
            xaxis_title="Date",
            yaxis_title="Value"
        )

        st.plotly_chart(fig, use_container_width=True)

    def _create_anomaly_chart(self, result: AnomalyResult):
        """Create an anomaly visualization chart."""
        # Generate sample data
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=60),
            end=datetime.now(),
            freq='D'
        )

        values = np.random.normal(loc=50, scale=10, size=60)
        values[-1] = 100 if result.is_outlier else 50  # Anomaly point

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            name='Values',
            line=dict(color='blue')
        ))

        # Highlight anomaly
        if result.is_outlier:
            fig.add_trace(go.Scatter(
                x=[dates[-1]],
                y=[values[-1]],
                name='Anomaly',
                mode='markers',
                marker=dict(color='red', size=10)
            ))

        fig.update_layout(
            title=f"{result.technology} Anomaly Detection",
            xaxis_title="Date",
            yaxis_title="Value"
        )

        st.plotly_chart(fig, use_container_width=True)

    def _create_evaluation_chart(self, results: Dict[str, Any]):
        """Create an evaluation visualization chart."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=results['actual_values'],
            y=results['predicted_values'],
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', opacity=0.6)
        ))

        # Perfect prediction line
        min_val = min(results['actual_values'] + results['predicted_values'])
        max_val = max(results['actual_values'] + results['predicted_values'])

        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title="Actual vs Predicted Values",
            xaxis_title="Actual",
            yaxis_title="Predicted"
        )

        st.plotly_chart(fig, use_container_width=True)

    def _show_prediction_history(self, technology: str):
        """Show historical predictions for a technology."""
        # This would load from a predictions log
        st.subheader("Recent Predictions")
        st.info("Prediction history will be displayed here")

    def _show_recent_anomalies(self, technology: str):
        """Show recent anomalies for a technology."""
        # This would load from an anomalies log
        st.subheader("Recent Anomalies")
        st.info("Recent anomalies will be displayed here")

    def _show_model_info(self, technology: str):
        """Show model information for a technology."""
        st.subheader("Model Information")
        try:
            info = self.model_trainer.get_model_info(technology)

            if info['models']:
                for model_type, model_info in info['models'].items():
                    st.write(f"**{model_type} Model**")
                    st.json(model_info)
            else:
                st.info("No trained models found")
        except Exception as e:
            st.error(f"Failed to get model info: {str(e)}")

    def _show_training_history(self):
        """Show training history."""
        st.subheader("Training History")
        # This would load from training logs
        st.info("Training history will be displayed here")

    def _show_feature_importance(self, technology: str):
        """Show feature importance for a technology."""
        st.subheader("Feature Importance")
        # This would load from model artifacts
        st.info("Feature importance will be displayed here")

    def _show_model_comparison(self, technology: str):
        """Show model comparison for a technology."""
        st.subheader("Model Comparison")
        # This would compare all trained models
        st.info("Model comparison will be displayed here")