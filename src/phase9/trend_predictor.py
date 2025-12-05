"""
Advanced Trend Predictor for Phase 9
Sophisticated time series forecasting and trend prediction
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedTrendPredictor:
    """
    Advanced trend predictor using machine learning
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_column = 'price'
        self.logger = logging.getLogger(__name__)
        self.logger.info("Advanced Trend Predictor initialized")

    def train_predictive_models(self, df: pd.DataFrame, target_col: str = 'price') -> Dict[str, Any]:
        """
        Train multiple predictive models

        Args:
            df: Training data
            target_col: Target column for prediction

        Returns:
            Dictionary with training results
        """
        try:
            if df.empty:
                return {"error": "Insufficient data for model training"}

            self.target_column = target_col
            self._prepare_features(df)

            # Prepare data
            X = df[self.feature_columns]
            y = df[target_col]

            # Split data for training
            tscv = TimeSeriesSplit(n_splits=5)
            train_indices, test_indices = next(tscv.split(X, y))

            X_train = X.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_train = y.iloc[train_indices]
            y_test = y.iloc[test_indices]

            # Train multiple models
            models_trained = {}

            # Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            models_trained['random_forest'] = {
                'model': rf,
                'mae': mean_absolute_error(y_test, rf.predict(X_test)),
                'r2': r2_score(y_test, rf.predict(X_test))
            }

            # Gradient Boosting
            gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            gb.fit(X_train, y_train)
            models_trained['gradient_boosting'] = {
                'model': gb,
                'mae': mean_absolute_error(y_test, gb.predict(X_test)),
                'r2': r2_score(y_test, gb.predict(X_test))
            }

            # Linear Model (baseline)
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            models_trained['linear_regression'] = {
                'model': lr_model,
                'mae': mean_absolute_error(y_test, lr_model.predict(X_test)),
                'r2': r2_score(y_test, lr_model.predict(X_test))
            }

            self.models = models_trained
            self.logger.info(f"Training completed for {len(models_trained)} models")

            return {
                'models_trained': models_trained,
                'feature_importance': self._calculate_feature_importance(models_trained),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'training_date': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {"error": str(e)}

    def predict_trends(self, df: pd.DataFrame, periods: int = 30, model_type: str = 'ensemble') -> Dict[str, Any]:
        """
        Predict future trends using trained models

        Args:
            df: Historical data for prediction
            periods: Number of periods to predict
            model_type: Type of model to use ('random_forest', 'gradient_boosting', 'linear_regression', 'ensemble')

        Returns:
            Dictionary with prediction results
        """
        try:
            if df.empty or not self.models:
                return {"error": "No trained models available for prediction"}

            self._prepare_features(df)
            X = df[self.feature_columns]

            # Select model
            if model_type == 'ensemble':
                predictions = self._ensemble_predict(X)
            elif model_type in self.models:
                model = self.models[model_type]['model']
                predictions = model.predict(X)
            else:
                return {"error": f"Unknown model type: {model_type}"}

            # Extend predictions to future periods
            future_predictions = self._extend_predictions(predictions, periods)

            return {
                'predictions': future_predictions,
                'model_type': model_type,
                'prediction_periods': len(future_predictions),
                'confidence_intervals': self._calculate_confidence_intervals(predictions),
                'prediction_date': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Trend prediction failed: {e}")
            return {"error": str(e)}

    def _prepare_features(self, df: pd.DataFrame) -> None:
        """Prepare features for machine learning"""
        # Select numeric columns for features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Important features for tech trend prediction
        important_features = ['score', 'descendants', 'sentiment_score']

        # Filter available features
        available_features = [col for col in important_features if col in numeric_cols]
        self.feature_columns = available_features + [col for col in numeric_cols if col not in self.feature_columns]

        # Scale features
        if self.feature_columns:
            X = df[self.feature_columns]
            if not hasattr(self, 'scaler'):
                self.scaler = StandardScaler()
                df[self.feature_columns] = self.scaler.fit_transform(X)
            else:
                df[self.feature_columns] = self.scaler.transform(X)

    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction from multiple models"""
        predictions = []

        for model_name, model_data in self.models.items():
            prediction = model_data['model'].predict(X)
            predictions.append(prediction)

        # Weighted average (weights based on model performance)
        weights = [model_data['r2'] for model_data in self.models.values()]
        weights = np.array(weights) / np.sum(weights)

        return np.average(predictions, axis=0, weights=weights)

    def _extend_predictions(self, base_predictions: np.ndarray, periods: int) -> np.ndarray:
        """Extend predictions to future periods"""
        extended_predictions = []

        for i in range(periods):
            # Simple linear extension
            if len(base_predictions) > 0:
                trend = base_predictions[-1] - base_predictions[0] if len(base_predictions) > 1 else 0
                next_prediction = base_predictions[-1] + trend * (i + 1) * 0.1
                extended_predictions.append(next_prediction)
            else:
                extended_predictions.append(0)

        return np.array(extended_predictions)

    def _calculate_confidence_intervals(self, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals for predictions"""
        std_error = np.std(predictions) if len(predictions) > 1 else np.ones(len(predictions))

        z_scores = [1.96, 1.645]  # 95% and 90% confidence
        confidence_intervals = {
            'lower_95': predictions - z_scores[0] * std_error,
            'upper_95': predictions + z_scores[0] * std_error,
            'lower_90': predictions - z_scores[1] * std_error,
            'upper_90': predictions + z_scores[1] * std_error
        }

        return confidence_intervals

    def _calculate_feature_importance(self, models_trained: Dict) -> Dict[str, Dict]:
        """Calculate feature importance across models"""
        importance_scores = {}

        for model_name, model_data in models_trained.items():
            model = model_data['model']

            if hasattr(model, 'feature_importances_'):
                # For tree-based models
                importance_dict = dict(zip(self.feature_columns, model.feature_importances_))
                importance_scores[model_name] = importance_dict
            elif hasattr(model, 'coef_'):
                # For linear models
                importance_dict = dict(zip(self.feature_columns, np.abs(model.coef_)))
                importance_scores[model_name] = importance_dict
            else:
                importance_scores[model_name] = {}

        return importance_scores

    def analyze_prediction_accuracy(self, test_data: pd.DataFrame, target_col: str = 'price') -> Dict[str, Any]:
        """
        Analyze prediction accuracy across all models

        Args:
            test_data: Test data for evaluation
            target_col: Target column for comparison

        Returns:
            Dictionary with accuracy metrics
        """
        try:
            if test_data.empty or not self.models:
                return {"error": "No trained models available for accuracy analysis"}

            self._prepare_features(test_data)
            X_test = test_data[self.feature_columns]
            y_actual = test_data[target_col]

            accuracy_analysis = {}

            for model_name, model_data in self.models.items():
                model = model_data['model']
                y_pred = model.predict(X_test)

                accuracy_analysis[model_name] = {
                    'mae': mean_absolute_error(y_actual, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_actual, y_pred)),
                    'r2_score': r2_score(y_actual, y_pred),
                    'mape': np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
                }

            return accuracy_analysis

        except Exception as e:
            logger.error(f"Accuracy analysis failed: {e}")
            return {"error": str(e)}

    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all trained models"""
        if not self.models:
            return {"status": "no_models_trained"}

        performance = {}
        for model_name, model_data in self.models.items():
            performance[model_name] = {
                'mae': model_data.get('mae', 0),
                'r2_score': model_data.get('r2', 0),
                'training_date': model_data.get('training_date', 'unknown')
            }

        best_model = max(performance['performance_summary'], key=performance['performance_summary'],
                        default=None, key=lambda x: performance['performance_summary'][x].get('r2_score', 0)) if performance else None

        return {
            'model_count': len(self.models),
            'available_models': list(self.models.keys()),
            'performance_summary': performance,
            'best_model': best_model
        }

    def predict_topic_trends(self, df: pd.DataFrame, periods: int = 30) -> Dict[str, Any]:
        """
        Predict technology topic trends

        Args:
            df: DataFrame with topic data
            periods: Number of periods to predict

        Returns:
            Dictionary with topic trend predictions
        """
        try:
            if df.empty or 'topic_keyword' not in df.columns:
                return {"error": "Insufficient topic data for prediction"}

            # Analyze topic trends
            topic_trends = {}

            for topic in df['topic_keyword'].unique():
                topic_data = df[df['topic_keyword'] == topic].copy()

                if len(topic_data) >= 5:  # Need sufficient data
                    # Sort by time if available
                    if 'time' in topic_data.columns:
                        topic_data = topic_data.sort_values('time')

                    # Simple trend prediction based on historical data
                    scores = topic_data['score'].values if 'score' in topic_data.columns else None

                    if scores is not None and len(scores) > 1:
                        # Calculate trend
                        trend = np.polyfit(range(len(scores)), scores, 1)[0]

                        # Predict future values
                        future_scores = []
                        last_score = scores[-1]

                        for i in range(periods):
                            predicted_score = last_score + trend * (i + 1)
                            future_scores.append(max(0, predicted_score))  # Ensure non-negative

                        topic_trends[topic] = {
                            'historical_scores': scores.tolist(),
                            'predicted_scores': future_scores,
                            'trend_direction': 'increasing' if trend > 0 else 'decreasing',
                            'trend_strength': abs(trend),
                            'confidence': min(0.9, len(scores) / 10.0)  # Confidence based on data points
                        }

            # Sort topics by trend strength
            sorted_trends = dict(sorted(topic_trends.items(),
                                       key=lambda x: x[1]['trend_strength'],
                                       reverse=True))

            return {
                'topic_trends': sorted_trends,
                'emerging_topics': [k for k, v in sorted_trends.items() if v['trend_direction'] == 'increasing'],
                'declining_topics': [k for k, v in sorted_trends.items() if v['trend_direction'] == 'decreasing'],
                'analysis_date': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Topic trend prediction failed: {e}")
            return {"error": str(e)}

    def generate_trend_forecast_report(self, df: pd.DataFrame, periods: int = 30) -> Dict[str, Any]:
        """
        Generate comprehensive trend forecast report

        Args:
            df: DataFrame with market data
            periods: Number of periods to forecast

        Returns:
            Dictionary with comprehensive trend report
        """
        try:
            # Train models if not already trained
            if not self.models:
                training_results = self.train_predictive_models(df)

            # Generate predictions
            predictions = self.predict_trends(df, periods)

            # Analyze topic trends
            topic_trends = self.predict_topic_trends(df, periods)

            # Create forecast report
            report = {
                'executive_summary': self._generate_forecast_summary(predictions, topic_trends),
                'market_predictions': predictions,
                'topic_forecasts': topic_trends,
                'key_insights': self._extract_key_insights(predictions, topic_trends),
                'risk_factors': self._identify_risk_factors(predictions, topic_trends),
                'opportunities': self._identify_opportunities(predictions, topic_trends),
                'recommendations': self._generate_forecast_recommendations(predictions, topic_trends),
                'confidence_assessment': self._assess_prediction_confidence(predictions, topic_trends),
                'methodology': {
                    'models_used': list(self.models.keys()) if self.models else [],
                    'features_used': self.feature_columns,
                    'forecast_periods': periods
                }
            }

            return report

        except Exception as e:
            logger.error(f"Trend forecast report generation failed: {e}")
            return {"error": str(e)}

    def _generate_forecast_summary(self, predictions: Dict, topic_trends: Dict) -> str:
        """Generate executive summary of forecast"""
        summary_parts = []

        if 'predictions' in predictions:
            pred_array = predictions['predictions']
            if len(pred_array) > 0:
                avg_prediction = np.mean(pred_array)
                summary_parts.append(f"Average market prediction: {avg_prediction:.2f}")

        if 'emerging_topics' in topic_trends:
            emerging = topic_trends['emerging_topics'][:3]
            if emerging:
                summary_parts.append(f"Key emerging topics: {', '.join(emerging)}")

        summary_parts.append(f"Forecast generated with {len(self.models)} prediction models")

        return ". ".join(summary_parts) if summary_parts else "Forecast summary generation pending"

    def _extract_key_insights(self, predictions: Dict, topic_trends: Dict) -> List[str]:
        """Extract key insights from predictions"""
        insights = []

        # Market insights
        if 'predictions' in predictions:
            pred_array = predictions['predictions']
            if len(pred_array) > 1:
                trend = pred_array[-1] - pred_array[0]
                if trend > 0:
                    insights.append("Market trend predicted to increase")
                elif trend < 0:
                    insights.append("Market trend predicted to decrease")
                else:
                    insights.append("Market trend predicted to remain stable")

        # Topic insights
        if 'emerging_topics' in topic_trends and topic_trends['emerging_topics']:
            insights.append(f"{len(topic_trends['emerging_topics'])} topics showing growth momentum")

        if 'declining_topics' in topic_trends and topic_trends['declining_topics']:
            insights.append(f"{len(topic_trends['declining_topics'])} topics showing declining interest")

        return insights

    def _identify_risk_factors(self, predictions: Dict, topic_trends: Dict) -> List[str]:
        """Identify potential risk factors"""
        risks = []

        # Prediction volatility risks
        if 'confidence_intervals' in predictions:
            intervals = predictions['confidence_intervals']
            if 'lower_95' in intervals and 'upper_95' in intervals:
                prediction_range = intervals['upper_95'] - intervals['lower_95']
                if np.mean(prediction_range) > np.mean(predictions['predictions']) * 0.5:
                    risks.append("High prediction volatility detected")

        # Topic concentration risks
        if 'topic_forecasts' in topic_trends:
            total_topics = len(topic_trends['topic_forecasts'])
            if total_topics < 5:
                risks.append("Limited topic diversity may indicate market concentration")

        return risks

    def _identify_opportunities(self, predictions: Dict, topic_trends: Dict) -> List[str]:
        """Identify opportunities from predictions"""
        opportunities = []

        # Growth opportunities
        if 'emerging_topics' in topic_trends and topic_trends['emerging_topics']:
            top_emerging = topic_trends['emerging_topics'][:3]
            opportunities.append(f"Growth opportunities in: {', '.join(top_emerging)}")

        # Market timing opportunities
        if 'predictions' in predictions:
            pred_array = predictions['predictions']
            if len(pred_array) > 1:
                if pred_array[-1] > pred_array[0] * 1.2:  # 20% growth
                    opportunities.append("Strong market growth predicted - consider expansion strategies")

        return opportunities

    def _generate_forecast_recommendations(self, predictions: Dict, topic_trends: Dict) -> List[str]:
        """Generate strategic recommendations based on forecasts"""
        recommendations = []

        if 'emerging_topics' in topic_trends and topic_trends['emerging_topics']:
            recommendations.append("Invest in emerging technologies to maintain competitive advantage")

        if 'declining_topics' in topic_trends and topic_trends['declining_topics']:
            recommendations.append("Monitor declining topics for potential portfolio adjustments")

        if self.models:
            best_model = max(self.models.keys(), key=lambda x: self.models[x].get('r2', 0))
            recommendations.append(f"Focus on {best_model} predictions for highest accuracy")

        return recommendations

    def _assess_prediction_confidence(self, predictions: Dict, topic_trends: Dict) -> Dict[str, Any]:
        """Assess overall confidence in predictions"""
        confidence_assessment = {
            'overall_confidence': 'moderate',
            'model_confidence': 0.0,
            'data_confidence': 0.0,
            'factors_affecting_confidence': []
        }

        # Model confidence based on R² scores
        if self.models:
            r2_scores = [model.get('r2', 0) for model in self.models.values()]
            avg_r2 = np.mean(r2_scores)
            confidence_assessment['model_confidence'] = avg_r2

        # Data confidence based on available data
        if 'topic_forecasts' in topic_trends:
            avg_confidence = np.mean([t.get('confidence', 0) for t in topic_trends['topic_forecasts'].values()])
            confidence_assessment['data_confidence'] = avg_confidence

        # Overall confidence
        overall = (confidence_assessment['model_confidence'] + confidence_assessment['data_confidence']) / 2
        if overall > 0.7:
            confidence_assessment['overall_confidence'] = 'high'
        elif overall > 0.5:
            confidence_assessment['overall_confidence'] = 'moderate'
        else:
            confidence_assessment['overall_confidence'] = 'low'

        return confidence_assessment