"""
Enhanced Predictive Analytics with Time Series Forecasting.

Extends the existing predictor with advanced time series forecasting
capabilities including ARIMA/SARIMA models, prediction intervals, and
uncertainty quantification.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
import warnings
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import existing predictor
from .predictor import PredictiveEngine, PredictionResult, AnomalyResult

# Import time series forecaster
from time_series.forecaster import TimeSeriesForecaster, ForecastResult

# Import cache manager
from cache_manager import CacheManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedPredictionResult(PredictionResult):
    """Enhanced prediction result with time series forecasting."""
    forecast_result: Optional[ForecastResult] = None
    forecast_method: str = 'auto'
    forecast_horizon: int = 30
    uncertainty_bounds: Optional[Dict[str, float]] = None
    model_diagnostics: Optional[Dict[str, Any]] = None
    seasonal_components: Optional[Dict[str, float]] = None

@dataclass
class TimeSeriesResult:
    """Container for time series analysis results."""
    timestamps: List[datetime]
    values: List[float]
    trend: Optional[List[float]] = None
    seasonal: Optional[List[float]] = None
    residuals: Optional[List[float]] = None
    trend_strength: Optional[float] = None
    seasonal_strength: Optional[float] = None
    period: Optional[int] = None

class EnhancedPredictiveEngine(PredictiveEngine):
    """
    Enhanced predictive engine with time series forecasting capabilities.

    Extends the original PredictiveEngine with:
    - ARIMA/SARIMA time series forecasting
    - Prediction intervals and uncertainty quantification
    - Seasonal decomposition and analysis
    - Multiple forecasting algorithms
    - Advanced model diagnostics
    """

    def __init__(self, models_dir: str = "src/phase7/models"):
        """
        Initialize the enhanced prediction engine.

        Args:
            models_dir: Directory to store trained models
        """
        # Initialize base class
        super().__init__(models_dir)

        # Initialize time series forecaster
        self.ts_forecaster = TimeSeriesForecaster(confidence_level=0.95)

        # Time series models storage
        self.ts_models = {}
        self.ts_model_metadata = {}

        # Enhanced parameters
        self.ts_forecast_params = {
            'confidence_level': 0.95,
            'bootstrap_samples': 1000,
            'auto_seasonal_detection': True,
            'max_seasonal_period': 365
        }

    def train_time_series_model(
        self,
        technology: str,
        time_series_data: Union[pd.Series, np.ndarray],
        target_column: Optional[str] = None,
        frequency: str = 'D',
        model_type: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Train a time series forecasting model.

        Args:
            technology: Name of the technology
            time_series_data: Time series data (index should be datetime)
            target_column: Column name if data is DataFrame
            frequency: Data frequency ('D', 'W', 'M', 'Q', 'Y')
            model_type: Model type ('arima', 'sarima', 'auto')

        Returns:
            Dictionary with training results and model diagnostics
        """
        logger.info(f"Training time series model for {technology}")

        # Prepare data
        if isinstance(time_series_data, dict):
            if target_column:
                time_series_data = time_series_data[target_column]
            else:
                # Convert dictionary to series, use first numeric column
                for key, value in time_series_data.items():
                    if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                        time_series_data = pd.Series(value)
                        break

        if isinstance(time_series_data, np.ndarray):
            time_series_data = pd.Series(time_series_data)

        # Ensure datetime index with specified frequency
        if not isinstance(time_series_data.index, pd.DatetimeIndex):
            time_series_data.index = pd.date_range(
                start='2020-01-01',
                periods=len(time_series_data),
                freq=frequency
            )
        else:
            # Reindex to ensure consistent frequency
            time_series_data = time_series_data.asfreq(frequency, method='ffill')

        # Remove missing values
        time_series_data = time_series_data.dropna()

        if len(time_series_data) < 50:
            logger.warning(f"Time series data too short for {technology}: {len(time_series_data)} points")
            return {'error': 'Insufficient data for time series modeling'}

        try:
            # Train ARIMA model with auto-selection
            auto_result = self.ts_forecaster.auto_arima(
                time_series_data,
                seasonal=True,
                m=self._detect_seasonal_period(time_series_data),
                information_criterion='aic'
            )

            # Store model
            model_key = f"{technology}_ts"
            self.ts_models[model_key] = auto_result['model']

            # Store metadata
            self.ts_model_metadata[model_key] = {
                'technology': technology,
                'model_type': 'sarima' if auto_result['params'] and len(auto_result['params']) > 3 else 'arima',
                'params': auto_result['params'],
                'aic': auto_result.get('aic'),
                'bic': auto_result.get('bic'),
                'log_likelihood': auto_result.get('log_likelihood'),
                'training_data_points': len(time_series_data),
                'frequency': frequency,
                'trained_at': datetime.now().isoformat(),
                'diagnostics': auto_result.get('diagnostics', {})
            }

            # Perform seasonal decomposition if applicable
            seasonal_result = None
            if self._detect_seasonal_period(time_series_data):
                try:
                    seasonal_result = self.ts_forecaster.seasonal_decompose(
                        time_series_data,
                        period=self._detect_seasonal_period(time_series_data)
                    )
                except Exception as e:
                    logger.warning(f"Seasonal decomposition failed for {technology}: {str(e)}")

            logger.info(f"Time series model trained for {technology}. "
                       f"AIC: {auto_result.get('aic', 'N/A'):.2f}")

            return {
                'model_key': model_key,
                'aic': auto_result.get('aic'),
                'bic': auto_result.get('bic'),
                'params': auto_result['params'],
                'diagnostics': auto_result.get('diagnostics'),
                'seasonal_analysis': seasonal_result,
                'training_samples': len(time_series_data),
                'model_type': 'arima/sarima'
            }

        except Exception as e:
            logger.error(f"Time series model training failed for {technology}: {str(e)}")
            return {'error': str(e)}

    def forecast_time_series(
        self,
        technology: str,
        current_data: Dict[str, float],
        historical_data: Optional[Union[pd.Series, np.ndarray]] = None,
        steps: int = 30,
        method: str = 'auto',
        confidence_level: float = 0.95
    ) -> EnhancedPredictionResult:
        """
        Generate time series forecast with uncertainty.

        Args:
            technology: Name of the technology
            current_data: Current feature values
            historical_data: Historical time series data
            steps: Number of steps to forecast
            method: Forecasting method
            confidence_level: Confidence level for prediction intervals

        Returns:
            EnhancedPredictionResult with forecast and uncertainty
        """
        # Get basic trend prediction from base class
        base_result = super().predict_trend(technology, current_data, time_horizon=steps)

        # Prepare time series data
        if historical_data is None:
            # Generate synthetic historical data from current features
            historical_data = self._generate_historical_from_features(
                current_data, periods=max(100, steps * 2)
            )

        # Convert to Series if needed
        if isinstance(historical_data, np.ndarray):
            historical_data = pd.Series(historical_data)

        # Ensure datetime index
        if not isinstance(historical_data.index, pd.DatetimeIndex):
            historical_data.index = pd.date_range(
                start=datetime.now() - timedelta(days=len(historical_data)),
                periods=len(historical_data),
                freq='D'
            )

        # Remove missing values
        historical_data = historical_data.dropna()

        # Generate time series forecast
        try:
            ts_forecast = self.ts_forecaster.forecast(
                historical_data,
                steps=steps,
                method=method,
                confidence_intervals=True
            )

            # Combine with base prediction
            # Use time series forecast if available, otherwise use base prediction
            if len(ts_forecast.predicted_values) > 0:
                predicted_values = ts_forecast.predicted_values
            else:
                # Extrapolate from base prediction
                predicted_values = [base_result.predicted_value +
                                   (i * 0.1 * base_result.predicted_value) for i in range(steps)]

            # Calculate uncertainty bounds
            uncertainty_bounds = self._calculate_uncertainty_bounds(
                ts_forecast, base_result, historical_data
            )

            return EnhancedPredictionResult(
                timestamp=base_result.timestamp,
                technology=technology,
                metric_type=base_result.metric_type,
                predicted_value=predicted_values[0] if predicted_values else base_result.predicted_value,
                confidence=min(base_result.confidence, confidence_level),
                time_horizon=steps,
                feature_importance=base_result.feature_importance,
                trend_direction=base_result.trend_direction,
                anomaly_score=base_result.anomaly_score,
                forecast_result=ts_forecast,
                forecast_method=method,
                forecast_horizon=steps,
                uncertainty_bounds=uncertainty_bounds,
                model_diagnostics=ts_forecast.model_metrics,
                seasonal_components=self._extract_seasonal_components(ts_forecast)
            )

        except Exception as e:
            logger.error(f"Time series forecasting failed for {technology}: {str(e)}")
            # Return enhanced result with only base prediction
            return EnhancedPredictionResult(
                timestamp=base_result.timestamp,
                technology=technology,
                metric_type=base_result.metric_type,
                predicted_value=base_result.predicted_value,
                confidence=base_result.confidence,
                time_horizon=steps,
                feature_importance=base_result.feature_importance,
                trend_direction=base_result.trend_direction,
                anomaly_score=base_result.anomaly_score,
                forecast_method=method,
                forecast_horizon=steps
            )

    def analyze_time_series_patterns(
        self,
        technology: str,
        data: Union[pd.Series, np.ndarray, Dict[str, float]],
        frequency: str = 'D'
    ) -> Dict[str, Any]:
        """
        Analyze time series patterns and characteristics.

        Args:
            technology: Name of the technology
            data: Time series data or features
            frequency: Data frequency

        Returns:
            Dictionary with pattern analysis results
        """
        # Prepare time series data
        if isinstance(data, dict):
            # Extract time series from features
            if 'historical_values' in data:
                ts_data = pd.Series(data['historical_values'])
            else:
                # Find the most time-like feature
                time_features = [k for k, v in data.items()
                                 if isinstance(v, (list, np.ndarray)) and len(v) > 10]
                if time_features:
                    ts_data = pd.Series(data[time_features[0]])
                else:
                    return {'error': 'No suitable time series data found'}

        elif isinstance(data, np.ndarray):
            ts_data = pd.Series(data)
        else:
            ts_data = data

        # Ensure datetime index
        if not isinstance(ts_data.index, pd.DatetimeIndex):
            ts_data.index = pd.date_range(
                start=datetime.now() - timedelta(days=len(ts_data)),
                periods=len(ts_data),
                freq=frequency
            )

        ts_data = ts_data.dropna()

        if len(ts_data) < 20:
            return {'error': 'Insufficient data for pattern analysis'}

        results = {
            'technology': technology,
            'data_points': len(ts_data),
            'date_range': (ts_data.index.min(), ts_data.index.max()),
            'frequency': frequency
        }

        # Basic statistics
        results['statistics'] = {
            'mean': float(ts_data.mean()),
            'std': float(ts_data.std()),
            'min': float(ts_data.min()),
            'max': float(ts_data.max()),
            'skewness': float(stats.skew(ts_data.dropna())),
            'kurtosis': float(stats.kurtosis(ts_data.dropna()))
        }

        # Trend analysis
        results['trend'] = self._analyze_trend(ts_data)

        # Seasonal decomposition
        seasonal_period = self._detect_seasonal_period(ts_data)
        if seasonal_period and seasonal_period > 1:
            try:
                seasonal_result = self.ts_forecaster.seasonal_decompose(
                    ts_data, period=seasonal_period
                )
                results['seasonal_decomposition'] = {
                    'period': seasonal_period,
                    'trend_strength': seasonal_result.get('trend_strength', 0),
                    'seasonal_strength': seasonal_result.get('seasonal_strength', 0),
                    'model': seasonal_result.get('model', 'additive')
                }
            except Exception as e:
                logger.warning(f"Seasonal decomposition failed: {str(e)}")
                results['seasonal_decomposition'] = None

        # Stationarity tests
        results['stationarity'] = self._test_stationarity(ts_data)

        # Autocorrelation analysis
        results['autocorrelation'] = self._analyze_autocorrelation(ts_data)

        # Change point detection (simplified)
        results['change_points'] = self._detect_change_points(ts_data)

        return results

    def get_time_series_model_info(self, technology: str) -> Dict[str, Any]:
        """Get information about trained time series models."""
        model_key = f"{technology}_ts"

        if model_key not in self.ts_models:
            return {
                'technology': technology,
                'model_trained': False,
                'message': 'No time series model trained'
            }

        if model_key in self.ts_model_metadata:
            return self.ts_model_metadata[model_key]

        return {
            'technology': technology,
            'model_trained': True,
            'message': 'Model metadata not found'
        }

    def _detect_seasonal_period(self, data: pd.Series) -> Optional[int]:
        """Detect seasonal period in time series."""
        try:
            # Try FFT method
            period = self.ts_forecaster._detect_seasonal_period(data)
            if period and 2 <= period <= min(365, len(data) // 3):
                return period

            # Try ACF method (simplified)
            # Look for peaks in autocorrelation
            from statsmodels.tsa.stattools import acf
            autocorr = acf(data.dropna(), nlags=min(100, len(data)//2), fft=False))

            # Find significant peaks beyond lag 1
            significant_peaks = []
            for lag in range(2, len(autocorr)):
                if autocorr[lag] > 0.3:  # Threshold for significance
                    significant_peaks.append(lag)

            if significant_peaks:
                # Return most common period
                from collections import Counter
                period_counter = Counter(significant_peaks)
                most_common = period_counter.most_common(1)[0]
                if 2 <= most_common <= min(52, len(data)//4):
                    return most_common

        except Exception:
            pass

        return None

    def _generate_historical_from_features(
        self,
        features: Dict[str, float],
        periods: int = 100
    ) -> pd.Series:
        """Generate synthetic historical data from current features."""
        # Use current value as base with some random variation
        base_value = features.get('current_value', features.get('value', 1.0))
        growth_rate = features.get('growth_rate', 0.01)
        volatility = features.get('volatility', 0.1)

        # Generate synthetic time series
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(0, volatility, periods)
        trend = np.linspace(0, growth_rate * periods, periods)

        values = base_value * (1 + trend) * (1 + returns)
        values = np.maximum(values, 0.01)  # Ensure positive values

        return pd.Series(values)

    def _calculate_uncertainty_bounds(
        self,
        forecast_result: ForecastResult,
        base_result: PredictionResult,
        historical_data: pd.Series
    ) -> Dict[str, float]:
        """Calculate uncertainty bounds combining multiple sources."""
        bounds = {}

        # From forecast confidence intervals
        if forecast_result.confidence_intervals:
            forecast_width = np.mean([
                ci[1] - ci[0] for ci in forecast_result.confidence_intervals
            ])
            bounds['forecast_uncertainty'] = forecast_width / 2
            bounds['confidence_level'] = self.ts_forecaster.confidence_level

        # From historical volatility
        if len(historical_data) > 10:
            historical_volatility = np.std(np.diff(historical_data))
            bounds['historical_volatility'] = historical_volatility
            bounds['prediction_error'] = historical_volatility * np.sqrt(30)  # 30-day prediction

        # From model confidence
        if hasattr(base_result, 'confidence'):
            bounds['model_confidence'] = base_result.confidence
            bounds['prediction_confidence'] = 1 - base_result.confidence

        return bounds

    def _extract_seasonal_components(self, forecast_result: ForecastResult) -> Dict[str, float]:
        """Extract seasonal components from forecast diagnostics."""
        components = {}

        if forecast_result.model_metrics:
            if 'aic' in forecast_result.model_metrics:
                components['aic'] = forecast_result.model_metrics['aic']
            if 'bic' in forecast_result.model_metrics:
                components['bic'] = forecast_result.model_metrics['bic']

        return components

    def _analyze_trend(self, data: pd.Series) -> Dict[str, Any]:
        """Analyze trend in time series."""
        # Simple linear trend analysis
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)

        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'trend_direction': 'up' if slope > 0 else 'down' if slope < 0 else 'stable',
            'trend_strength': abs(r_value),
            'is_significant': p_value < 0.05
        }

    def _test_stationarity(self, data: pd.Series) -> Dict[str, Any]:
        """Test for stationarity."""
        results = {}

        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(data)
            results['adf'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05,
                'critical_values': dict(zip(['1%', '5%', '10%'], adf_result[4]))
            }
        except Exception:
            results['adf'] = {'error': 'ADF test failed'}

        return results

    def _analyze_autocorrelation(self, data: pd.Series) -> Dict[str, Any]:
        """Analyze autocorrelation patterns."""
        try:
            from statsmodels.tsa.stattools import acf, pacf

            # Autocorrelation function
            acf_result = acf(data.dropna(), nlags=20, alpha=0.05)
            pacf_result = pacf(data.dropna(), nlags=20, alpha=0.05)

            return {
                'acf': acf_result.tolist(),
                'pacf': pacf_result.tolist(),
                'significant_lags_acf': np.where(np.abs(acf_result[1:]) > 0.2)[0].tolist(),
                'significant_lags_pacf': np.where(np.abs(pacf_result[1:]) > 0.2)[0].tolist()
            }
        except Exception as e:
            return {'error': f'Autocorrelation analysis failed: {str(e)}'}

    def _detect_change_points(self, data: pd.Series) -> List[int]:
        """Detect potential change points (simplified)."""
        try:
            # Simple change point detection using differences
            diff = np.diff(data)
            diff_abs = np.abs(diff)

            # Use median absolute deviation as threshold
            threshold = 3 * np.median(diff_abs)
            change_points = np.where(diff_abs > threshold)[0].tolist()

            # Adjust for indexing (diff shifts by 1)
            change_points = [cp + 1 for cp in change_points]

            return change_points
        except:
            return []

    def get_model_comparison(
        self,
        technology: str,
        test_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compare different forecasting models for a technology.

        Args:
            technology: Name of technology
            test_data: Test data for model evaluation

        Returns:
            Dictionary with model comparison results
        """
        results = {
            'technology': technology,
            'models': {},
            'comparison': {},
            'recommendation': None
        }

        # Test different forecasting methods
        methods = ['arima', 'ets', 'ensemble']
        time_series = test_data.get('time_series', test_data.iloc[:, 0])  # First column as time series

        if isinstance(time_series, pd.Series):
            time_series = time_series.dropna()

        for method in methods:
            try:
                forecast = self.ts_forecaster.forecast(
                    time_series,
                    steps=30,
                    method=method
                )

                # Calculate accuracy against actual values if available
                accuracy = {}
                if 'actual_values' in test_data.columns:
                    actual = test_data['actual_values'].dropna()
                    if len(actual) >= len(forecast.predicted_values):
                        actual_trimmed = actual.iloc[:len(forecast.predicted_values)]
                        accuracy = self.ts_forecaster.forecast_accuracy(
                            actual_trimmed,
                            pd.Series(forecast.predicted_values)
                        )

                results['models'][method] = {
                    'forecast': forecast.to_dict() if hasattr(forecast, 'to_dict') else str(forecast),
                    'accuracy': accuracy,
                    'model_metrics': forecast.model_metrics
                }

            except Exception as e:
                results['models'][method] = {
                    'error': str(e)
                }

        # Compare models
        if 'accuracy' in results['models'].get('arima', {}) and 'mae' in results['models']['arima']['accuracy']:
            best_model = 'arima'
            best_mae = results['models']['arima']['accuracy']['mae']

            for method, model_data in results['models'].items():
                if 'accuracy' in model_data and 'mae' in model_data['accuracy']:
                    if model_data['accuracy']['mae'] < best_mae:
                        best_model = method
                        best_mae = model_data['accuracy']['mae']

            results['comparison'] = {
                'best_model': best_model,
                'best_mae': best_mae,
                'all_models': list(results['models'].keys())
            }

            results['recommendation'] = f"Use {best_model} for {technology} forecasting"

        return results

    def export_time_series_results(
        self,
        technology: str,
        format: str = 'json'
    ) -> Union[str, Dict]:
        """
        Export time series analysis results.

        Args:
            technology: Technology name
            format: Export format ('json' or 'dict')

        Returns:
            Exported results in specified format
        """
        # Collect all time series related information
        results = {
            'technology': technology,
            'model_info': self.get_time_series_model_info(technology),
            'base_model_info': self.get_model_info(technology),
            'timestamp': datetime.now().isoformat()
        }

        if format == 'json':
            import json
            return json.dumps(results, indent=2, default=str)
        else:
            return results

    def cleanup_cache(self) -> None:
        """Clean up all caches."""
        super().cleanup_cache()
        # Clear any time series specific cache
        # (Implement if needed)