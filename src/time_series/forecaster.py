"""
Time Series Forecaster with Uncertainty Quantification.

Implements ARIMA/SARIMA models with prediction intervals,
bootstrap methods for uncertainty estimation, and various
forecasting techniques for time series data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import base64
from io import BytesIO

# Import statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class ForecastResult:
    """Container for forecast results with uncertainty."""
    timestamps: List[datetime]
    predicted_values: List[float]
    confidence_intervals: List[Tuple[float, float]]  # (lower, upper)
    prediction_method: str
    model_params: Dict[str, Any]
    model_metrics: Dict[str, float]
    residuals: Optional[np.ndarray] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecast results to DataFrame."""
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'predicted_value': self.predicted_values,
            'ci_lower': [ci[0] for ci in self.confidence_intervals],
            'ci_upper': [ci[1] for ci in self.confidence_intervals]
        })
        return df

class TimeSeriesForecaster:
    """
    Advanced time series forecasting with uncertainty quantification.

    Features:
    - ARIMA/SARIMA models with automatic parameter selection
    - Bootstrap methods for prediction intervals
    - Multiple forecasting algorithms (ARIMA, ETS, Prophet-like)
    - Seasonal decomposition and trend analysis
    - Model diagnostics and validation
    - Visualization of forecasts with confidence bands
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize the time series forecaster.

        Args:
            confidence_level: Confidence level for prediction intervals (0-1)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.fitted_models = {}
        self.forecast_history = []

        # Model selection criteria
        self.model_criteria = ['aic', 'bic']

    def auto_arima(
        self,
        data: Union[pd.Series, np.ndarray],
        seasonal: bool = False,
        m: Optional[int] = None,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        max_P: int = 2,
        max_D: int = 1,
        max_Q: int = 2,
        information_criterion: str = 'aic'
    ) -> Dict[str, Any]:
        """
        Automatic ARIMA model selection.

        Args:
            data: Time series data
            seasonal: Whether to fit seasonal model
            m: Seasonal period
            max_p: Maximum AR order
            max_d: Maximum differencing order
            max_q: Maximum MA order
            max_P: Maximum seasonal AR order
            max_D: Maximum seasonal differencing order
            max_Q: Maximum seasonal MA order
            information_criterion: Information criterion for model selection

        Returns:
            Dictionary with model selection results
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.date_range(
                start='2020-01-01',
                periods=len(data),
                freq='D'
            )

        best_model = None
        best_ic = np.inf
        best_params = None
        model_results = []

        # Grid search for best parameters
        p_range = range(0, max_p + 1)
        d_range = range(0, max_d + 1)
        q_range = range(0, max_q + 1)

        if seasonal and m:
            P_range = range(0, max_P + 1)
            D_range = range(0, max_D + 1)
            Q_range = range(0, max_Q + 1)
        else:
            P_range = [0]
            D_range = [0]
            Q_range = [0]

        total_combinations = len(p_range) * len(d_range) * len(q_range) * len(P_range) * len(D_range) * len(Q_range)

        print(f"Testing {total_combinations} model combinations...")

        for p in p_range:
            for d in d_range:
                for q in q_range:
                    for P in P_range:
                        for D in D_range:
                            for Q in Q_range:
                                try:
                                    if seasonal and m:
                                        model = SARIMAX(
                                            data,
                                            order=(p, d, q),
                                            seasonal_order=(P, D, Q, m),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False
                                        )
                                    else:
                                        model = ARIMA(
                                            data,
                                            order=(p, d, q)
                                        )

                                    fitted_model = model.fit(disp=False)
                                    ic = getattr(fitted_model, information_criterion)

                                    model_results.append({
                                        'params': (p, d, q) if not seasonal else (p, d, q, P, D, Q, m),
                                        'aic': fitted_model.aic if hasattr(fitted_model, 'aic') else None,
                                        'bic': fitted_model.bic if hasattr(fitted_model, 'bic') else None,
                                        'log_likelihood': fitted_model.llf if hasattr(fitted_model, 'llf') else None
                                    })

                                    if ic < best_ic:
                                        best_ic = ic
                                        best_model = fitted_model
                                        best_params = (p, d, q) if not seasonal else (p, d, q, P, D, Q, m)

                                except Exception as e:
                                    continue

        if best_model is None:
            raise ValueError("No suitable ARIMA model found")

        # Store the best model
        model_key = f"arima_{hash(str(data.values.tobytes())) % 10000}"
        self.fitted_models[model_key] = best_model

        # Model diagnostics
        diagnostics = self._model_diagnostics(best_model, data)

        return {
            'model': best_model,
            'params': best_params,
            'aic': best_model.aic if hasattr(best_model, 'aic') else None,
            'bic': best_model.bic if hasattr(best_model, 'bic') else None,
            'log_likelihood': best_model.llf if hasattr(best_model, 'llf') else None,
            'diagnostics': diagnostics,
            'models_tested': len(model_results)
        }

    def forecast(
        self,
        data: Union[pd.Series, np.ndarray],
        steps: int,
        method: str = 'arima',
        model_params: Optional[Tuple] = None,
        confidence_intervals: bool = True,
        bootstrap_samples: int = 1000
    ) -> ForecastResult:
        """
        Generate forecasts with prediction intervals.

        Args:
            data: Time series data
            steps: Number of steps to forecast
            method: Forecasting method ('arima', 'ets', 'ensemble')
            model_params: Pre-fitted model parameters
            confidence_intervals: Whether to compute confidence intervals
            bootstrap_samples: Number of bootstrap samples for CI

        Returns:
            ForecastResult with predictions and uncertainty
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.date_range(
                start='2020-01-01',
                periods=len(data),
                freq='D'
            )

        if method == 'arima':
            return self._forecast_arima(
                data, steps, model_params, confidence_intervals, bootstrap_samples
            )
        elif method == 'ets':
            return self._forecast_ets(data, steps, confidence_intervals)
        elif method == 'ensemble':
            return self._forecast_ensemble(
                data, steps, confidence_intervals, bootstrap_samples
            )
        else:
            raise ValueError(f"Unknown forecasting method: {method}")

    def _forecast_arima(
        self,
        data: pd.Series,
        steps: int,
        model_params: Optional[Tuple],
        confidence_intervals: bool,
        bootstrap_samples: int
    ) -> ForecastResult:
        """ARIMA forecasting with prediction intervals."""

        # Get or fit model
        model_key = f"arima_{hash(str(data.values.tobytes())) % 10000}"

        if model_key in self.fitted_models:
            model = self.fitted_models[model_key]
        else:
            if model_params:
                if len(model_params) == 7:  # SARIMA
                    model = SARIMAX(data, order=model_params[:3], seasonal_order=model_params[3:])
                else:  # ARIMA
                    model = ARIMA(data, order=model_params)
            else:
                # Auto-select model
                auto_result = self.auto_arima(data)
                model = auto_result['model']

            fitted_model = model.fit(disp=False)
            self.fitted_models[model_key] = fitted_model
            model = fitted_model

        # Generate forecast
        forecast_result = model.forecast(steps=steps, alpha=self.alpha)

        if isinstance(forecast_result, tuple):
            # statsmodels returns (forecast, conf_int)
            forecast, conf_int = forecast_result
            ci_lower = conf_int.iloc[:, 0].values
            ci_upper = conf_int.iloc[:, 1].values
        else:
            # Bootstrap for confidence intervals if not provided
            forecast = forecast_result
            if confidence_intervals:
                ci_lower, ci_upper = self._bootstrap_forecast_ci(
                    model, data, steps, bootstrap_samples
                )
            else:
                ci_lower = ci_upper = None

        # Generate future timestamps
        last_date = data.index[-1]
        freq = pd.infer_freq(data.index)
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq=freq
        )

        # Get residuals for diagnostics
        residuals = model.resid if hasattr(model, 'resid') else None

        return ForecastResult(
            timestamps=future_dates.tolist(),
            predicted_values=forecast.tolist() if hasattr(forecast, 'tolist') else [forecast] * steps,
            confidence_intervals=list(zip(ci_lower, ci_upper)) if confidence_intervals else [],
            prediction_method='ARIMA',
            model_params={
                'order': model.order if hasattr(model, 'order') else None,
                'seasonal_order': model.seasonal_order if hasattr(model, 'seasonal_order') else None
            },
            model_metrics={
                'aic': model.aic if hasattr(model, 'aic') else None,
                'bic': model.bic if hasattr(model, 'bic') else None,
                'log_likelihood': model.llf if hasattr(model, 'llf') else None
            },
            residuals=residuals
        )

    def _forecast_ets(
        self,
        data: pd.Series,
        steps: int,
        confidence_intervals: bool
    ) -> ForecastResult:
        """Exponential Smoothing forecasting."""

        try:
            # Try additive model first
            model = ExponentialSmoothing(
                data,
                trend='add',
                seasonal=None,
                damped_trend=True
            )
            fitted_model = model.fit()
        except:
            # Fall back to simple exponential smoothing
            model = ExponentialSmoothing(data)
            fitted_model = model.fit()

        # Generate forecast
        forecast_result = fitted_model.forecast(steps)

        # Simple confidence intervals based on residuals
        if confidence_intervals and hasattr(fitted_model, 'resid'):
            residuals = fitted_model.resid.dropna()
            std_error = np.std(residuals)
            z_score = stats.norm.ppf(1 - self.alpha / 2)
            ci_lower = forecast_result - z_score * std_error
            ci_upper = forecast_result + z_score * std_error
        else:
            ci_lower = ci_upper = None

        # Generate future timestamps
        last_date = data.index[-1]
        freq = pd.infer_freq(data.index)
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq=freq
        )

        return ForecastResult(
            timestamps=future_dates.tolist(),
            predicted_values=forecast_result.tolist(),
            confidence_intervals=list(zip(ci_lower, ci_upper)) if confidence_intervals else [],
            prediction_method='ETS',
            model_params={},
            model_metrics={
                'mse': np.mean(fitted_model.resid.dropna() ** 2) if hasattr(fitted_model, 'resid') else None
            },
            residuals=fitted_model.resid.dropna() if hasattr(fitted_model, 'resid') else None
        )

    def _forecast_ensemble(
        self,
        data: pd.Series,
        steps: int,
        confidence_intervals: bool,
        bootstrap_samples: int
    ) -> ForecastResult:
        """Ensemble forecasting combining multiple methods."""

        # Get forecasts from multiple methods
        forecasts = []

        try:
            arima_forecast = self._forecast_arima(
                data, steps, None, confidence_intervals, bootstrap_samples
            )
            forecasts.append(arima_forecast)
        except:
            pass

        try:
            ets_forecast = self._forecast_ets(data, steps, confidence_intervals)
            forecasts.append(ets_forecast)
        except:
            pass

        if not forecasts:
            raise ValueError("No forecasting methods succeeded")

        # Average predictions
        avg_predicted = np.mean([f.predicted_values for f in forecasts], axis=0)

        # Combine confidence intervals
        if confidence_intervals:
            avg_lower = np.mean([f.confidence_intervals for f in forecasts], axis=0)[:, 0]
            avg_upper = np.mean([f.confidence_intervals for f in forecasts], axis=0)[:, 1]
        else:
            avg_lower = avg_upper = None

        # Get timestamps from first forecast
        timestamps = forecasts[0].timestamps

        return ForecastResult(
            timestamps=timestamps,
            predicted_values=avg_predicted.tolist(),
            confidence_intervals=list(zip(avg_lower, avg_upper)) if confidence_intervals else [],
            prediction_method='Ensemble',
            model_params={'methods': [f.prediction_method for f in forecasts]},
            model_metrics={
                'forecast_variance': np.var([f.predicted_values for f in forecasts], axis=0)
            }
        )

    def _bootstrap_forecast_ci(
        self,
        model,
        data: pd.Series,
        steps: int,
        n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bootstrap prediction intervals."""

        forecasts = np.zeros((n_samples, steps))

        for i in range(n_samples):
            # Bootstrap sample
            boot_data = data.sample(n=len(data), replace=True)
            boot_data = boot_data.sort_index()

            # Refit model on bootstrap sample
            if hasattr(model, 'order'):
                boot_model = ARIMA(boot_data, order=model.order)
            else:
                boot_model = ARIMA(boot_data, order=(1, 0, 1))

            try:
                boot_fitted = boot_model.fit(disp=False)
                boot_forecast = boot_fitted.forecast(steps=steps)
                forecasts[i] = boot_forecast
            except:
                forecasts[i] = model.forecast(steps=steps)

        # Calculate percentiles
        ci_lower = np.percentile(forecasts, self.alpha / 2 * 100, axis=0)
        ci_upper = np.percentile(forecasts, (1 - self.alpha / 2) * 100, axis=0)

        return ci_lower, ci_upper

    def seasonal_decompose(
        self,
        data: Union[pd.Series, np.ndarray],
        model: str = 'additive',
        period: Optional[int] = None,
        extrapolate_trend: str = 'freq'
    ) -> Dict[str, Any]:
        """
        Perform seasonal decomposition of time series.

        Args:
            data: Time series data
            model: Decomposition model ('additive' or 'multiplicative')
            period: Seasonal period
            extrapolate_trend: Trend extrapolation method

        Returns:
            Dictionary with decomposition results
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        if period is None:
            # Auto-detect period using FFT
            period = self._detect_seasonal_period(data)

        if period is None or period <= 1:
            raise ValueError("Could not detect seasonal period")

        try:
            decomposition = seasonal_decompose(
                data,
                model=model,
                period=period,
                extrapolate_trend=extrapolate_trend
            )

            return {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'observed': decomposition.observed,
                'period': period,
                'model': model,
                'trend_strength': self._calculate_trend_strength(decomposition),
                'seasonal_strength': self._calculate_seasonal_strength(decomposition)
            }
        except Exception as e:
            raise ValueError(f"Seasonal decomposition failed: {str(e)}")

    def _detect_seasonal_period(self, data: pd.Series) -> Optional[int]:
        """Detect seasonal period using FFT."""
        try:
            # Remove trend
            differenced = data.diff().dropna()

            if len(differenced) < 10:
                return None

            # FFT to find dominant frequency
            fft = np.fft.fft(differenced.values)
            freqs = np.fft.fftfreq(len(differenced))

            # Find dominant frequency (excluding DC component)
            power = np.abs(fft[1:len(fft)//2])
            dominant_freq_idx = np.argmax(power) + 1
            dominant_freq = freqs[dominant_freq_idx]

            if dominant_freq > 0:
                period = int(1 / dominant_freq)
                # Limit to reasonable periods
                if 2 <= period <= len(data) // 3:
                    return period

            return None
        except:
            return None

    def _calculate_trend_strength(self, decomposition) -> float:
        """Calculate trend strength from decomposition."""
        try:
            trend_var = np.var(decomposition.trend.dropna())
            resid_var = np.var(decomposition.resid.dropna())
            return trend_var / (trend_var + resid_var)
        except:
            return 0.0

    def _calculate_seasonal_strength(self, decomposition) -> float:
        """Calculate seasonal strength from decomposition."""
        try:
            seasonal_var = np.var(decomposition.seasonal.dropna())
            resid_var = np.var(decomposition.resid.dropna())
            return seasonal_var / (seasonal_var + resid_var)
        except:
            return 0.0

    def _model_diagnostics(self, model, data: pd.Series) -> Dict[str, Any]:
        """Perform model diagnostics."""
        diagnostics = {}

        try:
            # Ljung-Box test for autocorrelation
            residuals = model.resid if hasattr(model, 'resid') else None
            if residuals is not None and len(residuals) > 10:
                lb_test = acorr_ljungbox(residuals, lags=10)
                diagnostics['ljung_box'] = {
                    'statistic': lb_test[0],
                    'p_value': lb_test[1],
                    'is_white_noise': lb_test[1] > 0.05
                }
        except:
            pass

        # Stationarity tests
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(data)
            diagnostics['augmented_dickey_fuller'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05
            }
        except:
            pass

        # Model fit statistics
        if hasattr(model, 'aic'):
            diagnostics['aic'] = model.aic
        if hasattr(model, 'bic'):
            diagnostics['bic'] = model.bic
        if hasattr(model, 'llf'):
            diagnostics['log_likelihood'] = model.llf

        return diagnostics

    def visualize_forecast(
        self,
        data: pd.Series,
        forecast_result: ForecastResult,
        title: Optional[str] = None,
        show_residuals: bool = True,
        save_plot: bool = True
    ) -> Optional[str]:
        """
        Visualize forecast with confidence intervals.

        Args:
            data: Original time series data
            forecast_result: Forecast results
            title: Plot title
            show_residuals: Whether to show residuals plot
            save_plot: Whether to save plot as base64

        Returns:
            Base64 encoded plot if save_plot=True
        """
        fig = plt.figure(figsize=(14, 8))

        # Main forecast plot
        ax1 = plt.subplot(2, 1, 1)

        # Plot original data
        ax1.plot(data.index, data.values, label='Historical', color='blue', alpha=0.7)

        # Plot forecast
        forecast_dates = pd.to_datetime(forecast_result.timestamps)
        ax1.plot(forecast_dates, forecast_result.predicted_values,
                label='Forecast', color='red', linewidth=2)

        # Plot confidence intervals
        if forecast_result.confidence_intervals:
            ci_lower = [ci[0] for ci in forecast_result.confidence_intervals]
            ci_upper = [ci[1] for ci in forecast_result.confidence_intervals]
            ax1.fill_between(
                forecast_dates,
                ci_lower,
                ci_upper,
                alpha=0.3,
                color='red',
                label=f'{int(self.confidence_level*100)}% CI'
            )

        ax1.set_title(title or f'Forecast - {forecast_result.prediction_method}')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Residuals plot
        if show_residuals and forecast_result.residuals is not None:
            ax2 = plt.subplot(2, 1, 2)
            ax2.plot(forecast_result.residuals)
            ax2.set_title('Model Residuals')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Residual Value')
            ax2.grid(True, alpha=0.3)

            # Add zero line
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()

        if save_plot:
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return plot_data
        else:
            plt.show()
            plt.close()
            return None

    def forecast_accuracy(
        self,
        actual: pd.Series,
        predicted: pd.Series,
        metrics: List[str] = ['mae', 'rmse', 'mape', 'mase']
    ) -> Dict[str, float]:
        """
        Calculate forecast accuracy metrics.

        Args:
            actual: Actual values
            predicted: Predicted values
            metrics: Metrics to calculate

        Returns:
            Dictionary of accuracy metrics
        """
        # Align series
        if len(actual) != len(predicted):
            min_len = min(len(actual), len(predicted))
            actual = actual.iloc[:min_len]
            predicted = predicted.iloc[:min_len]

        results = {}

        if 'mae' in metrics:
            results['mae'] = np.mean(np.abs(actual - predicted))

        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(np.mean((actual - predicted) ** 2))

        if 'mape' in metrics:
            # Avoid division by zero
            mask = actual != 0
            results['mape'] = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

        if 'mase' in metrics:
            # Mean Absolute Scaled Error
            naive_forecast = actual.shift(1).dropna()
            if len(naive_forecast) > 0:
                mae = np.mean(np.abs(actual[1:] - predicted[1:]))
                naive_mae = np.mean(np.abs(actual[1:] - naive_forecast))
                results['mase'] = mae / naive_mae if naive_mae > 0 else np.inf

        return results