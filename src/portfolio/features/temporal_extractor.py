"""
Temporal feature generation for time series analysis.

This module provides comprehensive temporal feature extraction capabilities
including lag features, rolling windows, time-based features, and trend analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from abc import ABC, abstractmethod

# Statistical imports
from scipy import stats
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class TemporalFeatures:
    """Container for temporal features."""
    # Basic time features
    hour_of_day: int
    day_of_week: int
    day_of_month: int
    day_of_year: int
    week_of_year: int
    month: int
    quarter: int
    year: int
    is_weekend: bool
    is_holiday: bool  # Simplified holiday detection

    # Cyclical features
    hour_sin: float
    hour_cos: float
    day_sin: float
    day_cos: float
    month_sin: float
    month_cos: float

    # Lag features (will be populated based on config)
    lag_features: Dict[str, float]

    # Rolling window features
    rolling_mean: Dict[str, float]
    rolling_std: Dict[str, float]
    rolling_min: Dict[str, float]
    rolling_max: Dict[str, float]
    rolling_median: Dict[str, float]

    # Trend features
    trend_slope: float
    trend_intercept: float
    trend_r2: float
    is_trending_up: bool
    is_trending_down: bool

    # Seasonality features
    seasonal_strength: float
    seasonal_period: Optional[int]
    has_seasonality: bool

    # Change point features
    change_point_detected: bool
    change_point_location: Optional[int]
    change_point_magnitude: float

    # Time-based aggregations
    time_since_first: float
    time_since_last: float
    time_to_next: Optional[float]

    # Frequency domain features
    dominant_frequency: float
    spectral_entropy: float

    # Volatility features
    volatility: float
    avg_volatility: float

    # Autocorrelation features
    autocorr_lag1: float
    autocorr_lag24: float  # 24-hour lag for hourly data

    # Missing time features
    missing_time_points: int
    longest_gap: float

    # Interaction features
    hour_day_interaction: float
    hour_month_interaction: float
    day_month_interaction: float

    # Additional metadata
    feature_count: int
    missing_value_count: int


@dataclass
class TemporalConfig:
    """Configuration for temporal feature generation."""
    # Lag configuration
    lag_periods: List[int] = None
    lag_features: List[str] = None

    # Rolling window configuration
    rolling_windows: List[int] = None
    rolling_functions: List[str] = None

    # Trend configuration
    trend_window: int = 24
    min_trend_periods: int = 3

    # Seasonality configuration
    test_seasonality: bool = True
    seasonal_periods: List[int] = None
    seasonal_threshold: float = 0.1

    # Change point configuration
    detect_change_points: bool = True
    change_point_method: str = 'mean_shift'
    change_point_threshold: float = 2.0

    # Frequency domain configuration
    compute_fft: bool = True
    max_frequency: int = 100

    # Volatility configuration
    volatility_window: int = 12
    volatility_method: str = 'std'

    # Autocorrelation configuration
    max_autocorr_lag: int = 48

    # Interaction features
    compute_interactions: bool = True

    # Holiday configuration (simplified)
    holiday_dates: List[datetime] = None

    # Missing data handling
    handle_missing: bool = True
    missing_value_method: str = 'interpolate'

    def __post_init__(self):
        """Set default values."""
        if self.lag_periods is None:
            self.lag_periods = [1, 2, 3, 6, 12, 24]

        if self.lag_features is None:
            self.lag_features = ['score', 'descendants', 'sentiment_score']

        if self.rolling_windows is None:
            self.rolling_windows = [3, 6, 12, 24]

        if self.rolling_functions is None:
            self.rolling_functions = ['mean', 'std', 'min', 'max', 'median']

        if self.seasonal_periods is None:
            self.seasonal_periods = [24, 168]  # Hourly, weekly

        if self.holiday_dates is None:
            # Major US holidays (simplified)
            self.holiday_dates = [
                datetime(2024, 1, 1),   # New Year's
                datetime(2024, 7, 4),   # Independence Day
                datetime(2024, 12, 25), # Christmas
                datetime(2024, 11, 28), # Thanksgiving (approximate)
            ]


class TemporalFeatureGenerator:
    """Generate comprehensive temporal features for time series data."""

    def __init__(self, config: Optional[TemporalConfig] = None):
        """
        Initialize the temporal feature generator.

        Args:
            config: Configuration for feature generation
        """
        self.config = config or TemporalConfig()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.feature_cache = {}

    def extract_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'time',
        value_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract temporal features from DataFrame.

        Args:
            df: Input DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            value_cols: Columns to compute features for

        Returns:
            DataFrame with temporal features
        """
        if df.empty:
            return pd.DataFrame()

        # Make a copy to avoid modifying original
        df = df.copy()

        # Ensure timestamp is datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Sort by timestamp
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        # Default value columns
        if value_cols is None:
            value_cols = ['score', 'descendants']
            # Add sentiment_score if available
            if 'sentiment_score' in df.columns:
                value_cols.append('sentiment_score')

        # Extract basic time features
        self._extract_basic_time_features(df, timestamp_col)

        # Extract cyclical features
        self._extract_cyclical_features(df)

        # Extract lag features
        if self.config.lag_periods:
            self._extract_lag_features(df, value_cols)

        # Extract rolling window features
        if self.config.rolling_windows:
            self._extract_rolling_features(df, value_cols)

        # Extract trend features
        self._extract_trend_features(df, value_cols)

        # Extract seasonality features
        if self.config.test_seasonality:
            self._extract_seasonality_features(df, value_cols)

        # Extract change point features
        if self.config.detect_change_points:
            self._extract_change_point_features(df, value_cols)

        # Extract frequency domain features
        if self.config.compute_fft:
            self._extract_frequency_features(df, value_cols)

        # Extract volatility features
        self._extract_volatility_features(df, value_cols)

        # Extract autocorrelation features
        self._extract_autocorrelation_features(df, value_cols)

        # Extract missing time features
        self._extract_missing_time_features(df, timestamp_col)

        # Extract interaction features
        if self.config.compute_interactions:
            self._extract_interaction_features(df)

        return df

    def _extract_basic_time_features(self, df: pd.DataFrame, timestamp_col: str):
        """Extract basic time-based features."""
        dt = df[timestamp_col].dt

        df['hour_of_day'] = dt.hour
        df['day_of_week'] = dt.dayofweek  # Monday=0, Sunday=6
        df['day_of_month'] = dt.day
        df['day_of_year'] = dt.dayofyear
        df['week_of_year'] = dt.isocalendar().week
        df['month'] = dt.month
        df['quarter'] = dt.quarter
        df['year'] = dt.year
        df['is_weekend'] = (dt.dayofweek >= 5).astype(int)

        # Simple holiday detection
        df['is_holiday'] = 0
        for holiday in self.config.holiday_dates:
            # Check if any timestamp matches holiday (same month and day)
            holiday_mask = (
                (dt.month == holiday.month) &
                (dt.day == holiday.day)
            )
            df.loc[holiday_mask, 'is_holiday'] = 1

    def _extract_cyclical_features(self, df: pd.DataFrame):
        """Extract cyclical features using sine/cosine transformation."""
        # Hour of day (24-hour cycle)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

        # Day of week (7-day cycle)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Month of year (12-month cycle)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    def _extract_lag_features(self, df: pd.DataFrame, value_cols: List[str]):
        """Extract lag features for specified columns."""
        for col in value_cols:
            if col not in df.columns:
                continue

            for lag in self.config.lag_periods:
                lag_col = f"{col}_lag_{lag}"
                df[lag_col] = df[col].shift(lag)

        # Handle missing values in lag features
        if self.config.handle_missing:
            lag_cols = [c for c in df.columns if '_lag_' in c]
            if self.config.missing_value_method == 'interpolate':
                df[lag_cols] = df[lag_cols].interpolate()
            elif self.config.missing_value_method == 'forward_fill':
                df[lag_cols] = df[lag_cols].ffill()
            else:
                df[lag_cols] = df[lag_cols].fillna(0)

    def _extract_rolling_features(self, df: pd.DataFrame, value_cols: List[str]):
        """Extract rolling window features."""
        for col in value_cols:
            if col not in df.columns:
                continue

            # Try to convert to numeric, skip if not possible
            try:
                series = pd.to_numeric(df[col], errors='coerce')
            except:
                continue

            for window in self.config.rolling_windows:
                for func in self.config.rolling_functions:
                    feature_name = f"{col}_rolling_{window}_{func}"

                    try:
                        if func == 'mean':
                            df[feature_name] = series.rolling(window=window).mean()
                        elif func == 'std':
                            df[feature_name] = series.rolling(window=window).std()
                        elif func == 'min':
                            df[feature_name] = series.rolling(window=window).min()
                        elif func == 'max':
                            df[feature_name] = series.rolling(window=window).max()
                        elif func == 'median':
                            df[feature_name] = series.rolling(window=window).median()
                    except:
                        # If rolling operation fails, fill with NaN
                        df[feature_name] = np.nan

        # Handle missing values in rolling features
        if self.config.handle_missing:
            rolling_cols = [c for c in df.columns if '_rolling_' in c]
            df[rolling_cols] = df[rolling_cols].bfill().fillna(0)

    def _extract_trend_features(self, df: pd.DataFrame, value_cols: List[str]):
        """Extract trend features using linear regression."""
        window = self.config.trend_window

        for col in value_cols:
            if col not in df.columns:
                continue

            # Initialize trend features
            df[f"{col}_trend_slope"] = 0.0
            df[f"{col}_trend_intercept"] = 0.0
            df[f"{col}_trend_r2"] = 0.0

            # Calculate rolling trend
            for i in range(window - 1, len(df)):
                y = df[col].iloc[i - window + 1:i + 1].values
                x = np.arange(len(y)).reshape(-1, 1)

                if len(y) > self.config.min_trend_periods and not np.all(y == y[0]):
                    try:
                        # Fit linear regression
                        lr = LinearRegression()
                        lr.fit(x, y)
                        r2 = lr.score(x, y)

                        df.loc[i, f"{col}_trend_slope"] = lr.coef_[0]
                        df.loc[i, f"{col}_trend_intercept"] = lr.intercept_
                        df.loc[i, f"{col}_trend_r2"] = r2
                    except:
                        pass

            # Trend direction features
            df[f"{col}_is_trending_up"] = (df[f"{col}_trend_slope"] > 0).astype(int)
            df[f"{col}_is_trending_down"] = (df[f"{col}_trend_slope"] < 0).astype(int)

    def _extract_seasonality_features(self, df: pd.DataFrame, value_cols: List[str]):
        """Extract seasonality features."""
        for col in value_cols:
            if col not in df.columns:
                continue

            # Test different seasonal periods
            best_strength = 0
            best_period = None

            for period in self.config.seasonal_periods:
                if len(df) >= 2 * period:
                    # Calculate seasonal strength
                    try:
                        # Simple seasonal decomposition
                        series = df[col].dropna()
                        if len(series) > 0:
                            # Autocorrelation at seasonal lag
                            autocorr = series.autocorr(lag=period)
                            if not np.isnan(autocorr):
                                strength = abs(autocorr)
                                if strength > best_strength:
                                    best_strength = strength
                                    best_period = period
                    except:
                        pass

            # Set seasonality features
            df[f"{col}_seasonal_strength"] = best_strength
            df[f"{col}_seasonal_period"] = best_period if best_period else -1
            has_seasonality = int(best_strength > self.config.seasonal_threshold)
            df[f"{col}_has_seasonality"] = has_seasonality

    def _extract_change_point_features(self, df: pd.DataFrame, value_cols: List[str]):
        """Extract change point detection features."""
        for col in value_cols:
            if col not in df.columns:
                continue

            # Simple change point detection using rolling mean
            window = max(12, len(df) // 10)
            rolling_mean = df[col].rolling(window=window, center=True).mean()

            # Calculate change points
            diff = rolling_mean.diff().abs()
            threshold = diff.std() * self.config.change_point_threshold

            # Detect change points
            change_points = diff > threshold

            # Set features
            df[f"{col}_change_point_detected"] = change_points.astype(int)

            # Find most recent change point
            change_point_locations = np.where(change_points)[0]
            if len(change_point_locations) > 0:
                last_change = change_point_locations[-1]
                df[f"{col}_change_point_location"] = last_change
                df[f"{col}_change_point_magnitude"] = diff.iloc[last_change]
            else:
                df[f"{col}_change_point_location"] = -1
                df[f"{col}_change_point_magnitude"] = 0.0

    def _extract_frequency_features(self, df: pd.DataFrame, value_cols: List[str]):
        """Extract frequency domain features using FFT."""
        for col in value_cols:
            if col not in df.columns:
                continue

            series = df[col].dropna()
            if len(series) < 10:
                # Set default values for short series
                df[f"{col}_dominant_frequency"] = 0.0
                df[f"{col}_spectral_entropy"] = 0.0
                continue

            try:
                # Compute FFT
                fft_values = fft(series.values)
                power_spectrum = np.abs(fft_values) ** 2

                # Normalize power spectrum
                power_spectrum = power_spectrum / power_spectrum.sum()

                # Find dominant frequency
                dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
                dominant_frequency = dominant_freq_idx / len(series)

                # Calculate spectral entropy
                spectral_entropy = -np.sum(power_spectrum * np.log(power_spectrum + 1e-10))

                df[f"{col}_dominant_frequency"] = dominant_frequency
                df[f"{col}_spectral_entropy"] = spectral_entropy
            except:
                df[f"{col}_dominant_frequency"] = 0.0
                df[f"{col}_spectral_entropy"] = 0.0

    def _extract_volatility_features(self, df: pd.DataFrame, value_cols: List[str]):
        """Extract volatility features."""
        window = self.config.volatility_window

        for col in value_cols:
            if col not in df.columns:
                continue

            if self.config.volatility_method == 'std':
                df[f"{col}_volatility"] = df[col].rolling(window=window).std()
                df[f"{col}_avg_volatility"] = df[f"{col}_volatility"].mean()
            else:
                # Use percentage change
                pct_change = df[col].pct_change()
                df[f"{col}_volatility"] = pct_change.rolling(window=window).std()
                df[f"{col}_avg_volatility"] = df[f"{col}_volatility"].mean()

            # Fill missing values
            df[f"{col}_volatility"] = df[f"{col}_volatility"].bfill().fillna(0)
            df[f"{col}_avg_volatility"] = df[f"{col}_avg_volatility"].fillna(0)

    def _extract_autocorrelation_features(self, df: pd.DataFrame, value_cols: List[str]):
        """Extract autocorrelation features."""
        max_lag = min(self.config.max_autocorr_lag, len(df) // 2)

        for col in value_cols:
            if col not in df.columns:
                continue

            series = df[col].dropna()
            if len(series) < 10:
                df[f"{col}_autocorr_lag1"] = 0.0
                df[f"{col}_autocorr_lag24"] = 0.0
                continue

            # Calculate autocorrelations
            try:
                autocorr_1 = series.autocorr(lag=1)
                autocorr_24 = series.autocorr(lag=min(24, max_lag)) if len(series) > 24 else 0

                df[f"{col}_autocorr_lag1"] = autocorr_1 if not np.isnan(autocorr_1) else 0.0
                df[f"{col}_autocorr_lag24"] = autocorr_24 if not np.isnan(autocorr_24) else 0.0
            except:
                df[f"{col}_autocorr_lag1"] = 0.0
                df[f"{col}_autocorr_lag24"] = 0.0

    def _extract_missing_time_features(self, df: pd.DataFrame, timestamp_col: str):
        """Extract features related to missing time points."""
        # Calculate time differences
        df['time_diff'] = df[timestamp_col].diff().dt.total_seconds() / 3600  # In hours

        # Regular time interval detection
        if len(df) > 1:
            mode_diff = df['time_diff'].mode()
            if len(mode_diff) > 0:
                regular_interval = mode_diff[0]
            else:
                regular_interval = df['time_diff'].median()
        else:
            regular_interval = 1

        # Missing time points
        large_gaps = df['time_diff'] > regular_interval * 1.5
        df['missing_time_points'] = large_gaps.astype(int)

        # Longest gap
        df['longest_gap'] = df['time_diff'].max()

        # Time since first/last
        df['time_since_first'] = (df[timestamp_col] - df[timestamp_col].min()).dt.total_seconds() / 3600
        df['time_since_last'] = (df[timestamp_col].max() - df[timestamp_col]).dt.total_seconds() / 3600

        # Time to next (for all but last row)
        df['time_to_next'] = df[timestamp_col].diff(-1).dt.total_seconds() / 3600 * -1
        df.loc[df.index[-1], 'time_to_next'] = np.nan

        # Clean up
        df = df.drop('time_diff', axis=1)

    def _extract_interaction_features(self, df: pd.DataFrame):
        """Extract interaction features between time components."""
        # Hour * Day of week interaction
        df['hour_day_interaction'] = df['hour_of_day'] * df['day_of_week']

        # Hour * Month interaction
        df['hour_month_interaction'] = df['hour_of_day'] * df['month']

        # Day * Month interaction
        df['day_month_interaction'] = df['day_of_month'] * df['month']

    def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of generated temporal features.

        Args:
            df: DataFrame with temporal features

        Returns:
            Dictionary with feature summary
        """
        feature_cols = [c for c in df.columns if c not in ['time', 'title', 'url', 'text']]

        summary = {
            'total_features': len(feature_cols),
            'feature_types': {},
            'missing_values': {},
            'feature_stats': {}
        }

        # Categorize features
        for col in feature_cols:
            if 'lag_' in col:
                feature_type = 'lag'
            elif 'rolling_' in col:
                feature_type = 'rolling'
            elif 'trend_' in col:
                feature_type = 'trend'
            elif 'seasonal_' in col:
                feature_type = 'seasonal'
            elif 'change_point' in col:
                feature_type = 'change_point'
            elif 'autocorr' in col:
                feature_type = 'autocorrelation'
            elif 'volatility' in col:
                feature_type = 'volatility'
            elif any(x in col for x in ['hour_', 'day_', 'month_', 'quarter', 'year']):
                feature_type = 'time_based'
            else:
                feature_type = 'other'

            summary['feature_types'][feature_type] = summary['feature_types'].get(feature_type, 0) + 1

        # Missing values
        for col in feature_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                summary['missing_values'][col] = missing_count

        # Basic statistics
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['feature_stats'] = df[numeric_cols].describe().to_dict()

        return summary

    def create_temporal_features_object(
        self,
        row: pd.Series,
        df: pd.DataFrame
    ) -> TemporalFeatures:
        """
        Create a TemporalFeatures object from a DataFrame row.

        Args:
            row: Single row of feature data
            df: Full DataFrame for context

        Returns:
            TemporalFeatures object
        """
        # Collect lag features
        lag_features = {}
        for col in self.config.lag_features or []:
            for lag in self.config.lag_periods or []:
                lag_key = f"{col}_lag_{lag}"
                lag_features[lag_key] = row.get(lag_key, 0.0)

        # Collect rolling features
        rolling_mean = {}
        rolling_std = {}
        rolling_min = {}
        rolling_max = {}
        rolling_median = {}

        for col in self.config.lag_features or []:
            for window in self.config.rolling_windows or []:
                rolling_mean[f"{col}_rolling_{window}_mean"] = row.get(f"{col}_rolling_{window}_mean", 0.0)
                rolling_std[f"{col}_rolling_{window}_std"] = row.get(f"{col}_rolling_{window}_std", 0.0)
                rolling_min[f"{col}_rolling_{window}_min"] = row.get(f"{col}_rolling_{window}_min", 0.0)
                rolling_max[f"{col}_rolling_{window}_max"] = row.get(f"{col}_rolling_{window}_max", 0.0)
                rolling_median[f"{col}_rolling_{window}_median"] = row.get(f"{col}_rolling_{window}_median", 0.0)

        return TemporalFeatures(
            # Basic time features
            hour_of_day=int(row.get('hour_of_day', 0)),
            day_of_week=int(row.get('day_of_week', 0)),
            day_of_month=int(row.get('day_of_month', 0)),
            day_of_year=int(row.get('day_of_year', 0)),
            week_of_year=int(row.get('week_of_year', 0)),
            month=int(row.get('month', 0)),
            quarter=int(row.get('quarter', 0)),
            year=int(row.get('year', 0)),
            is_weekend=bool(row.get('is_weekend', False)),
            is_holiday=bool(row.get('is_holiday', False)),

            # Cyclical features
            hour_sin=float(row.get('hour_sin', 0.0)),
            hour_cos=float(row.get('hour_cos', 0.0)),
            day_sin=float(row.get('day_sin', 0.0)),
            day_cos=float(row.get('day_cos', 0.0)),
            month_sin=float(row.get('month_sin', 0.0)),
            month_cos=float(row.get('month_cos', 0.0)),

            # Aggregated features
            lag_features=lag_features,
            rolling_mean=rolling_mean,
            rolling_std=rolling_std,
            rolling_min=rolling_min,
            rolling_max=rolling_max,
            rolling_median=rolling_median,

            # Trend features (using first value column)
            trend_slope=float(row.get(f"{self.config.lag_features[0]}_trend_slope", 0.0) if self.config.lag_features else 0.0),
            trend_intercept=float(row.get(f"{self.config.lag_features[0]}_trend_intercept", 0.0) if self.config.lag_features else 0.0),
            trend_r2=float(row.get(f"{self.config.lag_features[0]}_trend_r2", 0.0) if self.config.lag_features else 0.0),
            is_trending_up=bool(row.get(f"{self.config.lag_features[0]}_is_trending_up", False) if self.config.lag_features else False),
            is_trending_down=bool(row.get(f"{self.config.lag_features[0]}_is_trending_down", False) if self.config.lag_features else False),

            # Seasonality features
            seasonal_strength=float(row.get(f"{self.config.lag_features[0]}_seasonal_strength", 0.0) if self.config.lag_features else 0.0),
            seasonal_period=int(row.get(f"{self.config.lag_features[0]}_seasonal_period", -1) if self.config.lag_features else -1),
            has_seasonality=bool(row.get(f"{self.config.lag_features[0]}_has_seasonality", False) if self.config.lag_features else False),

            # Change point features
            change_point_detected=bool(row.get(f"{self.config.lag_features[0]}_change_point_detected", False) if self.config.lag_features else False),
            change_point_location=row.get(f"{self.config.lag_features[0]}_change_point_location", None) if self.config.lag_features else None,
            change_point_magnitude=float(row.get(f"{self.config.lag_features[0]}_change_point_magnitude", 0.0) if self.config.lag_features else 0.0),

            # Time-based features
            time_since_first=float(row.get('time_since_first', 0.0)),
            time_since_last=float(row.get('time_since_last', 0.0)),
            time_to_next=row.get('time_to_next', None),

            # Frequency domain features
            dominant_frequency=float(row.get(f"{self.config.lag_features[0]}_dominant_frequency", 0.0) if self.config.lag_features else 0.0),
            spectral_entropy=float(row.get(f"{self.config.lag_features[0]}_spectral_entropy", 0.0) if self.config.lag_features else 0.0),

            # Volatility features
            volatility=float(row.get(f"{self.config.lag_features[0]}_volatility", 0.0) if self.config.lag_features else 0.0),
            avg_volatility=float(row.get(f"{self.config.lag_features[0]}_avg_volatility", 0.0) if self.config.lag_features else 0.0),

            # Autocorrelation features
            autocorr_lag1=float(row.get(f"{self.config.lag_features[0]}_autocorr_lag1", 0.0) if self.config.lag_features else 0.0),
            autocorr_lag24=float(row.get(f"{self.config.lag_features[0]}_autocorr_lag24", 0.0) if self.config.lag_features else 0.0),

            # Missing time features
            missing_time_points=int(row.get('missing_time_points', 0)),
            longest_gap=float(row.get('longest_gap', 0.0)),

            # Interaction features
            hour_day_interaction=float(row.get('hour_day_interaction', 0.0)),
            hour_month_interaction=float(row.get('hour_month_interaction', 0.0)),
            day_month_interaction=float(row.get('day_month_interaction', 0.0)),

            # Metadata
            feature_count=len([c for c in df.columns if c not in ['time', 'title', 'url', 'text']]),
            missing_value_count=row.isna().sum()
        )