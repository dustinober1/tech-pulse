"""
Property-based tests for temporal feature extraction.

This module uses Hypothesis to verify that the temporal feature generator
produces valid, consistent temporal features across diverse time series data.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Union
from hypothesis import given, strategies as st, settings, HealthCheck

from src.portfolio.features.temporal_extractor import (
    TemporalFeatureGenerator,
    TemporalConfig,
    TemporalFeatures
)


class TestTemporalFeatureProperties:
    """Property-based tests for temporal feature extraction."""

    @pytest.fixture
    def generator(self):
        """Create a temporal feature generator for property testing."""
        # Use smaller windows for faster tests
        config = TemporalConfig(
            lag_periods=[1, 2],
            rolling_windows=[3],
            seasonal_periods=[12, 24],
            trend_window=6,
            test_seasonality=True,
            detect_change_points=True,
            compute_fft=True,
            compute_interactions=True
        )
        return TemporalFeatureGenerator(config)

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=20
    )
    @given(st.data())
    def test_time_feature_validity(self, generator, data):
        """
        Property: Basic time features have valid ranges and types.

        Given any time series, time-based features should be in valid ranges.
        """
        # Generate time series data
        n_points = data.draw(st.integers(min_value=10, max_value=100))
        start_time = datetime(2024, 1, 1, 0, 0, 0)

        # Create regular time intervals
        hours = data.draw(st.integers(min_value=1, max_value=6))
        times = [start_time + timedelta(hours=i * hours) for i in range(n_points)]

        # Create synthetic data
        np.random.seed(42)
        scores = np.cumsum(np.random.randn(n_points)) + 100

        df = pd.DataFrame({
            'time': times,
            'score': scores,
            'descendants': np.random.poisson(5, n_points)
        })

        # Extract features
        features_df = generator.extract_features(df)

        # Check basic time features are valid
        assert (features_df['hour_of_day'] >= 0).all()
        assert (features_df['hour_of_day'] <= 23).all()

        assert (features_df['day_of_week'] >= 0).all()
        assert (features_df['day_of_week'] <= 6).all()

        assert (features_df['day_of_month'] >= 1).all()
        assert (features_df['day_of_month'] <= 31).all()

        assert (features_df['month'] >= 1).all()
        assert (features_df['month'] <= 12).all()

        assert (features_df['quarter'] >= 1).all()
        assert (features_df['quarter'] <= 4).all()

        assert (features_df['year'] == 2024).all()

        assert features_df['is_weekend'].isin([0, 1]).all()
        assert features_df['is_holiday'].isin([0, 1]).all()

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=20
    )
    @given(st.data())
    def test_cyclical_feature_properties(self, generator, data):
        """
        Property: Cyclical features maintain sine-cosine relationships.

        Given any time series, sin^2 + cos^2 should be approximately 1.
        """
        # Generate time series data
        n_points = data.draw(st.integers(min_value=24, max_value=168))
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        times = [start_time + timedelta(hours=i) for i in range(n_points)]

        df = pd.DataFrame({
            'time': times,
            'score': np.random.randn(n_points).cumsum(),
            'descendants': np.random.poisson(5, n_points)
        })

        # Extract features
        features_df = generator.extract_features(df)

        # Check sine-cosine normalization
        hour_norm = np.sqrt(features_df['hour_sin']**2 + features_df['hour_cos']**2)
        day_norm = np.sqrt(features_df['day_sin']**2 + features_df['day_cos']**2)
        month_norm = np.sqrt(features_df['month_sin']**2 + features_df['month_cos']**2)

        # Should be close to 1 for all rows
        assert np.allclose(hour_norm, 1.0, rtol=0.1), f"Hour cyclical features not normalized: {hour_norm[:5]}"
        assert np.allclose(day_norm, 1.0, rtol=0.1), f"Day cyclical features not normalized: {day_norm[:5]}"
        assert np.allclose(month_norm, 1.0, rtol=0.1), f"Month cyclical features not normalized: {month_norm[:5]}"

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=15
    )
    @given(st.data())
    def test_lag_feature_consistency(self, generator, data):
        """
        Property: Lag features correctly shift values.

        Given any time series, lag features should match previous values.
        """
        # Generate time series data
        n_points = data.draw(st.integers(min_value=5, max_value=50))
        values = np.random.randn(n_points).cumsum()

        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=n_points, freq='h'),
            'score': values,
            'descendants': np.random.poisson(5, n_points)
        })

        # Extract features
        features_df = generator.extract_features(df)

        # Check lag consistency
        for lag in generator.config.lag_periods:
            lag_col = f'score_lag_{lag}'

            if lag_col in features_df.columns:
                # Verify lag values match original values
                for i in range(lag, len(features_df)):
                    expected = features_df['score'].iloc[i - lag]
                    actual = features_df[lag_col].iloc[i]

                    # Allow small floating point differences
                    assert np.isclose(expected, actual, rtol=1e-10), \
                        f"Lag {lag} mismatch at index {i}: expected {expected}, got {actual}"

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=15
    )
    @given(st.data())
    def test_rolling_window_properties(self, generator, data):
        """
        Property: Rolling window features maintain expected relationships.

        Given any time series, rolling statistics should have valid relationships.
        """
        # Generate time series data
        n_points = data.draw(st.integers(min_value=10, max_value=100))
        values = np.random.randn(n_points).cumsum()

        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=n_points, freq='h'),
            'score': values,
            'descendants': np.random.poisson(5, n_points)
        })

        # Extract features
        features_df = generator.extract_features(df)

        # Check rolling window relationships
        window = generator.config.rolling_windows[0]  # Use first window

        # Min <= Mean <= Max (where values exist)
        min_col = f'score_rolling_{window}_min'
        mean_col = f'score_rolling_{window}_mean'
        max_col = f'score_rolling_{window}_max'

        if all(col in features_df.columns for col in [min_col, mean_col, max_col]):
            # Find rows where all rolling windows are valid (not NaN)
            valid_mask = (
                features_df[min_col].notna() &
                features_df[mean_col].notna() &
                features_df[max_col].notna()
            )

            if valid_mask.any():
                valid_data = features_df[valid_mask]

                # Min <= Mean <= Max
                assert (valid_data[min_col] <= valid_data[mean_col] + 1e-10).all(), \
                    "Rolling min should be <= mean"
                assert (valid_data[mean_col] <= valid_data[max_col] + 1e-10).all(), \
                    "Rolling mean should be <= max"

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=15
    )
    @given(st.data())
    def test_feature_extraction_consistency(self, generator, data):
        """
        Property: Feature extraction is deterministic.

        Given the same data, extracting features twice yields identical results.
        """
        # Generate time series data
        n_points = data.draw(st.integers(min_value=10, max_value=50))
        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=n_points, freq='h'),
            'score': np.random.randn(n_points).cumsum(),
            'descendants': np.random.poisson(5, n_points)
        })

        # Extract features twice
        features1 = generator.extract_features(df)
        features2 = generator.extract_features(df)

        # Should be identical
        assert features1.equals(features2), "Feature extraction should be deterministic"

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=10
    )
    @given(st.data())
    def test_monotonic_time_features(self, generator, data):
        """
        Property: Time-based features are monotonic with time.

        Given a time series, time since start should be monotonically increasing.
        """
        # Generate time series data
        n_points = data.draw(st.integers(min_value=5, max_value=50))
        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=n_points, freq='h'),
            'score': np.random.randn(n_points).cumsum(),
            'descendants': np.random.poisson(5, n_points)
        })

        # Extract features
        features_df = generator.extract_features(df)

        # Check time since first is non-decreasing
        time_since = features_df['time_since_first']
        time_diffs = time_since.diff().dropna()
        assert (time_diffs >= 0).all(), "Time since first should be non-decreasing"

        # Check time since last is non-increasing
        time_since_last = features_df['time_since_last']
        last_diffs = time_since_last.diff().dropna()
        assert (last_diffs <= 0).all(), "Time since last should be non-increasing"

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=10
    )
    @given(st.data())
    def test_seasonality_feature_ranges(self, generator, data):
        """
        Property: Seasonality features are within valid ranges.

        Given any time series, seasonal strength should be between 0 and 1.
        """
        # Generate time series data with some points
        n_points = data.draw(st.integers(min_value=50, max_value=200))
        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=n_points, freq='h'),
            'score': np.random.randn(n_points).cumsum(),
            'descendants': np.random.poisson(5, n_points)
        })

        # Extract features
        features_df = generator.extract_features(df)

        # Check seasonality strength is valid
        strength_col = 'score_seasonal_strength'
        if strength_col in features_df.columns:
            strengths = features_df[strength_col]
            assert (strengths >= 0).all(), "Seasonal strength should be non-negative"
            assert (strengths <= 1).all(), "Seasonal strength should not exceed 1"

        # Check has seasonality is binary
        has_seasonality_col = 'score_has_seasonality'
        if has_seasonality_col in features_df.columns:
            has_seasonality = features_df[has_seasonality_col]
            assert has_seasonality.isin([0, 1]).all(), "Has seasonality should be binary"

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=10
    )
    @given(st.data())
    def test_volatility_non_negative(self, generator, data):
        """
        Property: Volatility features are non-negative.

        Given any time series, volatility should always be non-negative.
        """
        # Generate time series data
        n_points = data.draw(st.integers(min_value=10, max_value=100))
        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=n_points, freq='h'),
            'score': np.random.randn(n_points).cumsum(),
            'descendants': np.random.poisson(5, n_points)
        })

        # Extract features
        features_df = generator.extract_features(df)

        # Check volatility is non-negative
        vol_col = 'score_volatility'
        if vol_col in features_df.columns:
            volatility = features_df[vol_col]
            assert (volatility >= 0).all(), "Volatility should be non-negative"

        # Check average volatility is non-negative
        avg_vol_col = 'score_avg_volatility'
        if avg_vol_col in features_df.columns:
            avg_volatility = features_df[avg_vol_col]
            assert (avg_volatility >= 0).all(), "Average volatility should be non-negative"

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=10
    )
    @given(st.data())
    def test_autocorrelation_bounds(self, generator, data):
        """
        Property: Autocorrelation features are within valid bounds.

        Given any time series, autocorrelation should be between -1 and 1.
        """
        # Generate time series data
        n_points = data.draw(st.integers(min_value=20, max_value=100))
        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=n_points, freq='h'),
            'score': np.random.randn(n_points).cumsum(),
            'descendants': np.random.poisson(5, n_points)
        })

        # Extract features
        features_df = generator.extract_features(df)

        # Check autocorrelation bounds
        autocorr_cols = ['score_autocorr_lag1', 'score_autocorr_lag24']
        for col in autocorr_cols:
            if col in features_df.columns:
                autocorr = features_df[col]
                assert (autocorr >= -1.1).all(), f"{col} should be >= -1"
                assert (autocorr <= 1.1).all(), f"{col} should be <= 1"

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=10
    )
    @given(st.data())
    def test_frequency_feature_validity(self, generator, data):
        """
        Property: Frequency domain features are physically meaningful.

        Given any time series, dominant frequency and spectral entropy are valid.
        """
        # Generate time series data
        n_points = data.draw(st.integers(min_value=10, max_value=100))
        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=n_points, freq='h'),
            'score': np.random.randn(n_points).cumsum(),
            'descendants': np.random.poisson(5, n_points)
        })

        # Extract features
        features_df = generator.extract_features(df)

        # Check dominant frequency is valid
        freq_col = 'score_dominant_frequency'
        if freq_col in features_df.columns:
            freq = features_df[freq_col]
            assert (freq >= 0).all(), "Dominant frequency should be non-negative"
            assert (freq <= 0.5).all(), "Dominant frequency should not exceed Nyquist frequency"

        # Check spectral entropy is valid
        entropy_col = 'score_spectral_entropy'
        if entropy_col in features_df.columns:
            entropy = features_df[entropy_col]
            assert (entropy >= 0).all(), "Spectral entropy should be non-negative"

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=10
    )
    @given(st.data())
    def test_temporal_features_object_creation(self, generator, data):
        """
        Property: TemporalFeatures object contains all expected attributes.

        Given any extracted features, creating a TemporalFeatures object should succeed.
        """
        # Generate time series data
        n_points = data.draw(st.integers(min_value=10, max_value=50))
        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=n_points, freq='h'),
            'score': np.random.randn(n_points).cumsum(),
            'descendants': np.random.poisson(5, n_points),
            'sentiment_score': np.random.randn(n_points) * 0.1
        })

        # Extract features
        features_df = generator.extract_features(df)

        # Create TemporalFeatures object from middle row
        row = features_df.iloc[len(features_df) // 2]
        temporal_features = generator.create_temporal_features_object(row, features_df)

        # Check object was created successfully
        assert isinstance(temporal_features, TemporalFeatures)

        # Check all attributes are accessible
        attributes = [
            'hour_of_day', 'day_of_week', 'month', 'year',
            'hour_sin', 'hour_cos', 'lag_features',
            'rolling_mean', 'trend_slope', 'seasonal_strength',
            'volatility', 'autocorr_lag1', 'missing_time_points',
            'feature_count'
        ]

        for attr in attributes:
            assert hasattr(temporal_features, attr), f"Missing attribute: {attr}"

        # Check feature count is positive
        assert temporal_features.feature_count > 0

    def test_feature_summary_properties(self, generator):
        """
        Property: Feature summary provides comprehensive information.

        Given any feature set, the summary should include expected components.
        """
        # Create sample data
        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=50, freq='h'),
            'score': np.random.randn(50).cumsum(),
            'descendants': np.random.poisson(5, 50),
            'sentiment_score': np.random.randn(50) * 0.1
        })

        # Extract features
        features_df = generator.extract_features(df)

        # Get summary
        summary = generator.get_feature_summary(features_df)

        # Check summary structure
        required_keys = ['total_features', 'feature_types', 'missing_values', 'feature_stats']
        for key in required_keys:
            assert key in summary, f"Missing summary key: {key}"

        # Check total features is positive
        assert summary['total_features'] > 0

        # Check feature types include expected categories
        feature_types = summary['feature_types']
        expected_types = ['time_based', 'lag', 'rolling']
        for exp_type in expected_types:
            # Not all types may be present, but if present should be positive
            if exp_type in feature_types:
                assert feature_types[exp_type] > 0