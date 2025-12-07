"""
Unit tests for temporal feature extraction.

This module tests the TemporalFeatureGenerator class and its methods
for extracting comprehensive temporal features from time series data.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from src.portfolio.features.temporal_extractor import (
    TemporalFeatureGenerator,
    TemporalConfig,
    TemporalFeatures
)


class TestTemporalFeatureGenerator:
    """Test cases for TemporalFeatureGenerator."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame with temporal data."""
        # Create sample data over 7 days with hourly intervals
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        times = [start_time + timedelta(hours=i) for i in range(168)]  # 1 week

        # Create synthetic time series data
        np.random.seed(42)
        scores = np.cumsum(np.random.randn(168)) + 100  # Random walk trend
        descendants = np.random.poisson(5, 168)  # Random counts
        sentiment = np.random.randn(168) * 0.1  # Small sentiment values

        df = pd.DataFrame({
            'time': times,
            'title': [f'Story {i}' for i in range(168)],
            'url': [f'http://example.com/{i}' for i in range(168)],
            'score': scores,
            'descendants': descendants,
            'sentiment_score': sentiment
        })

        # Add some weekly seasonality
        df['score'] += 10 * np.sin(2 * np.pi * np.arange(168) / 24)

        return df

    @pytest.fixture
    def simple_dataframe(self):
        """Create a simple DataFrame for basic tests."""
        return pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=24, freq='h'),
            'score': [10, 12, 11, 13, 15, 14, 16, 18, 17, 19, 21, 20] * 2,
            'descendants': [5, 6, 5, 7, 8, 7, 9, 10, 9, 11, 12, 11] * 2
        })

    @pytest.fixture
    def generator(self):
        """Create a TemporalFeatureGenerator instance."""
        config = TemporalConfig(
            lag_periods=[1, 3, 6],
            rolling_windows=[3, 6],
            seasonal_periods=[12, 24],  # 12-hour and daily
            trend_window=6,
            detect_change_points=True,
            compute_fft=True
        )
        return TemporalFeatureGenerator(config)

    def test_initialization(self):
        """Test TemporalFeatureGenerator initialization."""
        # Test with default config
        gen = TemporalFeatureGenerator()
        assert gen.config is not None
        assert isinstance(gen.config, TemporalConfig)

        # Test with custom config
        config = TemporalConfig(lag_periods=[1, 2])
        gen = TemporalFeatureGenerator(config)
        assert gen.config.lag_periods == [1, 2]

    def test_config_post_init(self):
        """Test TemporalConfig post-initialization."""
        config = TemporalConfig()
        assert config.lag_periods == [1, 2, 3, 6, 12, 24]
        assert config.lag_features == ['score', 'descendants', 'sentiment_score']
        assert config.rolling_windows == [3, 6, 12, 24]
        assert config.rolling_functions == ['mean', 'std', 'min', 'max', 'median']
        assert config.seasonal_periods == [24, 168]

    def test_extract_basic_time_features(self, generator, simple_dataframe):
        """Test basic time feature extraction."""
        df = simple_dataframe.copy()
        generator._extract_basic_time_features(df, 'time')

        # Check that all basic time features exist
        expected_features = [
            'hour_of_day', 'day_of_week', 'day_of_month',
            'day_of_year', 'week_of_year', 'month',
            'quarter', 'year', 'is_weekend', 'is_holiday'
        ]

        for feature in expected_features:
            assert feature in df.columns, f"Missing feature: {feature}"

        # Validate ranges
        assert df['hour_of_day'].min() >= 0
        assert df['hour_of_day'].max() <= 23
        assert df['day_of_week'].min() >= 0
        assert df['day_of_week'].max() <= 6
        assert df['is_weekend'].isin([0, 1]).all()

    def test_extract_cyclical_features(self, generator, simple_dataframe):
        """Test cyclical feature extraction."""
        df = simple_dataframe.copy()
        # First add basic time features
        generator._extract_basic_time_features(df, 'time')
        # Then add cyclical features
        generator._extract_cyclical_features(df)

        # Check cyclical features
        cyclical_features = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month_sin', 'month_cos'
        ]

        for feature in cyclical_features:
            assert feature in df.columns, f"Missing cyclical feature: {feature}"
            assert df[feature].between(-1.1, 1.1).all(), f"Invalid range for {feature}"

        # Check sine-cosine relationship
        # For each row, sin^2 + cos^2 should be approximately 1
        hour_norm = np.sqrt(df['hour_sin']**2 + df['hour_cos']**2)
        assert np.allclose(hour_norm, 1.0, rtol=0.1), "Hour cyclical features not normalized"

    def test_extract_lag_features(self, generator, simple_dataframe):
        """Test lag feature extraction."""
        df = simple_dataframe.copy()
        generator._extract_lag_features(df, ['score', 'descendants'])

        # Check that lag features exist
        for col in ['score', 'descendants']:
            for lag in generator.config.lag_periods:
                lag_col = f"{col}_lag_{lag}"
                assert lag_col in df.columns, f"Missing lag feature: {lag_col}"

                # Verify lag values (first lag should match previous value)
                if lag == 1:
                    assert df[lag_col].iloc[1] == df[col].iloc[0], "Lag 1 feature incorrect"

    def test_extract_rolling_features(self, generator, simple_dataframe):
        """Test rolling window feature extraction."""
        df = simple_dataframe.copy()
        generator._extract_rolling_features(df, ['score', 'descendants'])

        # Check that rolling features exist
        for col in ['score', 'descendants']:
            for window in generator.config.rolling_windows:
                for func in generator.config.rolling_functions:
                    feature_name = f"{col}_rolling_{window}_{func}"
                    assert feature_name in df.columns, f"Missing rolling feature: {feature_name}"

        # Verify rolling mean
        rolling_mean_3 = df['score_rolling_3_mean']
        assert rolling_mean_3.iloc[2] == df['score'].iloc[:3].mean(), "Rolling mean incorrect"

    def test_extract_trend_features(self, generator, simple_dataframe):
        """Test trend feature extraction."""
        df = simple_dataframe.copy()
        generator._extract_trend_features(df, ['score'])

        # Check trend features
        trend_features = [
            'score_trend_slope', 'score_trend_intercept',
            'score_trend_r2', 'score_is_trending_up', 'score_is_trending_down'
        ]

        for feature in trend_features:
            assert feature in df.columns, f"Missing trend feature: {feature}"

        # Check that trend direction flags are mutually exclusive
        assert not ((df['score_is_trending_up'] == 1) & (df['score_is_trending_down'] == 1)).any()

    def test_extract_seasonality_features(self, generator, sample_dataframe):
        """Test seasonality feature extraction."""
        df = sample_dataframe.copy()
        generator._extract_seasonality_features(df, ['score'])

        # Check seasonality features
        seasonality_features = [
            'score_seasonal_strength', 'score_seasonal_period', 'score_has_seasonality'
        ]

        for feature in seasonality_features:
            assert feature in df.columns, f"Missing seasonality feature: {feature}"

        # Validate ranges
        assert df['score_seasonal_strength'].between(0, 1).all()
        assert df['score_has_seasonality'].isin([0, 1]).all()

    def test_extract_change_point_features(self, generator, sample_dataframe):
        """Test change point detection features."""
        df = sample_dataframe.copy()
        # Add an artificial change point
        df.loc[df.index[84:], 'score'] += 50  # Sudden jump at midpoint

        generator._extract_change_point_features(df, ['score'])

        # Check change point features
        change_features = [
            'score_change_point_detected', 'score_change_point_location',
            'score_change_point_magnitude'
        ]

        for feature in change_features:
            assert feature in df.columns, f"Missing change point feature: {feature}"

        # Should detect at least one change point
        assert df['score_change_point_detected'].sum() > 0

    def test_extract_frequency_features(self, generator, sample_dataframe):
        """Test frequency domain feature extraction."""
        df = sample_dataframe.copy()
        generator._extract_frequency_features(df, ['score'])

        # Check frequency features
        freq_features = ['score_dominant_frequency', 'score_spectral_entropy']

        for feature in freq_features:
            assert feature in df.columns, f"Missing frequency feature: {feature}"
            assert df[feature].notna().all(), f"Frequency feature {feature} has NaN values"

        # Validate ranges
        assert (df['score_dominant_frequency'] >= 0).all()  # Should be non-negative
        assert (df['score_dominant_frequency'] <= 0.5).all()  # Nyquist frequency
        assert (df['score_spectral_entropy'] >= 0).all()  # Entropy should be non-negative

    def test_extract_volatility_features(self, generator, simple_dataframe):
        """Test volatility feature extraction."""
        df = simple_dataframe.copy()
        generator._extract_volatility_features(df, ['score'])

        # Check volatility features
        volatility_features = ['score_volatility', 'score_avg_volatility']

        for feature in volatility_features:
            assert feature in df.columns, f"Missing volatility feature: {feature}"
            assert df[feature].notna().all(), f"Volatility feature {feature} has NaN values"
            assert (df[feature] >= 0).all(), f"Volatility feature {feature} has negative values"

    def test_extract_autocorrelation_features(self, generator, simple_dataframe):
        """Test autocorrelation feature extraction."""
        df = simple_dataframe.copy()
        generator._extract_autocorrelation_features(df, ['score'])

        # Check autocorrelation features
        autocorr_features = ['score_autocorr_lag1', 'score_autocorr_lag24']

        for feature in autocorr_features:
            assert feature in df.columns, f"Missing autocorrelation feature: {feature}"
            assert df[feature].notna().all(), f"Autocorrelation feature {feature} has NaN values"
            assert df[feature].between(-1.1, 1.1).all(), f"Autocorrelation feature {feature} out of range"

    def test_extract_missing_time_features(self, generator, simple_dataframe):
        """Test missing time feature extraction."""
        df = simple_dataframe.copy()
        generator._extract_missing_time_features(df, 'time')

        # Check missing time features
        missing_features = [
            'missing_time_points', 'longest_gap',
            'time_since_first', 'time_since_last', 'time_to_next'
        ]

        for feature in missing_features:
            assert feature in df.columns, f"Missing time feature: {feature}"

        # Validate ranges
        assert df['missing_time_points'].isin([0, 1]).all()
        assert (df['longest_gap'] >= 0).all()
        assert (df['time_since_first'] >= 0).all()
        assert (df['time_since_last'] >= 0).all()

    def test_extract_interaction_features(self, generator, simple_dataframe):
        """Test interaction feature extraction."""
        df = simple_dataframe.copy()
        # First add basic time features
        generator._extract_basic_time_features(df, 'time')
        # Then add interactions
        generator._extract_interaction_features(df)

        # Check interaction features
        interaction_features = [
            'hour_day_interaction', 'hour_month_interaction', 'day_month_interaction'
        ]

        for feature in interaction_features:
            assert feature in df.columns, f"Missing interaction feature: {feature}"

    def test_extract_features_comprehensive(self, generator, sample_dataframe):
        """Test comprehensive feature extraction."""
        features_df = generator.extract_features(
            sample_dataframe,
            timestamp_col='time',
            value_cols=['score', 'descendants', 'sentiment_score']
        )

        # Should have many more columns
        assert len(features_df.columns) > len(sample_dataframe.columns)

        # Check that all expected feature types exist
        feature_types = [
            'hour_', 'day_', 'month_', '_lag_', '_rolling_',
            '_trend_', '_seasonal_', '_change_point',
            '_autocorr', '_volatility'
        ]

        for feature_type in feature_types:
            has_feature = any(feature_type in col for col in features_df.columns)
            assert has_feature, f"No features of type: {feature_type}"

        # No NaN values in critical features
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Allow some NaN in lag features at the beginning
            if '_lag_' in col:
                assert features_df[col].iloc[generator.config.lag_periods[-1]:].notna().all()
            else:
                # Other features should have minimal NaN
                nan_count = features_df[col].isna().sum()
                assert nan_count <= len(features_df) * 0.1, f"Too many NaN in {col}: {nan_count}"

    def test_extract_features_empty_dataframe(self, generator):
        """Test feature extraction with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = generator.extract_features(empty_df)
        assert result.empty

    def test_extract_features_missing_columns(self, generator):
        """Test feature extraction with missing value columns."""
        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=10, freq='h'),
            'title': [f'Story {i}' for i in range(10)]
        })

        # Should not crash when value columns are missing
        result = generator.extract_features(df, value_cols=['missing_column'])
        assert len(result) == len(df)
        assert 'time' in result.columns

    def test_get_feature_summary(self, generator, sample_dataframe):
        """Test feature summary generation."""
        features_df = generator.extract_features(sample_dataframe)
        summary = generator.get_feature_summary(features_df)

        # Check summary structure
        expected_keys = ['total_features', 'feature_types', 'missing_values', 'feature_stats']
        for key in expected_keys:
            assert key in summary, f"Missing summary key: {key}"

        assert summary['total_features'] > 0
        assert len(summary['feature_types']) > 0
        assert isinstance(summary['missing_values'], dict)
        assert isinstance(summary['feature_stats'], dict)

    def test_create_temporal_features_object(self, generator, sample_dataframe):
        """Test creation of TemporalFeatures object."""
        features_df = generator.extract_features(sample_dataframe)
        row = features_df.iloc[100]  # Get a sample row

        temporal_features = generator.create_temporal_features_object(row, features_df)

        # Check object type
        assert isinstance(temporal_features, TemporalFeatures)

        # Check that all attributes are accessible
        assert temporal_features.hour_of_day is not None
        assert temporal_features.day_of_week is not None
        assert temporal_features.lag_features is not None
        assert temporal_features.rolling_mean is not None
        assert temporal_features.feature_count > 0

    def test_custom_config(self):
        """Test TemporalFeatureGenerator with custom configuration."""
        custom_config = TemporalConfig(
            lag_periods=[2, 4],
            rolling_windows=[5, 10],
            test_seasonality=False,
            detect_change_points=False,
            compute_interactions=False
        )

        gen = TemporalFeatureGenerator(custom_config)

        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=20, freq='h'),
            'score': np.random.randn(20).cumsum()
        })

        features_df = gen.extract_features(df)

        # Should have fewer features due to disabled options
        assert 'score_lag_1' not in features_df.columns  # lag 1 not in config
        assert 'score_lag_2' in features_df.columns     # lag 2 in config
        assert 'score_rolling_3_mean' not in features_df.columns  # window 3 not in config
        assert 'score_rolling_5_mean' in features_df.columns     # window 5 in config

        # Should not have disabled features
        assert not any('change_point' in col for col in features_df.columns)
        assert not any('seasonal_' in col for col in features_df.columns)

    def test_holiday_detection(self):
        """Test holiday detection functionality."""
        # Create a config with specific holidays
        config = TemporalConfig(
            holiday_dates=[datetime(2024, 1, 1), datetime(2024, 12, 25)]
        )
        gen = TemporalFeatureGenerator(config)

        # Create DataFrame spanning holidays
        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', '2024-01-02', freq='h'),
            'score': range(25)
        })

        gen._extract_basic_time_features(df, 'time')

        # Check that New Year's is detected
        assert df.loc[df['time'].dt.date == datetime(2024, 1, 1).date(), 'is_holiday'].any()

    def test_error_handling(self, generator):
        """Test error handling in feature extraction."""
        # Test with missing values and mixed data types
        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=10, freq='h'),
            'score': [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
            'descendants': range(10)
        })

        # Should not crash with missing values
        result = generator.extract_features(df)
        assert len(result) == len(df)

    def test_memory_efficiency(self, generator, sample_dataframe):
        """Test that feature extraction doesn't use excessive memory."""
        import sys

        # Get initial memory usage
        initial_size = sys.getsizeof(sample_dataframe)

        # Extract features
        features_df = generator.extract_features(sample_dataframe)

        # Check that memory usage is reasonable (shouldn't be more than 10x original)
        final_size = sys.getsizeof(features_df)
        assert final_size < initial_size * 10, "Memory usage too high after feature extraction"

    def test_reproducibility(self, generator, sample_dataframe):
        """Test that feature extraction is reproducible."""
        # Extract features twice
        features1 = generator.extract_features(sample_dataframe)
        features2 = generator.extract_features(sample_dataframe)

        # Results should be identical
        assert features1.equals(features2), "Feature extraction not reproducible"

    @pytest.mark.parametrize("feature_type", [
        'basic_time', 'cyclical', 'lag', 'rolling', 'trend',
        'seasonality', 'change_point', 'frequency', 'volatility',
        'autocorrelation', 'missing_time', 'interaction'
    ])
    def test_individual_feature_types(self, generator, simple_dataframe, feature_type):
        """Test each feature type individually."""
        df = simple_dataframe.copy()

        if feature_type == 'basic_time':
            generator._extract_basic_time_features(df, 'time')
            assert 'hour_of_day' in df.columns

        elif feature_type == 'cyclical':
            generator._extract_basic_time_features(df, 'time')
            generator._extract_cyclical_features(df)
            assert 'hour_sin' in df.columns

        elif feature_type == 'lag':
            generator._extract_lag_features(df, ['score'])
            assert 'score_lag_1' in df.columns

        elif feature_type == 'rolling':
            generator._extract_rolling_features(df, ['score'])
            assert 'score_rolling_3_mean' in df.columns

        elif feature_type == 'trend':
            generator._extract_trend_features(df, ['score'])
            assert 'score_trend_slope' in df.columns

        elif feature_type == 'seasonality':
            generator._extract_seasonality_features(df, ['score'])
            assert 'score_seasonal_strength' in df.columns

        elif feature_type == 'change_point':
            generator._extract_change_point_features(df, ['score'])
            assert 'score_change_point_detected' in df.columns

        elif feature_type == 'frequency':
            generator._extract_frequency_features(df, ['score'])
            assert 'score_dominant_frequency' in df.columns

        elif feature_type == 'volatility':
            generator._extract_volatility_features(df, ['score'])
            assert 'score_volatility' in df.columns

        elif feature_type == 'autocorrelation':
            generator._extract_autocorrelation_features(df, ['score'])
            assert 'score_autocorr_lag1' in df.columns

        elif feature_type == 'missing_time':
            generator._extract_missing_time_features(df, 'time')
            assert 'missing_time_points' in df.columns

        elif feature_type == 'interaction':
            generator._extract_basic_time_features(df, 'time')
            generator._extract_interaction_features(df)
            assert 'hour_day_interaction' in df.columns