"""Tests for the feature transformation pipeline."""

import pytest
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from unittest.mock import Mock, patch

from src.portfolio.features.transformer import (
    FeatureTransformer,
    PipelineConfig,
    LogTransformer,
    BinningTransformer,
    CyclicalTransformer,
    TransformationError
)


class TestPipelineConfig:
    """Test cases for PipelineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()

        assert config.numeric_scaling_method == 'standard'
        assert config.categorical_encoding_method == 'onehot'
        assert config.handle_unknown_categories == 'ignore'
        assert config.numeric_imputation_strategy == 'mean'
        assert config.categorical_imputation_strategy == 'most_frequent'
        assert config.create_polynomial_features is False
        assert config.polynomial_degree == 2
        assert config.verbose is True

    def test_config_customization(self):
        """Test configuration customization."""
        # Custom config
        config = PipelineConfig(
            numeric_scaling_method='minmax',
            categorical_encoding_method='ordinal',
            numeric_imputation_strategy='median'
        )
        assert config.numeric_scaling_method == 'minmax'
        assert config.categorical_encoding_method == 'ordinal'
        assert config.numeric_imputation_strategy == 'median'


class TestCustomTransformers:
    """Test cases for custom transformers."""

    def test_log_transformer(self):
        """Test log transformer."""
        transformer = LogTransformer()

        # Test positive values
        X = pd.DataFrame({'positive': [1, 2, 3, 4, 5]})
        result = transformer.transform(X)

        assert result.shape == X.shape
        assert np.allclose(result['positive'], np.log(X['positive']))

        # Test zero values
        X_zero = pd.DataFrame({'zero': [0, 1, 2]})
        result_zero = transformer.transform(X_zero)

        assert result_zero['zero'].iloc[0] == 0  # log(1) after adding 1

        # Test negative values
        X_neg = pd.DataFrame({'negative': [-1, 0, 1]})
        result_neg = transformer.transform(X_neg)

        assert not np.isinf(result_neg).any().any()

    def test_binning_transformer(self):
        """Test binning transformer."""
        transformer = BinningTransformer(n_bins=3)

        # Test with fixed number of bins
        X = pd.DataFrame({'numeric': np.random.randn(100)})
        result = transformer.fit_transform(X)

        assert result.shape == (100, 1)
        assert len(result['numeric'].unique()) <= 3
        assert result['numeric'].min() >= 0
        assert result['numeric'].max() <= 2

        # Test with custom quantiles
        transformer_quantile = BinningTransformer(bins=[0, 0.5, 1.0], strategy='quantile')
        result_quantile = transformer_quantile.fit_transform(X)

        assert result_quantile['numeric'].nunique() <= 3

    def test_cyclical_transformer(self):
        """Test cyclical transformer."""
        transformer = CyclicalTransformer()

        # Test with month data
        X = pd.DataFrame({'month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})
        result = transformer.transform(X)

        assert result.shape == (12, 2)
        assert 'month_sin' in result.columns
        assert 'month_cos' in result.columns

        # Check sine and cosine properties
        sin_vals = result['month_sin'].values
        cos_vals = result['month_cos'].values

        # Values should be between -1 and 1
        assert np.all(sin_vals >= -1) and np.all(sin_vals <= 1)
        assert np.all(cos_vals >= -1) and np.all(cos_vals <= 1)

        # Check that sine and cosine are approximately unit vectors
        radii = np.sqrt(sin_vals**2 + cos_vals**2)
        assert np.allclose(radii, 1.0, rtol=1e-10)


class TestFeatureTransformer:
    """Test cases for FeatureTransformer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'numeric1': np.random.randn(100),
            'numeric2': np.random.randn(100) * 2 + 5,
            'categorical1': np.random.choice(['A', 'B', 'C'], 100),
            'categorical2': np.random.choice(['X', 'Y'], 100),
            'binary': np.random.choice([0, 1], 100),
            'date_col': pd.date_range('2020-01-01', periods=100, freq='D')
        })

    @pytest.fixture
    def sample_target(self):
        """Create sample target for testing."""
        np.random.seed(42)
        return pd.Series(np.random.randn(100))

    def test_initialization(self):
        """Test FeatureTransformer initialization."""
        # Default config
        transformer = FeatureTransformer()
        assert transformer.config.numeric_scaling_method == 'standard'

        # Custom config
        config = PipelineConfig(numeric_scaling_method='minmax')
        transformer = FeatureTransformer(config)
        assert transformer.config.numeric_scaling_method == 'minmax'

    def test_fit_basic(self, sample_data):
        """Test basic fit functionality."""
        transformer = FeatureTransformer()
        transformer.fit(sample_data)

        # Check that transformers are fitted
        assert transformer.is_fitted_ == True
        assert hasattr(transformer, 'numeric_features_')
        assert hasattr(transformer, 'categorical_features_')
        assert hasattr(transformer, 'scaler_')
        assert hasattr(transformer, 'encoder_')
        assert hasattr(transformer, 'imputer_')

        # Check feature identification
        assert 'numeric1' in transformer.numeric_features_
        assert 'numeric2' in transformer.numeric_features_
        assert 'categorical1' in transformer.categorical_features_
        assert 'binary' in transformer.numeric_features_  # Binary treated as numeric

    def test_transform_basic(self, sample_data):
        """Test basic transform functionality."""
        transformer = FeatureTransformer()
        transformer.fit(sample_data)

        transformed = transformer.transform(sample_data)

        # Check output structure
        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(sample_data)

        # Check that transformations were applied
        # Numeric features should be scaled (approximate check)
        numeric_cols = [col for col in transformed.columns if col.startswith('numeric')]
        if numeric_cols:
            assert abs(transformed[numeric_cols].mean().mean()) < 1.0  # Standard scaling

    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        transformer = FeatureTransformer()
        transformed = transformer.fit_transform(sample_data)

        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(sample_data)
        assert transformer.is_fitted_ == True

    def test_column_identification(self, sample_data):
        """Test column type identification."""
        transformer = FeatureTransformer()
        transformer._identify_column_types(sample_data)

        # Check numeric columns
        assert 'numeric1' in transformer.numeric_features_
        assert 'numeric2' in transformer.numeric_features_
        assert 'binary' in transformer.numeric_features_

        # Check categorical columns
        assert 'categorical1' in transformer.categorical_features_
        assert 'categorical2' in transformer.categorical_features_

        # Check date columns
        assert 'date_col' in transformer.date_features_

    def test_outlier_detection(self):
        """Test outlier detection and handling."""
        transformer = FeatureTransformer(
            PipelineConfig(handle_outliers=True, outlier_method='iqr')
        )

        # Create data with outliers
        np.random.seed(42)
        normal_data = np.random.randn(95)
        outliers = np.array([10, -10, 15, -15, 20])
        data = np.concatenate([normal_data, outliers])

        X = pd.DataFrame({'with_outliers': data})

        # Test outlier detection
        outlier_mask = transformer._detect_outliers(X['with_outliers'])
        assert len(outlier_mask) == len(X)
        assert outlier_mask.sum() >= 5  # Should detect at least the manual outliers

        # Test outlier capping
        X_capped = transformer._handle_outliers(X)
        assert X_capped['with_outliers'].max() < 20  # Outliers should be capped

    def test_encoding_methods(self):
        """Test different encoding methods."""
        # Test data
        X = pd.DataFrame({
            'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
            'numeric': np.random.randn(100)
        })

        # Test One-Hot encoding
        transformer_onehot = FeatureTransformer(
            PipelineConfig(encoding_method='onehot')
        )
        X_onehot = transformer_onehot.fit_transform(X)

        # Should have columns for each category
        cat_cols = [col for col in X_onehot.columns if col.startswith('category_')]
        assert len(cat_cols) == 4  # A, B, C, D

        # Test Label encoding
        transformer_label = FeatureTransformer(
            PipelineConfig(encoding_method='label')
        )
        X_label = transformer_label.fit_transform(X)

        # Should have single column with integer values
        assert 'category' in X_label.columns
        assert X_label['category'].dtype in ['int64', 'float64']

    def test_scaling_methods(self):
        """Test different scaling methods."""
        X = pd.DataFrame({
            'feature1': np.random.randn(100) * 5 + 10,
            'feature2': np.random.randn(100) * 2 + 5
        })

        # Test StandardScaler
        transformer_std = FeatureTransformer(
            PipelineConfig(scaling_method='standard')
        )
        X_std = transformer_std.fit_transform(X)

        # Check mean close to 0 and std close to 1
        assert abs(X_std.mean().mean()) < 0.1
        assert abs(X_std.std().mean() - 1.0) < 0.1

        # Test MinMaxScaler
        transformer_minmax = FeatureTransformer(
            PipelineConfig(scaling_method='minmax')
        )
        X_minmax = transformer_minmax.fit_transform(X)

        # Check values are between 0 and 1
        assert X_minmax.min().min() >= 0
        assert X_minmax.max().max() <= 1

        # Test RobustScaler
        transformer_robust = FeatureTransformer(
            PipelineConfig(scaling_method='robust')
        )
        X_robust = transformer_robust.fit_transform(X)

        # Should be scaled but not necessarily in [0, 1]
        assert isinstance(X_robust, pd.DataFrame)

    def test_missing_value_imputation(self):
        """Test missing value imputation."""
        # Create data with missing values
        X = pd.DataFrame({
            'numeric': [1, 2, np.nan, 4, 5],
            'categorical': ['A', 'B', None, 'B', 'C']
        })

        # Test mean imputation
        transformer_mean = FeatureTransformer(
            PipelineConfig(imputation_strategy='mean')
        )
        X_filled = transformer_mean.fit_transform(X)

        assert not X_filled.isnull().any().any()
        assert np.isclose(X_filled['numeric'].mean(), 3.0)  # Mean of [1,2,4,5]

        # Test median imputation
        transformer_median = FeatureTransformer(
            PipelineConfig(imputation_strategy='median')
        )
        X_filled_median = transformer_median.fit_transform(X)

        assert np.isclose(X_filled_median['numeric'].median(), 3.0)

        # Test most_frequent for categorical
        transformer_freq = FeatureTransformer(
            PipelineConfig(imputation_strategy='most_frequent')
        )
        X_filled_freq = transformer_freq.fit_transform(X)

        # Check categorical column has no missing values
        cat_cols = [col for col in X_filled_freq.columns
                   if col.startswith('categorical')]
        for col in cat_cols:
            assert not X_filled_freq[col].isnull().any()

    def test_custom_transformers(self):
        """Test custom transformer integration."""
        # Create data with log-normal and cyclical features
        X = pd.DataFrame({
            'log_normal': np.random.lognormal(0, 1, 100),
            'month': np.random.randint(1, 13, 100),
            'regular': np.random.randn(100)
        })

        # Define custom transformers
        custom_transformers = {
            'log_normal': LogTransformer(),
            'month': CyclicalTransformer()
        }

        config = PipelineConfig(custom_transformers=custom_transformers)
        transformer = FeatureTransformer(config)

        X_transformed = transformer.fit_transform(X)

        # Check that transformations were applied
        assert 'log_normal' in X_transformed.columns  # Should be preserved
        assert 'month_sin' in X_transformed.columns  # Cyclical transformation
        assert 'month_cos' in X_transformed.columns
        assert 'regular' in X_transformed.columns  # Should remain unchanged

    def test_train_test_consistency(self):
        """Test consistency between train and test transformations."""
        # Create train and test data
        np.random.seed(42)
        X_train = pd.DataFrame({
            'numeric': np.random.randn(100),
            'categorical': np.random.choice(['A', 'B', 'C'], 100)
        })
        X_test = pd.DataFrame({
            'numeric': np.random.randn(50),
            'categorical': np.random.choice(['A', 'B', 'C'], 50)
        })

        transformer = FeatureTransformer()
        transformer.fit(X_train)

        X_train_transformed = transformer.transform(X_train)
        X_test_transformed = transformer.transform(X_test)

        # Check consistent column names
        assert set(X_train_transformed.columns) == set(X_test_transformed.columns)

        # Check consistent shapes
        assert len(X_train_transformed) == len(X_train)
        assert len(X_test_transformed) == len(X_test)

        # Check that test data is properly scaled
        # (should use parameters learned from training data)
        numeric_cols = [col for col in X_test_transformed.columns
                       if col.startswith('numeric')]
        if numeric_cols:
            # Values shouldn't be exactly 0 mean, 1 std (using train params)
            assert isinstance(X_test_transformed[numeric_cols], pd.DataFrame)

    def test_get_feature_names(self, sample_data):
        """Test getting feature names."""
        transformer = FeatureTransformer()
        transformer.fit(sample_data)

        feature_names = transformer.get_feature_names_out()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 0

        # Check that all returned names exist in transformed data
        transformed = transformer.transform(sample_data)
        assert all(name in transformed.columns for name in feature_names)

    def test_get_transform_info(self, sample_data):
        """Test getting transformation information."""
        transformer = FeatureTransformer()
        transformer.fit(sample_data)

        info = transformer.get_transform_info()

        assert isinstance(info, dict)
        assert 'numeric_features' in info
        assert 'categorical_features' in info
        assert 'date_features' in info
        assert 'outliers_handled' in info
        assert 'transformers_applied' in info
        assert 'final_feature_count' in info

    def test_error_handling(self):
        """Test error handling."""
        transformer = FeatureTransformer()

        # Test transform before fit
        with pytest.raises(TransformationError):
            transformer.transform(pd.DataFrame({'a': [1, 2, 3]}))

        # Test with empty DataFrame
        with pytest.raises(TransformationError):
            transformer.fit(pd.DataFrame())

        # Test with non-DataFrame input
        with pytest.raises(TransformationError):
            transformer.fit([1, 2, 3])

    def test_edge_cases(self):
        """Test edge cases."""
        transformer = FeatureTransformer()

        # Test with single column
        X_single = pd.DataFrame({'single': [1, 2, 3, 4, 5]})
        X_transformed = transformer.fit_transform(X_single)
        assert X_transformed.shape == (5, 1)

        # Test with all categorical
        X_cat = pd.DataFrame({
            'cat1': ['A', 'B'] * 5,
            'cat2': ['X', 'Y'] * 5
        })
        X_cat_transformed = transformer.fit_transform(X_cat)
        assert len(X_cat_transformed) == 10

        # Test with all numeric
        X_num = pd.DataFrame({
            'num1': [1, 2, 3, 4, 5],
            'num2': [10, 20, 30, 40, 50]
        })
        X_num_transformed = transformer.fit_transform(X_num)
        assert X_num_transformed.shape == (5, 2)

    def test_inverse_transform(self, sample_data):
        """Test inverse transformation."""
        transformer = FeatureTransformer()

        # Only test with standard scaler (has inverse)
        config = PipelineConfig(scaling_method='standard')
        transformer = FeatureTransformer(config)

        X_transformed = transformer.fit_transform(sample_data[['numeric1', 'numeric2']])
        X_original = transformer.inverse_transform(X_transformed)

        # Check that inverse transform returns similar values
        # (allowing for small numerical differences)
        assert isinstance(X_original, pd.DataFrame)
        assert X_original.shape == X_transformed.shape

        # Compare with original (within tolerance)
        for col in ['numeric1', 'numeric2']:
            if col in X_original.columns:
                np.testing.assert_allclose(
                    X_original[col].values,
                    sample_data[col].values,
                    rtol=1e-10,
                    atol=1e-10
                )

    def test_feature_selection_integration(self, sample_data, sample_target):
        """Test integration with feature selection."""
        config = PipelineConfig(
            feature_selection=True,
            selection_method='mutual_info',
            n_features_to_select=5
        )
        transformer = FeatureTransformer(config)

        # Need to provide target for selection
        X_transformed = transformer.fit_transform(sample_data, sample_target)

        # Should have reduced number of features
        assert len(X_transformed.columns) <= 5