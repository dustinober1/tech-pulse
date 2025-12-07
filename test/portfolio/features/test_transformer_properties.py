"""Property-based tests for feature transformation pipeline.

This module uses Hypothesis to verify that the FeatureTransformer
produces valid, consistent transformations across diverse datasets.
"""

import pytest
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from hypothesis import given, strategies as st, settings, HealthCheck

from src.portfolio.features.transformer import (
    FeatureTransformer,
    PipelineConfig,
    TransformationError
)


class TestFeatureTransformerProperties:
    """Property-based tests for FeatureTransformer."""

    @pytest.fixture
    def transformer(self):
        """Create a feature transformer for property testing."""
        config = PipelineConfig(
            scaling_method='standard',
            encoding_method='onehot',
            handle_outliers=False,
            feature_selection=False,
            random_state=42
        )
        return FeatureTransformer(config)

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=15
    )
    @given(st.data())
    def test_transformation_preserves_rows(self, transformer, data):
        """
        Property: Transformation preserves the number of rows.

        Given any dataset, fitting and transforming should not change row count.
        """
        # Generate sample data
        n_rows = data.draw(st.integers(min_value=20, max_value=100))
        n_cols = data.draw(st.integers(min_value=2, max_value=8))

        # Create mixed data
        np.random.seed(42)
        df_dict = {}

        # Add numeric columns
        for i in range(n_cols // 2):
            df_dict[f'numeric_{i}'] = np.random.randn(n_rows)

        # Add categorical columns
        if n_cols > 1:
            categories = [f'cat_{j}' for j in range(5)]
            for i in range(n_cols - n_cols // 2):
                df_dict[f'cat_{i}'] = np.random.choice(categories, n_rows)

        df = pd.DataFrame(df_dict)

        # Apply transformation
        df_transformed = transformer.fit_transform(df)

        # Check row preservation
        assert len(df_transformed) == n_rows

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=10
    )
    @given(st.data())
    def test_transformed_data_has_no_nulls(self, transformer, data):
        """
        Property: Transformed data has no missing values.

        Given any dataset with missing values, transformation should handle them.
        """
        # Generate data with missing values
        n_rows = data.draw(st.integers(min_value=30, max_value=100))
        missing_ratio = data.draw(st.floats(min_value=0, max_value=0.3))

        np.random.seed(42)
        data_dict = {
            'numeric': np.random.randn(n_rows),
            'categorical': np.random.choice(['A', 'B', 'C'], n_rows)
        }

        df = pd.DataFrame(data_dict)

        # Add missing values
        n_missing = int(n_rows * missing_ratio)
        if n_missing > 0:
            missing_idx = np.random.choice(n_rows, n_missing, replace=False)
            df.loc[missing_idx, 'numeric'] = np.nan

        # Apply transformation
        df_transformed = transformer.fit_transform(df)

        # Check no missing values
        assert not df_transformed.isnull().any().any()

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=10
    )
    @given(st.data())
    def test_scaling_properties(self, data):
        """
        Property: Scaling transforms numeric data appropriately.

        Given numeric data, scaling should produce values with expected properties.
        """
        n_rows = data.draw(st.integers(min_value=50, max_value=200))
        scaling_method = data.draw(st.sampled_from(['standard', 'minmax', 'robust']))

        # Generate numeric data
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(n_rows) * 10 + 5,
            'feature2': np.random.randn(n_rows) * 5 - 2,
            'feature3': np.random.uniform(0, 100, n_rows)
        })

        # Apply specific scaling
        config = PipelineConfig(scaling_method=scaling_method, encoding_method=None)
        transformer = FeatureTransformer(config)
        df_scaled = transformer.fit_transform(df)

        # Check scaling properties
        if scaling_method == 'standard':
            # Mean should be close to 0, std close to 1
            assert abs(df_scaled.mean().mean()) < 0.1
            assert abs(df_scaled.std().mean() - 1.0) < 0.1

        elif scaling_method == 'minmax':
            # Values should be between 0 and 1
            assert df_scaled.min().min() >= 0
            assert df_scaled.max().max() <= 1

        elif scaling_method == 'robust':
            # Should be scaled but more robust to outliers
            assert isinstance(df_scaled, pd.DataFrame)

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=10
    )
    @given(st.data())
    def test_encoding_creates_correct_columns(self, data):
        """
        Property: Categorical encoding creates appropriate columns.

        Given categorical data, encoding should create correct number of columns.
        """
        n_rows = data.draw(st.integers(min_value=20, max_value=100))
        n_categories = data.draw(st.integers(min_value=2, max_value=10))
        encoding_method = data.draw(st.sampled_from(['onehot', 'label']))

        # Generate categorical data
        categories = [f'cat_{i}' for i in range(n_categories)]
        np.random.seed(42)
        df = pd.DataFrame({
            'category': np.random.choice(categories, n_rows),
            'numeric': np.random.randn(n_rows)
        })

        # Apply encoding
        config = PipelineConfig(encoding_method=encoding_method)
        transformer = FeatureTransformer(config)
        df_encoded = transformer.fit_transform(df)

        # Check encoding results
        cat_cols = [col for col in df_encoded.columns if col.startswith('category_')]

        if encoding_method == 'onehot':
            # Should have columns for each category (or n-1 depending on drop_first)
            assert len(cat_cols) >= n_categories - 1
            assert len(cat_cols) <= n_categories

        elif encoding_method == 'label':
            # Should have single column with integer values
            assert 'category' in df_encoded.columns
            assert df_encoded['category'].dtype in ['int64', 'float64']

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=10
    )
    @given(st.data())
    def test_outlier_handling_properties(self, data):
        """
        Property: Outlier handling caps extreme values appropriately.

        Given data with outliers, outlier handling should reduce extreme values.
        """
        n_rows = data.draw(st.integers(min_value=50, max_value=200))
        outlier_method = data.draw(st.sampled_from(['iqr', 'zscore']))

        # Generate data with outliers
        np.random.seed(42)
        normal_data = np.random.randn(n_rows - 10)
        outliers = np.random.uniform(10, 20, 10)
        data_with_outliers = np.concatenate([normal_data, outliers])

        df = pd.DataFrame({
            'with_outliers': data_with_outliers,
            'normal': np.random.randn(n_rows)
        })

        # Apply outlier handling
        config = PipelineConfig(
            handle_outliers=True,
            outlier_method=outlier_method
        )
        transformer = FeatureTransformer(config)
        df_handled = transformer.fit_transform(df)

        # Check that extreme values were reduced
        outlier_col = [col for col in df_handled.columns if 'outliers' in col][0]

        # Max value should be significantly reduced
        assert df_handled[outlier_col].max() < df['with_outliers'].max()

        # Should preserve most data (not all values should be identical)
        assert df_handled[outlier_col].nunique() > 1

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=10
    )
    @given(st.data())
    def test_train_test_consistency_properties(self, transformer, data):
        """
        Property: Train and test transformations are consistent.

        Given train and test splits, transformations use same parameters.
        """
        n_train = data.draw(st.integers(min_value=50, max_value=150))
        n_test = data.draw(st.integers(min_value=20, max_value=50))

        # Generate train and test data
        np.random.seed(42)
        X_train = pd.DataFrame({
            'numeric1': np.random.randn(n_train),
            'numeric2': np.random.randn(n_train) * 2 + 1,
            'categorical': np.random.choice(['A', 'B', 'C'], n_train)
        })

        X_test = pd.DataFrame({
            'numeric1': np.random.randn(n_test),
            'numeric2': np.random.randn(n_test) * 2 + 1,
            'categorical': np.random.choice(['A', 'B', 'C'], n_test)
        })

        # Fit on train, transform both
        transformer.fit(X_train)
        X_train_transformed = transformer.transform(X_train)
        X_test_transformed = transformer.transform(X_test)

        # Check column consistency
        assert set(X_train_transformed.columns) == set(X_test_transformed.columns)

        # Check shape consistency (same columns, different rows)
        assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
        assert X_train_transformed.shape[0] == n_train
        assert X_test_transformed.shape[0] == n_test

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=8
    )
    @given(st.data())
    def test_fit_transform_idempotency(self, transformer, data):
        """
        Property: Multiple fits produce consistent results.

        Given same data, multiple fits should produce identical parameters.
        """
        # Generate data
        n_rows = data.draw(st.integers(min_value=30, max_value=100))
        np.random.seed(42)
        df = pd.DataFrame({
            'feature': np.random.randn(n_rows),
            'category': np.random.choice(['X', 'Y'], n_rows)
        })

        # Fit and transform twice
        X1 = transformer.fit_transform(df.copy())
        transformer_copy = FeatureTransformer(transformer.config)
        X2 = transformer_copy.fit_transform(df.copy())

        # Results should be identical (within numerical precision)
        assert X1.shape == X2.shape
        assert list(X1.columns) == list(X2.columns)

        # Compare values (allowing for small numerical differences)
        np.testing.assert_allclose(
            X1.values,
            X2.values,
            rtol=1e-10,
            atol=1e-10
        )

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=8
    )
    @given(st.data())
    def test_feature_names_consistency(self, transformer, data):
        """
        Property: Feature names are consistent and descriptive.

        Given any transformation, feature names should be consistent.
        """
        # Generate data
        n_rows = data.draw(st.integers(min_value=20, max_value=80))
        n_numeric = data.draw(st.integers(min_value=1, max_value=4))
        n_categorical = data.draw(st.integers(min_value=1, max_value=4))

        np.random.seed(42)
        df_dict = {}

        for i in range(n_numeric):
            df_dict[f'num_{i}'] = np.random.randn(n_rows)

        for i in range(n_categorical):
            df_dict[f'cat_{i}'] = np.random.choice(['A', 'B'], n_rows)

        df = pd.DataFrame(df_dict)

        # Apply transformation
        X_transformed = transformer.fit_transform(df)

        # Get feature names
        feature_names = transformer.get_feature_names_out()

        # Check consistency
        assert len(feature_names) == X_transformed.shape[1]
        assert all(name in X_transformed.columns for name in feature_names)

        # Check that names are strings
        assert all(isinstance(name, str) for name in feature_names)

        # Check that original features are traceable
        for original_col in df.columns:
            # At least one transformed feature should reference original
            matching_features = [name for name in feature_names
                               if original_col in name]
            assert len(matching_features) > 0

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=8
    )
    @given(st.data())
    def test_transformation_info_properties(self, transformer, data):
        """
        Property: Transformation info provides accurate metadata.

        Given any fitted transformer, info should be accurate.
        """
        # Generate diverse data
        n_rows = data.draw(st.integers(min_value=30, max_value=100))
        n_cols = data.draw(st.integers(min_value=3, max_value=7))

        np.random.seed(42)
        df_dict = {}

        # Mix of feature types
        has_numeric = data.draw(st.booleans())
        has_categorical = data.draw(st.booleans())
        has_datetime = data.draw(st.booleans())

        if has_numeric:
            df_dict['numeric'] = np.random.randn(n_rows)
        if has_categorical:
            df_dict['categorical'] = np.random.choice(['A', 'B'], n_rows)
        if has_datetime:
            df_dict['datetime'] = pd.date_range('2020-01-01', periods=n_rows, freq='D')

        # Add more columns to reach target
        for i in range(len(df_dict), n_cols):
            if i % 2 == 0:
                df_dict[f'num_{i}'] = np.random.randn(n_rows)
            else:
                df_dict[f'cat_{i}'] = np.random.choice(['X', 'Y'], n_rows)

        df = pd.DataFrame(df_dict)

        # Fit transformer
        transformer.fit(df)

        # Get info
        info = transformer.get_transform_info()

        # Check info structure
        required_keys = [
            'numeric_features',
            'categorical_features',
            'date_features',
            'outliers_handled',
            'transformers_applied',
            'final_feature_count'
        ]
        for key in required_keys:
            assert key in info

        # Check info accuracy
        if has_numeric:
            assert len(info['numeric_features']) > 0
        if has_categorical:
            assert len(info['categorical_features']) > 0
        if has_datetime:
            assert len(info['date_features']) > 0

        # Feature counts should be reasonable
        assert info['final_feature_count'] > 0
        assert isinstance(info['outliers_handled'], bool)

    def test_error_handling_properties(self):
        """
        Property: Error handling provides clear messages.

        Given invalid inputs, should raise appropriate errors.
        """
        transformer = FeatureTransformer()

        # Empty DataFrame
        with pytest.raises(TransformationError):
            transformer.fit(pd.DataFrame())

        # Transform before fit
        with pytest.raises(TransformationError):
            transformer.transform(pd.DataFrame({'a': [1, 2, 3]}))

        # Invalid input type
        with pytest.raises(TransformationError):
            transformer.fit([1, 2, 3])

        # Valid but challenging case - single column
        df_single = pd.DataFrame({'single': [1, 2, 3]})
        X_transformed = transformer.fit_transform(df_single)
        assert X_transformed.shape == (3, 1)

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=5
    )
    @given(st.data())
    def test_high_cardinality_handling(self, data):
        """
        Property: High cardinality categorical features are handled.

        Given categorical features with many unique values, transformation should work.
        """
        n_rows = data.draw(st.integers(min_value=50, max_value=200))
        n_categories = data.draw(st.integers(min_value=50, max_value=100))

        # Create high cardinality categorical data
        categories = [f'category_{i}' for i in range(n_categories)]
        np.random.seed(42)
        df = pd.DataFrame({
            'high_card': np.random.choice(categories, n_rows),
            'low_card': np.random.choice(['A', 'B', 'C'], n_rows),
            'numeric': np.random.randn(n_rows)
        })

        # Apply transformation
        transformer = FeatureTransformer()
        X_transformed = transformer.fit_transform(df)

        # Should handle high cardinality (might use label encoding or limit)
        assert X_transformed.shape[0] == n_rows
        assert X_transformed.shape[1] >= 1  # At least some features remain

        # Check no missing values
        assert not X_transformed.isnull().any().any()