"""
Property-based tests for feature documentation.

This module uses Hypothesis to verify that the feature documenter
produces valid, consistent documentation across diverse feature types.
"""

import pytest
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from hypothesis import given, strategies as st, settings, HealthCheck

from src.portfolio.features.feature_documenter import (
    FeatureDocumenter,
    DocumentationConfig,
    FeatureDescription,
    FeatureStats
)


class TestFeatureDocumentationProperties:
    """Property-based tests for feature documentation."""

    @pytest.fixture
    def documenter(self):
        """Create a feature documenter for property testing."""
        config = DocumentationConfig(
            correlation_threshold=0.8,
            outlier_method='iqr',
            output_dir="test_output",
            include_plots=False,
            include_statistics=True,
            include_analysis=True
        )
        return FeatureDocumenter(config)

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=15
    )
    @given(st.data())
    def test_documentation_completeness(self, documenter, data):
        """
        Property: Feature documentation contains all required fields.

        Given any analyzed feature, documentation should have required information.
        """
        # Generate sample data
        n_samples = data.draw(st.integers(min_value=50, max_value=200))

        # Create numeric feature
        np.random.seed(42)
        X = pd.DataFrame({
            'test_feature': np.random.randn(n_samples)
        })
        y = pd.Series(np.random.randn(n_samples))

        # Analyze feature
        feature_doc = documenter.analyze_feature(
            'test_feature', X, y,
            description='Test numeric feature'
        )

        # Check required fields
        assert feature_doc.name == 'test_feature'
        assert feature_doc.description is not None
        assert feature_doc.feature_type is not None
        assert feature_doc.source is not None
        assert feature_doc.creation_date is not None
        assert isinstance(feature_doc.creation_date, str)

        # Check statistics
        assert feature_doc.statistics is not None
        assert isinstance(feature_doc.statistics, FeatureStats)
        assert feature_doc.statistics.name == 'test_feature'
        assert feature_doc.statistics.count == n_samples

        # Check analysis (may be None for some edge cases)
        if feature_doc.analysis is not None:
            assert feature_doc.analysis.name == 'test_feature'

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=10
    )
    @given(st.data())
    def test_numeric_statistics_properties(self, documenter, data):
        """
        Property: Numeric statistics have valid ranges and relationships.

        Given any numeric feature, statistics should be mathematically valid.
        """
        # Generate numeric data
        n_samples = data.draw(st.integers(min_value=100, max_value=500))
        mean = data.draw(st.floats(min_value=-10, max_value=10))
        std = data.draw(st.floats(min_value=0.1, max_value=10))

        # Create controlled numeric feature
        np.random.seed(42)
        X = pd.DataFrame({
            'numeric_feature': np.random.randn(n_samples) * std + mean
        })

        stats = documenter._calculate_statistics(X['numeric_feature'])

        # Check basic properties
        assert stats.count == n_samples
        assert stats.mean is not None
        assert stats.std is not None
        assert stats.min_value is not None
        assert stats.max_value is not None

        # Check ordering of quartiles
        if stats.q25 is not None and stats.q50 is not None and stats.q75 is not None:
            assert stats.q25 <= stats.q50 <= stats.q75

        # Check range consistency
        if stats.min_value is not None and stats.max_value is not None:
            assert stats.min_value <= stats.max_value

        # Check standard deviation is non-negative
        assert stats.std >= 0

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=10
    )
    @given(st.data())
    def test_categorical_statistics_properties(self, documenter, data):
        """
        Property: Categorical statistics correctly summarize categories.

        Given any categorical feature, statistics should reflect categories.
        """
        # Generate categorical data
        n_samples = data.draw(st.integers(min_value=50, max_value=200))
        n_categories = data.draw(st.integers(min_value=2, max_value=10))

        categories = [f'cat_{i}' for i in range(n_categories)]
        np.random.seed(42)
        X = pd.DataFrame({
            'categorical_feature': np.random.choice(categories, n_samples)
        })

        stats = documenter._calculate_statistics(X['categorical_feature'])

        # Check basic properties
        assert stats.count == n_samples
        assert stats.most_frequent is not None
        assert stats.most_frequent_count is not None
        assert stats.most_frequent_percentage is not None
        assert stats.n_categories is not None

        # Should not exceed actual categories
        assert stats.n_categories <= len(categories)

        # Percentage should be valid
        if stats.n_categories > 0:
            assert 0 <= stats.most_frequent_percentage <= 100

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=10
    )
    @given(st.data())
    def test_missing_value_properties(self, documenter, data):
        """
        Property: Missing value statistics are accurate.

        Given any feature with missing values, null statistics should be correct.
        """
        # Generate data with missing values
        n_samples = data.draw(st.integers(min_value=100, max_value=300))
        missing_ratio = data.draw(st.floats(min_value=0, max_value=0.5))
        n_missing = int(n_samples * missing_ratio)

        np.random.seed(42)
        feature = np.random.randn(n_samples)
        feature[:n_missing] = np.nan

        X = pd.DataFrame({'feature_with_missing': feature})
        stats = documenter._calculate_statistics(X['feature_with_missing'])

        # Check missing value calculations
        assert stats.null_count == n_missing
        assert abs(stats.null_percentage - (100 * n_missing / n_samples)) < 0.01

        # Valid count should be total minus missing
        assert stats.count == n_samples - n_missing

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=10
    )
    @given(st.data())
    def test_feature_type_detection_consistency(self, documenter, data):
        """
        Property: Feature type detection is consistent with data characteristics.

        Given any data, detected type should match data properties.
        """
        # Generate different types of features
        n_samples = data.draw(st.integers(min_value=50, max_value=200))

        # Numeric feature
        numeric_data = np.random.randn(n_samples)
        numeric_type = documenter._detect_feature_type(pd.Series(numeric_data))
        assert numeric_type in ['numeric', 'ordinal']

        # Categorical feature (few unique values)
        cat_data = np.random.choice(['A', 'B', 'C'], n_samples)
        cat_type = documenter._detect_feature_type(pd.Series(cat_data))
        assert cat_type in ['categorical']

        # Boolean feature
        bool_data = np.random.choice([True, False], n_samples)
        bool_type = documenter._detect_feature_type(pd.Series(bool_data))
        assert bool_type == 'boolean'

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=8
    )
    @given(st.data())
    def test_documentation_generation_format(self, documenter, data):
        """
        Property: Generated documentation follows expected format.

        Given any feature, exported documentation should be valid format.
        """
        # Generate test data
        n_samples = 100
        X = pd.DataFrame({'test_feature': np.random.randn(n_samples)})
        y = pd.Series(np.random.randn(n_samples))

        # Analyze feature
        documenter.analyze_feature(
            'test_feature', X, y,
            description='Test feature for format validation'
        )

        # Test markdown export
        md_content = documenter.export_feature_documentation(
            'test_feature', format='markdown'
        )
        assert isinstance(md_content, str)
        assert '# test_feature' in md_content

        # Test JSON export
        json_content = documenter.export_feature_documentation(
            'test_feature', format='json'
        )
        assert isinstance(json_content, str)
        # Should be valid JSON
        import json
        parsed = json.loads(json_content)
        assert 'name' in parsed
        assert parsed['name'] == 'test_feature'

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=8
    )
    @given(st.data())
    def test_catalog_structure(self, documenter, data):
        """
        Property: Feature catalog has consistent structure.

        Given any set of documented features, catalog should have required columns.
        """
        # Generate multiple features
        n_features = data.draw(st.integers(min_value=3, max_value=10))
        n_samples = 100

        feature_data = {}
        for i in range(n_features):
            feature_data[f'feature_{i}'] = np.random.randn(n_samples)

        X = pd.DataFrame(feature_data)
        y = pd.Series(np.random.randn(n_samples))

        # Analyze all features
        documenter.analyze_dataset(X, y)

        # Generate catalog
        catalog = documenter.generate_catalog()

        # Check catalog structure
        assert isinstance(catalog, pd.DataFrame)
        assert len(catalog) == n_features

        # Check required columns
        required_cols = ['name', 'type', 'description', 'source']
        for col in required_cols:
            assert col in catalog.columns

        # All feature names should be present
        for i in range(n_features):
            assert f'feature_{i}' in catalog['name'].values

    def test_outlier_detection_bounds(self, documenter):
        """
        Property: Outlier detection returns valid counts.

        Outlier counts should be non-negative and not exceed sample size.
        """
        # Test with normal data (few outliers)
        normal_data = np.random.randn(100)
        outlier_count, outlier_pct = documenter._detect_outliers(
            pd.Series(normal_data), 'iqr', 1.5
        )
        assert 0 <= outlier_count <= 100
        assert 0 <= outlier_pct <= 100

        # Test with extreme outliers
        extreme_data = np.concatenate([normal_data, [50, -50, 100]])
        outlier_count, outlier_pct = documenter._detect_outliers(
            pd.Series(extreme_data), 'iqr', 1.5
        )
        assert outlier_count >= 3  # At least our manual outliers
        assert 0 <= outlier_pct <= 100

    def test_consistency_score_range(self, documenter):
        """
        Property: Consistency score is always between 0 and 1.

        For any data, consistency score should be a valid percentage.
        """
        # Test with consistent data
        consistent_data = pd.Series([1] * 100)
        consistency = documenter._calculate_consistency(consistent_data)
        assert consistency == 1.0

        # Test with variable data
        variable_data = pd.Series(np.random.randn(100) * 100)
        consistency = documenter._calculate_consistency(variable_data)
        assert 0 <= consistency <= 1

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        max_examples=5
    )
    @given(st.data())
    def test_search_filter_logic(self, documenter, data):
        """
        Property: Feature search filters work correctly.

        Given search criteria, only matching features should be returned.
        """
        # Create features with different properties
        X = pd.DataFrame({
            'numeric_test': np.random.randn(100),
            'categorical_test': np.random.choice(['A', 'B'], 100)
        })
        y = pd.Series(np.random.randn(100))

        # Analyze with different metadata
        documenter.analyze_feature(
            'numeric_test', X, y,
            tags=['important', 'numeric'],
            owner='analyst'
        )
        documenter.analyze_feature(
            'categorical_test', X, y,
            tags=['categorical', 'test'],
            owner='scientist'
        )

        # Test search by tag
        results = documenter.search_features(tag='important')
        assert 'numeric_test' in results
        assert 'categorical_test' not in results

        # Test search by owner
        results = documenter.search_features(owner='scientist')
        assert 'categorical_test' in results
        assert 'numeric_test' not in results

        # Test search by type
        results = documenter.search_features(feature_type='numeric')
        assert 'numeric_test' in results
        assert 'categorical_test' not in results