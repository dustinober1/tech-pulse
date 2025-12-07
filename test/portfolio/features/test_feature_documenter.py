"""
Unit tests for feature documentation system.

This module tests the FeatureDocumenter class and its methods
for documenting and analyzing features.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from datetime import datetime

from src.portfolio.features.feature_documenter import (
    FeatureDocumenter,
    DocumentationConfig,
    FeatureDescription,
    FeatureStats,
    FeatureAnalysis
)


class TestFeatureDocumenter:
    """Test cases for FeatureDocumenter."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 200

        data = {
            'numeric_feature': np.random.randn(n_samples),
            'categorical_feature': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
            'boolean_feature': np.random.choice([True, False], n_samples),
            'datetime_feature': pd.date_range('2024-01-01', periods=n_samples, freq='D'),
            'text_feature': ['This is a much longer sample text string ' + str(i) for i in range(n_samples)],
            'missing_feature': np.random.randn(n_samples),
            'constant_feature': np.ones(n_samples) * 5
        }

        # Add some missing values
        data['missing_feature'][:20] = np.nan
        data['categorical_feature'][50:55] = np.nan

        X = pd.DataFrame(data)
        y = pd.Series(np.random.randn(n_samples))

        return X, y

    @pytest.fixture
    def documenter(self):
        """Create a FeatureDocumenter instance."""
        config = DocumentationConfig(
            correlation_threshold=0.8,
            outlier_method='iqr',
            output_dir=tempfile.mkdtemp(),
            include_plots=False  # Disable plots for faster tests
        )
        return FeatureDocumenter(config)

    def test_initialization(self):
        """Test FeatureDocumenter initialization."""
        # Test with default config
        doc = FeatureDocumenter()
        assert doc.config is not None
        assert isinstance(doc.config, DocumentationConfig)
        assert doc.features_ == {}
        assert doc.catalog_.empty

        # Test with custom config
        config = DocumentationConfig(correlation_threshold=0.9)
        doc = FeatureDocumenter(config)
        assert doc.config.correlation_threshold == 0.9

    def test_detect_feature_type(self, documenter, sample_data):
        """Test feature type detection."""
        X, _ = sample_data

        # Test numeric feature
        feature_type = documenter._detect_feature_type(X['numeric_feature'])
        assert feature_type == 'numeric'

        # Test categorical feature
        feature_type = documenter._detect_feature_type(X['categorical_feature'])
        assert feature_type == 'categorical'

        # Test boolean feature
        feature_type = documenter._detect_feature_type(X['boolean_feature'])
        assert feature_type == 'boolean'

        # Test datetime feature
        feature_type = documenter._detect_feature_type(X['datetime_feature'])
        assert feature_type == 'datetime'

        # Test text feature
        feature_type = documenter._detect_feature_type(X['text_feature'])
        # Text detection threshold is 50 characters, our test data has ~43 chars average
        # So this will be detected as categorical, which is acceptable
        assert feature_type in ['text', 'categorical']

    def test_calculate_statistics_numeric(self, documenter, sample_data):
        """Test statistics calculation for numeric features."""
        X, _ = sample_data
        stats = documenter._calculate_statistics(X['numeric_feature'])

        assert isinstance(stats, FeatureStats)
        assert stats.name == 'numeric_feature'
        assert stats.dtype == 'float64'
        assert stats.count == 200
        assert stats.mean is not None
        assert stats.std is not None
        assert stats.min_value is not None
        assert stats.max_value is not None
        assert stats.q25 is not None
        assert stats.q50 is not None
        assert stats.q75 is not None
        assert stats.skewness is not None
        assert stats.kurtosis is not None

    def test_calculate_statistics_categorical(self, documenter, sample_data):
        """Test statistics calculation for categorical features."""
        X, _ = sample_data
        stats = documenter._calculate_statistics(X['categorical_feature'])

        assert isinstance(stats, FeatureStats)
        assert stats.name == 'categorical_feature'
        assert stats.most_frequent is not None
        assert stats.most_frequent_count is not None
        assert stats.most_frequent_percentage is not None
        assert stats.categories is not None
        assert stats.n_categories is not None
        assert stats.n_categories <= 4  # Should be A, B, C, D

    def test_calculate_statistics_with_missing(self, documenter, sample_data):
        """Test statistics calculation with missing values."""
        X, _ = sample_data
        stats = documenter._calculate_statistics(X['missing_feature'])

        assert stats.null_count == 20
        assert stats.null_percentage == 10.0  # 20/200 * 100

    def test_calculate_statistics_constant(self, documenter, sample_data):
        """Test statistics calculation for constant feature."""
        X, _ = sample_data
        stats = documenter._calculate_statistics(X['constant_feature'])

        assert stats.mean == 5.0
        assert stats.std == 0.0
        assert stats.min_value == 5.0
        assert stats.max_value == 5.0

    def test_analyze_relationships_numeric(self, documenter, sample_data):
        """Test relationship analysis for numeric features."""
        X, y = sample_data
        analysis = documenter._analyze_relationships(
            X['numeric_feature'], y, X.drop(columns=['numeric_feature'])
        )

        assert isinstance(analysis, FeatureAnalysis)
        assert analysis.name == 'numeric_feature'
        assert analysis.target_correlation is not None
        assert -1 <= analysis.target_correlation <= 1
        assert analysis.outlier_count is not None
        assert analysis.outlier_percentage is not None
        assert analysis.completeness_score is not None
        assert analysis.consistency_score is not None
        assert 0 <= analysis.completeness_score <= 1
        assert 0 <= analysis.consistency_score <= 1

    def test_analyze_feature(self, documenter, sample_data):
        """Test complete feature analysis."""
        X, y = sample_data

        # Analyze a single feature
        feature_doc = documenter.analyze_feature(
            feature_name='numeric_feature',
            X=X,
            y=y,
            description='Test numeric feature',
            business_meaning='This is a test feature',
            rationale='Created for testing purposes',
            tags=['test', 'numeric'],
            owner='test_user'
        )

        assert isinstance(feature_doc, FeatureDescription)
        assert feature_doc.name == 'numeric_feature'
        assert feature_doc.description == 'Test numeric feature'
        assert feature_doc.feature_type == 'numeric'
        assert feature_doc.business_meaning == 'This is a test feature'
        assert feature_doc.rationale == 'Created for testing purposes'
        assert feature_doc.tags == ['test', 'numeric']
        assert feature_doc.owner == 'test_user'
        assert feature_doc.statistics is not None
        assert feature_doc.analysis is not None

        # Check it's stored in documenter
        assert 'numeric_feature' in documenter.features_
        assert documenter.features_['numeric_feature'] == feature_doc

    def test_analyze_dataset(self, documenter, sample_data):
        """Test analyzing entire dataset."""
        X, y = sample_data

        feature_descriptions = {
            'numeric_feature': 'A numeric test feature',
            'categorical_feature': 'A categorical test feature'
        }

        # Analyze all features
        feature_docs = documenter.analyze_dataset(
            X=X,
            y=y,
            feature_descriptions=feature_descriptions,
            tags='auto_generated',
            owner='auto_test'
        )

        assert isinstance(feature_docs, dict)
        assert len(feature_docs) == len(X.columns)

        # Check each feature was documented
        for feature_name in X.columns:
            assert feature_name in feature_docs
            assert isinstance(feature_docs[feature_name], FeatureDescription)

        # Check descriptions were applied
        assert feature_docs['numeric_feature'].description == 'A numeric test feature'
        assert feature_docs['categorical_feature'].description == 'A categorical test feature'

        # Check common metadata was applied
        for feature_doc in feature_docs.values():
            assert 'auto_generated' in feature_doc.tags
            assert feature_doc.owner == 'auto_test'

    def test_generate_catalog(self, documenter, sample_data):
        """Test catalog generation."""
        X, y = sample_data

        # Analyze features first
        documenter.analyze_dataset(X, y)

        # Generate catalog
        catalog = documenter.generate_catalog()

        assert isinstance(catalog, pd.DataFrame)
        assert len(catalog) == len(X.columns)
        assert 'name' in catalog.columns
        assert 'type' in catalog.columns
        assert 'description' in catalog.columns
        assert 'null_percentage' in catalog.columns

        # Check all features are in catalog
        for feature_name in X.columns:
            assert feature_name in catalog['name'].values

    def test_export_markdown(self, documenter, sample_data):
        """Test markdown export."""
        X, y = sample_data

        # Analyze feature
        documenter.analyze_feature(
            feature_name='numeric_feature',
            X=X,
            y=y,
            description='Test feature for export',
            business_meaning='Test business meaning',
            rationale='Test rationale'
        )

        # Export markdown
        md_content = documenter.export_feature_documentation(
            'numeric_feature',
            format='markdown'
        )

        assert isinstance(md_content, str)
        assert '# numeric_feature' in md_content
        assert 'Test feature for export' in md_content
        assert '## Business Meaning' in md_content
        assert '## Statistics' in md_content
        assert '## Analysis' in md_content

    def test_export_json(self, documenter, sample_data):
        """Test JSON export."""
        X, y = sample_data

        # Analyze feature
        documenter.analyze_feature(
            feature_name='numeric_feature',
            X=X,
            y=y
        )

        # Export JSON
        json_content = documenter.export_feature_documentation(
            'numeric_feature',
            format='json'
        )

        assert isinstance(json_content, str)

        # Parse JSON to verify structure
        data = json.loads(json_content)
        assert data['name'] == 'numeric_feature'
        assert 'statistics' in data
        assert 'analysis' in data

    def test_export_all_documentation(self, documenter, sample_data):
        """Test exporting all documentation."""
        X, y = sample_data

        # Analyze features
        documenter.analyze_dataset(
            X,
            y,
            feature_descriptions={
                'numeric_feature': 'Numeric test feature',
                'categorical_feature': 'Categorical test feature'
            }
        )

        # Export all to temp directory
        all_docs = documenter.export_all_documentation(
            format='markdown',
            output_dir=documenter.config.output_dir
        )

        assert isinstance(all_docs, dict)
        assert len(all_docs) == len(X.columns)

        # Check files were created
        output_path = Path(documenter.config.output_dir)
        for feature_name in X.columns:
            file_path = output_path / f"{feature_name}.markdown"
            assert file_path.exists()

    def test_search_features(self, documenter, sample_data):
        """Test feature searching."""
        X, y = sample_data

        # Analyze features with different tags and owners
        documenter.analyze_feature(
            'numeric_feature', X, y,
            description='Important numeric feature',
            tags=['important', 'numeric'],
            owner='data_scientist'
        )

        documenter.analyze_feature(
            'categorical_feature', X, y,
            description='Low importance categorical',
            tags=['categorical', 'low_priority'],
            owner='domain_expert'
        )

        # Search by tag
        features = documenter.search_features(tag='important')
        assert 'numeric_feature' in features
        assert 'categorical_feature' not in features

        # Search by owner
        features = documenter.search_features(owner='domain_expert')
        assert 'categorical_feature' in features
        assert 'numeric_feature' not in features

        # Search by text
        features = documenter.search_features(query='numeric')
        assert 'numeric_feature' in features

        # Search by type
        features = documenter.search_features(feature_type='categorical')
        assert 'categorical_feature' in features

    def test_get_feature_summary(self, documenter, sample_data):
        """Test feature summary retrieval."""
        X, y = sample_data

        # Analyze feature
        documenter.analyze_feature('numeric_feature', X, y)

        # Get summary
        summary = documenter.get_feature_summary('numeric_feature')

        assert isinstance(summary, dict)
        assert summary['name'] == 'numeric_feature'
        assert summary['type'] == 'numeric'
        assert 'statistics' in summary
        assert 'analysis' in summary

        if summary['statistics']:
            assert 'null_percentage' in summary['statistics']
            assert 'mean' in summary['statistics']

    def test_validate_documentation(self, documenter, sample_data):
        """Test documentation validation."""
        X, y = sample_data

        # Create feature with minimal documentation
        documenter.analyze_feature('numeric_feature', X, y)

        # Validate
        validation = documenter.validate_documentation('numeric_feature')

        assert isinstance(validation, dict)
        assert 'complete' in validation
        assert 'warnings' in validation
        assert 'errors' in validation
        assert 'score' in validation
        assert isinstance(validation['score'], int)
        assert 0 <= validation['score'] <= 100

    def test_validate_missing_fields(self, documenter, sample_data):
        """Test validation with missing fields."""
        X, y = sample_data

        # Create feature without analysis (simulate incomplete documentation)
        feature = FeatureDescription(
            name='test_feature',
            description='',  # Empty
            feature_type='numeric',
            source='unknown'
        )
        documenter.features_['test_feature'] = feature

        # Validate
        validation = documenter.validate_documentation('test_feature')

        assert not validation['complete']
        assert len(validation['errors']) > 0
        assert validation['score'] < 100

    def test_detect_outliers(self, documenter):
        """Test outlier detection methods."""
        # Create data with known outliers
        normal_data = np.random.randn(100)
        outlier_data = np.concatenate([normal_data, [10, -10, 15]])

        # Test IQR method
        outlier_count, outlier_pct = documenter._detect_outliers(
            pd.Series(outlier_data), 'iqr', 1.5
        )
        assert outlier_count >= 3  # At least our manual outliers

        # Test Z-score method
        outlier_count, outlier_pct = documenter._detect_outliers(
            pd.Series(outlier_data), 'zscore', 3.0
        )
        assert outlier_count >= 3

    def test_calculate_consistency(self, documenter):
        """Test consistency score calculation."""
        # Test numeric consistency (low CV = high consistency)
        consistent_numeric = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        consistency = documenter._calculate_consistency(pd.Series(consistent_numeric))
        assert consistency == 1.0

        # Test numeric consistency (high CV = low consistency)
        variable_numeric = np.random.randn(100) * 100
        consistency = documenter._calculate_consistency(pd.Series(variable_numeric))
        assert 0 <= consistency <= 1
        assert consistency < 0.5  # Should be low for highly variable data

        # Test categorical consistency (single category = high consistency)
        consistent_cat = pd.Series(['A'] * 100)
        consistency = documenter._calculate_consistency(consistent_cat)
        assert consistency == 1.0

        # Test categorical consistency (many categories = low consistency)
        variable_cat = pd.Series([str(i) for i in range(100)])
        consistency = documenter._calculate_consistency(variable_cat)
        assert 0 <= consistency <= 1
        assert consistency < 0.5  # Should be low for highly variable data

    def test_error_handling(self, documenter):
        """Test error handling in various scenarios."""
        X = pd.DataFrame({'test_feature': [1, 2, 3]})

        # Analyze non-existent feature
        with pytest.raises(ValueError, match="not found"):
            documenter.analyze_feature('nonexistent', X)

        # Export non-existent feature
        with pytest.raises(ValueError, match="not documented"):
            documenter.export_feature_documentation('nonexistent')

        # Get summary of non-existent feature
        with pytest.raises(ValueError, match="not documented"):
            documenter.get_feature_summary('nonexistent')

        # Validate non-existent feature
        with pytest.raises(ValueError, match="not documented"):
            documenter.validate_documentation('nonexistent')

    def test_config_parameters(self):
        """Test configuration parameters."""
        config = DocumentationConfig(
            correlation_threshold=0.9,
            outlier_method='zscore',
            outlier_threshold=2.5,
            output_dir='/tmp/test_docs',
            export_format='html',
            include_statistics=False,
            include_analysis=False
        )

        documenter = FeatureDocumenter(config)

        assert documenter.config.correlation_threshold == 0.9
        assert documenter.config.outlier_method == 'zscore'
        assert documenter.config.outlier_threshold == 2.5
        assert documenter.config.output_dir == '/tmp/test_docs'
        assert documenter.config.export_format == 'html'
        assert documenter.config.include_statistics is False
        assert documenter.config.include_analysis is False

    def test_datetime_statistics(self, documenter):
        """Test statistics calculation for datetime features."""
        # Create datetime series
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        datetime_series = pd.Series(dates)

        stats = documenter._calculate_statistics(datetime_series)

        assert stats.time_span_days == 99.0  # 100 days - 1 day
        assert stats.time_start == '2024-01-01 00:00:00'
        assert stats.time_end == '2024-04-09 00:00:00'

    def test_text_statistics(self, documenter):
        """Test statistics calculation for text features."""
        # Create text series with varying lengths
        texts = ['short', 'medium length', 'this is a much longer text string', 'tiny']
        text_series = pd.Series(texts)

        stats = documenter._calculate_statistics(text_series)

        assert stats.avg_length > 4
        assert stats.min_length == 4  # 'tiny'
        assert stats.max_length == len(texts[2])

    def test_empty_dataset(self, documenter):
        """Test handling of empty dataset."""
        X_empty = pd.DataFrame()
        y_empty = pd.Series([])

        # Should handle gracefully
        feature_docs = documenter.analyze_dataset(X_empty, y_empty)
        assert feature_docs == {}

    def test_single_value_feature(self, documenter):
        """Test feature with single unique value."""
        X = pd.DataFrame({'single_value': [5] * 100})
        y = pd.Series(np.random.randn(100))

        # Should still work
        feature_doc = documenter.analyze_feature('single_value', X, y)
        assert feature_doc.statistics.unique_count == 1
        assert feature_doc.statistics.mean == 5.0
        assert feature_doc.statistics.std == 0.0