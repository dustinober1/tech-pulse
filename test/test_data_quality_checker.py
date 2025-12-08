"""Tests for data quality checker."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime
import tempfile
import os

from src.data_quality.data_quality_checker import DataQualityChecker


class TestDataQualityChecker:
    """Test cases for DataQualityChecker class."""

    @pytest.fixture
    def clean_data(self):
        """Create clean test data."""
        return pd.DataFrame({
            'id': range(100),
            'age': np.random.randint(18, 80, 100),
            'salary': np.random.normal(50000, 15000, 100),
            'name': [f'Person_{i}' for i in range(100)],
            'active': np.random.choice([True, False], 100),
            'join_date': pd.date_range('2020-01-01', periods=100, freq='D')
        })

    @pytest.fixture
    def data_with_missing(self):
        """Create data with missing values."""
        data = pd.DataFrame({
            'id': range(100),
            'age': np.random.randint(18, 80, 100),
            'salary': np.random.normal(50000, 15000, 100),
            'department': ['A', 'B', 'C'] * 33 + ['A'],
            'score': np.random.randn(100)
        })
        # Introduce missing values
        data.loc[10:20, 'salary'] = np.nan
        data.loc[30:35, 'department'] = np.nan
        data.loc[[1, 5, 10, 15], 'score'] = np.nan
        return data

    @pytest.fixture
    def data_with_duplicates(self):
        """Create data with duplicates."""
        base_data = pd.DataFrame({
            'id': range(50),
            'name': [f'Person_{i}' for i in range(50)],
            'value': np.random.randn(50)
        })
        # Add duplicates
        duplicates = base_data.iloc[10:15].copy()
        duplicates['id'] = range(50, 55)
        return pd.concat([base_data, duplicates], ignore_index=True)

    @pytest.fixture
    def data_with_outliers(self):
        """Create data with outliers."""
        data = pd.DataFrame({
            'normal': np.random.normal(0, 1, 100),
            'skewed': np.random.exponential(2, 100)
        })
        # Add outliers
        data.loc[95, 'normal'] = 10  # Extreme outlier
        data.loc[96, 'normal'] = -8
        data.loc[97, 'skewed'] = 50  # Extreme outlier
        return data

    @pytest.fixture
    def schema(self):
        """Create sample schema."""
        return {
            'id': {'type': 'int', 'nullable': False},
            'age': {'type': 'int', 'nullable': True, 'range': [0, 150]},
            'salary': {'type': 'float', 'nullable': True, 'range': [0, 1000000]},
            'name': {'type': 'string', 'max_length': 50},
            'department': {'type': 'string', 'enum': ['A', 'B', 'C'], 'nullable': True}
        }

    def test_initialization(self, clean_data):
        """Test DataQualityChecker initialization."""
        checker = DataQualityChecker(clean_data)

        assert checker.data.equals(clean_data)
        assert checker.schema == {}
        assert 'timestamp' in checker.quality_report
        assert checker.quality_report['data_shape'] == (100, 6)

    def test_initialization_with_schema(self, clean_data, schema):
        """Test initialization with schema."""
        checker = DataQualityChecker(clean_data, schema)

        assert checker.schema == schema

    def test_check_missing_values_clean(self, clean_data):
        """Test missing value check on clean data."""
        checker = DataQualityChecker(clean_data)
        result = checker.check_missing_values()

        assert result['total_missing'] == 0
        assert result['missing_percentage'] == 0.0
        assert result['completeness_score'] == 1.0

    def test_check_missing_values_with_missing(self, data_with_missing):
        """Test missing value check with missing data."""
        checker = DataQualityChecker(data_with_missing)
        result = checker.check_missing_values()

        assert result['total_missing'] > 0
        assert result['missing_percentage'] > 0
        assert 'salary' in result['by_column']
        assert result['by_column']['salary']['count'] == 10
        assert result['by_column']['department']['count'] == 5
        assert 'patterns' in result

    def test_check_duplicates_clean(self, clean_data):
        """Test duplicate check on clean data."""
        checker = DataQualityChecker(clean_data)
        result = checker.check_duplicates()

        assert result['total_duplicates'] == 0
        assert result['duplicate_percentage'] == 0.0
        assert len(result['duplicate_rows']) == 0

    def test_check_duplicates_with_duplicates(self, data_with_duplicates):
        """Test duplicate check with duplicate data."""
        checker = DataQualityChecker(data_with_duplicates)
        result = checker.check_duplicates()

        assert result['total_duplicates'] == 5
        assert result['duplicate_percentage'] > 0
        assert len(result['duplicate_rows']) <= 10  # Limited to first 10
        assert 'uniqueness_score' in result
        assert result['uniqueness_score'] < 1.0

    def test_check_duplicates_subset(self, data_with_duplicates):
        """Test duplicate check with subset of columns."""
        checker = DataQualityChecker(data_with_duplicates)
        result = checker.check_duplicates(subset=['name'])

        assert result['total_duplicates'] == 5  # Names are duplicated

    def test_detect_outliers_clean(self, clean_data):
        """Test outlier detection on clean data."""
        checker = DataQualityChecker(clean_data)
        result = checker.detect_outliers()

        assert 'methods_used' in result
        assert 'outliers_by_column' in result
        assert 'cleanliness_score' in result
        assert 0.0 <= result['cleanliness_score'] <= 1.0

    def test_detect_outliers_with_outliers(self, data_with_outliers):
        """Test outlier detection with outliers."""
        checker = DataQualityChecker(data_with_outliers)
        result = checker.detect_outliers()

        assert 'normal' in result['outlier_counts']
        assert 'skewed' in result['outlier_counts']
        assert result['cleanliness_score'] < 1.0

    def test_detect_outliers_methods(self, data_with_outliers):
        """Test specific outlier detection methods."""
        checker = DataQualityChecker(data_with_outliers)
        result = checker.detect_outliers(methods=['iqr', 'zscore'])

        assert set(result['methods_used']) == {'iqr', 'zscore'}
        assert 'iqr' in result['outliers_by_column']
        assert 'zscore' in result['outliers_by_column']

    def test_validate_data_types(self, clean_data):
        """Test data type validation."""
        checker = DataQualityChecker(clean_data)
        result = checker.validate_data_types()

        assert 'column_types' in result
        assert 'type_issues' in result
        assert 'type_consistency_score' in result

        # Check specific column types
        assert result['column_types']['id']['actual_type'] == 'int64'
        assert result['column_types']['name']['actual_type'] == 'object'
        assert result['column_types']['active']['actual_type'] == 'bool'

    def test_validate_data_types_with_schema(self, clean_data, schema):
        """Test data type validation with schema."""
        checker = DataQualityChecker(clean_data, schema)
        result = checker.validate_data_types()

        # Should have type issues since schema doesn't match all columns
        assert 'type_issues' in result

    def test_validate_schema_no_schema(self, clean_data):
        """Test schema validation without schema."""
        checker = DataQualityChecker(clean_data)
        result = checker.validate_schema()

        assert result['schema_compliance_score'] == 1.0
        assert result['message'] == "No schema provided for validation"

    def test_validate_schema_with_schema(self, data_with_missing, schema):
        """Test schema validation with schema."""
        checker = DataQualityChecker(data_with_missing, schema)
        result = checker.validate_schema()

        assert 'schema_compliant' in result
        assert 'missing_columns' in result
        assert 'extra_columns' in result
        assert 'constraint_violations' in result

    def test_validate_schema_constraints(self):
        """Test schema constraint validation."""
        # Create data that violates constraints
        data = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'age': [25, -5, 150, 200],  # Age violations
            'department': ['A', 'B', 'X', 'Y'],  # Enum violations
            'name': ['A', 'BB', 'CCC', 'DDDDD']  # Length violations
        })

        schema = {
            'id': {'type': 'int', 'nullable': False},
            'age': {'type': 'int', 'range': [0, 120]},
            'department': {'type': 'string', 'enum': ['A', 'B', 'C']},
            'name': {'type': 'string', 'max_length': 3}
        }

        checker = DataQualityChecker(data, schema)
        result = checker.validate_schema()

        assert not result['schema_compliant']
        assert len(result['constraint_violations']) > 0
        assert 'age' in result['constraint_violations']
        assert 'department' in result['constraint_violations']
        assert 'name' in result['constraint_violations']

    def test_generate_quality_report(self, clean_data):
        """Test comprehensive quality report generation."""
        checker = DataQualityChecker(clean_data)
        report = checker.generate_quality_report()

        assert 'timestamp' in report
        assert 'data_shape' in report
        assert 'summary' in report
        assert 'missing_values' in report
        assert 'duplicates' in report
        assert 'outliers' in report
        assert 'data_types' in report
        assert 'schema_validation' in report
        assert 'quality_score' in report
        assert 'scores' in report

        # Check summary
        summary = report['summary']
        assert summary['total_records'] == 100
        assert summary['total_columns'] == 6
        assert 0.0 <= summary['overall_quality_score'] <= 1.0
        assert summary['quality_grade'] in ['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor']

    def test_generate_quality_report_comprehensive(self, data_with_missing):
        """Test quality report with various issues."""
        checker = DataQualityChecker(data_with_missing)
        report = checker.generate_quality_report()

        # Should have lower quality score due to missing values
        assert report['quality_score'] < 1.0
        assert report['scores']['completeness'] < 1.0

    def test_get_quality_grade(self, clean_data):
        """Test quality grade calculation."""
        checker = DataQualityChecker(clean_data)

        assert checker._get_quality_grade(0.95) == 'Excellent'
        assert checker._get_quality_grade(0.85) == 'Good'
        assert checker._get_quality_grade(0.75) == 'Fair'
        assert checker._get_quality_grade(0.65) == 'Poor'
        assert checker._get_quality_grade(0.55) == 'Very Poor'

    def test_save_report_json(self, clean_data):
        """Test saving report as JSON."""
        checker = DataQualityChecker(clean_data)
        report = checker.generate_quality_report()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            checker.save_report(tmp_path, 'json')
            assert os.path.exists(tmp_path)

            # Verify file content
            import json
            with open(tmp_path, 'r') as f:
                saved_report = json.load(f)
            assert 'summary' in saved_report
            assert saved_report['summary']['total_records'] == 100

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_save_report_csv(self, clean_data):
        """Test saving report as CSV."""
        checker = DataQualityChecker(clean_data)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            checker.save_report(tmp_path, 'csv')
            assert os.path.exists(tmp_path)

            # Verify CSV content
            saved_df = pd.read_csv(tmp_path)
            assert 'total_records' in saved_df.columns
            assert saved_df['total_records'].iloc[0] == 100

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_get_quality_recommendations_clean(self, clean_data):
        """Test recommendations for clean data."""
        checker = DataQualityChecker(clean_data)
        recommendations = checker.get_quality_recommendations()

        assert isinstance(recommendations, list)
        if recommendations:
            assert all('category' in rec for rec in recommendations)
            assert all('issue' in rec for rec in recommendations)
            assert all('recommendation' in rec for rec in recommendations)

    def test_get_quality_recommendations_with_issues(self, data_with_missing, data_with_duplicates):
        """Test recommendations for data with issues."""
        # Combine data with issues
        combined_data = pd.concat([data_with_missing, data_with_duplicates], ignore_index=True)
        checker = DataQualityChecker(combined_data)
        recommendations = checker.get_quality_recommendations()

        assert len(recommendations) > 0
        categories = [rec['category'] for rec in recommendations]
        assert 'Missing Values' in categories or 'Duplicates' in categories

    def test_check_type_compatibility(self):
        """Test type compatibility checking."""
        checker = DataQualityChecker(pd.DataFrame({'a': [1, 2, 3]}))

        assert checker._check_type_compatibility('int64', 'int')
        assert checker._check_type_compatibility('float64', 'float')
        assert checker._check_type_compatibility('object', 'string')
        assert not checker._check_type_compatibility('int64', 'string')

    def test_check_mcar_pattern(self):
        """Test MCAR pattern detection."""
        # Test with no missing data
        clean_data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        checker = DataQualityChecker(clean_data)
        result = checker._check_mcar_pattern()
        assert result == "no_missing_data"

        # Test with missing data
        missing_data = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [5, np.nan, 7, 8]
        })
        checker = DataQualityChecker(missing_data)
        result = checker._check_mcar_pattern()
        assert result in ["likely_mcar", "likely_mar"]

    def test_detect_outliers_iqr(self):
        """Test IQR outlier detection."""
        checker = DataQualityChecker(pd.DataFrame())
        data = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is an outlier
        outliers = checker._detect_outliers_iqr(data)
        assert len(outliers) == 1

    def test_detect_outliers_zscore(self):
        """Test Z-score outlier detection."""
        checker = DataQualityChecker(pd.DataFrame())
        data = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is an outlier
        outliers = checker._detect_outliers_zscore(data)
        assert len(outliers) == 1

    def test_detect_outliers_small_sample(self):
        """Test outlier detection with small sample."""
        checker = DataQualityChecker(pd.DataFrame())
        data = pd.Series([1, 2, 3])  # Very small sample

        # Isolation Forest and LOF should handle small samples gracefully
        outliers_if = checker._detect_outliers_isolation_forest(data)
        outliers_lof = checker._detect_outliers_lof(data)
        assert isinstance(outliers_if, np.ndarray)
        assert isinstance(outliers_lof, np.ndarray)

    @patch('src.data_quality.data_quality_checker.IsolationForest')
    def test_detect_outliers_isolation_forest_mock(self, mock_iso_forest):
        """Test Isolation Forest outlier detection with mock."""
        mock_iso_forest.return_value.fit_predict.return_value = np.array([1, 1, 1, -1, 1])
        checker = DataQualityChecker(pd.DataFrame())
        data = pd.Series([1, 2, 3, 4, 5])

        outliers = checker._detect_outliers_isolation_forest(data)
        assert len(outliers) == 1
        mock_iso_forest.assert_called_once()

    @patch('src.data_quality.data_quality_checker.LocalOutlierFactor')
    def test_detect_outliers_lof_mock(self, mock_lof):
        """Test LOF outlier detection with mock."""
        mock_lof.return_value.fit_predict.return_value = np.array([1, 1, 1, -1, 1])
        checker = DataQualityChecker(pd.DataFrame())
        data = pd.Series([1, 2, 3, 4, 5])

        outliers = checker._detect_outliers_lof(data)
        assert len(outliers) == 1
        mock_lof.assert_called_once()

    def test_quality_report_immutability(self, clean_data):
        """Test that quality report is properly updated."""
        checker = DataQualityChecker(clean_data)

        # Initial report should be empty except for metadata
        assert not checker.quality_report['missing_values']
        assert not checker.quality_report['duplicates']

        # Generate report
        report = checker.generate_quality_report()

        # Report should now be complete
        assert checker.quality_report['missing_values']
        assert checker.quality_report['duplicates']
        assert 'quality_score' in checker.quality_report