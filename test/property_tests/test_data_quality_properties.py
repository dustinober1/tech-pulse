"""Property tests for data quality checker."""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule
from typing import Dict, List, Any

from src.data_quality.data_quality_checker import DataQualityChecker


class Property23DataValidationExecution:
    """
    Property 23: Data validation execution

    Validates: Requirements 6.2

    Ensures that data quality checker correctly executes data validation
    processes and identifies all types of data quality issues.
    """

    @pytest.mark.property
    @given(
        n_rows=st.integers(min_value=50, max_value=500),
        n_cols=st.integers(min_value=3, max_value=15),
        missing_pct=st.floats(min_value=0.0, max_value=0.3),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_missing_value_validation_completeness(self, n_rows, n_cols, missing_pct, random_state):
        """Test that missing value validation is complete and accurate."""
        np.random.seed(random_state)

        # Create base data
        data = {}
        for i in range(n_cols):
            if i % 3 == 0:  # Numeric column
                data[f'num_{i}'] = np.random.randn(n_rows)
            else:  # Categorical column
                data[f'cat_{i}'] = [f'val_{j}' for j in range(n_rows)]

        df = pd.DataFrame(data)

        # Introduce missing values
        n_missing = int(n_rows * n_cols * missing_pct)
        if n_missing > 0:
            missing_indices = np.random.choice(n_rows * n_cols, n_missing, replace=False)
            for idx in missing_indices:
                row = idx // n_cols
                col = idx % n_cols
                df.iloc[row, col] = np.nan

        # Run missing value check
        checker = DataQualityChecker(df)
        result = checker.check_missing_values()

        # Validate completeness
        assert 'total_missing' in result, "Total missing count missing"
        assert 'missing_percentage' in result, "Missing percentage missing"
        assert 'by_column' in result, "Missing by column analysis missing"
        assert 'patterns' in result, "Missing patterns analysis missing"
        assert 'completeness_score' in result, "Completeness score missing"

        # Validate counts
        actual_missing = df.isnull().sum().sum()
        assert result['total_missing'] == actual_missing, \
            f"Expected {actual_missing} missing values, got {result['total_missing']}"

        expected_pct = actual_missing / (n_rows * n_cols) * 100
        assert abs(result['missing_percentage'] - expected_pct) < 0.01, \
            f"Missing percentage mismatch: expected {expected_pct}, got {result['missing_percentage']}"

        # Validate column-wise analysis
        for col in df.columns:
            assert col in result['by_column'], f"Column {col} missing from by_column analysis"
            col_missing = df[col].isnull().sum()
            assert result['by_column'][col]['count'] == col_missing, \
                f"Mismatch in missing count for column {col}"

        # Validate completeness score
        expected_score = (100 - result['missing_percentage']) / 100
        assert abs(result['completeness_score'] - expected_score) < 0.01, \
            "Completeness score calculation incorrect"

    @pytest.mark.property
    @given(
        n_rows=st.integers(min_value=50, max_value=500),
        n_cols=st.integers(min_value=3, max_value=10),
        duplicate_pct=st.floats(min_value=0.0, max_value=0.2),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_duplicate_detection_accuracy(self, n_rows, n_cols, duplicate_pct, random_state):
        """Test that duplicate detection accurately identifies all duplicates."""
        np.random.seed(random_state)

        # Create base data
        data = {}
        for i in range(n_cols):
            data[f'col_{i}'] = [f'val_{j}' for j in range(n_rows)]

        df = pd.DataFrame(data)

        # Introduce duplicates
        n_duplicates = int(n_rows * duplicate_pct)
        if n_duplicates > 0:
            duplicate_indices = np.random.choice(n_rows, n_duplicates, replace=True)
            duplicate_rows = df.iloc[duplicate_indices].copy()
            df = pd.concat([df, duplicate_rows], ignore_index=True)

        # Run duplicate check
        checker = DataQualityChecker(df)
        result = checker.check_duplicates()

        # Validate completeness
        assert 'total_duplicates' in result, "Total duplicates missing"
        assert 'duplicate_percentage' in result, "Duplicate percentage missing"
        assert 'duplicate_rows' in result, "Duplicate rows missing"
        assert 'duplicate_columns' in result, "Duplicate columns analysis missing"
        assert 'uniqueness_score' in result, "Uniqueness score missing"

        # Validate duplicate count
        expected_duplicates = len(df) - len(df.drop_duplicates())
        assert result['total_duplicates'] == expected_duplicates, \
            f"Expected {expected_duplicates} duplicates, got {result['total_duplicates']}"

        # Validate duplicate percentage
        expected_pct = expected_duplicates / len(df) * 100
        assert abs(result['duplicate_percentage'] - expected_pct) < 0.01, \
            "Duplicate percentage calculation incorrect"

        # Validate column uniqueness
        for col in df.columns:
            assert col in result['duplicate_columns'], f"Column {col} missing from uniqueness analysis"
            unique_count = df[col].nunique()
            assert result['duplicate_columns'][col]['unique_count'] == unique_count, \
                f"Uniqueness count mismatch for column {col}"
            assert result['duplicate_columns'][col]['total_count'] == len(df), \
                f"Total count mismatch for column {col}"

    @pytest.mark.property
    @given(
        n_rows=st.integers(min_value=100, max_value=500),
        n_numeric_cols=st.integers(min_value=2, max_value=5),
        outlier_method=st.sampled_from(['iqr', 'zscore', 'isolation_forest']),
        outlier_pct=st.floats(min_value=0.0, max_value=0.1),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_outlier_detection_execution(self, n_rows, n_numeric_cols, outlier_method, outlier_pct, random_state):
        """Test that outlier detection correctly executes for different methods."""
        np.random.seed(random_state)

        # Create data with numeric columns
        data = {}
        for i in range(n_numeric_cols):
            # Normal data
            normal_data = np.random.normal(0, 1, n_rows)
            # Add outliers
            n_outliers = int(n_rows * outlier_pct)
            if n_outliers > 0:
                outlier_indices = np.random.choice(n_rows, n_outliers, replace=False)
                normal_data[outlier_indices] = np.random.choice([10, -10], n_outliers)
            data[f'num_{i}'] = normal_data

        df = pd.DataFrame(data)

        # Run outlier detection
        checker = DataQualityChecker(df)
        result = checker.detect_outliers(methods=[outlier_method])

        # Validate execution
        assert 'methods_used' in result, "Methods used missing"
        assert 'outliers_by_column' in result, "Outliers by column missing"
        assert 'outlier_counts' in result, "Outlier counts missing"
        assert 'cleanliness_score' in result, "Cleanliness score missing"

        # Verify method was used
        assert outlier_method in result['methods_used'], f"Method {outlier_method} not in used methods"
        assert outlier_method in result['outliers_by_column'], f"Outliers not found for method {outlier_method}"

        # Validate column analysis
        for col in df.columns:
            if col in result['outliers_by_column'][outlier_method]:
                outliers = result['outliers_by_column'][outlier_method][col]
                assert isinstance(outliers, np.ndarray), f"Outliers for {col} not a numpy array"
                if len(outliers) > 0:
                    assert all(0 <= idx < len(df) for idx in outliers), \
                        f"Invalid outlier index for column {col}"

        # Validate cleanliness score range
        assert 0.0 <= result['cleanliness_score'] <= 1.0, \
            "Cleanliness score out of valid range"

    @pytest.mark.property
    @given(
        n_rows=st.integers(min_value=50, max_value=300),
        type_mixture=st.lists(
            st.sampled_from(['int', 'float', 'string', 'boolean', 'datetime']),
            min_size=3,
            max_size=8,
            unique=True
        ),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_data_type_validation_execution(self, n_rows, type_mixture, random_state):
        """Test that data type validation correctly identifies all types."""
        np.random.seed(random_state)

        # Create data with mixed types
        data = {}
        for i, col_type in enumerate(type_mixture):
            col_name = f'col_{i}_{col_type}'

            if col_type == 'int':
                data[col_name] = np.random.randint(0, 100, n_rows)
            elif col_type == 'float':
                data[col_name] = np.random.randn(n_rows)
            elif col_type == 'string':
                data[col_name] = [f'str_{j}' for j in range(n_rows)]
            elif col_type == 'boolean':
                data[col_name] = np.random.choice([True, False], n_rows)
            elif col_type == 'datetime':
                data[col_name] = pd.date_range('2020-01-01', periods=n_rows, freq='D')

        df = pd.DataFrame(data)

        # Run type validation
        checker = DataQualityChecker(df)
        result = checker.validate_data_types()

        # Validate execution
        assert 'column_types' in result, "Column types analysis missing"
        assert 'type_issues' in result, "Type issues analysis missing"
        assert 'type_consistency_score' in result, "Type consistency score missing"

        # Validate each column
        for col in df.columns:
            assert col in result['column_types'], f"Column {col} missing from type analysis"

            col_info = result['column_types'][col]
            assert 'actual_type' in col_info, f"Actual type missing for {col}"
            assert 'expected_type' in col_info, f"Expected type missing for {col}"
            assert 'is_correct' in col_info, f"Type correctness flag missing for {col}"

            # Validate numeric column info
            if df[col].dtype.kind in 'biufc':
                assert 'min' in col_info, f"Min value missing for numeric column {col}"
                assert 'max' in col_info, f"Max value missing for numeric column {col}"
                assert 'mean' in col_info, f"Mean value missing for numeric column {col}"
                assert isinstance(col_info['min'], (int, float)), f"Min not numeric for {col}"
                assert isinstance(col_info['max'], (int, float)), f"Max not numeric for {col}"

            # Validate string column info
            elif df[col].dtype == 'object':
                assert 'unique_values' in col_info, f"Unique values missing for string column {col}"
                assert isinstance(col_info['unique_values'], int), f"Unique values not integer for {col}"

        # Validate consistency score range
        assert 0.0 <= result['type_consistency_score'] <= 1.0, \
            "Type consistency score out of valid range"

    @pytest.mark.property
    @given(
        n_rows=st.integers(min_value=50, max_value=300),
        schema_fields=st.integers(min_value=2, max_value=6),
        constraint_types=st.lists(
            st.sampled_from(['nullable', 'range', 'enum', 'max_length']),
            min_size=1,
            max_size=3,
            unique=True
        ),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_schema_validation_execution(self, n_rows, schema_fields, constraint_types, random_state):
        """Test that schema validation correctly executes constraint checks."""
        np.random.seed(random_state)

        # Create data
        data = {
            'id': range(n_rows),
            'value': np.random.randn(n_rows),
            'category': np.random.choice(['A', 'B', 'C'], n_rows),
            'name': [f'item_{i}' for i in range(n_rows)]
        }

        # Create schema with constraints
        schema = {}
        for i, field in enumerate(['id', 'value', 'category', 'name'][:schema_fields]):
            schema[field] = {'type': 'int' if field == 'id' else 'string' if field == 'name' else 'float' if field == 'value' else 'string'}

            if 'nullable' in constraint_types and i == 0:
                schema[field]['nullable'] = False
            if 'range' in constraint_types and field == 'value':
                schema[field]['range'] = [-10, 10]
            if 'enum' in constraint_types and field == 'category':
                schema[field]['enum'] = ['A', 'B', 'C']
            if 'max_length' in constraint_types and field == 'name':
                schema[field]['max_length'] = 20

        df = pd.DataFrame(data)

        # Run schema validation
        checker = DataQualityChecker(df, schema)
        result = checker.validate_schema()

        # Validate execution
        assert 'schema_compliant' in result, "Schema compliance flag missing"
        assert 'missing_columns' in result, "Missing columns analysis missing"
        assert 'extra_columns' in result, "Extra columns analysis missing"
        assert 'constraint_violations' in result, "Constraint violations missing"
        assert 'schema_compliance_score' in result, "Schema compliance score missing"

        # Validate missing/extra columns
        expected_columns = set(schema.keys())
        actual_columns = set(df.columns)
        missing = expected_columns - actual_columns
        extra = actual_columns - expected_columns

        assert set(result['missing_columns']) == missing, "Missing columns mismatch"
        assert set(result['extra_columns']) == extra, "Extra columns mismatch"

        # Validate constraint checks
        if 'nullable' in constraint_types:
            # Check if nullable constraints were evaluated
            for field in schema:
                if 'nullable' in schema[field] and not schema[field]['nullable'] and field in df.columns:
                    if field in result['constraint_violations']:
                        violations = result['constraint_violations'][field]
                        nullable_violations = [v for v in violations if v['constraint'] == 'nullable']
                        assert len(nullable_violations) > 0, f"Nullable violation not detected for {field}"

        # Validate score range
        assert 0.0 <= result['schema_compliance_score'] <= 1.0, \
            "Schema compliance score out of valid range"

    @pytest.mark.property
    @given(
        n_rows=st.integers(min_value=50, max_value=300),
        n_issues=st.integers(min_value=0, max_value=5),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_quality_report_generation_completeness(self, n_rows, n_issues, random_state):
        """Test that quality report includes all required components."""
        np.random.seed(random_state)

        # Create base data
        data = {
            'id': range(n_rows),
            'value': np.random.randn(n_rows),
            'category': np.random.choice(['A', 'B', 'C'], n_rows)
        }
        df = pd.DataFrame(data)

        # Introduce issues based on n_issues
        if n_issues >= 1:
            # Add missing values
            df.loc[0:5, 'value'] = np.nan
        if n_issues >= 2:
            # Add duplicates
            df = pd.concat([df, df.iloc[0:2]], ignore_index=True)
        if n_issues >= 3:
            # Add outliers
            df.loc[0, 'value'] = 100

        # Generate quality report
        checker = DataQualityChecker(df)
        report = checker.generate_quality_report()

        # Validate report completeness
        required_sections = [
            'timestamp', 'data_shape', 'summary', 'missing_values',
            'duplicates', 'outliers', 'data_types', 'schema_validation',
            'quality_score', 'scores'
        ]

        for section in required_sections:
            assert section in report, f"Required section '{section}' missing from quality report"

        # Validate summary
        summary = report['summary']
        required_summary_fields = [
            'total_records', 'total_columns', 'numeric_columns',
            'categorical_columns', 'overall_quality_score', 'quality_grade'
        ]

        for field in required_summary_fields:
            assert field in summary, f"Required summary field '{field}' missing"

        # Validate summary values
        assert summary['total_records'] == len(df), "Total records incorrect"
        assert summary['total_columns'] == len(df.columns), "Total columns incorrect"
        assert 0.0 <= summary['overall_quality_score'] <= 1.0, "Quality score out of range"
        assert summary['quality_grade'] in ['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor'], \
            "Invalid quality grade"

        # Validate individual scores
        scores = report['scores']
        required_scores = ['completeness', 'uniqueness', 'cleanliness', 'type_consistency', 'schema_compliance']
        for score in required_scores:
            assert score in scores, f"Score '{score}' missing"
            assert 0.0 <= scores[score] <= 1.0, f"Score '{score}' out of range"

    @pytest.mark.property
    @given(
        n_rows=st.integers(min_value=50, max_value=300),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_quality_recommendations_generation(self, n_rows, random_state):
        """Test that quality recommendations are generated for all issue types."""
        np.random.seed(random_state)

        # Create data with various issues
        data = {
            'id': range(n_rows),
            'value': np.random.randn(n_rows),
            'category': np.random.choice(['A', 'B', 'C'], n_rows)
        }
        df = pd.DataFrame(data)

        # Add multiple types of issues
        df.loc[0:10, 'value'] = np.nan  # Missing values
        df = pd.concat([df, df.iloc[0:5]], ignore_index=True)  # Duplicates
        df.loc[0, 'value'] = 100  # Outlier

        # Get recommendations
        checker = DataQualityChecker(df)
        recommendations = checker.get_quality_recommendations()

        # Validate recommendations structure
        assert isinstance(recommendations, list), "Recommendations should be a list"

        if recommendations:
            for rec in recommendations:
                assert 'category' in rec, "Category missing from recommendation"
                assert 'issue' in rec, "Issue description missing from recommendation"
                assert 'recommendation' in rec, "Recommendation text missing from recommendation"
                assert isinstance(rec['category'], str), "Category should be string"
                assert isinstance(rec['issue'], str), "Issue should be string"
                assert isinstance(rec['recommendation'], str), "Recommendation should be string"

        # Check for expected categories based on introduced issues
        categories = [rec['category'] for rec in recommendations]
        assert 'Missing Values' in categories, "Missing values recommendation missing"
        assert 'Duplicates' in categories, "Duplicates recommendation missing"
        assert 'Outliers' in categories, "Outliers recommendation missing"

    @pytest.mark.property
    @given(
        data_size=st.integers(min_value=100, max_value=1000),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_data_validation_idempotency(self, data_size, random_state):
        """Test that data validation results are consistent across multiple runs."""
        np.random.seed(random_state)

        # Create deterministic data
        data = {
            'id': range(data_size),
            'value': np.random.RandomState(random_state).randn(data_size),
            'category': np.random.RandomState(random_state).choice(['A', 'B'], data_size)
        }
        df = pd.DataFrame(data)

        # Create two checkers with the same data
        checker1 = DataQualityChecker(df)
        checker2 = DataQualityChecker(df)

        # Run validations
        report1 = checker1.generate_quality_report()
        report2 = checker2.generate_quality_report()

        # Results should be identical
        assert report1['quality_score'] == report2['quality_score'], \
            "Quality scores differ between identical data"
        assert report1['summary']['quality_grade'] == report2['summary']['quality_grade'], \
            "Quality grades differ between identical data"

        # Individual scores should match
        for score in report1['scores']:
            assert report1['scores'][score] == report2['scores'][score], \
                f"Score {score} differs between identical data"

    @pytest.mark.property
    @given(
        data_size=st.integers(min_value=50, max_value=200),
        validation_steps=st.lists(
            st.sampled_from(['missing', 'duplicates', 'outliers', 'types', 'schema']),
            min_size=1,
            max_size=5,
            unique=True
        ),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_stepwise_validation_execution(self, data_size, validation_steps, random_state):
        """Test that validation steps can be executed independently."""
        np.random.seed(random_state)

        # Create test data
        data = {
            'id': range(data_size),
            'value': np.random.randn(data_size),
            'category': np.random.choice(['A', 'B'], data_size)
        }
        df = pd.DataFrame(data)

        checker = DataQualityChecker(df)

        # Execute validation steps independently
        step_results = {}
        for step in validation_steps:
            if step == 'missing':
                step_results[step] = checker.check_missing_values()
            elif step == 'duplicates':
                step_results[step] = checker.check_duplicates()
            elif step == 'outliers':
                step_results[step] = checker.detect_outliers()
            elif step == 'types':
                step_results[step] = checker.validate_data_types()
            elif step == 'schema':
                step_results[step] = checker.validate_schema()

        # Validate each step result
        for step, result in step_results.items():
            assert result is not None, f"Validation step '{step}' returned None"
            assert isinstance(result, dict), f"Validation step '{step}' should return dictionary"

        # Specific validations per step
        if 'missing' in step_results:
            assert 'total_missing' in step_results['missing']
        if 'duplicates' in step_results:
            assert 'total_duplicates' in step_results['duplicates']
        if 'outliers' in step_results:
            assert 'cleanliness_score' in step_results['outliers']
        if 'types' in step_results:
            assert 'type_consistency_score' in step_results['types']
        if 'schema' in step_results:
            assert 'schema_compliance_score' in step_results['schema']