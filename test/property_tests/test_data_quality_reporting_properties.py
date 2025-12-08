"""Property tests for data quality reporting."""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule
from typing import Dict, List, Any
import tempfile
import os
import json
from datetime import datetime

from src.data_quality.data_quality_checker import DataQualityChecker


class Property51DataQualityReporting:
    """
    Property 51: Data quality reporting

    Validates: Requirements 12.4

    Ensures that data quality reporting provides comprehensive,
    actionable insights and properly formats quality metrics.
    """

    @pytest.mark.property
    @given(
        n_rows=st.integers(min_value=100, max_value=1000),
        n_cols=st.integers(min_value=5, max_value=20),
        quality_level=st.sampled_from(['excellent', 'good', 'fair', 'poor', 'very_poor']),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_quality_score_calculation(self, n_rows, n_cols, quality_level, random_state):
        """Test that quality score is calculated correctly across different data quality levels."""
        np.random.seed(random_state)

        # Create base data
        data = {}
        for i in range(n_cols):
            if i % 2 == 0:
                data[f'num_{i}'] = np.random.randn(n_rows)
            else:
                data[f'cat_{i}'] = [f'val_{j}' for j in range(n_rows)]

        df = pd.DataFrame(data)

        # Adjust data quality based on level
        if quality_level == 'excellent':
            # No issues - perfect data
            pass
        elif quality_level == 'good':
            # Minor issues - 5% missing values
            missing_indices = np.random.choice(n_rows * n_cols, int(0.05 * n_rows * n_cols), replace=False)
            for idx in missing_indices:
                df.iloc[idx // n_cols, idx % n_cols] = np.nan
        elif quality_level == 'fair':
            # Moderate issues - 15% missing values, some duplicates
            missing_indices = np.random.choice(n_rows * n_cols, int(0.15 * n_rows * n_cols), replace=False)
            for idx in missing_indices:
                df.iloc[idx // n_cols, idx % n_cols] = np.nan
            df = pd.concat([df, df.iloc[0:10]], ignore_index=True)
        elif quality_level == 'poor':
            # Many issues - 30% missing values, duplicates, outliers
            missing_indices = np.random.choice(n_rows * n_cols, int(0.30 * n_rows * n_cols), replace=False)
            for idx in missing_indices:
                df.iloc[idx // n_cols, idx % n_cols] = np.nan
            df = pd.concat([df, df.iloc[0:20]], ignore_index=True)
            if 'num_0' in df.columns:
                df.loc[0:10, 'num_0'] = 100  # Outliers
        else:  # very_poor
            # Severe issues - 50% missing values, many duplicates, outliers, type issues
            missing_indices = np.random.choice(n_rows * n_cols, int(0.50 * n_rows * n_cols), replace=False)
            for idx in missing_indices:
                df.iloc[idx // n_cols, idx % n_cols] = np.nan
            df = pd.concat([df, df.iloc[0:50]], ignore_index=True)
            if 'num_0' in df.columns:
                df.loc[0:20, 'num_0'] = [1000 if i % 2 == 0 else -1000 for i in range(20)]

        # Generate quality report
        checker = DataQualityChecker(df)
        report = checker.generate_quality_report()

        # Validate quality score is in valid range
        assert 0.0 <= report['quality_score'] <= 1.0, "Quality score out of valid range"

        # Validate quality grade matches score
        score = report['quality_score']
        grade = report['summary']['quality_grade']

        if score >= 0.9:
            assert grade == 'Excellent', f"Score {score:.2f} should map to 'Excellent', got {grade}"
        elif score >= 0.8:
            assert grade == 'Good', f"Score {score:.2f} should map to 'Good', got {grade}"
        elif score >= 0.7:
            assert grade == 'Fair', f"Score {score:.2f} should map to 'Fair', got {grade}"
        elif score >= 0.6:
            assert grade == 'Poor', f"Score {score:.2f} should map to 'Poor', got {grade}"
        else:
            assert grade == 'Very Poor', f"Score {score:.2f} should map to 'Very Poor', got {grade}"

        # Validate score composition
        scores = report['scores']
        assert all(0.0 <= score <= 1.0 for score in scores.values()), "Individual scores out of range"

        # Score should be roughly consistent with quality level
        expected_ranges = {
            'excellent': (0.85, 1.0),
            'good': (0.7, 0.9),
            'fair': (0.5, 0.8),
            'poor': (0.3, 0.7),
            'very_poor': (0.0, 0.6)
        }
        min_expected, max_expected = expected_ranges[quality_level]
        assert min_expected <= score <= max_expected, \
            f"Score {score:.2f} not in expected range [{min_expected}, {max_expected}] for {quality_level}"

    @pytest.mark.property
    @given(
        n_rows=st.integers(min_value=100, max_value=500),
        issue_types=st.lists(
            st.sampled_from(['missing', 'duplicates', 'outliers', 'type_issues']),
            min_size=1,
            max_size=4,
            unique=True
        ),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_recommendations_relevance(self, n_rows, issue_types, random_state):
        """Test that recommendations are relevant to detected issues."""
        np.random.seed(random_state)

        # Create base data
        data = {
            'id': range(n_rows),
            'value': np.random.randn(n_rows),
            'category': np.random.choice(['A', 'B', 'C'], n_rows),
            'description': [f'Item {i}' for i in range(n_rows)]
        }
        df = pd.DataFrame(data)

        # Introduce issues based on specified types
        if 'missing' in issue_types:
            df.loc[0:20, 'value'] = np.nan
        if 'duplicates' in issue_types:
            df = pd.concat([df, df.iloc[0:15]], ignore_index=True)
        if 'outliers' in issue_types:
            df.loc[0:5, 'value'] = 100
        if 'type_issues' in issue_types:
            # Simulate type issues by setting strings in numeric column
            df.loc[0:3, 'value'] = 'invalid'

        # Get recommendations
        checker = DataQualityChecker(df)
        recommendations = checker.get_quality_recommendations()

        # Validate recommendations exist for each issue type
        assert isinstance(recommendations, list), "Recommendations should be a list"
        assert len(recommendations) > 0, "Should have at least one recommendation"

        # Check that recommendations match issue types
        categories = [rec['category'] for rec in recommendations]
        if 'missing' in issue_types:
            assert any('Missing Values' in cat for cat in categories), \
                "Missing values recommendation missing"
        if 'duplicates' in issue_types:
            assert any('Duplicates' in cat for cat in categories), \
                "Duplicates recommendation missing"
        if 'outliers' in issue_types:
            assert any('Outliers' in cat for cat in categories), \
                "Outliers recommendation missing"
        if 'type_issues' in issue_types:
            assert any('Data Types' in cat for cat in categories), \
                "Data types recommendation missing"

        # Validate recommendation structure
        for rec in recommendations:
            assert all(key in rec for key in ['category', 'issue', 'recommendation']), \
                "Recommendation missing required fields"
            assert all(isinstance(rec[key], str) for key in ['category', 'issue', 'recommendation']), \
                "Recommendation fields should be strings"
            assert len(rec['recommendation']) > 10, "Recommendation should be meaningful"

    @pytest.mark.property
    @given(
        n_rows=st.integers(min_value=50, max_value=500),
        n_cols=st.integers(min_value=3, max_value=10),
        save_format=st.sampled_from(['json', 'csv']),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_report_export_functionality(self, n_rows, n_cols, save_format, random_state):
        """Test that quality reports can be exported in different formats."""
        np.random.seed(random_state)

        # Create test data
        data = {}
        for i in range(n_cols):
            data[f'col_{i}'] = np.random.randn(n_rows) if i % 2 == 0 else [f'val_{j}' for j in range(n_rows)]
        df = pd.DataFrame(data)

        # Generate quality report
        checker = DataQualityChecker(df)
        report = checker.generate_quality_report()

        # Test report export
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{save_format}', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            checker.save_report(tmp_path, save_format)
            assert os.path.exists(tmp_path), f"Report file not created for {save_format} format"
            assert os.path.getsize(tmp_path) > 0, f"Report file is empty for {save_format} format"

            # Validate content based on format
            if save_format == 'json':
                with open(tmp_path, 'r') as f:
                    saved_data = json.load(f)
                assert 'summary' in saved_data, "JSON report missing summary"
                assert saved_data['summary']['total_records'] == n_rows, \
                    "JSON report incorrect record count"
            elif save_format == 'csv':
                saved_df = pd.read_csv(tmp_path)
                assert 'total_records' in saved_df.columns, "CSV report missing total_records column"
                assert saved_df['total_records'].iloc[0] == n_rows, \
                    "CSV report incorrect record count"

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @pytest.mark.property
    @given(
        dataset_variations=st.integers(min_value=1, max_value=10),
        base_quality=st.sampled_from(['clean', 'noisy', 'mixed']),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_report_comparative_analysis(self, dataset_variations, base_quality, random_state):
        """Test that quality reports enable comparative analysis across datasets."""
        np.random.seed(random_state)

        reports = {}

        # Generate multiple datasets with varying quality
        for i in range(dataset_variations):
            if base_quality == 'clean':
                # Clean data
                data = {
                    'id': range(100),
                    'value': np.random.randn(100),
                    'category': np.random.choice(['A', 'B', 'C'], 100)
                }
            elif base_quality == 'noisy':
                # Noisy data with issues
                data = {
                    'id': range(100),
                    'value': np.random.randn(100),
                    'category': np.random.choice(['A', 'B', 'C'], 100)
                }
                df_temp = pd.DataFrame(data)
                df_temp.loc[0:10, 'value'] = np.nan  # Missing values
                df_temp = pd.concat([df_temp, df_temp.iloc[0:5]], ignore_index=True)  # Duplicates
                data = df_temp.to_dict('list')
            else:  # mixed
                if i % 2 == 0:
                    # Clean variation
                    data = {
                        'id': range(100),
                        'value': np.random.randn(100),
                        'category': np.random.choice(['A', 'B'], 100)
                    }
                else:
                    # Noisy variation
                    data = {
                        'id': range(100),
                        'value': np.random.randn(100),
                        'category': np.random.choice(['A', 'B'], 100)
                    }
                    df_temp = pd.DataFrame(data)
                    df_temp.loc[0:5, 'value'] = np.nan
                    data = df_temp.to_dict('list')

            df = pd.DataFrame(data)
            checker = DataQualityChecker(df)
            reports[f'dataset_{i}'] = checker.generate_quality_report()

        # Validate comparative properties
        assert len(reports) == dataset_variations, "Not all reports generated"

        # All reports should have same structure
        required_keys = ['summary', 'scores', 'quality_score']
        for report in reports.values():
            assert all(key in report for key in required_keys), "Report structure inconsistent"

        # Quality scores should vary across datasets if they're different
        scores = [report['quality_score'] for report in reports.values()]
        if base_quality == 'mixed':
            assert len(set(round(s, 2) for s in scores)) > 1, "Quality scores should vary for mixed quality"

        # Validate comparability
        for name, report in reports.items():
            assert 'timestamp' in report, f"Report {name} missing timestamp"
            assert isinstance(report['timestamp'], str), f"Report {name} timestamp not string"

    @pytest.mark.property
    @given(
        data_size=st.integers(min_value=100, max_value=1000),
        schema_complexity=st.sampled_from(['simple', 'complex']),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_schema_compliance_reporting(self, data_size, schema_complexity, random_state):
        """Test that schema compliance is properly reported."""
        np.random.seed(random_state)

        # Create test data
        data = {
            'id': range(data_size),
            'value': np.random.randn(data_size),
            'category': np.random.choice(['A', 'B', 'C', 'D'], data_size),
            'status': np.random.choice([0, 1], data_size),
            'description': [f'Item {i}' for i in range(data_size)]
        }
        df = pd.DataFrame(data)

        # Create schema based on complexity
        if schema_complexity == 'simple':
            schema = {
                'id': {'type': 'int', 'nullable': False},
                'value': {'type': 'float'},
                'category': {'type': 'string'}
            }
        else:  # complex
            schema = {
                'id': {'type': 'int', 'nullable': False, 'range': [0, data_size]},
                'value': {'type': 'float', 'range': [-10, 10], 'nullable': True},
                'category': {'type': 'string', 'enum': ['A', 'B', 'C']},
                'status': {'type': 'int', 'enum': [0, 1], 'nullable': False},
                'description': {'type': 'string', 'max_length': 20},
                'timestamp': {'type': 'datetime', 'nullable': True}  # Missing column
            }

        # Generate report with schema
        checker = DataQualityChecker(df, schema)
        report = checker.generate_quality_report()

        # Validate schema reporting
        schema_validation = report['schema_validation']
        assert 'schema_compliant' in schema_validation, "Schema compliance flag missing"
        assert 'schema_compliance_score' in schema_validation, "Schema compliance score missing"
        assert 'missing_columns' in schema_validation, "Missing columns list missing"
        assert 'extra_columns' in schema_validation, "Extra columns list missing"
        assert 'constraint_violations' in schema_validation, "Constraint violations missing"

        # Validate compliance score
        assert 0.0 <= schema_validation['schema_compliance_score'] <= 1.0, \
            "Schema compliance score out of range"

        # Check missing columns (complex schema has timestamp)
        if schema_complexity == 'complex':
            assert 'timestamp' in schema_validation['missing_columns'], \
                "Expected missing column not reported"

        # Validate constraint violations
        violations = schema_validation['constraint_violations']
        if schema_complexity == 'complex':
            # Should have violations for category enum and description length
            assert len(violations) > 0, "Expected constraint violations not found"
            for col, col_violations in violations.items():
                assert isinstance(col_violations, list), f"Violations for {col} not a list"

    @pytest.mark.property
    @given(
        n_rows=st.integers(min_value=100, max_value=500),
        report_format=st.sampled_from(['json', 'csv']),
        include_timestamp=st.booleans(),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_report_metadata_completeness(self, n_rows, report_format, include_timestamp, random_state):
        """Test that report metadata is complete and accurate."""
        np.random.seed(random_state)

        # Create test data
        data = {
            'id': range(n_rows),
            'value': np.random.randn(n_rows),
            'category': np.random.choice(['A', 'B'], n_rows)
        }
        df = pd.DataFrame(data)

        # Generate report
        checker = DataQualityChecker(df)
        report = checker.generate_quality_report()

        # Validate metadata completeness
        assert 'timestamp' in report, "Timestamp missing from report"
        assert 'data_shape' in report, "Data shape missing from report"

        # Validate timestamp format
        if include_timestamp:
            try:
                timestamp = datetime.fromisoformat(report['timestamp'])
                assert isinstance(timestamp, datetime), "Invalid timestamp format"
            except ValueError:
                pytest.fail("Timestamp not in ISO format")

        # Validate data shape
        shape = report['data_shape']
        assert isinstance(shape, tuple), "Data shape should be tuple"
        assert len(shape) == 2, "Data shape should have 2 dimensions"
        assert shape[0] == n_rows, f"Row count mismatch: expected {n_rows}, got {shape[0]}"
        assert shape[1] == 3, f"Column count mismatch: expected 3, got {shape[1]}"

        # Validate summary metadata
        summary = report['summary']
        assert 'total_records' in summary, "Total records missing from summary"
        assert 'total_columns' in summary, "Total columns missing from summary"
        assert 'numeric_columns' in summary, "Numeric columns count missing"
        assert 'categorical_columns' in summary, "Categorical columns count missing"

        assert summary['total_records'] == n_rows, "Summary record count incorrect"
        assert summary['total_columns'] == 3, "Summary column count incorrect"
        assert summary['numeric_columns'] >= 2, "Numeric columns count too low"
        assert summary['categorical_columns'] >= 1, "Categorical columns count too low"

    @pytest.mark.property
    @given(
        score_components=st.lists(
            st.sampled_from(['completeness', 'uniqueness', 'cleanliness', 'type_consistency', 'schema_compliance']),
            min_size=2,
            max_size=5,
            unique=True
        ),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_score_component_accuracy(self, score_components, random_state):
        """Test that individual score components are calculated accurately."""
        np.random.seed(random_state)

        # Create test data
        data = {
            'id': range(100),
            'value': np.random.randn(100),
            'category': np.random.choice(['A', 'B'], 100)
        }
        df = pd.DataFrame(data)

        # Start with clean data
        checker = DataQualityChecker(df)
        report = checker.generate_quality_report()

        # Get base scores (should be high for clean data)
        base_scores = {comp: report['scores'].get(comp, 0) for comp in score_components}

        # Test each component independently
        for component in score_components:
            # Create new checker for isolated testing
            checker_test = DataQualityChecker(df)

            if component == 'completeness':
                # Introduce missing values
                checker_test.data.loc[0:10, 'value'] = np.nan
                result = checker_test.check_missing_values()
                expected_score = result['completeness_score']
            elif component == 'uniqueness':
                # Introduce duplicates
                checker_test.data = pd.concat([checker_test.data, checker_test.data.iloc[0:5]], ignore_index=True)
                result = checker_test.check_duplicates()
                expected_score = result['uniqueness_score']
            elif component == 'cleanliness':
                # Introduce outliers
                checker_test.data.loc[0:5, 'value'] = 100
                result = checker_test.detect_outliers()
                expected_score = result['cleanliness_score']
            elif component == 'type_consistency':
                # This should remain high for our clean data
                result = checker_test.validate_data_types()
                expected_score = result['type_consistency_score']
            elif component == 'schema_compliance':
                # This should remain high without schema constraints
                result = checker_test.validate_schema()
                expected_score = result['schema_compliance_score']

            # Validate score is in valid range
            assert 0.0 <= expected_score <= 1.0, f"Component {component} score out of range"

    @pytest.mark.property
    @given(
        n_records=st.integers(min_value=50, max_value=300),
        recommendation_categories=st.lists(
            st.sampled_from(['Missing Values', 'Duplicates', 'Outliers', 'Data Types', 'Schema', 'Overall']),
            min_size=1,
            max_size=6,
            unique=True
        ),
        random_state=st.integers(min_value=0, max_value=100)
    )
    def test_recommendation_actionability(self, n_records, recommendation_categories, random_state):
        """Test that recommendations are actionable and specific."""
        np.random.seed(random_state)

        # Create data
        data = {
            'id': range(n_records),
            'value': np.random.randn(n_records),
            'category': np.random.choice(['A', 'B'], n_records)
        }
        df = pd.DataFrame(data)

        # Get recommendations
        checker = DataQualityChecker(df)
        recommendations = checker.get_quality_recommendations()

        # Validate that recommendations are actionable
        for rec in recommendations:
            assert rec['category'] in recommendation_categories or 'Overall' in rec['category'], \
                f"Unexpected recommendation category: {rec['category']}"

            # Issue description should be specific
            issue = rec['issue']
            assert len(issue) > 10, f"Issue description too short: {issue}"
            assert any(keyword in issue.lower() for keyword in ['%', 'duplicate', 'outlier', 'missing', 'type', 'schema']), \
                f"Issue description lacks specific metrics: {issue}"

            # Recommendation should be actionable
            recommendation = rec['recommendation']
            assert len(recommendation) > 15, f"Recommendation too short: {recommendation}"
            action_words = ['consider', 'remove', 'investigate', 'review', 'fix', 'improve', 'check', 'validate', 'correct']
            assert any(word in recommendation.lower() for word in action_words), \
                f"Recommendation lacks action words: {recommendation}"

        # Check that recommendations don't duplicate unnecessarily
        category_counts = {}
        for rec in recommendations:
            cat = rec['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Should not have multiple recommendations for the same category unless justified
        for cat, count in category_counts.items():
            if count > 1:
                # Multiple recommendations for same category should have different issues
                issues = [rec['issue'] for rec in recommendations if rec['category'] == cat]
                assert len(set(issues)) == count, f"Duplicate recommendation for {cat}"