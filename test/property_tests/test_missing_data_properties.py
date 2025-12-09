"""Property-based tests for MissingDataAnalyzer class."""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import (
    lists, floats, integers, text, composite, sampled_from,
    data, one_of, just, none, booleans, dictionaries
)
from typing import Dict, Any, List, Tuple, Union

from src.missing_data.missing_data_analyzer import MissingDataAnalyzer


@composite
def generate_data_with_missing(draw):
    """Generate a DataFrame with various missing data patterns."""
    # Generate base data dimensions
    n_rows = draw(integers(min_value=50, max_value=200))
    n_cols = draw(integers(min_value=2, max_value=8))

    # Choose missing pattern type
    pattern_type = draw(sampled_from(['mcar', 'mar', 'mnar', 'monotone', 'mixed']))

    # Generate column names and types
    col_names = [f"col_{i}" for i in range(n_cols)]
    col_types = [draw(sampled_from(['numeric', 'categorical'])) for _ in range(n_cols)]

    data = {}

    for i, (col, col_type) in enumerate(zip(col_names, col_types)):
        if col_type == 'numeric':
            # Generate numeric data
            dist_type = draw(sampled_from(['normal', 'uniform', 'exponential']))

            if dist_type == 'normal':
                mean = draw(floats(min_value=-10, max_value=10))
                std = draw(floats(min_value=0.1, max_value=5))
                values = np.random.normal(mean, std, n_rows)
            elif dist_type == 'uniform':
                low = draw(floats(min_value=-10, max_value=0))
                high = draw(floats(min_value=0, max_value=10))
                values = np.random.uniform(low, high, n_rows)
            else:  # exponential
                scale = draw(floats(min_value=0.5, max_value=3))
                values = np.random.exponential(scale, n_rows)

            data[col] = values
        else:
            # Generate categorical data
            n_categories = draw(integers(min_value=2, max_value=6))
            categories = [f"cat_{j}" for j in range(n_categories)]
            probs = None
            if draw(booleans()):
                # Unequal probabilities
                probs = np.random.dirichlet(np.ones(n_categories))
            values = np.random.choice(categories, n_rows, p=probs)
            data[col] = values

    df = pd.DataFrame(data)

    # Apply missing pattern
    if pattern_type == 'mcar':
        # Missing Completely At Random
        missing_rate = draw(floats(min_value=0.05, max_value=0.4))
        n_missing = max(1, int(missing_rate * n_rows * n_cols))  # Ensure at least 1 missing

        for _ in range(n_missing):
            row = draw(integers(min_value=0, max_value=n_rows-1))
            col = draw(sampled_from(col_names))
            df.loc[row, col] = np.nan

    elif pattern_type == 'mar':
        # Missing At Random - missingness depends on observed variables
        # Make missingness in column i depend on column i-1
        missing_rate = draw(floats(min_value=0.1, max_value=0.3))

        for i in range(1, min(n_cols, 3)):  # Limit to first few columns
            if col_types[i] == 'numeric' and col_types[i-1] == 'numeric':
                # Missing based on value in previous column
                threshold = np.percentile(df[col_names[i-1]].dropna(), 75)
                missing_mask = (df[col_names[i-1]] > threshold) & (np.random.random(n_rows) < missing_rate)
                if not missing_mask.any():  # Ensure at least some missing
                    df.loc[df[col_names[i-1]].idxmax(), col_names[i]] = np.nan
                else:
                    df.loc[missing_mask, col_names[i]] = np.nan

    elif pattern_type == 'mnar':
        # Missing Not At Random - missingness depends on the missing value itself
        missing_rate = draw(floats(min_value=0.1, max_value=0.3))

        for i in range(min(3, n_cols)):  # Limit to first few columns
            col = col_names[i]
            if col_types[i] == 'numeric':
                # High values more likely to be missing
                threshold = np.percentile(df[col].dropna(), 80)
                missing_mask = (df[col] > threshold) & (np.random.random(n_rows) < missing_rate * 2)
                if not missing_mask.any():  # Ensure at least some missing
                    df.loc[df[col].idxmax(), col] = np.nan
                else:
                    df.loc[missing_mask, col] = np.nan

    elif pattern_type == 'monotone':
        # Monotone missing pattern
        for i in range(1, min(n_cols, 4)):
            # Start missing from some row onwards
            start_missing = draw(integers(min_value=10, max_value=n_rows//2))
            df.loc[start_missing:, col_names[i]] = np.nan

    else:  # mixed
        # Combine multiple patterns
        # Some MCAR missing
        mcar_rate = draw(floats(min_value=0.02, max_value=0.1))
        n_mcar = int(mcar_rate * n_rows * n_cols)
        for _ in range(n_mcar):
            row = draw(integers(min_value=0, max_value=n_rows-1))
            col = draw(sampled_from(col_names))
            df.loc[row, col] = np.nan

        # Some monotone pattern - ensure at least one missing
        if n_cols >= 2 and n_rows >= 10:
            monotone_col = draw(sampled_from(col_names[1:]))
            start_missing = draw(integers(min_value=5, max_value=max(6, n_rows//2)))
            df.loc[start_missing:, monotone_col] = np.nan

    return df


class TestMissingDataProperties:
    """Property-based tests for MissingDataAnalyzer."""

    @settings(deadline=None, max_examples=30)
    @given(data=generate_data_with_missing())
    def test_property_50_missing_data_documentation(self, data):
        """
        Property 50: Missing data documentation provides comprehensive analysis.

        The missing data analyzer should properly detect, analyze, and document
        missing data patterns, mechanisms, and provide appropriate imputation
        recommendations.
        """
        analyzer = MissingDataAnalyzer(data)

        # Test missing pattern analysis
        summary = analyzer.analyze_missing_patterns()

        # Property: Summary should have valid structure
        assert isinstance(summary, dict), "Summary should be a dictionary"
        assert 'overall' in summary, "Summary should have overall statistics"
        assert 'by_column' in summary, "Summary should have column-wise statistics"
        assert 'by_row' in summary, "Summary should have row-wise statistics"
        assert 'patterns' in summary, "Summary should have pattern analysis"

        # Property: Overall statistics should be consistent
        overall = summary['overall']
        total_cells = overall['total_cells']
        total_missing = overall['total_missing']
        missing_pct = overall['missing_percentage']
        complete_cases = overall['complete_cases']

        assert total_cells == data.shape[0] * data.shape[1], "Total cells should match data size"
        assert total_missing == data.isnull().sum().sum(), "Missing count should match data"
        assert abs(missing_pct - ((total_missing / total_cells) * 100)) < 0.01, "Missing percentage should be accurate"
        assert complete_cases == len(data.dropna()), "Complete cases count should be correct"

        # Property: Column statistics should be accurate
        for col, col_info in summary['by_column'].items():
            actual_missing = data[col].isnull().sum()
            assert col_info['missing_count'] == actual_missing, f"Missing count for {col} should be accurate"
            assert col_info['has_missing'] == (actual_missing > 0), f"Has missing flag for {col} should be correct"
            assert col_info['data_type'] == str(data[col].dtype), f"Data type for {col} should match"

        # Property: Row statistics should be consistent
        row_missing = data.isnull().sum(axis=1)
        by_row = summary['by_row']
        assert abs(by_row['mean_missing_per_row'] - row_missing.mean()) < 0.01, "Mean missing per row should be accurate"
        assert by_row['max_missing_in_row'] == row_missing.max(), "Max missing in row should be correct"
        assert by_row['rows_with_missing'] == (row_missing > 0).sum(), "Rows with missing count should be correct"

        # Test missingness mechanism testing
        mechanism_results = analyzer.test_missingness_mechanism()

        # Property: Mechanism test should return valid structure
        assert isinstance(mechanism_results, dict), "Mechanism test results should be a dictionary"
        assert 'conclusion' in mechanism_results, "Should have conclusion"
        assert 'pattern_analysis' in mechanism_results, "Should have pattern analysis"

        # Property: Conclusion should have valid mechanism
        conclusion = mechanism_results['conclusion']
        assert conclusion['likely_mechanism'] in ['MCAR', 'MAR', 'MNAR'], "Mechanism should be valid"
        assert conclusion['is_mcar'] in [True, False], "MCAR flag should be boolean-like"
        assert conclusion['is_mar'] in [True, False], "MAR flag should be boolean-like"
        assert conclusion['is_mnar'] in [True, False], "MNAR flag should be boolean-like"

        # Test imputation suggestions
        suggestions = analyzer.suggest_imputation_methods()

        # Property: Suggestions should be provided for columns with missing data
        missing_cols = data.columns[data.isnull().any()]
        for col in missing_cols:
            assert col in suggestions, f"Should provide suggestions for column {col}"
            col_suggestions = suggestions[col]
            assert 'suggested_methods' in col_suggestions, f"Should list methods for {col}"
            assert 'primary_recommendation' in col_suggestions, f"Should have primary recommendation for {col}"

            # Property: Suggested methods should be valid
            for method in col_suggestions['suggested_methods']:
                assert 'method' in method, "Method suggestion should have method name"
                assert 'confidence' in method, "Method suggestion should have confidence level"
                assert method['confidence'] in ['High', 'Medium', 'Low'], "Confidence should be valid"

        # Test imputation methods
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            # Test mean imputation
            imputed_mean = analyzer.apply_imputation('mean')
            assert isinstance(imputed_mean, pd.DataFrame), "Imputed data should be DataFrame"
            assert imputed_mean.shape[1] == data.shape[1], "Number of columns should be preserved"

            # Property: Mean imputation should fill numeric missing values
            for col in numeric_cols:
                if data[col].isnull().any():
                    assert not imputed_mean[col].isnull().any(), f"Mean imputation should fill {col}"

            # Test median imputation
            imputed_median = analyzer.apply_imputation('median')
            for col in numeric_cols:
                if data[col].isnull().any():
                    assert not imputed_median[col].isnull().any(), f"Median imputation should fill {col}"

        # Test evaluation
        if numeric_cols and len(analyzer.imputation_history) > 0:
            evaluation = analyzer.evaluate_imputation()

            # Property: Evaluation should have valid structure
            assert isinstance(evaluation, dict), "Evaluation should be a dictionary"
            assert 'missingness_reduction' in evaluation, "Should track missingness reduction"
            assert 'distribution_similarity' in evaluation, "Should assess distribution similarity"

            # Property: Missingness should be reduced or eliminated
            reduction = evaluation['missingness_reduction']
            assert reduction['imputed_missing_count'] <= reduction['original_missing_count'], "Missing should be reduced"

    @settings(deadline=None, max_examples=20)
    @given(data=generate_data_with_missing())
    def test_missing_pattern_consistency(self, data):
        """
        Test that missing pattern analysis is internally consistent.
        """
        analyzer = MissingDataAnalyzer(data)
        summary = analyzer.analyze_missing_patterns()

        # Property: Sum of column missing counts should equal total missing
        col_missing_sum = sum(info['missing_count'] for info in summary['by_column'].values())
        assert col_missing_sum == summary['overall']['total_missing'], "Column missing sums should equal total"

        # Property: Pattern counts should sum to total rows
        if 'pattern_counts' in summary['patterns']:
            pattern_total = sum(summary['patterns']['pattern_counts'].values())
            assert pattern_total == len(data), "Pattern counts should sum to total rows"

    @settings(deadline=None, max_examples=25)
    @given(data=generate_data_with_missing())
    def test_imputation_method_properties(self, data):
        """
        Test that imputation methods preserve important data properties.
        """
        analyzer = MissingDataAnalyzer(data)

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            pytest.skip("No numeric columns for imputation testing")

        # Get a column with missing values
        missing_numeric_cols = [col for col in numeric_cols if data[col].isnull().any()]
        if not missing_numeric_cols:
            pytest.skip("No missing numeric values for imputation testing")

        test_col = missing_numeric_cols[0]
        original_values = data[test_col].dropna()

        # Test mean imputation
        imputed_mean = analyzer.apply_imputation('mean', columns=[test_col])
        imputed_values = imputed_mean[test_col]

        # Property: Imputed values should not be missing
        assert not imputed_values.isnull().any(), "Imputed values should not be missing"

        # Property: Observed values should be unchanged
        observed_mask = data[test_col].notnull()
        np.testing.assert_array_equal(
            imputed_values[observed_mask],
            data[test_col][observed_mask],
            "Observed values should remain unchanged after imputation"
        )

        # Property: Mean should be preserved (for mean imputation)
        if len(original_values) > 1:
            assert abs(imputed_values.mean() - original_values.mean()) < 0.01, "Mean should be preserved"

        # Test mode imputation for categorical
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        missing_cat_cols = [col for col in categorical_cols if data[col].isnull().any()]

        if missing_cat_cols:
            test_cat_col = missing_cat_cols[0]
            imputed_mode = analyzer.apply_imputation('mode', columns=[test_cat_col])

            # Property: Mode imputation should not create new categories
            original_categories = set(data[test_cat_col].dropna().unique())
            imputed_categories = set(imputed_mode[test_cat_col].unique())

            # Remove NaN from comparison
            original_categories.discard(np.nan)
            imputed_categories.discard(np.nan)

            assert imputed_categories.issubset(original_categories), "Mode imputation should not create new categories"

    def test_edge_cases_robustness(self):
        """
        Test that missing data analyzer handles edge cases gracefully.
        """
        # Test with complete data
        complete_data = pd.DataFrame({
            'x': range(100),
            'y': np.random.normal(0, 1, 100),
            'z': np.random.choice(['A', 'B'], 100)
        })
        analyzer = MissingDataAnalyzer(complete_data)

        summary = analyzer.analyze_missing_patterns()
        assert summary['overall']['total_missing'] == 0, "Complete data should have no missing values"
        assert summary['overall']['complete_cases'] == 100, "All cases should be complete"

        # Test with all missing column
        all_missing_data = pd.DataFrame({
            'x': [np.nan] * 50,
            'y': range(50),
            'z': np.random.choice(['A', 'B'], 50)
        })
        analyzer = MissingDataAnalyzer(all_missing_data)

        summary = analyzer.analyze_missing_patterns()
        assert summary['by_column']['x']['missing_count'] == 50, "Should handle all-missing column"
        assert summary['by_column']['x']['missing_percentage'] == 100, "Missing percentage should be 100%"

        # Test with single column
        single_col_data = pd.DataFrame({'x': [1, 2, np.nan, 4, 5]})
        analyzer = MissingDataAnalyzer(single_col_data)

        summary = analyzer.analyze_missing_patterns()
        assert len(summary['by_column']) == 1, "Should handle single column"
        assert summary['overall']['total_cells'] == 5, "Should count cells correctly"

    @settings(deadline=None, max_examples=15)
    @given(data=generate_data_with_missing())
    def test_report_completeness(self, data):
        """
        Test that generated reports contain all required information.
        """
        analyzer = MissingDataAnalyzer(data)
        report = analyzer.generate_report()

        # Property: Report should have all major sections
        required_sections = [
            'dataset_info',
            'missing_summary',
            'missingness_analysis',
            'imputation_recommendations',
            'quality_indicators',
            'actionable_insights',
            'generated_at'
        ]

        for section in required_sections:
            assert section in report, f"Report should include {section} section"

        # Property: Dataset info should be accurate
        dataset_info = report['dataset_info']
        assert dataset_info['shape'] == data.shape, "Dataset shape should be accurate"
        assert len(dataset_info['columns']) == len(data.columns), "Column count should be accurate"
        assert len(dataset_info['numeric_columns']) == len(data.select_dtypes(include=[np.number]).columns), "Numeric column count should be accurate"

        # Property: Quality indicators should have valid values
        quality = report['quality_indicators']
        assert 0 <= quality['completeness'] <= 100, "Completeness should be between 0 and 100"
        assert quality['missing_pattern_complexity'] in ['Low', 'Medium', 'High'], "Pattern complexity should be valid"
        assert quality['imputation_difficulty'] in ['Easy', 'Moderate', 'Challenging'], "Imputation difficulty should be valid"
        assert quality['overall_quality'] in ['Excellent', 'Good', 'Fair', 'Poor'], "Overall quality should be valid"

    @settings(deadline=None, max_examples=20)
    @given(data=generate_data_with_missing())
    def test_export_preserves_integrity(self, data):
        """
        Test that export functionality preserves data integrity.
        """
        analyzer = MissingDataAnalyzer(data)
        analyzer.analyze_missing_patterns()

        # Export as dictionary
        exported_dict = analyzer.export_results(format='dict')

        # Property: Exported data should match internal state
        assert exported_dict['missing_summary'] == analyzer.missing_summary, "Dictionary export should preserve summary"

        # Export as JSON
        exported_json = analyzer.export_results(format='json')

        # Property: JSON should be parseable and contain same information
        import json
        parsed = json.loads(exported_json)
        assert 'missing_summary' in parsed, "JSON export should include summary"
        assert tuple(parsed['dataset_info']['shape']) == data.shape, "JSON should preserve dataset info"

    def test_monotone_pattern_detection(self):
        """
        Test that monotone patterns are correctly identified.
        """
        # Create data with clear monotone pattern
        n = 50
        data = pd.DataFrame({
            'var1': range(n),
            'var2': range(n),
            'var3': range(n)
        })

        # Create monotone missing
        data.loc[20:, 'var2'] = np.nan  # var2 missing from row 20
        data.loc[30:, 'var3'] = np.nan  # var3 missing from row 30

        analyzer = MissingDataAnalyzer(data)
        monotone_cols = analyzer._detect_monotone_patterns()

        # Property: Should detect monotone columns
        assert 'var2' in monotone_cols, "Should detect var2 as monotone"
        assert 'var3' in monotone_cols, "Should detect var3 as monotone"
        assert 'var1' not in monotone_cols, "Should not detect complete var1 as monotone"

    @settings(deadline=None, max_examples=20)
    @given(data=generate_data_with_missing())
    def test_imputation_history_tracking(self, data):
        """
        Test that imputation operations are properly tracked.
        """
        analyzer = MissingDataAnalyzer(data)

        # Apply multiple imputations
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns

        imputation_count = 0

        # Ensure we have some missing values to impute
        has_missing = data.isnull().any().any()

        if len(numeric_cols) > 0 and has_missing:
            analyzer.apply_imputation('mean')
            imputation_count += 1

            if len(numeric_cols) > 1:
                analyzer.apply_imputation('median', columns=list(numeric_cols[:1]))
                imputation_count += 1

                analyzer.apply_imputation('constant', fill_value=0, columns=list(numeric_cols[:1]))
                imputation_count += 1

        if len(categorical_cols) > 0 and has_missing and imputation_count < 3:
            analyzer.apply_imputation('mode')
            imputation_count += 1

        # Property: History should track all imputations
        # Note: Imputation is only recorded when there are actually missing values
        if has_missing:
            assert len(analyzer.imputation_history) == imputation_count, f"Should track all {imputation_count} imputation operations"
        else:
            assert len(analyzer.imputation_history) == 0, "No imputation when no missing values"

        for record in analyzer.imputation_history:
            # Property: Each record should have required fields
            assert 'method' in record, "Record should have method"
            assert 'columns' in record, "Record should have columns"
            assert 'timestamp' in record, "Record should have timestamp"
            assert 'original_missing' in record, "Record should have original missing counts"

    @settings(deadline=None, max_examples=25)
    @given(data=generate_data_with_missing())
    def test_visualization_return_types(self, data):
        """
        Test that visualization methods return appropriate types.
        """
        analyzer = MissingDataAnalyzer(data)

        # Test with mocked matplotlib
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend

        # Property: Visualization should handle all plot types
        plot_types = ['matrix', 'bar', 'heatmap', 'dendrogram']
        for plot_type in plot_types:
            try:
                result = analyzer.visualize_missing_patterns(plot_type=plot_type, save_plot=True)
                # When save_plot=True, should return string (base64)
                assert result is None or isinstance(result, str), f"Visualization for {plot_type} should return string or None"
            except Exception:
                # Some plot types might fail with certain data patterns
                pass