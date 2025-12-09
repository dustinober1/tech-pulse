"""Property-based tests for OutlierHandler class."""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies, settings
from hypothesis.strategies import (
    lists, floats, integers, text, composite, sampled_from,
    data, one_of, just, none
)
from typing import Dict, Any, List, Tuple

from src.outliers.outlier_handler import OutlierHandler


@composite
def generate_data_with_outliers(draw):
    """Generate a DataFrame with potential outliers."""
    # Generate base data
    n_rows = draw(integers(min_value=50, max_value=200))
    n_cols = draw(integers(min_value=1, max_value=5))

    # Generate column names
    col_names = [f"col_{i}" for i in range(n_cols)]

    data = {}
    for col in col_names:
        # Choose distribution type
        dist_type = draw(sampled_from(['normal', 'skewed', 'uniform']))

        if dist_type == 'normal':
            mean = draw(floats(min_value=-5, max_value=5))
            std = draw(floats(min_value=0.1, max_value=3))
            base_data = np.random.normal(mean, std, n_rows)
        elif dist_type == 'skewed':
            # Create skewed data (exponential)
            base_data = np.random.exponential(draw(floats(min_value=0.5, max_value=3)), n_rows)
            # Add some shifts
            if draw(strategies.booleans()):
                base_data += draw(floats(min_value=-5, max_value=5))
        else:  # uniform
            low = draw(floats(min_value=-10, max_value=0))
            high = draw(floats(min_value=0, max_value=10))
            base_data = np.random.uniform(low, high, n_rows)

        # Add outliers with some probability
        if draw(strategies.booleans()):
            n_outliers = draw(integers(min_value=1, max_value=min(20, n_rows // 5)))
            outlier_indices = np.random.choice(n_rows, n_outliers, replace=False)
            outlier_values = draw(lists(
                floats(min_value=-50, max_value=50),
                min_size=n_outliers,
                max_size=n_outliers
            ))
            base_data[outlier_indices] = outlier_values

        data[col] = base_data

    return pd.DataFrame(data)


class TestOutlierHandlerProperties:
    """Property-based tests for OutlierHandler."""

    @settings(deadline=None, max_examples=50)
    @given(data=generate_data_with_outliers())
    def test_property_49_outlier_analysis_and_treatment(self, data):
        """
        Property 49: Outlier analysis and treatment provides robust handling.

        The outlier handler should properly detect, analyze, and provide
        multiple treatment strategies for outliers across different data distributions.
        """
        handler = OutlierHandler(data)

        # Test IQR method
        iqr_results = handler.detect_iqr_outliers()

        # Property: IQR detection should return valid structure
        assert isinstance(iqr_results, dict), "IQR results should be a dictionary"
        assert 'method' in iqr_results, "IQR results should have method field"
        assert iqr_results['method'] == 'IQR', "Method should be 'IQR'"
        assert 'outliers_by_column' in iqr_results, "Should have outliers_by_column field"
        assert 'summary' in iqr_results, "Should have summary field"

        # Property: Summary should be consistent
        summary = iqr_results['summary']
        total_outliers = 0
        columns_with_outliers = 0

        for col_info in iqr_results['outliers_by_column'].values():
            total_outliers += col_info.get('count', 0)
            if col_info.get('count', 0) > 0:
                columns_with_outliers += 1

            # Property: Each column outlier info should have required fields
            assert 'count' in col_info, f"Missing count in {col_info}"
            assert 'percentage' in col_info, f"Missing percentage in {col_info}"
            assert 'indices' in col_info, f"Missing indices in {col_info}"
            assert 'values' in col_info, f"Missing values in {col_info}"

            # Property: Bounds should be valid for IQR
            if 'bounds' in col_info:
                bounds = col_info['bounds']
                assert 'lower' in bounds, "Missing lower bound"
                assert 'upper' in bounds, "Missing upper bound"
                assert 'Q1' in bounds, "Missing Q1"
                assert 'Q3' in bounds, "Missing Q3"
                assert 'IQR' in bounds, "Missing IQR"
                assert bounds['lower'] < bounds['upper'], "Lower bound should be less than upper"
                assert bounds['Q1'] <= bounds['Q3'], "Q1 should be less than or equal to Q3"
                assert bounds['IQR'] == bounds['Q3'] - bounds['Q1'], "IQR should equal Q3 - Q1"

        assert summary['total_outliers'] == total_outliers, "Summary outlier count should match"
        assert summary['columns_with_outliers'] == columns_with_outliers, "Summary columns count should match"

        # Test Z-score method
        zscore_results = handler.detect_zscore_outliers()

        # Property: Z-score detection should return valid structure
        assert isinstance(zscore_results, dict), "Z-score results should be a dictionary"
        assert zscore_results['method'] == 'Z-Score', "Method should be 'Z-Score'"
        assert 'outliers_by_column' in zscore_results, "Should have outliers_by_column field"
        assert 'summary' in zscore_results, "Should have summary field"

        # Test multivariate methods if we have enough data
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2 and len(data) >= 20:
            isolation_results = handler.detect_isolation_forest_outliers()

            # Property: Isolation Forest should return valid structure
            assert isinstance(isolation_results, dict), "Isolation Forest results should be a dictionary"
            assert 'method' in isolation_results, "Should have method field"
            assert isolation_results['method'] == 'Isolation Forest', "Method should be 'Isolation Forest'"

            if 'error' not in isolation_results:
                assert 'outliers' in isolation_results, "Should have outliers field"
                assert 'count' in isolation_results['outliers'], "Should have outlier count"
                assert 'percentage' in isolation_results['outliers'], "Should have outlier percentage"
                assert 0 <= isolation_results['outliers']['percentage'] <= 100, "Percentage should be between 0 and 100"

        # Test outlier treatment strategies
        handler.detect_iqr_outliers()

        # Property: Remove outliers should reduce data size
        data_removed = handler.handle_outliers(method='remove', method_used='iqr')
        assert len(data_removed) <= len(data), "Data size should reduce or stay same when removing outliers"

        # Property: Clipping outliers should maintain data size
        data_clipped = handler.handle_outliers(method='clip', method_used='iqr')
        assert len(data_clipped) == len(data), "Data size should remain same when clipping outliers"

        # Property: Replacement should maintain data size
        data_replaced = handler.handle_outliers(method='replace', method_used='iqr', replacement_value='median')
        assert len(data_replaced) == len(data), "Data size should remain same when replacing outliers"

        # Property: Treatment should be recorded
        assert len(handler.treatment_history) > 0, "Treatment should be recorded in history"
        treatment = handler.treatment_history[-1]
        assert 'method' in treatment, "Treatment method should be recorded"
        assert 'method_used' in treatment, "Detection method used should be recorded"
        assert 'original_shape' in treatment, "Original shape should be recorded"
        assert 'new_shape' in treatment, "New shape should be recorded"
        assert 'timestamp' in treatment, "Timestamp should be recorded"

    @settings(deadline=None, max_examples=30)
    @given(data=generate_data_with_outliers())
    def test_outlier_detection_consistency(self, data):
        """
        Test that outlier detection methods are consistent within their parameters.
        """
        handler = OutlierHandler(data, z_threshold=2.5, iqr_factor=2.0)

        # Test with different parameters
        iqr_default = handler.detect_iqr_outliers()
        iqr_custom = handler.detect_iqr_outliers(iqr_factor=3.0)

        # Property: Higher IQR factor should detect fewer outliers
        for col in iqr_default['outliers_by_column']:
            if col in iqr_custom['outliers_by_column']:
                default_count = iqr_default['outliers_by_column'][col]['count']
                custom_count = iqr_custom['outliers_by_column'][col]['count']
                assert custom_count <= default_count, f"Higher IQR factor should detect fewer outliers in {col}"

        # Test Z-score with different thresholds
        zscore_default = handler.detect_zscore_outliers()
        zscore_custom = handler.detect_zscore_outliers(z_threshold=2.0)

        # Property: Lower Z-score threshold should detect more outliers
        for col in zscore_default['outliers_by_column']:
            if col in zscore_custom['outliers_by_column']:
                default_count = zscore_default['outliers_by_column'][col]['count']
                custom_count = zscore_custom['outliers_by_column'][col]['count']
                assert custom_count >= default_count, f"Lower Z-score threshold should detect more outliers in {col}"

    @settings(deadline=None, max_examples=30)
    @given(data=generate_data_with_outliers())
    def test_outlier_summary_aggregation(self, data):
        """
        Test that outlier summary properly aggregates results from multiple methods.
        """
        handler = OutlierHandler(data)

        # Run multiple detection methods
        handler.detect_iqr_outliers()
        handler.detect_zscore_outliers()
        handler.detect_isolation_forest_outliers()

        summary = handler.get_outlier_summary()

        # Property: Summary should include all applied methods
        assert 'methods_applied' in summary, "Summary should list methods applied"
        assert set(summary['methods_applied']) == {'iqr', 'zscore', 'isolation_forest'}, "Should include all detection methods"

        # Property: Summary should include outlier counts
        assert 'outlier_counts' in summary, "Summary should include outlier counts"
        assert 'iqr' in summary['outlier_counts'], "Should include IQR outlier count"
        assert 'zscore' in summary['outlier_counts'], "Should include Z-score outlier count"

        # Property: Summary should include dataset info
        assert 'dataset_info' in summary, "Summary should include dataset info"
        assert 'shape' in summary['dataset_info'], "Should include data shape"
        assert summary['dataset_info']['shape'] == data.shape, "Shape should match original data"
        assert 'numeric_columns' in summary['dataset_info'], "Should list numeric columns"

        # Property: Summary should provide recommendations when appropriate
        assert 'recommendations' in summary, "Summary should include recommendations"
        assert isinstance(summary['recommendations'], list), "Recommendations should be a list"

    @settings(deadline=None, max_examples=20)
    @given(data=generate_data_with_outliers())
    def test_outlier_treatment_preserves_structure(self, data):
        """
        Test that outlier treatment preserves data structure and relationships.
        """
        handler = OutlierHandler(data)
        handler.detect_iqr_outliers()

        # Test each treatment method
        treatments = ['remove', 'clip', 'replace']

        for treatment in treatments:
            if treatment == 'replace':
                data_treated = handler.handle_outliers(
                    method=treatment,
                    method_used='iqr',
                    replacement_value='median'
                )
            else:
                data_treated = handler.handle_outliers(method=treatment, method_used='iqr')

            # Property: Treated data should maintain column structure
            assert set(data_treated.columns) == set(data.columns), "Treated data should maintain original columns"

            # Property: Categorical columns should remain unchanged
            cat_cols = data.select_dtypes(exclude=[np.number]).columns
            for col in cat_cols:
                assert col in data_treated.columns, f"Categorical column {col} should be preserved"
                assert data_treated[col].equals(data[col]), f"Categorical column {col} should be unchanged"

            # Property: Data types should be preserved for numeric columns
            num_cols = data.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                assert col in data_treated.columns, f"Numeric column {col} should be preserved"
                assert data_treated[col].dtype == data[col].dtype, f"Data type for {col} should be preserved"

    @settings(deadline=None, max_examples=30)
    @given(data=generate_data_with_outliers())
    def test_outlier_bounds_validity(self, data):
        """
        Test that outlier bounds are mathematically valid.
        """
        handler = OutlierHandler(data)
        iqr_results = handler.detect_iqr_outliers()

        for col, info in iqr_results['outliers_by_column'].items():
            if 'bounds' in info:
                bounds = info['bounds']
                Q1 = bounds['Q1']
                Q3 = bounds['Q3']
                lower = bounds['lower']
                upper = bounds['upper']

                # Property: Bounds should be consistent with quartiles
                assert lower == Q1 - handler.iqr_factor * (Q3 - Q1), \
                    f"Lower bound should equal Q1 - IQR_factor * (Q3 - Q1) for {col}"
                assert upper == Q3 + handler.iqr_factor * (Q3 - Q1), \
                    f"Upper bound should equal Q3 + IQR_factor * (Q3 - Q1) for {col}"

                # Property: All outliers should be outside bounds
                outlier_values = info['values']
                for val in outlier_values:
                    assert val < lower or val > upper, \
                        f"Outlier value {val} should be outside bounds [{lower}, {upper}] for {col}"

    @settings(deadline=None, max_examples=20)
    @given(data=generate_data_with_outliers())
    def test_outlier_indices_accuracy(self, data):
        """
        Test that outlier indices correctly reference actual data points.
        """
        handler = OutlierHandler(data)
        iqr_results = handler.detect_iqr_outliers()

        for col, info in iqr_results['outliers_by_column'].items():
            if info['count'] > 0:
                indices = info['indices']
                outlier_values = info['values']
                actual_values = data[col].iloc[indices].values

                # Property: Indices should point to correct outlier values
                np.testing.assert_array_equal(
                    actual_values, outlier_values,
                    err_msg=f"Outlier indices should correctly reference data values for {col}",
                    rtol=1e-10
                )

                # Property: All indices should be valid
                for idx in indices:
                    assert idx in data.index, f"Index {idx} should exist in data index"

    @settings(deadline=None, max_examples=30)
    @given(data=generate_data_with_outliers())
    def test_export_results_completeness(self, data):
        """
        Test that export functionality preserves all important information.
        """
        handler = OutlierHandler(data)
        handler.detect_iqr_outliers()
        handler.detect_zscore_outliers()

        # Export as dictionary
        results_dict = handler.export_results(format='dict')

        # Property: Export should include all sections
        assert 'summary' in results_dict, "Export should include summary"
        assert 'detailed_results' in results_dict, "Export should include detailed results"
        assert 'treatment_history' in results_dict, "Export should include treatment history"
        assert 'parameters' in results_dict, "Export should include parameters"

        # Property: Exported results should be internally consistent
        assert results_dict['detailed_results'] == handler.outlier_results, \
            "Exported results should match internal results"
        assert results_dict['treatment_history'] == handler.treatment_history, \
            "Exported history should match internal history"

        # Export as JSON
        results_json = handler.export_results(format='json')

        # Property: JSON export should be parseable
        import json
        parsed = json.loads(results_json)
        assert isinstance(parsed, dict), "JSON export should be parseable"
        assert 'summary' in parsed, "JSON export should include summary"

    def test_edge_cases_robustness(self):
        """
        Test that outlier handler handles edge cases gracefully.
        """
        # Test with constant data
        constant_data = pd.DataFrame({'x': [5.0] * 100})
        handler = OutlierHandler(constant_data)
        results = handler.detect_iqr_outliers()
        assert isinstance(results, dict), "Should handle constant data gracefully"

        # Test with single data point
        single_data = pd.DataFrame({'x': [1.0]})
        handler = OutlierHandler(single_data)
        results = handler.detect_iqr_outliers()
        assert isinstance(results, dict), "Should handle single point data gracefully"

        # Test with mixed data types
        mixed_data = pd.DataFrame({
            'numeric': [1, 2, 3, 100, 4, 5],
            'text': ['a', 'b', 'c', 'd', 'e', 'f'],
            'dates': pd.date_range('2020-01-01', periods=6)
        })
        handler = OutlierHandler(mixed_data)
        results = handler.detect_iqr_outliers()
        assert isinstance(results, dict), "Should handle mixed data types gracefully"
        assert 'numeric' in results['outliers_by_column'], "Should process numeric columns only"
        assert len(results['outliers_by_column']) == 1, "Should only process one numeric column"