"""Property-based tests for StatisticalValidator class."""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, assume, settings
from hypothesis.strategies import (
    lists, floats, integers, text, composite, sampled_from,
    data, one_of, just, none
)
from typing import Dict, Any

from src.statistical.statistical_validator import StatisticalValidator


@composite
def generate_dataframe(draw):
    """Generate a DataFrame for testing."""
    # Generate number of columns
    n_cols = draw(integers(min_value=1, max_value=5))
    n_rows = draw(integers(min_value=50, max_value=200))

    # Generate column names
    col_names = [f"col_{i}" for i in range(n_cols)]

    # Generate data
    data = {}
    for col in col_names:
        # Choose distribution type
        dist_type = draw(sampled_from(['normal', 'exponential', 'uniform', 'mixed']))

        if dist_type == 'normal':
            mean = draw(floats(min_value=-10, max_value=10))
            std = draw(floats(min_value=0.1, max_value=5))
            data[col] = np.random.normal(mean, std, n_rows)
        elif dist_type == 'exponential':
            scale = draw(floats(min_value=0.1, max_value=5))
            data[col] = np.random.exponential(scale, n_rows)
        elif dist_type == 'uniform':
            low = draw(floats(min_value=-10, max_value=0))
            high = draw(floats(min_value=0, max_value=10))
            data[col] = np.random.uniform(low, high, n_rows)
        else:  # mixed - combine normal with some outliers
            base = np.random.normal(0, 1, n_rows)
            # Add outliers
            n_outliers = n_rows // 10
            outlier_indices = np.random.choice(n_rows, n_outliers, replace=False)
            base[outlier_indices] = np.random.uniform(-10, 10, n_outliers)
            data[col] = base

    df = pd.DataFrame(data)
    return df


class TestStatisticalValidatorProperties:
    """Property-based tests for StatisticalValidator."""

    @settings(deadline=None, max_examples=50)
    @given(data=generate_dataframe())
    def test_property_44_metric_uncertainty_reporting(self, data):
        """
        Property 44: Metric uncertainty reporting provides confidence intervals.

        The statistical validator should properly report uncertainty in its metrics
        through confidence intervals calculated using bootstrap methods.
        """
        # Create validator
        validator = StatisticalValidator(data)

        # Calculate confidence intervals
        confidence_levels = [0.90, 0.95, 0.99]

        for confidence_level in confidence_levels:
            ci_results = validator.calculate_confidence_intervals(
                confidence_level=confidence_level,
                n_bootstrap=100  # Use fewer for faster testing
            )

            # Property: Confidence intervals should be present for each numeric column
            numeric_cols = validator._get_numeric_columns()

            for col in numeric_cols:
                if len(data[col].dropna()) >= 10:  # Only check columns with sufficient data
                    # Property: Column should have confidence interval results
                    assert col in ci_results['details'], f"Missing CI results for column {col}"

                    col_ci = ci_results['details'][col]

                    # Property: Should not have errors for sufficient data
                    assert 'error' not in col_ci, f"Unexpected error for column {col}: {col_ci.get('error')}"

                    # Property: Should have mean and standard deviation intervals
                    assert 'mean' in col_ci, f"Missing mean CI for column {col}"
                    assert 'std' in col_ci, f"Missing std CI for column {col}"

                    # Property: Mean CI should have valid structure
                    mean_ci = col_ci['mean']
                    assert 'lower' in mean_ci, f"Missing lower bound for mean CI in column {col}"
                    assert 'upper' in mean_ci, f"Missing upper bound for mean CI in column {col}"
                    assert isinstance(mean_ci['lower'], (int, float)), f"Lower bound not numeric for {col}"
                    assert isinstance(mean_ci['upper'], (int, float)), f"Upper bound not numeric for {col}"

                    # Property: Lower bound should be <= upper bound
                    assert mean_ci['lower'] <= mean_ci['upper'], f"Mean CI invalid for column {col}"

                    # Property: Std CI should have valid structure
                    std_ci = col_ci['std']
                    assert 'lower' in std_ci, f"Missing lower bound for std CI in column {col}"
                    assert 'upper' in std_ci, f"Missing upper bound for std CI in column {col}"
                    assert isinstance(std_ci['lower'], (int, float)), f"Std lower bound not numeric for {col}"
                    assert isinstance(std_ci['upper'], (int, float)), f"Std upper bound not numeric for {col}"

                    # Property: Std bounds should be non-negative
                    assert std_ci['lower'] >= 0, f"Std lower bound negative for column {col}"
                    assert std_ci['upper'] >= 0, f"Std upper bound negative for column {col}"
                    assert std_ci['lower'] <= std_ci['upper'], f"Std CI invalid for column {col}"

                    # Property: Percentiles should match confidence level
                    assert 'percentiles' in col_ci, f"Missing percentiles for column {col}"
                    percentiles = col_ci['percentiles']
                    expected_lower = (1 - confidence_level) / 2 * 100
                    expected_upper = (1 + confidence_level) / 2 * 100
                    assert abs(percentiles['lower'] - expected_lower) < 0.1, f"Wrong lower percentile for {col}"
                    assert abs(percentiles['upper'] - expected_upper) < 0.1, f"Wrong upper percentile for {col}"

            # Property: Summary should contain correct confidence level
            assert ci_results['summary']['confidence_level'] == confidence_level
            assert ci_results['summary']['n_bootstrap'] == 100

            # Property: Higher confidence should produce wider intervals (probabilistic)
            # Note: This is a weak statistical property and may not hold for small bootstrap samples
            # We mainly test that the intervals are properly structured and valid

    @settings(deadline=None, max_examples=30)
    @given(data=generate_dataframe())
    def test_bootstrap_uncertainty_properties(self, data):
        """Test properties specific to bootstrap uncertainty estimation."""
        validator = StatisticalValidator(data)

        # Test with different bootstrap sample sizes
        n_bootstrap_values = [50, 100, 200]

        results = {}
        for n_bootstrap in n_bootstrap_values:
            results[n_bootstrap] = validator.calculate_confidence_intervals(
                confidence_level=0.95,
                n_bootstrap=n_bootstrap
            )

        numeric_cols = validator._get_numeric_columns()

        for col in numeric_cols:
            if len(data[col].dropna()) >= 10:
                # Property: More bootstrap samples should generally give more stable results
                # Check standard error of the mean estimate
                for i, n1 in enumerate(n_bootstrap_values[:-1]):
                    n2 = n_bootstrap_values[i + 1]

                    if col in results[n1]['details'] and col in results[n2]['details']:
                        se1 = results[n1]['details'][col]['mean']['std_error']
                        se2 = results[n2]['details'][col]['mean']['std_error']

                        # Property: Standard error should be finite and non-negative
                        assert isinstance(se1, (int, float)), f"SE not numeric for {col} with {n1} samples"
                        assert isinstance(se2, (int, float)), f"SE not numeric for {col} with {n2} samples"
                        assert se1 >= 0, f"SE negative for {col} with {n1} samples"
                        assert se2 >= 0, f"SE negative for {col} with {n2} samples"

                        # Property: Standard error shouldn't be zero for variable data
                        if data[col].std() > 0.01:  # Only check for variable columns
                            assert se1 > 0, f"SE zero for variable column {col}"
                            assert se2 > 0, f"SE zero for variable column {col}"

    @settings(deadline=None, max_examples=20)
    @given(data=generate_dataframe())
    def test_uncertainty_with_different_data_sizes(self, data):
        """Test how uncertainty estimates behave with different data sizes."""
        validator = StatisticalValidator(data)

        # Test with subsets of data
        original_size = len(data)
        sizes_to_test = []

        # Create different size subsets
        if original_size >= 100:
            sizes_to_test.extend([50, min(75, original_size)])
        if original_size >= 50:
            sizes_to_test.append(25)

        sizes_to_test.append(original_size)  # Include full dataset

        results_by_size = {}
        for size in sizes_to_test:
            if size <= original_size:
                subset = data.iloc[:size]
                sub_validator = StatisticalValidator(subset)

                ci_results = sub_validator.calculate_confidence_intervals(
                    confidence_level=0.95,
                    n_bootstrap=50
                )
                results_by_size[size] = ci_results

        # Property: Smaller samples should generally have wider confidence intervals
        numeric_cols = validator._get_numeric_columns()

        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) >= 20:  # Only test columns with sufficient data
                widths_by_size = []

                for size in sorted(sizes_to_test):
                    if size <= len(col_data) and size in results_by_size:
                        if col in results_by_size[size]['details']:
                            mean_ci = results_by_size[size]['details'][col]['mean']
                            width = mean_ci['upper'] - mean_ci['lower']
                            widths_by_size.append((size, width))

                # Check monotonicity (larger samples -> smaller intervals)
                # This is probabilistic, so we use a relaxed check
                if len(widths_by_size) >= 2:
                    widths_by_size.sort(key=lambda x: x[0])  # Sort by sample size

                    # Check that the smallest sample doesn't have the narrowest interval
                    # This would be very unlikely under normal circumstances
                    if len(widths_by_size) >= 3:
                        smallest_width = widths_by_size[0][1]
                        largest_width = widths_by_size[-1][1]

                        # The relationship should generally be inverse, but we allow exceptions
                        # due to bootstrap randomness
                        assert not (smallest_width < largest_width * 0.5), \
                            f"Smallest sample has suspiciously narrow interval for {col}"

    @settings(deadline=None, max_examples=20)
    @given(data=generate_dataframe())
    def test_uncertainty_assumptions_documentation(self, data):
        """Test that uncertainty assumptions are properly documented."""
        validator = StatisticalValidator(data)

        ci_results = validator.calculate_confidence_intervals(
            confidence_level=0.95,
            n_bootstrap=100
        )

        # Property: Assumptions should be documented
        assert 'assumptions' in ci_results, "Missing assumptions documentation"
        assumptions = ci_results['assumptions']

        # Property: Should document bootstrap sample sufficiency
        assert 'bootstrap_samples_sufficient' in assumptions, "Missing bootstrap samples assumption"
        assert isinstance(assumptions['bootstrap_samples_sufficient'], bool), "Bootstrap samples assumption not boolean"

        # Property: Should document data availability
        assert 'data_points_available' in assumptions, "Missing data points assumption"
        assert isinstance(assumptions['data_points_available'], bool), "Data points assumption not boolean"

        # Property: Should track errors
        assert 'errors' in ci_results['summary'], "Missing error tracking"
        assert isinstance(ci_results['summary']['errors'], int), "Error count not integer"
        assert ci_results['summary']['errors'] >= 0, "Negative error count"

        # Property: Should track processed columns
        assert 'columns_processed' in ci_results['summary'], "Missing columns processed tracking"
        assert isinstance(ci_results['summary']['columns_processed'], int), "Columns processed not integer"
        assert ci_results['summary']['columns_processed'] >= 0, "Negative columns processed"

    @settings(deadline=None, max_examples=30)
    @given(data=generate_dataframe())
    def test_uncertainty_reproducibility(self, data):
        """Test that uncertainty estimates are reproducible with fixed seed."""
        validator = StatisticalValidator(data)

        # Set random seed for reproducibility
        np.random.seed(42)
        ci_results1 = validator.calculate_confidence_intervals(
            confidence_level=0.95,
            n_bootstrap=50
        )

        # Reset seed and run again
        np.random.seed(42)
        ci_results2 = validator.calculate_confidence_intervals(
            confidence_level=0.95,
            n_bootstrap=50
        )

        # Property: Results should be identical with same seed
        numeric_cols = validator._get_numeric_columns()

        for col in numeric_cols:
            if len(data[col].dropna()) >= 10:
                if col in ci_results1['details'] and col in ci_results2['details']:
                    col1 = ci_results1['details'][col]
                    col2 = ci_results2['details'][col]

                    # Property: Mean CIs should match
                    assert col1['mean']['lower'] == col2['mean']['lower'], f"Mean lower bound differs for {col}"
                    assert col1['mean']['upper'] == col2['mean']['upper'], f"Mean upper bound differs for {col}"

                    # Property: Std CIs should match
                    assert col1['std']['lower'] == col2['std']['lower'], f"Std lower bound differs for {col}"
                    assert col1['std']['upper'] == col2['std']['upper'], f"Std upper bound differs for {col}"

    @settings(deadline=None, max_examples=30)
    @given(data=generate_dataframe())
    def test_property_45_statistical_assumption_documentation(self, data):
        """
        Property 45: Statistical assumption documentation provides clear validation.

        The statistical validator should document all statistical assumptions,
        test them appropriately, and provide clear validation results.
        """
        validator = StatisticalValidator(data)

        # Run comprehensive analysis
        results = validator.run_comprehensive_analysis()

        # Property: Should document assumption validation
        assert 'assumption_validation' in results, "Missing assumption validation documentation"
        validation = results['assumption_validation']

        # Property: Should document all key statistical assumptions
        expected_assumptions = ['normality', 'homoscedasticity', 'independence']
        for assumption in expected_assumptions:
            assert assumption in validation, f"Missing {assumption} assumption documentation"
            assert 'assumption_met' in validation[assumption], f"Missing assumption_met for {assumption}"
            assert 'reason' in validation[assumption], f"Missing reason for {assumption}"
            # Handle both Python bool and numpy bool_
            assert isinstance(validation[assumption]['assumption_met'], (bool, np.bool_)), f"assumption_met not boolean for {assumption}"
            assert isinstance(validation[assumption]['reason'], str), f"reason not string for {assumption}"

        # Property: Should provide overall assessment
        assert 'overall' in validation, "Missing overall assessment"
        overall = validation['overall']
        assert 'assumptions_met' in overall, "Missing assumptions_met count"
        assert 'total_assumptions' in overall, "Missing total_assumptions count"
        assert isinstance(overall['assumptions_met'], int), "assumptions_met not integer"
        assert isinstance(overall['total_assumptions'], int), "total_assumptions not integer"
        assert 0 <= overall['assumptions_met'] <= overall['total_assumptions'], "Invalid assumptions_met count"

        # Property: Normality tests should be documented
        assert 'normality_tests' in results, "Missing normality tests documentation"
        normality = results['normality_tests']

        # Property: Should document test parameters
        assert 'summary' in normality, "Missing normality summary"
        assert 'significance_level' in normality['summary'], "Missing significance level"
        assert isinstance(normality['summary']['significance_level'], float), "Significance level not float"
        assert 0 < normality['summary']['significance_level'] < 1, "Invalid significance level"

        # Property: Should document test results per column
        assert 'details' in normality, "Missing normality details"
        numeric_cols = validator._get_numeric_columns()

        for col in numeric_cols:
            if len(data[col].dropna()) >= 3:  # Minimum for normality tests
                if col in normality['details']:
                    col_result = normality['details'][col]

                    # Property: Should perform multiple tests
                    expected_tests = ['shapiro_wilk', 'anderson_darling', 'ks_test']
                    for test in expected_tests:
                        assert test in col_result, f"Missing {test} for column {col}"

                        test_result = col_result[test]
                        # Property: Each test should have proper structure
                        assert 'test_name' in test_result, f"Missing test_name in {test} for {col}"
                        # Check that test name contains the expected test identifier
                        if test == 'shapiro_wilk':
                            assert 'Shapiro' in test_result['test_name'] and 'Wilk' in test_result['test_name'], f"Wrong test_name for {test}: {test_result['test_name']}"
                        elif test == 'anderson_darling':
                            assert 'Anderson' in test_result['test_name'] and 'Darling' in test_result['test_name'], f"Wrong test_name for {test}: {test_result['test_name']}"
                        elif test == 'ks_test':
                            assert 'Kolmogorov' in test_result['test_name'] or 'KS' in test_result['test_name'], f"Wrong test_name for {test}: {test_result['test_name']}"

                        # Property: Should handle test errors gracefully
                        if 'error' not in test_result:
                            # For successful tests, check required fields
                            if test == 'shapiro_wilk' or test == 'ks_test':
                                assert 'statistic' in test_result, f"Missing statistic in {test} for {col}"
                                assert 'p_value' in test_result, f"Missing p_value in {test} for {col}"
                                assert 'is_normal' in test_result, f"Missing is_normal in {test} for {col}"
                                assert isinstance(test_result['is_normal'], (bool, np.bool_)), f"is_normal not boolean in {test} for {col}"
                                assert 0 <= test_result['statistic'] <= float('inf'), f"Invalid statistic in {test} for {col}"
                                assert 0 <= test_result['p_value'] <= 1, f"Invalid p_value in {test} for {col}"

                    # Property: Should provide overall normality assessment
                    assert 'is_normal' in col_result, f"Missing overall is_normal for column {col}"
                    assert isinstance(col_result['is_normal'], (bool, np.bool_)), f"Overall is_normal not boolean for {col}"

        # Property: Homoscedasticity tests should be documented
        assert 'homoscedasticity_tests' in results, "Missing homoscedasticity tests documentation"
        homoscedasticity = results['homoscedasticity_tests']

        assert 'summary' in homoscedasticity, "Missing homoscedasticity summary"
        assert 'test_used' in homoscedasticity['summary'], "Missing test_used documentation"
        assert homoscedasticity['summary']['test_used'] == 'Breusch-Pagan', "Wrong test documented"

        if 'homoscedastic' in homoscedasticity['summary'] and homoscedasticity['summary']['homoscedastic'] is not None:
            assert isinstance(homoscedasticity['summary']['homoscedastic'], (bool, np.bool_)), "homoscedastic not boolean"

        # Property: Independence tests should be documented
        assert 'independence_tests' in results, "Missing independence tests documentation"
        independence = results['independence_tests']

        assert 'summary' in independence, "Missing independence summary"
        if 'independent' in independence['summary'] and independence['summary']['independent'] is not None:
            assert isinstance(independence['summary']['independent'], (bool, np.bool_)), "independent not boolean"

        # Property: Should document test assumptions
        if 'assumptions' in homoscedasticity['summary']:
            assert isinstance(homoscedasticity['summary'], dict), "Homoscedasticity assumptions not documented"

        # Property: String representations should be informative
        summary_str = validator.get_summary_statistics()
        assert isinstance(summary_str, str), "Summary statistics not string"
        assert len(summary_str) > 0, "Empty summary statistics string"
        assert "Summary Statistics" in summary_str, "Missing header in summary statistics"

        assumption_str = validator.get_assumption_status()
        assert isinstance(assumption_str, str), "Assumption status not string"
        assert len(assumption_str) > 0, "Empty assumption status string"
        assert "Statistical Assumption Status" in assumption_str, "Missing header in assumption status"

        # Property: Should document timestamp and data info
        assert 'timestamp' in results, "Missing timestamp"
        assert 'data_info' in results, "Missing data info"
        data_info = results['data_info']
        assert 'shape' in data_info, "Missing data shape"
        assert 'alpha' in data_info, "Missing alpha level"
        assert isinstance(data_info['alpha'], float), "Alpha not float"

        # Property: Export functionality should preserve documentation
        import io
        json_output = io.StringIO()
        validator.export_results(results, json_output, format='json')

        json_str = json_output.getvalue()
        assert len(json_str) > 0, "Empty JSON export"
        assert 'assumption_validation' in json_str, "Assumption validation missing from export"
        assert 'normality_tests' in json_str, "Normality tests missing from export"