"""Unit tests for StatisticalValidator class."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from scipy import stats

from src.statistical.statistical_validator import StatisticalValidator


class TestStatisticalValidator:
    """Test cases for StatisticalValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a StatisticalValidator instance for testing."""
        np.random.seed(42)
        data = pd.DataFrame({
            'numeric1': np.random.normal(0, 1, 100),
            'numeric2': np.random.normal(5, 2, 100),
            'numeric3': np.random.exponential(1, 100),
            'categorical': np.random.choice(['A', 'B', 'C'], 100)
        })
        return StatisticalValidator(data)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'x1': np.random.normal(0, 1, 50),
            'x2': np.random.normal(2, 1, 50),
            'y': np.random.normal(1 + 0.5 * np.random.normal(0, 1, 50), 0.5, 50)
        })

    def test_initialization(self, sample_data):
        """Test StatisticalValidator initialization."""
        validator = StatisticalValidator(sample_data)

        assert isinstance(validator.data, pd.DataFrame)
        assert len(validator.data) == 50
        assert list(validator.data.columns) == ['x1', 'x2', 'y']
        assert validator.alpha == 0.05

    def test_initialization_with_alpha(self, sample_data):
        """Test StatisticalValidator initialization with custom alpha."""
        validator = StatisticalValidator(sample_data, alpha=0.01)
        assert validator.alpha == 0.01

    def test_initialization_empty_data(self):
        """Test initialization with empty data."""
        with pytest.raises(ValueError, match="Data cannot be None"):
            StatisticalValidator(None)

    def test_get_numeric_columns(self, validator):
        """Test numeric column detection."""
        numeric_cols = validator._get_numeric_columns()

        assert isinstance(numeric_cols, list)
        assert 'numeric1' in numeric_cols
        assert 'numeric2' in numeric_cols
        assert 'numeric3' in numeric_cols
        assert 'categorical' not in numeric_cols

    def test_get_numeric_columns_empty(self):
        """Test numeric column detection with no numeric columns."""
        data = pd.DataFrame({'text': ['a', 'b', 'c']})
        validator = StatisticalValidator(data)

        numeric_cols = validator._get_numeric_columns()
        assert numeric_cols == []

    def test_test_normality_all_columns(self, validator):
        """Test normality testing on all numeric columns."""
        result = validator.test_normality()

        assert isinstance(result, dict)
        assert 'summary' in result
        assert 'details' in result
        assert 'assumptions' in result

        # Check summary
        summary = result['summary']
        assert 'columns_tested' in summary
        assert 'columns_normal' in summary
        assert 'columns_non_normal' in summary
        assert 'normality_percentage' in summary
        assert summary['columns_tested'] == 3  # All numeric columns

        # Check details
        details = result['details']
        for col in ['numeric1', 'numeric2', 'numeric3']:
            assert col in details
            assert 'shapiro_wilk' in details[col]
            assert 'anderson_darling' in details[col]
            assert 'ks_test' in details[col]
            assert 'is_normal' in details[col]

    def test_test_normality_specific_columns(self, validator):
        """Test normality testing on specific columns."""
        columns = ['numeric1', 'numeric2']
        result = validator.test_normality(columns=columns)

        assert result['summary']['columns_tested'] == 2
        assert set(result['details'].keys()) == set(columns)

    def test_test_normality_invalid_columns(self, validator):
        """Test normality testing with invalid columns."""
        result = validator.test_normality(columns=['nonexistent'])

        assert result['summary']['columns_tested'] == 0
        assert result['summary']['errors'] > 0

    def test_test_normality_custom_alpha(self, validator):
        """Test normality testing with custom alpha."""
        result = validator.test_normality(alpha=0.01)

        # Verify the alpha is used correctly
        for col_details in result['details'].values():
            for test_result in col_details.values():
                if isinstance(test_result, dict) and 'p_value' in test_result:
                    assert test_result['significance_level'] == 0.01

    def test_perform_shapiro_wilk(self, validator):
        """Test Shapiro-Wilk test execution."""
        data = np.random.normal(0, 1, 100)
        result = validator._perform_shapiro_wilk(data, alpha=0.05)

        assert isinstance(result, dict)
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'is_normal' in result
        assert 'test_name' in result
        assert result['test_name'] == 'Shapiro-Wilk'
        assert 0 <= result['statistic'] <= 1
        assert 0 <= result['p_value'] <= 1

    def test_perform_anderson_darling(self, validator):
        """Test Anderson-Darling test execution."""
        data = np.random.normal(0, 1, 100)
        result = validator._perform_anderson_darling(data, alpha=0.05)

        assert isinstance(result, dict)
        assert 'statistic' in result
        assert 'critical_values' in result
        assert 'significance_levels' in result
        assert 'is_normal' in result
        assert 'test_name' in result
        assert result['test_name'] == 'Anderson-Darling'

    def test_perform_ks_test(self, validator):
        """Test Kolmogorov-Smirnov test execution."""
        data = np.random.normal(0, 1, 100)
        result = validator._perform_ks_test(data, alpha=0.05)

        assert isinstance(result, dict)
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'is_normal' in result
        assert 'test_name' in result
        assert result['test_name'] == 'Kolmogorov-Smirnov'

    def test_test_homoscedasticity_with_X_y(self, validator):
        """Test homoscedasticity testing with X and y provided."""
        X = np.random.normal(0, 1, (100, 3))
        y = np.random.normal(0, 1, 100)

        result = validator.test_homoscedasticity(X=X, y=y)

        assert isinstance(result, dict)
        assert 'summary' in result
        assert 'details' in result
        assert 'assumptions' in result

        summary = result['summary']
        assert 'homoscedastic' in summary
        assert 'test_statistic' in summary
        assert 'p_value' in summary
        assert 'test_used' in summary

    def test_test_homoscedasticity_from_data(self, validator):
        """Test homoscedasticity testing using data columns."""
        # Use first numeric columns as X, last as y
        result = validator.test_homoscedasticity(X=None, y=None)

        assert isinstance(result, dict)
        assert 'summary' in result
        # Should work with available numeric columns
        assert result['summary']['test_used'] == 'Breusch-Pagan'

    def test_test_homoscedasticity_insufficient_data(self, validator):
        """Test homoscedasticity testing with insufficient data."""
        X = np.random.normal(0, 1, (5, 1))  # Too few samples
        y = np.random.normal(0, 1, 5)

        result = validator.test_homoscedasticity(X=X, y=y)

        assert result['summary']['homoscedastic'] is False
        assert 'error' in result['summary']

    @patch('src.statistical.statistical_validator.stats')
    def test_test_homoscedasticity_statsmodels_error(self, mock_stats, validator):
        """Test homoscedasticity testing when statsmodels fails."""
        # Mock statsmodels to raise an exception
        mock_stats.api.stats.diagnostic.het_breuschpagan.side_effect = Exception("Import error")

        result = validator.test_homoscedasticity()

        assert 'error' in result['summary']

    def test_test_independence_with_series(self, validator):
        """Test independence testing with a series."""
        series = pd.Series(np.random.normal(0, 1, 100))
        result = validator.test_independence(data=series)

        assert isinstance(result, dict)
        assert 'summary' in result
        assert 'details' in result

        summary = result['summary']
        assert 'independent' in summary
        assert 'durbin_watson_stat' in summary
        assert 'ljung_box_pvalue' in summary

    def test_test_independence_from_data(self, validator):
        """Test independence testing using data columns."""
        result = validator.test_independence(data=None)

        assert isinstance(result, dict)
        assert 'summary' in result
        assert result['summary']['independent'] is not None

    def test_test_independence_insufficient_data(self):
        """Test independence testing with insufficient data."""
        data = pd.DataFrame({'x': [1, 2]})
        validator = StatisticalValidator(data)

        result = validator.test_independence()

        assert result['summary']['independent'] is False
        assert 'error' in result['summary']

    def test_calculate_confidence_intervals(self, validator):
        """Test bootstrap confidence interval calculation."""
        result = validator.calculate_confidence_intervals()

        assert isinstance(result, dict)
        assert 'summary' in result
        assert 'details' in result
        assert 'assumptions' in result

        summary = result['summary']
        assert 'confidence_level' in summary
        assert 'n_bootstrap' in summary
        assert 'columns_processed' in summary
        assert summary['confidence_level'] == 0.95
        assert summary['n_bootstrap'] == 1000

        details = result['details']
        for col in ['numeric1', 'numeric2', 'numeric3']:
            if col in details:  # Only check if column was processed
                assert 'mean' in details[col]
                assert 'std' in details[col]
                assert 'percentiles' in details[col]

    def test_calculate_confidence_intervals_custom_parameters(self, validator):
        """Test bootstrap confidence intervals with custom parameters."""
        result = validator.calculate_confidence_intervals(
            confidence_level=0.90,
            n_bootstrap=100
        )

        assert result['summary']['confidence_level'] == 0.90
        assert result['summary']['n_bootstrap'] == 100

    def test_calculate_confidence_intervals_insufficient_data(self):
        """Test confidence intervals with insufficient data."""
        data = pd.DataFrame({'x': [1, 2]})
        validator = StatisticalValidator(data)

        result = validator.calculate_confidence_intervals()

        assert result['summary']['columns_processed'] == 0
        assert 'errors' in result['summary']

    def test_bootstrap_confidence_interval(self, validator):
        """Test individual bootstrap confidence interval calculation."""
        data = np.random.normal(0, 1, 100)
        result = validator._bootstrap_confidence_interval(data, n_bootstrap=100, lower_percentile=2.5, upper_percentile=97.5)

        assert isinstance(result, dict)
        assert 'mean' in result
        assert 'std' in result
        assert 'percentiles' in result

        mean_ci = result['mean']
        assert isinstance(mean_ci, dict)
        assert 'lower' in mean_ci
        assert 'upper' in mean_ci
        assert mean_ci['lower'] <= mean_ci['upper']

    def test_bootstrap_confidence_interval_very_small_sample(self, validator):
        """Test bootstrap with very small sample."""
        data = np.array([1])
        result = validator._bootstrap_confidence_interval(data, n_bootstrap=100, lower_percentile=2.5, upper_percentile=97.5)

        assert result['mean']['lower'] == result['mean']['upper']  # Should be equal for single value

    def test_generate_summary_statistics(self, validator):
        """Test summary statistics generation."""
        result = validator._generate_summary_statistics()

        assert isinstance(result, dict)
        assert 'numeric_columns' in result
        assert 'numeric_summary' in result
        assert 'data_shape' in result

        numeric_summary = result['numeric_summary']
        for col in ['numeric1', 'numeric2', 'numeric3']:
            assert col in numeric_summary
            col_stats = numeric_summary[col]
            assert 'count' in col_stats
            assert 'mean' in col_stats
            assert 'std' in col_stats
            assert 'min' in col_stats
            assert 'max' in col_stats
            assert 'q25' in col_stats
            assert 'median' in col_stats
            assert 'q75' in col_stats
            assert 'missing_values' in col_stats
            assert 'missing_percentage' in col_stats

    def test_calculate_descriptive_stats(self, validator):
        """Test descriptive statistics calculation."""
        data = np.array([1, 2, 3, 4, 5])
        result = validator._calculate_descriptive_stats(data)

        assert result['count'] == 5
        assert result['mean'] == 3.0
        assert result['median'] == 3.0
        assert result['min'] == 1.0
        assert result['max'] == 5.0
        assert result['q25'] == 2.0
        assert result['q75'] == 4.0

    def test_validate_assumptions(self, validator):
        """Test assumption validation."""
        test_results = {
            'normality': {
                'summary': {
                    'columns_normal': 2,
                    'columns_tested': 3,
                    'normality_percentage': 66.67
                }
            },
            'homoscedasticity': {
                'summary': {
                    'homoscedastic': True
                }
            },
            'independence': {
                'summary': {
                    'independent': True
                }
            }
        }

        result = validator._validate_assumptions(test_results)

        assert isinstance(result, dict)
        assert 'normality' in result
        assert 'homoscedasticity' in result
        assert 'independence' in result
        assert 'overall' in result

        assert result['normality']['assumption_met'] is False  # Not all normal
        assert result['homoscedasticity']['assumption_met'] is True
        assert result['independence']['assumption_met'] is True

    def test_validate_assumptions_missing_tests(self, validator):
        """Test assumption validation with missing test results."""
        test_results = {}

        result = validator._validate_assumptions(test_results)

        assert result['normality']['assumption_met'] is False
        assert result['normality']['reason'] == 'Test not performed'

    def test_run_comprehensive_analysis(self, validator):
        """Test comprehensive statistical analysis."""
        result = validator.run_comprehensive_analysis()

        assert isinstance(result, dict)
        assert 'summary_statistics' in result
        assert 'normality_tests' in result
        assert 'homoscedasticity_tests' in result
        assert 'independence_tests' in result
        assert 'confidence_intervals' in result
        assert 'assumption_validation' in result
        assert 'timestamp' in result
        assert 'data_info' in result

    def test_run_comprehensive_analysis_subset(self, validator):
        """Test comprehensive analysis with column subset."""
        columns = ['numeric1', 'numeric2']
        result = validator.run_comprehensive_analysis(columns=columns)

        # Check that only specified columns are included
        normality_details = result['normality_tests']['details']
        assert set(normality_details.keys()) == set(columns)

    def test_run_comprehensive_analysis_custom_alpha(self, validator):
        """Test comprehensive analysis with custom alpha."""
        result = validator.run_comprehensive_analysis(alpha=0.01)

        # Verify alpha is propagated
        assert result['normality_tests']['summary']['significance_level'] == 0.01

    def test_get_summary_statistics_string(self, validator):
        """Test string representation of summary statistics."""
        result = validator.get_summary_statistics()

        assert isinstance(result, str)
        assert 'Summary Statistics' in result
        assert 'Data Shape:' in result
        assert 'Numeric Columns:' in result

    def test_get_assumption_status_string(self, validator):
        """Test string representation of assumption status."""
        result = validator.get_assumption_status()

        assert isinstance(result, str)
        assert 'Statistical Assumption Status' in result
        assert 'Normality:' in result
        assert 'Homoscedasticity:' in result
        assert 'Independence:' in result

    def test_export_results(self, validator):
        """Test exporting results to JSON."""
        analysis_results = validator.run_comprehensive_analysis()

        # Test export to buffer
        import io
        output = io.StringIO()
        validator.export_results(analysis_results, output, format='json')

        output.seek(0)
        content = output.read()
        assert len(content) > 0
        assert 'summary_statistics' in content

    def test_export_results_unsupported_format(self, validator):
        """Test exporting with unsupported format."""
        analysis_results = {'test': 'data'}

        import io
        output = io.StringIO()

        with pytest.raises(ValueError, match="Unsupported format"):
            validator.export_results(analysis_results, output, format='xml')