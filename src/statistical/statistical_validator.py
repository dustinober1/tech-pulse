"""Statistical validator for testing statistical assumptions and reporting uncertainty."""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
import warnings
from datetime import datetime
import json

from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


class StatisticalValidator:
    """
    Statistical validator for testing assumptions and reporting uncertainty.

    Performs normality tests, homoscedasticity tests, independence tests,
    and calculates confidence intervals using bootstrap methods.
    """

    def __init__(self, data: Union[pd.DataFrame, np.ndarray], alpha: float = 0.05):
        """
        Initialize the statistical validator.

        Args:
            data: Data to validate
            alpha: Significance level for statistical tests
        """
        if data is None:
            raise ValueError("Data cannot be None")

        if isinstance(data, np.ndarray) and data.ndim == 1:
            self.data = pd.DataFrame({'value': data})
        else:
            self.data = pd.DataFrame(data) if isinstance(data, np.ndarray) else data.copy()

        self.alpha = alpha
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': self.data.shape,
            'normality_tests': {},
            'homoscedasticity_tests': {},
            'independence_tests': {},
            'confidence_intervals': {},
            'assumption_violations': [],
            'statistical_summary': {}
        }

    def _get_numeric_columns(self) -> List[str]:
        """Get list of numeric columns in the data."""
        return self.data.select_dtypes(include=[np.number]).columns.tolist()

    def test_normality(self, columns: Optional[List[str]] = None, alpha: Optional[float] = None) -> Dict[str, Any]:
        """
        Test normality assumption using multiple tests.

        Args:
            columns: Columns to test (default: all numeric columns)
            alpha: Significance level for tests (default: self.alpha)

        Returns:
            Dictionary containing normality test results
        """
        if alpha is None:
            alpha = self.alpha

        if columns is None:
            columns = self._get_numeric_columns()

        results = {
            'summary': {
                'columns_tested': 0,
                'columns_normal': 0,
                'columns_non_normal': 0,
                'normality_percentage': 0.0,
                'errors': 0,
                'significance_level': alpha
            },
            'details': {},
            'assumptions': {}
        }

        for col in columns:
            if col not in self.data.columns:
                results['summary']['errors'] += 1
                continue

            data_col = self.data[col].dropna()
            if len(data_col) < 3:
                results['details'][col] = {
                    'error': f'Insufficient data points (need at least 3, have {len(data_col)})'
                }
                results['summary']['errors'] += 1
                continue

            results['summary']['columns_tested'] += 1

            # Perform all tests
            shapiro_result = self._perform_shapiro_wilk(data_col, alpha)
            anderson_result = self._perform_anderson_darling(data_col, alpha)
            ks_result = self._perform_ks_test(data_col, alpha)

            # Determine if column is normal (majority of tests say it's normal)
            test_results = [shapiro_result, anderson_result, ks_result]
            normal_count = sum(1 for test in test_results if test.get('is_normal', False))
            is_normal = normal_count >= len(test_results) / 2

            if is_normal:
                results['summary']['columns_normal'] += 1
            else:
                results['summary']['columns_non_normal'] += 1

            results['details'][col] = {
                'shapiro_wilk': shapiro_result,
                'anderson_darling': anderson_result,
                'ks_test': ks_result,
                'is_normal': is_normal
            }

        # Calculate percentages
        if results['summary']['columns_tested'] > 0:
            results['summary']['normality_percentage'] = (
                results['summary']['columns_normal'] / results['summary']['columns_tested'] * 100
            )

        # Set assumptions
        results['assumptions'] = {
            'data_points_sufficient': results['summary']['columns_tested'] > 0,
            'tests_performed': results['summary']['columns_tested'] > 0
        }

        return results

    def _perform_shapiro_wilk(self, data: np.ndarray, alpha: float) -> Dict[str, Any]:
        """Perform Shapiro-Wilk test for normality."""
        try:
            statistic, p_value = stats.shapiro(data)
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'is_normal': p_value > alpha,
                'test_name': 'Shapiro-Wilk',
                'significance_level': alpha
            }
        except Exception as e:
            return {
                'error': str(e),
                'test_name': 'Shapiro-Wilk',
                'is_normal': False
            }

    def _perform_anderson_darling(self, data: np.ndarray, alpha: float) -> Dict[str, Any]:
        """Perform Anderson-Darling test for normality."""
        try:
            result = stats.anderson(data, dist='norm')
            statistic = result.statistic
            critical_values = result.critical_values

            # AndersonResult has significance_level attribute (not significance_levels)
            if hasattr(result, 'significance_level'):
                significance_levels = result.significance_level
            else:
                # Default significance levels for Anderson-Darling
                significance_levels = np.array([15.0, 10.0, 5.0, 2.5, 1.0])

            # Find the critical value for 5% significance (or closest)
            idx = np.argmin(np.abs(significance_levels - 5.0))
            critical_value = critical_values[idx]
            is_normal = statistic < critical_value

            return {
                'statistic': float(statistic),
                'critical_values': critical_values.tolist(),
                'significance_levels': significance_levels.tolist(),
                'is_normal': is_normal,
                'test_name': 'Anderson-Darling'
            }
        except Exception as e:
            return {
                'error': str(e),
                'test_name': 'Anderson-Darling',
                'is_normal': False
            }

    def _perform_ks_test(self, data: np.ndarray, alpha: float) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test for normality."""
        try:
            # Standardize the data
            standardized = (data - np.mean(data)) / np.std(data)
            statistic, p_value = stats.kstest(standardized, 'norm')
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'is_normal': p_value > alpha,
                'test_name': 'Kolmogorov-Smirnov',
                'significance_level': alpha
            }
        except Exception as e:
            return {
                'error': str(e),
                'test_name': 'Kolmogorov-Smirnov',
                'is_normal': False
            }

    def test_homoscedasticity(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                             alpha: Optional[float] = None) -> Dict[str, Any]:
        """
        Test homoscedasticity assumption (equal variance).

        Args:
            X: Independent variables
            y: Dependent variable
            alpha: Significance level (default: self.alpha)

        Returns:
            Dictionary containing homoscedasticity test results
        """
        if alpha is None:
            alpha = self.alpha

        results = {
            'summary': {
                'homoscedastic': None,
                'test_statistic': None,
                'p_value': None,
                'test_used': 'Breusch-Pagan',
                'significance_level': alpha
            },
            'details': {},
            'assumptions': {}
        }

        try:
            # If X and y are not provided, use the first two numeric columns
            if X is None or y is None:
                numeric_cols = self._get_numeric_columns()
                if len(numeric_cols) < 2:
                    results['summary']['homoscedastic'] = False
                    results['summary']['error'] = 'Need at least 2 numeric columns'
                    return results

                X_data = self.data[numeric_cols[:-1]].values
                y_data = self.data[numeric_cols[-1]].values
            else:
                X_data = X
                y_data = y

            # Ensure X is 2D
            if X_data.ndim == 1:
                X_data = X_data.reshape(-1, 1)

            # Check for sufficient data
            if len(y_data) < 10:
                results['summary']['homoscedastic'] = False
                results['summary']['error'] = f'Insufficient data: need at least 10 samples, have {len(y_data)}'
                return results

            # Perform Breusch-Pagan test
            bp_stat, bp_p, f_stat, f_p = het_breuschpagan(y_data, X_data)

            results['summary']['test_statistic'] = float(bp_stat)
            results['summary']['p_value'] = float(bp_p)
            results['summary']['homoscedastic'] = bp_p > alpha

            results['details'] = {
                'breusch_pagan': {
                    'statistic': float(bp_stat),
                    'p_value': float(bp_p),
                    'f_statistic': float(f_stat),
                    'f_p_value': float(f_p)
                }
            }

            results['assumptions'] = {
                'sample_size_sufficient': len(y_data) >= 10,
                'independent_variables_present': X_data.shape[1] > 0
            }

        except Exception as e:
            results['summary']['homoscedastic'] = False
            results['summary']['error'] = str(e)

        return results

    def test_independence(self, data: Optional[pd.Series] = None, alpha: Optional[float] = None) -> Dict[str, Any]:
        """
        Test independence assumption.

        Args:
            data: Time series data to test
            alpha: Significance level (default: self.alpha)

        Returns:
            Dictionary containing independence test results
        """
        if alpha is None:
            alpha = self.alpha

        results = {
            'summary': {
                'independent': None,
                'durbin_watson_stat': None,
                'ljung_box_pvalue': None
            },
            'details': {},
            'assumptions': {}
        }

        try:
            # If data is not provided, use the first numeric column
            if data is None:
                numeric_cols = self._get_numeric_columns()
                if not numeric_cols:
                    results['summary']['independent'] = False
                    results['summary']['error'] = 'No numeric columns available'
                    return results

                series_data = self.data[numeric_cols[0]].dropna()
            else:
                series_data = data.dropna() if hasattr(data, 'dropna') else pd.Series(data).dropna()

            # Check for sufficient data
            if len(series_data) < 10:
                results['summary']['independent'] = False
                results['summary']['error'] = f'Insufficient data: need at least 10 samples, have {len(series_data)}'
                return results

            # Durbin-Watson test
            diff = np.diff(series_data)
            durbin_watson = np.sum(diff**2) / np.sum(series_data**2)
            dw_independent = 1.5 <= durbin_watson <= 2.5

            # Ljung-Box test for autocorrelation
            try:
                ljung_box_result = acorr_ljungbox(series_data, lags=min(10, len(series_data)//5), return_df=True)
                ljung_box_p = ljung_box_result['lb_pvalue'].iloc[-1]
                lb_independent = ljung_box_p > alpha
            except:
                ljung_box_p = None
                lb_independent = True  # Assume independent if test fails

            results['summary']['durbin_watson_stat'] = float(durbin_watson)
            results['summary']['ljung_box_pvalue'] = float(ljung_box_p) if ljung_box_p is not None else None
            results['summary']['independent'] = dw_independent and lb_independent

            results['details'] = {
                'durbin_watson': {
                    'statistic': float(durbin_watson),
                    'independent': dw_independent,
                    'interpretation': 'independent' if dw_independent else 'autocorrelated'
                },
                'ljung_box': {
                    'p_value': float(ljung_box_p) if ljung_box_p is not None else None,
                    'independent': lb_independent,
                    'interpretation': 'independent' if lb_independent else 'autocorrelated'
                }
            }

            results['assumptions'] = {
                'sample_size_sufficient': len(series_data) >= 10,
                'time_series_data': True
            }

        except Exception as e:
            results['summary']['independent'] = False
            results['summary']['error'] = str(e)

        return results

    def calculate_confidence_intervals(self, confidence_level: float = 0.95,
                                     n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Calculate bootstrap confidence intervals.

        Args:
            confidence_level: Confidence level for intervals (default: 0.95)
            n_bootstrap: Number of bootstrap samples (default: 1000)

        Returns:
            Dictionary containing confidence intervals
        """
        results = {
            'summary': {
                'confidence_level': confidence_level,
                'n_bootstrap': n_bootstrap,
                'columns_processed': 0,
                'errors': 0
            },
            'details': {},
            'assumptions': {}
        }

        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        for col in self._get_numeric_columns():
            data_col = self.data[col].dropna()

            if len(data_col) < 2:
                results['details'][col] = {
                    'error': f'Insufficient data for bootstrap (need at least 2, have {len(data_col)})'
                }
                results['summary']['errors'] += 1
                continue

            # Only process if we have sufficient data for meaningful bootstrap
            if len(data_col) < 10:
                results['details'][col] = {
                    'error': f'Insufficient data for meaningful bootstrap (need at least 10, have {len(data_col)})'
                }
                results['summary']['errors'] += 1
                continue

            results['summary']['columns_processed'] += 1

            # Calculate bootstrap confidence intervals
            ci_result = self._bootstrap_confidence_interval(
                data_col, n_bootstrap, lower_percentile, upper_percentile
            )
            results['details'][col] = ci_result

        results['assumptions'] = {
            'bootstrap_samples_sufficient': n_bootstrap >= 100,
            'data_points_available': results['summary']['columns_processed'] > 0
        }

        return results

    def _bootstrap_confidence_interval(self, data: np.ndarray, n_bootstrap: int,
                                     lower_percentile: float, upper_percentile: float) -> Dict[str, Any]:
        """Calculate bootstrap confidence interval for a single variable."""
        try:
            n = len(data)
            bootstrap_means = []
            bootstrap_stds = []

            for _ in range(n_bootstrap):
                # Bootstrap sample
                sample = np.random.choice(data, size=n, replace=True)
                bootstrap_means.append(np.mean(sample))
                bootstrap_stds.append(np.std(sample, ddof=1))

            bootstrap_means = np.array(bootstrap_means)
            bootstrap_stds = np.array(bootstrap_stds)

            # Calculate percentiles
            mean_ci = np.percentile(bootstrap_means, [lower_percentile, upper_percentile])
            std_ci = np.percentile(bootstrap_stds, [lower_percentile, upper_percentile])

            return {
                'mean': {
                    'lower': float(mean_ci[0]),
                    'upper': float(mean_ci[1]),
                    'std_error': float(np.std(bootstrap_means))
                },
                'std': {
                    'lower': float(std_ci[0]),
                    'upper': float(std_ci[1])
                },
                'percentiles': {
                    'lower': lower_percentile,
                    'upper': upper_percentile
                }
            }
        except Exception as e:
            return {
                'error': str(e),
                'mean': {'lower': None, 'upper': None, 'std_error': None},
                'std': {'lower': None, 'upper': None},
                'percentiles': {'lower': lower_percentile, 'upper': upper_percentile}
            }

    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for the data."""
        numeric_cols = self._get_numeric_columns()

        summary = {
            'numeric_columns': numeric_cols,
            'numeric_summary': {},
            'data_shape': self.data.shape
        }

        for col in numeric_cols:
            data_col = self.data[col].dropna()
            if len(data_col) > 0:
                summary['numeric_summary'][col] = self._calculate_descriptive_stats(data_col)

        return summary

    def _calculate_descriptive_stats(self, data: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Calculate descriptive statistics for a single column."""
        if isinstance(data, np.ndarray):
            return {
                'count': len(data),
                'mean': float(np.mean(data)),
                'std': float(np.std(data, ddof=1)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'q25': float(np.percentile(data, 25)),
                'median': float(np.median(data)),
                'q75': float(np.percentile(data, 75)),
                'missing_values': 0,
                'missing_percentage': 0.0
            }
        else:
            return {
                'count': len(data),
                'mean': float(data.mean()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'q25': float(data.quantile(0.25)),
                'median': float(data.median()),
                'q75': float(data.quantile(0.75)),
                'missing_values': int(self.data[data.name].isna().sum()) if data.name in self.data.columns else 0,
                'missing_percentage': float(self.data[data.name].isna().mean() * 100) if data.name in self.data.columns else 0
            }

    def _validate_assumptions(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate statistical assumptions based on test results."""
        validation = {
            'normality': {
                'assumption_met': False,
                'reason': 'Test not performed'
            },
            'homoscedasticity': {
                'assumption_met': False,
                'reason': 'Test not performed'
            },
            'independence': {
                'assumption_met': False,
                'reason': 'Test not performed'
            },
            'overall': {
                'assumptions_met': 0,
                'total_assumptions': 3
            }
        }

        # Check normality
        if 'normality' in test_results:
            normality_result = test_results['normality']
            if 'summary' in normality_result:
                normality_pct = normality_result['summary'].get('normality_percentage', 0)
                validation['normality']['assumption_met'] = normality_pct >= 80
                validation['normality']['reason'] = (
                    f'{normality_pct:.1f}% of columns are normally distributed'
                )
                if validation['normality']['assumption_met']:
                    validation['overall']['assumptions_met'] += 1

        # Check homoscedasticity
        if 'homoscedasticity' in test_results:
            homoscedasticity_result = test_results['homoscedasticity']
            if 'summary' in homoscedasticity_result:
                validation['homoscedasticity']['assumption_met'] = homoscedasticity_result['summary'].get('homoscedastic', False)
                validation['homoscedasticity']['reason'] = (
                    'Equal variance assumption satisfied' if validation['homoscedasticity']['assumption_met']
                    else 'Unequal variance detected'
                )
                if validation['homoscedasticity']['assumption_met']:
                    validation['overall']['assumptions_met'] += 1

        # Check independence
        if 'independence' in test_results:
            independence_result = test_results['independence']
            if 'summary' in independence_result:
                validation['independence']['assumption_met'] = independence_result['summary'].get('independent', False)
                validation['independence']['reason'] = (
                    'No significant autocorrelation' if validation['independence']['assumption_met']
                    else 'Autocorrelation detected'
                )
                if validation['independence']['assumption_met']:
                    validation['overall']['assumptions_met'] += 1

        return validation

    def run_comprehensive_analysis(self, columns: Optional[List[str]] = None,
                                 alpha: Optional[float] = None) -> Dict[str, Any]:
        """
        Run comprehensive statistical analysis.

        Args:
            columns: Columns to analyze (default: all numeric columns)
            alpha: Significance level for tests (default: self.alpha)

        Returns:
            Dictionary containing all analysis results
        """
        if alpha is None:
            alpha = self.alpha

        # Run all tests
        normality_results = self.test_normality(columns=columns, alpha=alpha)
        homoscedasticity_results = self.test_homoscedasticity(alpha=alpha)
        independence_results = self.test_independence(alpha=alpha)
        ci_results = self.calculate_confidence_intervals()

        # Validate assumptions
        test_results = {
            'normality': normality_results,
            'homoscedasticity': homoscedasticity_results,
            'independence': independence_results
        }
        assumption_validation = self._validate_assumptions(test_results)

        # Generate summary statistics
        summary_stats = self._generate_summary_statistics()

        # Compile comprehensive results
        comprehensive_results = {
            'summary_statistics': summary_stats,
            'normality_tests': normality_results,
            'homoscedasticity_tests': homoscedasticity_results,
            'independence_tests': independence_results,
            'confidence_intervals': ci_results,
            'assumption_validation': assumption_validation,
            'timestamp': datetime.now().isoformat(),
            'data_info': {
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'numeric_columns': self._get_numeric_columns(),
                'alpha': alpha
            }
        }

        # Update validation results
        self.validation_results.update(comprehensive_results)

        return comprehensive_results

    def get_summary_statistics(self) -> str:
        """Get formatted summary statistics as a string."""
        summary = self._generate_summary_statistics()

        output = []
        output.append("Summary Statistics")
        output.append("=" * 50)
        output.append(f"Data Shape: {summary['data_shape']}")
        output.append(f"Numeric Columns: {len(summary['numeric_columns'])}")
        output.append("")

        for col, stats in summary['numeric_summary'].items():
            output.append(f"Column: {col}")
            output.append(f"  Count: {stats['count']}")
            output.append(f"  Mean: {stats['mean']:.4f}")
            output.append(f"  Std Dev: {stats['std']:.4f}")
            output.append(f"  Min: {stats['min']:.4f}")
            output.append(f"  Max: {stats['max']:.4f}")
            output.append(f"  Median: {stats['median']:.4f}")
            output.append(f"  Missing: {stats['missing_values']} ({stats['missing_percentage']:.1f}%)")
            output.append("")

        return "\n".join(output)

    def get_assumption_status(self) -> str:
        """Get formatted assumption status as a string."""
        # Run quick analysis
        normality = self.test_normality()
        homoscedasticity = self.test_homoscedasticity()
        independence = self.test_independence()

        test_results = {
            'normality': normality,
            'homoscedasticity': homoscedasticity,
            'independence': independence
        }
        validation = self._validate_assumptions(test_results)

        output = []
        output.append("Statistical Assumption Status")
        output.append("=" * 50)
        output.append(f"Normality: {'✓' if validation['normality']['assumption_met'] else '✗'}")
        output.append(f"  {validation['normality']['reason']}")
        output.append("")
        output.append(f"Homoscedasticity: {'✓' if validation['homoscedasticity']['assumption_met'] else '✗'}")
        output.append(f"  {validation['homoscedasticity']['reason']}")
        output.append("")
        output.append(f"Independence: {'✓' if validation['independence']['assumption_met'] else '✗'}")
        output.append(f"  {validation['independence']['reason']}")
        output.append("")
        output.append(f"Overall: {validation['overall']['assumptions_met']}/{validation['overall']['total_assumptions']} assumptions met")

        return "\n".join(output)

    def export_results(self, results: Dict[str, Any], output, format: str = 'json') -> None:
        """
        Export analysis results.

        Args:
            results: Results dictionary to export
            output: Output file-like object
            format: Export format ('json' or 'csv')
        """
        if format == 'json':
            # Convert numpy bools to Python bools for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj

            json.dump(convert_numpy(results), output, indent=2)
        elif format == 'csv':
            # Export summary statistics as CSV
            if 'summary_statistics' in results and 'numeric_summary' in results['summary_statistics']:
                summary_df = pd.DataFrame(results['summary_statistics']['numeric_summary']).T
                summary_df.to_csv(output)
            else:
                output.write("No summary statistics available")
        else:
            raise ValueError(f"Unsupported format: {format}")