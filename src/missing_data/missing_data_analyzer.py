"""Missing Data Analyzer for analyzing and handling missing data patterns.

This module provides comprehensive missing data analysis capabilities including:
- Missingness pattern analysis (MCAR, MAR, MNAR)
- Statistical tests for missingness mechanisms
- Visualization of missing data patterns
- Multiple imputation strategies
- Missing data documentation and reporting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import base64
from io import BytesIO
import json
from datetime import datetime
import missingno as msno

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MissingDataAnalyzer:
    """Comprehensive missing data analysis and imputation toolkit.

    This class provides tools for analyzing missing data patterns,
    determining missingness mechanisms, visualizing missingness,
    and applying appropriate imputation strategies.

    Attributes:
        data (pd.DataFrame): The input dataset
        original_shape (Tuple[int, int]): Original shape of the data
        missing_summary (Dict): Summary of missing data analysis
        missingness_tests (Dict): Results of statistical tests
        imputation_history (List): History of applied imputations
        missingness_mechanism (Dict): Determined missingness mechanisms
    """

    def __init__(self, data: Union[pd.DataFrame, np.ndarray]):
        """Initialize the MissingDataAnalyzer.

        Args:
            data: Input dataset containing missing values
        """
        if isinstance(data, np.ndarray):
            self.data = pd.DataFrame(data)
        else:
            self.data = data.copy()

        self.original_shape = self.data.shape
        self.missing_summary = {}
        self.missingness_tests = {}
        self.imputation_history = []
        self.missingness_mechanism = {}

    def analyze_missing_patterns(self) -> Dict[str, Any]:
        """Analyze missing data patterns across the dataset.

        Returns:
            Dictionary containing comprehensive missing pattern analysis
        """
        summary = {
            'overall': {},
            'by_column': {},
            'by_row': {},
            'patterns': {},
            'correlations': {}
        }

        # Overall missing statistics
        total_cells = self.data.size
        total_missing = self.data.isnull().sum().sum()
        summary['overall'] = {
            'total_cells': total_cells,
            'total_missing': total_missing,
            'missing_percentage': (total_missing / total_cells) * 100,
            'complete_cases': self.data.dropna().shape[0],
            'complete_cases_percentage': (self.data.dropna().shape[0] / len(self.data)) * 100
        }

        # Missing by column
        missing_counts = self.data.isnull().sum()
        missing_percentages = (missing_counts / len(self.data)) * 100

        for col in self.data.columns:
            summary['by_column'][col] = {
                'missing_count': int(missing_counts[col]),
                'missing_percentage': float(missing_percentages[col]),
                'data_type': str(self.data[col].dtype),
                'unique_values': int(self.data[col].nunique()),
                'has_missing': missing_counts[col] > 0
            }

        # Missing by row
        row_missing = self.data.isnull().sum(axis=1)
        summary['by_row'] = {
            'mean_missing_per_row': float(row_missing.mean()),
            'max_missing_in_row': int(row_missing.max()),
            'rows_with_missing': int((row_missing > 0).sum()),
            'rows_fully_missing': int((row_missing == len(self.data.columns)).sum()),
            'row_missing_distribution': row_missing.value_counts().to_dict()
        }

        # Missing patterns
        # Create binary matrix of missingness
        missing_matrix = self.data.isnull().astype(int)
        pattern_counts = missing_matrix.value_counts().to_dict()

        # Convert tuple keys to strings for JSON serialization
        pattern_summary = {}
        for pattern, count in pattern_counts.items():
            if isinstance(pattern, tuple):
                pattern_str = ','.join(['1' if x else '0' for x in pattern])
            else:
                pattern_str = str(int(pattern))
            pattern_summary[pattern_str] = count

        summary['patterns'] = {
            'unique_patterns': len(pattern_summary),
            'pattern_counts': pattern_summary,
            'most_common_pattern': max(pattern_summary.items(), key=lambda x: x[1])[0] if pattern_summary else None
        }

        # Missing correlations
        if missing_matrix.shape[1] > 1:
            missing_corr = missing_matrix.corr()
            # Find strong correlations (> 0.5)
            strong_corrs = []
            for i in range(len(missing_corr.columns)):
                for j in range(i+1, len(missing_corr.columns)):
                    corr_val = missing_corr.iloc[i, j]
                    if abs(corr_val) > 0.5 and not np.isnan(corr_val):
                        strong_corrs.append({
                            'column1': missing_corr.columns[i],
                            'column2': missing_corr.columns[j],
                            'correlation': float(corr_val)
                        })

            summary['correlations'] = {
                'correlation_matrix': missing_corr.fillna(0).to_dict(),
                'strong_correlations': strong_corrs,
                'max_correlation': float(missing_corr.abs().max().max()) if not missing_corr.empty else 0
            }

        self.missing_summary = summary
        return summary

    def test_missingness_mechanism(self, significance_level: float = 0.05) -> Dict[str, Any]:
        """Test for missingness mechanisms (MCAR, MAR, MNAR).

        Args:
            significance_level: Alpha level for statistical tests

        Returns:
            Dictionary containing test results for missingness mechanisms
        """
        if self.data.empty:
            return {'error': 'No data available for testing'}

        results = {
            'little_mcar_test': None,
            't_tests': {},
            'correlation_tests': {},
            'pattern_analysis': {},
            'conclusion': {}
        }

        # Little's MCAR test (simplified version using pattern analysis)
        # This is an approximation since Little's exact test is complex
        try:
            missing_matrix = self.data.isnull()
            complete_data = self.data.dropna()

            if len(complete_data) > 0:
                # Test if missingness patterns are random across variables
                chi_square_stats = []
                degrees_of_freedom = []

                for col in self.data.columns:
                    if self.data[col].isnull().any():
                        # Create contingency table for missing vs observed
                        other_cols = [c for c in self.data.columns if c != col]
                        if other_cols:
                            # Test missingness against other variables
                            for other_col in other_cols[:3]:  # Limit to avoid too many tests
                                if not self.data[other_col].isnull().all():
                                    # Discretize continuous variables for chi-square test
                                    temp_data = self.data[[col, other_col]].copy()

                                    # Create missing indicator
                                    temp_data[f'{col}_missing'] = temp_data[col].isnull()

                                    # Discretize if continuous
                                    if temp_data[other_col].dtype in ['float64', 'int64']:
                                        temp_data[f'{other_col}_cat'] = pd.cut(
                                            temp_data[other_col].dropna(),
                                            bins=5,
                                            labels=False
                                        )
                                        temp_data[f'{other_col}_cat'] = temp_data[f'{other_col}_cat'].fillna('Unknown')
                                    else:
                                        temp_data[f'{other_col}_cat'] = temp_data[other_col].fillna('Unknown')

                                    # Create contingency table
                                    contingency_table = pd.crosstab(
                                        temp_data[f'{col}_missing'],
                                        temp_data[f'{other_col}_cat']
                                    )

                                    if contingency_table.size > 0:
                                        chi2, p_value, dof, _ = chi2_contingency(contingency_table)
                                        chi_square_stats.append(chi2)
                                        degrees_of_freedom.append(dof)

                # Combine test results
                if chi_square_stats:
                    total_chi2 = sum(chi_square_stats)
                    total_dof = sum(degrees_of_freedom)
                    p_value_combined = 1 - stats.chi2.cdf(total_chi2, total_dof)

                    results['little_mcar_test'] = {
                        'chi_square_statistic': total_chi2,
                        'degrees_of_freedom': total_dof,
                        'p_value': p_value_combined,
                        'is_mcar': p_value_combined > significance_level,
                        'significance_level': significance_level
                    }

        except Exception as e:
            results['little_mcar_test'] = {'error': str(e)}

        # T-tests for MCAR (compare means of observed vs missing groups)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if self.data[col].isnull().any() and self.data[col].notnull().any():
                # Test if other variables differ between missing and observed groups
                missing_indicator = self.data[col].isnull()

                for other_col in numeric_cols:
                    if other_col != col and not self.data[other_col].isnull().all():
                        observed_values = self.data[other_col][~missing_indicator].dropna()
                        missing_values = self.data[other_col][missing_indicator].dropna()

                        if len(observed_values) > 5 and len(missing_values) > 5:
                            try:
                                t_stat, p_value = stats.ttest_ind(observed_values, missing_values)

                                if f'{col}_vs_{other_col}' not in results['t_tests']:
                                    results['t_tests'][f'{col}_vs_{other_col}'] = {
                                        'statistic': t_stat,
                                        'p_value': p_value,
                                        'significant_difference': p_value < significance_level
                                    }
                            except:
                                continue

        # Correlation tests for MAR
        missing_matrix = self.data.isnull().astype(int)

        for col in self.data.columns:
            if self.data[col].isnull().any():
                # Correlate missingness with other variables
                missing_indicator = missing_matrix[col]

                for other_col in numeric_cols:
                    if other_col != col:
                        observed_mask = self.data[other_col].notnull()

                        if observed_mask.sum() > 10:  # Need enough data
                            try:
                                correlation, p_value = stats.pearsonr(
                                    missing_indicator[observed_mask],
                                    self.data[other_col][observed_mask]
                                )

                                if abs(correlation) > 0.1:  # Only include meaningful correlations
                                    if col not in results['correlation_tests']:
                                        results['correlation_tests'][col] = {}

                                    results['correlation_tests'][col][other_col] = {
                                        'correlation': correlation,
                                        'p_value': p_value,
                                        'significant_correlation': p_value < significance_level
                                    }
                            except:
                                continue

        # Pattern analysis
        complete_cases_mask = self.data.notnull().all(axis=1)
        complete_cases = self.data[complete_cases_mask]
        incomplete_cases = self.data[~complete_cases_mask]

        results['pattern_analysis'] = {
            'complete_cases_count': len(complete_cases),
            'incomplete_cases_count': len(incomplete_cases),
            'monotone_patterns': self._detect_monotone_patterns(),
            'arbitrary_patterns': len(self.missing_summary.get('patterns', {}).get('pattern_counts', {}))
        }

        # Determine likely mechanism
        is_mcar = False
        is_mar = False
        is_mnar = False

        if results['little_mcar_test'] and not results['little_mcar_test'].get('error'):
            is_mcar = results['little_mcar_test']['is_mcar']

        # Check for MAR patterns
        significant_correlations = 0
        for col, corr_tests in results['correlation_tests'].items():
            for other_col, test_result in corr_tests.items():
                if test_result['significant_correlation']:
                    significant_correlations += 1

        is_mar = significant_correlations > 0

        # If not MCAR and no clear MAR patterns, likely MNAR
        if not is_mcar and not is_mar:
            is_mnar = True

        results['conclusion'] = {
            'likely_mechanism': 'MCAR' if is_mcar else ('MAR' if is_mar else 'MNAR'),
            'is_mcar': is_mcar,
            'is_mar': is_mar,
            'is_mnar': is_mnar,
            'confidence': 'High' if (is_mcar or is_mar) else 'Medium',
            'significant_correlations_count': significant_correlations
        }

        self.missingness_tests = results
        self.missingness_mechanism = results['conclusion']
        return results

    def _detect_monotone_patterns(self) -> List[str]:
        """Detect monotone missing patterns.

        Returns:
            List of columns with monotone missing patterns
        """
        missing_matrix = self.data.isnull()
        monotone_cols = []

        for col in missing_matrix.columns:
            missing_series = missing_matrix[col]

            # Check if missingness is monotone (all missing values come after all observed values)
            if missing_series.any():
                # Find indices where missing starts
                missing_indices = missing_series[missing_series].index

                # Get the first and last missing indices
                first_missing_idx = missing_indices.min()
                last_missing_idx = missing_indices.max()

                # Check if all values before first_missing_idx are not missing
                # and all values from first_missing_idx onwards are missing (monotone pattern)
                before_first = missing_series.loc[:first_missing_idx-1] if first_missing_idx > 0 else missing_series.loc[:0]
                from_first = missing_series.loc[first_missing_idx:]

                is_monotone = (not before_first.any()) and from_first.all()

                if is_monotone:
                    monotone_cols.append(col)

        return monotone_cols

    def visualize_missing_patterns(self, plot_type: str = 'matrix',
                                 save_plot: bool = True) -> Optional[str]:
        """Visualize missing data patterns.

        Args:
            plot_type: Type of plot ('matrix', 'bar', 'heatmap', 'dendrogram')
            save_plot: Whether to save the plot as base64 string

        Returns:
            Base64 encoded plot if save_plot=True, None otherwise
        """
        plt.figure(figsize=(12, 8))

        if plot_type == 'matrix':
            # Missing data matrix plot
            msno.matrix(self.data, figsize=(12, 8), sparkline=False, fontsize=10)
            plt.title('Missing Data Pattern Matrix', fontsize=14, fontweight='bold')

        elif plot_type == 'bar':
            # Bar plot of missing values by column
            msno.bar(self.data, figsize=(12, 8), fontsize=10)
            plt.title('Missing Values by Column', fontsize=14, fontweight='bold')

        elif plot_type == 'heatmap':
            # Heatmap of missingness correlations
            msno.heatmap(self.data, figsize=(12, 8), fontsize=10)
            plt.title('Missingness Correlation Heatmap', fontsize=14, fontweight='bold')

        elif plot_type == 'dendrogram':
            # Dendrogram of missingness patterns
            msno.dendrogram(self.data, figsize=(12, 8), fontsize=10)
            plt.title('Missing Data Dendrogram', fontsize=14, fontweight='bold')

        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

        plt.tight_layout()

        if save_plot:
            # Save plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return plot_data
        else:
            plt.show()
            plt.close()
            return None

    def suggest_imputation_methods(self) -> Dict[str, Any]:
        """Suggest appropriate imputation methods based on missingness analysis.

        Returns:
            Dictionary containing suggested imputation methods for each column
        """
        suggestions = {}

        if not self.missing_summary:
            self.analyze_missing_patterns()

        if not self.missingness_mechanism:
            self.test_missingness_mechanism()

        mechanism = self.missingness_mechanism.get('likely_mechanism', 'Unknown')

        for col, col_info in self.missing_summary['by_column'].items():
            if col_info['has_missing']:
                missing_pct = col_info['missing_percentage']
                data_type = col_info['data_type']
                unique_vals = col_info['unique_values']

                col_suggestions = []

                # Based on missingness mechanism
                if mechanism == 'MCAR':
                    # MCAR: Can use simpler methods
                    if missing_pct < 5:
                        col_suggestions.append({
                            'method': 'mean_median_mode',
                            'confidence': 'High',
                            'reason': 'Low missing percentage with MCAR mechanism'
                        })
                    elif missing_pct < 20:
                        col_suggestions.extend([
                            {
                                'method': 'knn',
                                'confidence': 'High',
                                'reason': 'Moderate missing percentage with MCAR mechanism'
                            },
                            {
                                'method': 'iterative',
                                'confidence': 'Medium',
                                'reason': 'Iterative imputation works well with MCAR'
                            }
                        ])
                    else:
                        col_suggestions.append({
                            'method': 'multiple_imputation',
                            'confidence': 'Medium',
                            'reason': 'High missing percentage requires robust approach'
                        })

                elif mechanism == 'MAR':
                    # MAR: Should use methods that account for relationships
                    col_suggestions.extend([
                        {
                            'method': 'iterative',
                            'confidence': 'High',
                            'reason': 'Iterative imputation captures relationships in MAR'
                        },
                        {
                            'method': 'knn',
                            'confidence': 'Medium',
                            'reason': 'KNN can account for observed correlations'
                        }
                    ])

                    if missing_pct > 30:
                        col_suggestions.append({
                            'method': 'multiple_imputation',
                            'confidence': 'High',
                            'reason': 'High missing percentage with MAR requires MI'
                        })

                else:  # MNAR
                    # MNAR: Most challenging, requires sophisticated methods
                    col_suggestions.extend([
                        {
                            'method': 'multiple_imputation',
                            'confidence': 'High',
                            'reason': 'MNAR requires multiple imputation with auxiliary variables'
                        },
                        {
                            'method': 'model_based',
                            'confidence': 'Medium',
                            'reason': 'Model-based approaches can handle MNAR patterns'
                        }
                    ])

                # Based on data type
                if 'int' in str(data_type) or 'float' in str(data_type):
                    # Numeric data
                    if missing_pct < 10:
                        col_suggestions.append({
                            'method': 'mean_median',
                            'confidence': 'Medium',
                            'reason': 'Simple imputation suitable for low missingness'
                        })
                else:
                    # Categorical data
                    col_suggestions.append({
                        'method': 'mode',
                        'confidence': 'High',
                        'reason': 'Mode imputation standard for categorical data'
                    })

                    if unique_vals > 10 and missing_pct < 20:
                        col_suggestions.append({
                            'method': 'frequent_category',
                            'confidence': 'Medium',
                            'reason': 'Add missing as separate category for high-cardinality data'
                        })

                # Based on missing percentage
                if missing_pct > 50:
                    col_suggestions.append({
                        'method': 'drop_column',
                        'confidence': 'High',
                        'reason': 'Very high missing percentage suggests dropping column'
                    })
                elif missing_pct > 30:
                    col_suggestions.append({
                        'method': 'flag_missing',
                        'confidence': 'Medium',
                        'reason': 'Create missing indicator for high missingness'
                    })

                # Remove duplicates and rank by confidence
                unique_suggestions = {}
                for s in col_suggestions:
                    method = s['method']
                    if method not in unique_suggestions or unique_suggestions[method]['confidence'] == 'Low':
                        unique_suggestions[method] = s

                # Sort by confidence
                confidence_order = {'High': 3, 'Medium': 2, 'Low': 1}
                sorted_suggestions = sorted(
                    unique_suggestions.values(),
                    key=lambda x: (confidence_order.get(x['confidence'], 0), -x.get('priority', 0)),
                    reverse=True
                )

                suggestions[col] = {
                    'missing_percentage': missing_pct,
                    'data_type': data_type,
                    'suggested_methods': sorted_suggestions[:3],  # Top 3 suggestions
                    'primary_recommendation': sorted_suggestions[0] if sorted_suggestions else None
                }

        return suggestions

    def apply_imputation(self, method: str, columns: Optional[List[str]] = None,
                        **kwargs) -> pd.DataFrame:
        """Apply imputation to the data.

        Args:
            method: Imputation method ('mean', 'median', 'mode', 'knn', 'iterative', 'drop')
            columns: Specific columns to impute (None for all)
            **kwargs: Additional parameters for imputation methods

        Returns:
            DataFrame with imputed values
        """
        imputed_data = self.data.copy()

        if columns is None:
            # Get columns with missing values
            columns = self.data.columns[self.data.isnull().any()].tolist()

        # Convert columns to list if needed
        if columns is None or (isinstance(columns, (list, tuple)) and len(columns) == 0):
            return imputed_data  # No imputation needed
        elif isinstance(columns, (pd.Index, pd.Series)):
            columns = columns.tolist()
        elif not isinstance(columns, (list, tuple)):
            columns = [columns]

        # Check again after conversion
        if len(columns) == 0:
            return imputed_data  # No imputation needed

        # Record original missingness
        original_missing = imputed_data[columns].isnull().sum().to_dict()

        # Apply imputation
        if method == 'mean':
            imputer = SimpleImputer(strategy='mean')
            numeric_cols = imputed_data[columns].select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                imputed_data[numeric_cols] = imputer.fit_transform(imputed_data[numeric_cols])

        elif method == 'median':
            imputer = SimpleImputer(strategy='median')
            numeric_cols = imputed_data[columns].select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                imputed_data[numeric_cols] = imputer.fit_transform(imputed_data[numeric_cols])

        elif method == 'mode':
            imputer = SimpleImputer(strategy='most_frequent')
            imputed_data[columns] = imputer.fit_transform(imputed_data[columns])

        elif method == 'knn':
            n_neighbors = kwargs.get('n_neighbors', 5)
            imputer = KNNImputer(n_neighbors=n_neighbors)

            # Only apply to numeric columns
            numeric_cols = imputed_data[columns].select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                imputed_data[numeric_cols] = imputer.fit_transform(imputed_data[numeric_cols])

        elif method == 'iterative':
            max_iter = kwargs.get('max_iter', 10)
            random_state = kwargs.get('random_state', 42)

            imputer = IterativeImputer(
                max_iter=max_iter,
                random_state=random_state,
                estimator=RandomForestRegressor(n_estimators=10, random_state=random_state)
            )

            # Only apply to numeric columns
            numeric_cols = imputed_data[columns].select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                imputed_data[numeric_cols] = imputer.fit_transform(imputed_data[numeric_cols])

        elif method == 'drop':
            # Drop rows with missing values in specified columns
            imputed_data = imputed_data.dropna(subset=columns)

        elif method == 'forward_fill':
            imputed_data[columns] = imputed_data[columns].fillna(method='ffill')

        elif method == 'backward_fill':
            imputed_data[columns] = imputed_data[columns].fillna(method='bfill')

        elif method == 'constant':
            fill_value = kwargs.get('fill_value', 0)
            imputed_data[columns] = imputed_data[columns].fillna(fill_value)

        else:
            raise ValueError(f"Unknown imputation method: {method}")

        # Record imputation details
        imputation_record = {
            'method': method,
            'columns': columns,
            'parameters': kwargs,
            'original_missing': original_missing,
            'timestamp': datetime.now().isoformat(),
            'imputed_shape': imputed_data.shape,
            'original_shape': self.data.shape
        }

        self.imputation_history.append(imputation_record)

        return imputed_data

    def evaluate_imputation(self, original_data: Optional[pd.DataFrame] = None,
                          imputed_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Evaluate the quality of imputation.

        Args:
            original_data: Original data with missing values (if None, uses self.data)
            imputed_data: Imputed data to evaluate (if None, uses last imputed data)

        Returns:
            Dictionary containing evaluation metrics
        """
        if original_data is None:
            original_data = self.data

        if imputed_data is None and self.imputation_history:
            # Get the last applied imputation
            last_imputation = self.imputation_history[-1]
            imputed_data = self.apply_imputation(
                last_imputation['method'],
                last_imputation['columns'],
                **last_imputation['parameters']
            )
        elif imputed_data is None:
            return {'error': 'No imputed data available for evaluation'}

        evaluation = {
            'missingness_reduction': {},
            'distribution_similarity': {},
            'correlation_preservation': {}
        }

        # Missingness reduction
        original_missing = original_data.isnull().sum().sum()
        imputed_missing = imputed_data.isnull().sum().sum()

        evaluation['missingness_reduction'] = {
            'original_missing_count': int(original_missing),
            'imputed_missing_count': int(imputed_missing),
            'missing_reduction_percentage': ((original_missing - imputed_missing) / max(original_missing, 1)) * 100
        }

        # Distribution similarity for numeric columns
        numeric_cols = original_data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if original_data[col].isnull().any():
                original_values = original_data[col].dropna()
                imputed_values = imputed_data[col]

                # Kolmogorov-Smirnov test for distribution similarity
                ks_statistic, ks_p_value = stats.ks_2samp(original_values, imputed_values)

                # Compare basic statistics
                original_stats = {
                    'mean': original_values.mean(),
                    'std': original_values.std(),
                    'min': original_values.min(),
                    'max': original_values.max()
                }

                imputed_stats = {
                    'mean': imputed_values.mean(),
                    'std': imputed_values.std(),
                    'min': imputed_values.min(),
                    'max': imputed_values.max()
                }

                evaluation['distribution_similarity'][col] = {
                    'ks_statistic': ks_statistic,
                    'ks_p_value': ks_p_value,
                    'distributions_similar': ks_p_value > 0.05,
                    'mean_difference': abs(original_stats['mean'] - imputed_stats['mean']),
                    'std_difference': abs(original_stats['std'] - imputed_stats['std']),
                    'original_stats': original_stats,
                    'imputed_stats': imputed_stats
                }

        # Correlation preservation
        original_corr = original_data.select_dtypes(include=[np.number]).corr()
        imputed_corr = imputed_data.select_dtypes(include=[np.number]).corr()

        if not original_corr.empty and not imputed_corr.empty:
            corr_diff = np.abs(original_corr - imputed_corr)
            mean_corr_diff = corr_diff.mean().mean()
            max_corr_diff = corr_diff.max().max()

            evaluation['correlation_preservation'] = {
                'mean_correlation_difference': float(mean_corr_diff),
                'max_correlation_difference': float(max_corr_diff),
                'correlations_preserved': mean_corr_diff < 0.1,
                'original_correlation_matrix': original_corr.fillna(0).to_dict(),
                'imputed_correlation_matrix': imputed_corr.fillna(0).to_dict()
            }

        return evaluation

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive missing data report.

        Returns:
            Dictionary containing all analysis results and recommendations
        """
        # Ensure all analyses are run
        if not self.missing_summary:
            self.analyze_missing_patterns()

        if not self.missingness_tests:
            self.test_missingness_mechanism()

        imputation_suggestions = self.suggest_imputation_methods()

        report = {
            'dataset_info': {
                'shape': self.original_shape,
                'columns': list(self.data.columns),
                'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(self.data.select_dtypes(include=['object', 'category']).columns)
            },
            'missing_summary': self.missing_summary,
            'missingness_analysis': self.missingness_tests,
            'imputation_recommendations': imputation_suggestions,
            'imputation_history': self.imputation_history,
            'quality_indicators': self._calculate_quality_indicators(),
            'actionable_insights': self._generate_insights(),
            'generated_at': datetime.now().isoformat()
        }

        return report

    def _calculate_quality_indicators(self) -> Dict[str, Any]:
        """Calculate quality indicators for the dataset.

        Returns:
            Dictionary of quality indicators
        """
        indicators = {
            'completeness': 0,
            'missing_pattern_complexity': 'Low',
            'imputation_difficulty': 'Easy',
            'overall_quality': 'Good'
        }

        if self.missing_summary:
            # Completeness percentage
            total_cells = self.missing_summary['overall']['total_cells']
            total_missing = self.missing_summary['overall']['total_missing']
            completeness = ((total_cells - total_missing) / total_cells) * 100
            indicators['completeness'] = completeness

            # Missing pattern complexity
            unique_patterns = self.missing_summary['patterns']['unique_patterns']
            if unique_patterns <= 3:
                indicators['missing_pattern_complexity'] = 'Low'
            elif unique_patterns <= 10:
                indicators['missing_pattern_complexity'] = 'Medium'
            else:
                indicators['missing_pattern_complexity'] = 'High'

            # Imputation difficulty based on mechanism
            if self.missingness_mechanism:
                mechanism = self.missingness_mechanism.get('likely_mechanism', 'Unknown')
                if mechanism == 'MCAR':
                    indicators['imputation_difficulty'] = 'Easy'
                elif mechanism == 'MAR':
                    indicators['imputation_difficulty'] = 'Moderate'
                else:
                    indicators['imputation_difficulty'] = 'Challenging'

            # Overall quality assessment
            if completeness > 95 and indicators['missing_pattern_complexity'] == 'Low':
                indicators['overall_quality'] = 'Excellent'
            elif completeness > 85 and indicators['missing_pattern_complexity'] in ['Low', 'Medium']:
                indicators['overall_quality'] = 'Good'
            elif completeness > 70:
                indicators['overall_quality'] = 'Fair'
            else:
                indicators['overall_quality'] = 'Poor'

        return indicators

    def _generate_insights(self) -> List[str]:
        """Generate actionable insights from the analysis.

        Returns:
            List of insights and recommendations
        """
        insights = []

        if not self.missing_summary:
            return insights

        # Overall missingness insights
        overall_missing_pct = self.missing_summary['overall']['missing_percentage']

        if overall_missing_pct < 5:
            insights.append("Dataset has excellent completeness with minimal missing values.")
        elif overall_missing_pct < 15:
            insights.append("Dataset has good completeness with manageable missing values.")
        elif overall_missing_pct < 30:
            insights.append("Dataset has moderate missingness requiring careful imputation.")
        else:
            insights.append("Dataset has high missingness requiring sophisticated imputation strategies.")

        # Column-specific insights
        high_missing_cols = []
        for col, info in self.missing_summary['by_column'].items():
            if info['missing_percentage'] > 50:
                high_missing_cols.append(col)

        if high_missing_cols:
            insights.append(f"Columns {high_missing_cols} have >50% missing values - consider dropping or creating indicators.")

        # Pattern complexity insights
        if self.missing_summary['patterns']['unique_patterns'] > 10:
            insights.append("Multiple missing data patterns detected - consider advanced imputation methods.")

        # Missingness mechanism insights
        if self.missingness_mechanism:
            mechanism = self.missingness_mechanism.get('likely_mechanism', 'Unknown')
            if mechanism == 'MCAR':
                insights.append("Missing data appears to be completely random - simpler imputation methods appropriate.")
            elif mechanism == 'MAR':
                insights.append("Missing data shows relationships with observed variables - use relationship-aware methods.")
            else:
                insights.append("Missing data may not be random - consider sensitivity analysis.")

        return insights

    def export_results(self, format: str = 'dict') -> Union[Dict, str]:
        """Export analysis results.

        Args:
            format: Export format ('dict' or 'json')

        Returns:
            Results in specified format
        """
        report = self.generate_report()

        if format == 'json':
            return json.dumps(report, indent=2, default=str)
        elif format == 'dict':
            return report
        else:
            raise ValueError(f"Unknown export format: {format}")