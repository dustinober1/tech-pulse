"""Data quality checker for validating and reporting data quality issues."""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
import warnings
from datetime import datetime
import json

from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class DataQualityChecker:
    """
    Comprehensive data quality checker for detecting and reporting data issues.

    Checks for missing values, duplicates, outliers, data type issues,
    and generates detailed quality reports.
    """

    def __init__(self, data: pd.DataFrame, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize the data quality checker.

        Args:
            data: DataFrame to check
            schema: Optional schema definition with expected types and constraints
        """
        self.data = data.copy()
        self.schema = schema or {}
        self.quality_report = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': data.shape,
            'summary': {},
            'missing_values': {},
            'duplicates': {},
            'outliers': {},
            'data_types': {},
            'schema_validation': {},
            'quality_score': 0.0
        }

    def check_missing_values(self, threshold: float = 0.05) -> Dict[str, Any]:
        """
        Check for missing values and patterns.

        Args:
            threshold: Threshold for flagging high missing value percentage

        Returns:
            Dictionary containing missing value analysis
        """
        missing_analysis = {
            'total_missing': int(self.data.isnull().sum().sum()),
            'missing_percentage': float(self.data.isnull().sum().sum() / self.data.size * 100),
            'by_column': {},
            'patterns': {},
            'completeness_score': 0.0
        }

        # Missing values by column
        for col in self.data.columns:
            missing_count = self.data[col].isnull().sum()
            missing_pct = missing_count / len(self.data) * 100
            missing_analysis['by_column'][col] = {
                'count': int(missing_count),
                'percentage': float(missing_pct),
                'high_missing': missing_pct > threshold * 100
            }

        # Missing value patterns (correlations in missingness)
        missing_mask = self.data.isnull()
        if missing_mask.any().any():
            # Find columns with similar missing patterns
            missing_corr = missing_mask.corr()
            if not missing_corr.empty:
                high_corr_pairs = []
                for i in range(len(missing_corr.columns)):
                    for j in range(i+1, len(missing_corr.columns)):
                        corr_val = missing_corr.iloc[i, j]
                        if not np.isnan(corr_val) and abs(corr_val) > 0.7:
                            high_corr_pairs.append({
                                'columns': [missing_corr.columns[i], missing_corr.columns[j]],
                                'correlation': float(corr_val)
                            })
                missing_analysis['patterns']['correlated_missingness'] = high_corr_pairs

            # Check for MCAR (Missing Completely At Random) patterns
            missing_analysis['patterns']['mcar_likelihood'] = self._check_mcar_pattern()

        # Calculate completeness score
        completeness_score = (100 - missing_analysis['missing_percentage']) / 100
        missing_analysis['completeness_score'] = float(completeness_score)

        self.quality_report['missing_values'] = missing_analysis
        return missing_analysis

    def _check_mcar_pattern(self) -> str:
        """Check if missing values are MCAR, MAR, or MNAR."""
        missing_mask = self.data.isnull()
        if not missing_mask.any().any():
            return "no_missing_data"

        # Simple heuristic based on missing value correlations
        # This is a simplified check - real MCAR testing requires more sophisticated methods
        missing_corr = missing_mask.corr()
        if missing_corr.isnull().all().all():
            return "likely_mcar"
        elif missing_corr.abs().mean().mean() > 0.3:
            return "likely_mar"
        else:
            return "likely_mcar"

    def check_duplicates(self, subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Check for duplicate records.

        Args:
            subset: List of columns to check for duplicates

        Returns:
            Dictionary containing duplicate analysis
        """
        duplicate_analysis = {
            'total_duplicates': 0,
            'duplicate_percentage': 0.0,
            'duplicate_rows': [],
            'duplicate_columns': {},
            'uniqueness_score': 0.0
        }

        if subset:
            # Check duplicates based on subset of columns
            duplicates = self.data.duplicated(subset=subset, keep=False)
        else:
            # Check complete row duplicates
            duplicates = self.data.duplicated(keep=False)

        duplicate_rows = self.data[duplicates]
        duplicate_analysis['total_duplicates'] = int(len(duplicate_rows))
        duplicate_analysis['duplicate_percentage'] = float(
            len(duplicate_rows) / len(self.data) * 100
        )

        if len(duplicate_rows) > 0:
            # Show first 10 duplicate rows
            duplicate_analysis['duplicate_rows'] = duplicate_rows.head(10).to_dict('records')

        # Check column-wise uniqueness
        for col in self.data.columns:
            unique_count = self.data[col].nunique()
            uniqueness_rate = unique_count / len(self.data)
            duplicate_analysis['duplicate_columns'][col] = {
                'unique_count': int(unique_count),
                'total_count': int(len(self.data)),
                'uniqueness_rate': float(uniqueness_rate),
                'has_duplicates': unique_count < len(self.data)
            }

        # Calculate uniqueness score
        avg_uniqueness = np.mean([
            info['uniqueness_rate'] for info in duplicate_analysis['duplicate_columns'].values()
        ])
        duplicate_analysis['uniqueness_score'] = float(avg_uniqueness)

        self.quality_report['duplicates'] = duplicate_analysis
        return duplicate_analysis

    def detect_outliers(self, methods: List[str] = ['iqr', 'zscore', 'isolation_forest']) -> Dict[str, Any]:
        """
        Detect outliers using multiple methods.

        Args:
            methods: List of outlier detection methods to use

        Returns:
            Dictionary containing outlier analysis
        """
        outlier_analysis = {
            'methods_used': methods,
            'outliers_by_column': {},
            'outlier_counts': {},
            'outlier_summary': {},
            'cleanliness_score': 0.0
        }

        # Only process numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        for method in methods:
            method_outliers = {}

            for col in numeric_cols:
                col_data = self.data[col].dropna()

                if len(col_data) == 0:
                    continue

                if method == 'iqr':
                    outliers = self._detect_outliers_iqr(col_data)
                elif method == 'zscore':
                    outliers = self._detect_outliers_zscore(col_data)
                elif method == 'isolation_forest':
                    outliers = self._detect_outliers_isolation_forest(col_data)
                elif method == 'lof':
                    outliers = self._detect_outliers_lof(col_data)
                else:
                    warnings.warn(f"Unknown outlier detection method: {method}")
                    continue

                method_outliers[col] = outliers

            outlier_analysis['outliers_by_column'][method] = method_outliers

        # Aggregate outlier results
        for col in numeric_cols:
            col_outliers = {}
            for method in methods:
                if method in outlier_analysis['outliers_by_column'] and col in outlier_analysis['outliers_by_column'][method]:
                    col_outliers[method] = outlier_analysis['outliers_by_column'][method][col]

            outlier_analysis['outlier_counts'][col] = {
                method: len(indices) if indices is not None else 0
                for method, indices in col_outliers.items()
            }

            # Consensus outliers (detected by multiple methods)
            if len(col_outliers) > 1:
                consensus_indices = set()
                for indices in col_outliers.values():
                    if indices is not None:
                        consensus_indices.update(indices)
                outlier_analysis['outlier_summary'][col] = {
                    'consensus_count': len(consensus_indices),
                    'consensus_percentage': len(consensus_indices) / len(self.data) * 100
                }

        # Calculate cleanliness score (inverse of outlier percentage)
        if outlier_analysis['outlier_summary']:
            avg_outlier_pct = np.mean([
                info['consensus_percentage']
                for info in outlier_analysis['outlier_summary'].values()
            ])
            outlier_analysis['cleanliness_score'] = float(max(0, (100 - avg_outlier_pct) / 100))
        else:
            outlier_analysis['cleanliness_score'] = 1.0

        self.quality_report['outliers'] = outlier_analysis
        return outlier_analysis

    def _detect_outliers_iqr(self, data: pd.Series) -> np.ndarray:
        """Detect outliers using IQR method."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        return data[outlier_mask].index.values

    def _detect_outliers_zscore(self, data: pd.Series, threshold: float = 3.0) -> np.ndarray:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(data))
        outlier_mask = z_scores > threshold
        return data[outlier_mask].index.values

    def _detect_outliers_isolation_forest(self, data: pd.Series, contamination: float = 0.1) -> np.ndarray:
        """Detect outliers using Isolation Forest."""
        if len(data) < 10:
            return np.array([])

        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(data.values.reshape(-1, 1))
        outlier_mask = outlier_labels == -1
        return data[outlier_mask].index.values

    def _detect_outliers_lof(self, data: pd.Series, contamination: float = 0.1) -> np.ndarray:
        """Detect outliers using Local Outlier Factor."""
        if len(data) < 10:
            return np.array([])

        lof = LocalOutlierFactor(contamination=contamination)
        outlier_labels = lof.fit_predict(data.values.reshape(-1, 1))
        outlier_mask = outlier_labels == -1
        return data[outlier_mask].index.values

    def validate_data_types(self) -> Dict[str, Any]:
        """
        Validate data types against expected schema.

        Returns:
            Dictionary containing data type validation results
        """
        type_validation = {
            'column_types': {},
            'type_issues': {},
            'type_consistency_score': 0.0
        }

        for col in self.data.columns:
            actual_type = str(self.data[col].dtype)
            expected_type = self.schema.get(col, {}).get('type', actual_type)

            type_info = {
                'actual_type': actual_type,
                'expected_type': expected_type,
                'is_correct': self._check_type_compatibility(actual_type, expected_type)
            }

            # Check for type consistency
            if self.data[col].dtype.kind in 'biufc':  # numeric types
                type_info['min'] = float(self.data[col].min())
                type_info['max'] = float(self.data[col].max())
                type_info['mean'] = float(self.data[col].mean())
                type_info['std'] = float(self.data[col].std())
            elif self.data[col].dtype == 'object':
                type_info['unique_values'] = int(self.data[col].nunique())
                type_info['most_common'] = str(self.data[col].mode().iloc[0]) if not self.data[col].mode().empty else None

            type_validation['column_types'][col] = type_info

            if not type_info['is_correct']:
                type_validation['type_issues'][col] = {
                    'expected': expected_type,
                    'actual': actual_type,
                    'issue': f"Expected {expected_type} but got {actual_type}"
                }

        # Calculate type consistency score
        if self.data.columns.size > 0:
            correct_types = sum(1 for info in type_validation['column_types'].values() if info['is_correct'])
            type_validation['type_consistency_score'] = float(correct_types / len(self.data.columns))

        self.quality_report['data_types'] = type_validation
        return type_validation

    def _check_type_compatibility(self, actual: str, expected: str) -> bool:
        """Check if actual type is compatible with expected type."""
        type_mapping = {
            'int64': ['int', 'integer', 'numeric'],
            'int32': ['int', 'integer', 'numeric'],
            'float64': ['float', 'numeric', 'decimal'],
            'float32': ['float', 'numeric', 'decimal'],
            'object': ['string', 'str', 'categorical'],
            'bool': ['boolean', 'bool'],
            'datetime64[ns]': ['datetime', 'date', 'timestamp']
        }

        actual_lower = actual.lower()
        expected_lower = expected.lower()

        if actual_lower == expected_lower:
            return True

        for compatible_types in type_mapping.values():
            if actual_lower in compatible_types and expected_lower in compatible_types:
                return True

        return False

    def validate_schema(self) -> Dict[str, Any]:
        """
        Validate data against provided schema.

        Returns:
            Dictionary containing schema validation results
        """
        schema_validation = {
            'schema_compliant': True,
            'missing_columns': [],
            'extra_columns': [],
            'constraint_violations': {},
            'schema_compliance_score': 0.0
        }

        if not self.schema:
            schema_validation['schema_compliance_score'] = 1.0
            schema_validation['message'] = "No schema provided for validation"
            self.quality_report['schema_validation'] = schema_validation
            return schema_validation

        # Check for missing columns
        expected_columns = set(self.schema.keys())
        actual_columns = set(self.data.columns)

        schema_validation['missing_columns'] = list(expected_columns - actual_columns)
        schema_validation['extra_columns'] = list(actual_columns - expected_columns)

        # Check constraints
        for col, constraints in self.schema.items():
            if col not in self.data.columns:
                continue

            col_violations = []

            # Check nullable constraint
            if 'nullable' in constraints:
                is_nullable = constraints['nullable']
                has_nulls = self.data[col].isnull().any()
                if not is_nullable and has_nulls:
                    col_violations.append({
                        'constraint': 'nullable',
                        'expected': False,
                        'actual': True,
                        'null_count': int(self.data[col].isnull().sum())
                    })

            # Check range constraint
            if 'range' in constraints and pd.api.types.is_numeric_dtype(self.data[col]):
                min_val, max_val = constraints['range']
                actual_min = self.data[col].min()
                actual_max = self.data[col].max()
                if actual_min < min_val or actual_max > max_val:
                    col_violations.append({
                        'constraint': 'range',
                        'expected': [min_val, max_val],
                        'actual': [float(actual_min), float(actual_max)]
                    })

            # Check enum constraint
            if 'enum' in constraints:
                allowed_values = set(constraints['enum'])
                actual_values = set(self.data[col].dropna().unique())
                invalid_values = actual_values - allowed_values
                if invalid_values:
                    col_violations.append({
                        'constraint': 'enum',
                        'allowed_values': list(allowed_values),
                        'invalid_values': list(invalid_values)
                    })

            # Check length constraint
            if 'max_length' in constraints and self.data[col].dtype == 'object':
                max_length = constraints['max_length']
                actual_max_length = self.data[col].astype(str).str.len().max()
                if actual_max_length > max_length:
                    col_violations.append({
                        'constraint': 'max_length',
                        'expected': max_length,
                        'actual': int(actual_max_length)
                    })

            if col_violations:
                schema_validation['constraint_violations'][col] = col_violations

        # Calculate overall compliance
        total_checks = len(self.schema)
        passed_checks = total_checks - len(schema_validation['constraint_violations'])
        schema_validation['schema_compliant'] = len(schema_validation['constraint_violations']) == 0
        schema_validation['schema_compliance_score'] = float(passed_checks / max(1, total_checks))

        self.quality_report['schema_validation'] = schema_validation
        return schema_validation

    def generate_quality_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.

        Returns:
            Dictionary containing complete quality assessment
        """
        # Run all checks if not already done
        if not self.quality_report['missing_values']:
            self.check_missing_values()
        if not self.quality_report['duplicates']:
            self.check_duplicates()
        if not self.quality_report['outliers']:
            self.detect_outliers()
        if not self.quality_report['data_types']:
            self.validate_data_types()
        if not self.quality_report['schema_validation']:
            self.validate_schema()

        # Calculate overall quality score
        scores = {
            'completeness': self.quality_report['missing_values'].get('completeness_score', 0),
            'uniqueness': self.quality_report['duplicates'].get('uniqueness_score', 0),
            'cleanliness': self.quality_report['outliers'].get('cleanliness_score', 0),
            'type_consistency': self.quality_report['data_types'].get('type_consistency_score', 0),
            'schema_compliance': self.quality_report['schema_validation'].get('schema_compliance_score', 0)
        }

        overall_score = np.mean(list(scores.values()))
        self.quality_report['quality_score'] = float(overall_score)
        self.quality_report['scores'] = scores
        self.quality_report['summary'] = {
            'total_records': int(len(self.data)),
            'total_columns': int(len(self.data.columns)),
            'numeric_columns': int(len(self.data.select_dtypes(include=[np.number]).columns)),
            'categorical_columns': int(len(self.data.select_dtypes(include=['object', 'category']).columns)),
            'overall_quality_score': float(overall_score),
            'quality_grade': self._get_quality_grade(overall_score)
        }

        return self.quality_report

    def _get_quality_grade(self, score: float) -> str:
        """Get quality grade based on score."""
        if score >= 0.9:
            return 'Excellent'
        elif score >= 0.8:
            return 'Good'
        elif score >= 0.7:
            return 'Fair'
        elif score >= 0.6:
            return 'Poor'
        else:
            return 'Very Poor'

    def save_report(self, filepath: str, format: str = 'json') -> None:
        """
        Save quality report to file.

        Args:
            filepath: Path to save the report
            format: Format to save ('json' or 'csv')
        """
        report = self.generate_quality_report()

        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif format.lower() == 'csv':
            # Save summary as CSV
            summary_df = pd.DataFrame([self.quality_report['summary']])
            summary_df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_quality_recommendations(self) -> List[Dict[str, str]]:
        """
        Generate recommendations for improving data quality.

        Returns:
            List of quality improvement recommendations
        """
        recommendations = []
        report = self.generate_quality_report()

        # Missing values recommendations
        if report['missing_values']['missing_percentage'] > 5:
            recommendations.append({
                'category': 'Missing Values',
                'issue': f"High missing value percentage ({report['missing_values']['missing_percentage']:.1f}%)",
                'recommendation': 'Consider imputation strategies or data collection improvements'
            })

        # Duplicates recommendations
        if report['duplicates']['duplicate_percentage'] > 1:
            recommendations.append({
                'category': 'Duplicates',
                'issue': f"{report['duplicates']['duplicate_percentage']:.1f}% duplicate records found",
                'recommendation': 'Remove duplicate records or investigate data collection process'
            })

        # Outliers recommendations
        outlier_summary = report['outliers'].get('outlier_summary', {})
        high_outlier_cols = [
            col for col, info in outlier_summary.items()
            if info['consensus_percentage'] > 5
        ]
        if high_outlier_cols:
            recommendations.append({
                'category': 'Outliers',
                'issue': f"High outlier percentage in columns: {', '.join(high_outlier_cols)}",
                'recommendation': 'Investigate outliers for data entry errors or legitimate extreme values'
            })

        # Data type recommendations
        type_issues = report['data_types'].get('type_issues', {})
        if type_issues:
            recommendations.append({
                'category': 'Data Types',
                'issue': f"Type inconsistencies in {len(type_issues)} columns",
                'recommendation': 'Review and correct data type assignments'
            })

        # Schema recommendations
        if not report['schema_validation']['schema_compliant']:
            recommendations.append({
                'category': 'Schema',
                'issue': 'Schema validation failed',
                'recommendation': 'Review schema constraints and data compliance'
            })

        if not recommendations:
            recommendations.append({
                'category': 'Overall',
                'issue': 'Data quality is acceptable',
                'recommendation': 'Continue monitoring data quality on an ongoing basis'
            })

        return recommendations