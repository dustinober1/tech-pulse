"""Outlier detection and handling utilities."""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
import warnings
from datetime import datetime
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO


class OutlierHandler:
    """
    Comprehensive outlier detection and handling system.

    Provides multiple methods for outlier detection including:
    - IQR-based detection
    - Z-score method
    - Isolation Forest for multivariate outliers
    - Local Outlier Factor
    """

    def __init__(self, data: Union[pd.DataFrame, np.ndarray],
                 z_threshold: float = 3.0,
                 iqr_factor: float = 1.5,
                 contamination: float = 'auto'):
        """
        Initialize the outlier handler.

        Args:
            data: Input data for outlier detection
            z_threshold: Threshold for z-score method (default: 3.0)
            iqr_factor: IQR multiplier for outlier detection (default: 1.5)
            contamination: Expected proportion of outliers for isolation methods
        """
        self.data = pd.DataFrame(data) if isinstance(data, np.ndarray) else data.copy()
        self.z_threshold = z_threshold
        self.iqr_factor = iqr_factor
        self.contamination = contamination
        self.outlier_results = {}
        self.treatment_history = []

    def detect_iqr_outliers(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect outliers using the Interquartile Range (IQR) method.

        Args:
            columns: Columns to analyze (default: all numeric columns)

        Returns:
            Dictionary containing IQR outlier detection results
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        results = {
            'method': 'IQR',
            'parameters': {
                'iqr_factor': self.iqr_factor
            },
            'outliers_by_column': {},
            'summary': {
                'total_columns': len(columns),
                'columns_with_outliers': 0,
                'total_outliers': 0
            },
            'timestamp': datetime.now().isoformat()
        }

        for col in columns:
            if col not in self.data.columns:
                continue

            col_data = self.data[col].dropna()
            if len(col_data) == 0:
                continue

            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - self.iqr_factor * IQR
            upper_bound = Q3 + self.iqr_factor * IQR

            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            outlier_indices = outliers.index.tolist()

            results['outliers_by_column'][col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(col_data) * 100,
                'indices': outlier_indices,
                'values': outliers.tolist(),
                'bounds': {
                    'lower': float(lower_bound),
                    'upper': float(upper_bound),
                    'Q1': float(Q1),
                    'Q3': float(Q3),
                    'IQR': float(IQR)
                }
            }

            if len(outliers) > 0:
                results['summary']['columns_with_outliers'] += 1
                results['summary']['total_outliers'] += len(outliers)

        self.outlier_results['iqr'] = results
        return results

    def detect_zscore_outliers(self, columns: Optional[List[str]] = None,
                           z_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Detect outliers using the Z-score method.

        Args:
            columns: Columns to analyze (default: all numeric columns)
            z_threshold: Z-score threshold (default: self.z_threshold)

        Returns:
            Dictionary containing Z-score outlier detection results
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        if z_threshold is None:
            z_threshold = self.z_threshold

        results = {
            'method': 'Z-Score',
            'parameters': {
                'threshold': z_threshold
            },
            'outliers_by_column': {},
            'summary': {
                'total_columns': len(columns),
                'columns_with_outliers': 0,
                'total_outliers': 0
            },
            'timestamp': datetime.now().isoformat()
        }

        for col in columns:
            if col not in self.data.columns:
                continue

            col_data = self.data[col].dropna()
            if len(col_data) == 0:
                continue

            z_scores = np.abs(stats.zscore(col_data))
            outlier_mask = z_scores > z_threshold
            outliers = col_data[outlier_mask]
            outlier_indices = col_data[outlier_mask].index.tolist()

            results['outliers_by_column'][col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(col_data) * 100,
                'indices': outlier_indices,
                'values': outliers.tolist(),
                'z_scores': z_scores[outlier_mask].tolist(),
                'statistics': {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min_z': float(z_scores.min()),
                    'max_z': float(z_scores.max())
                }
            }

            if len(outliers) > 0:
                results['summary']['columns_with_outliers'] += 1
                results['summary']['total_outliers'] += len(outliers)

        self.outlier_results['zscore'] = results
        return results

    def detect_isolation_forest_outliers(self, columns: Optional[List[str]] = None,
                                        contamination: Optional[float] = None) -> Dict[str, Any]:
        """
        Detect outliers using Isolation Forest algorithm.

        Args:
            columns: Columns to analyze (default: all numeric columns)
            contamination: Expected proportion of outliers

        Returns:
            Dictionary containing Isolation Forest outlier detection results
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        if contamination is None:
            contamination = self.contamination

        # Prepare data
        data_subset = self.data[columns].dropna()
        if len(data_subset) == 0:
            return {
                'method': 'Isolation Forest',
                'error': 'No data available after removing missing values'
            }

        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )

        outlier_labels = iso_forest.fit_predict(data_subset)
        outlier_scores = iso_forest.decision_function(data_subset)

        # -1 indicates outliers
        outlier_mask = outlier_labels == -1
        outlier_indices = data_subset[outlier_mask].index.tolist()

        results = {
            'method': 'Isolation Forest',
            'parameters': {
                'contamination': contamination,
                'n_estimators': 100,
                'random_state': 42
            },
            'outliers': {
                'count': int(outlier_mask.sum()),
                'percentage': float(outlier_mask.mean() * 100),
                'indices': outlier_indices,
                'scores': outlier_scores[outlier_mask].tolist(),
                'average_score': float(outlier_scores.mean()),
                'score_std': float(outlier_scores.std())
            },
            'model_info': {
                'columns_used': columns,
                'samples_analyzed': len(data_subset)
            },
            'timestamp': datetime.now().isoformat()
        }

        self.outlier_results['isolation_forest'] = results
        return results

    def detect_lof_outliers(self, columns: Optional[List[str]] = None,
                           n_neighbors: int = 20,
                           contamination: Optional[float] = None) -> Dict[str, Any]:
        """
        Detect outliers using Local Outlier Factor (LOF).

        Args:
            columns: Columns to analyze (default: all numeric columns)
            n_neighbors: Number of neighbors for LOF
            contamination: Expected proportion of outliers

        Returns:
            Dictionary containing LOF outlier detection results
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        if contamination is None:
            contamination = self.contamination

        # Prepare data
        data_subset = self.data[columns].dropna()
        if len(data_subset) == 0:
            return {
                'method': 'Local Outlier Factor',
                'error': 'No data available after removing missing values'
            }

        # Fit LOF
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=False
        )

        outlier_predictions = lof.fit_predict(data_subset)
        lof_scores = lof.negative_outlier_factor_

        # Convert predictions (-1 for outliers, 1 for inliers)
        outlier_mask = outlier_predictions == -1
        outlier_indices = data_subset[outlier_mask].index.tolist()

        results = {
            'method': 'Local Outlier Factor',
            'parameters': {
                'n_neighbors': n_neighbors,
                'contamination': contamination
            },
            'outliers': {
                'count': int(outlier_mask.sum()),
                'percentage': float(outlier_mask.mean() * 100),
                'indices': outlier_indices,
                'lof_scores': lof_scores[outlier_mask].tolist(),
                'average_lof': float(lof_scores.mean()),
                'lof_std': float(lof_scores.std())
            },
            'model_info': {
                'columns_used': columns,
                'samples_analyzed': len(data_subset)
            },
            'timestamp': datetime.now().isoformat()
        }

        self.outlier_results['lof'] = results
        return results

    def get_outlier_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of all outlier detection methods.

        Returns:
            Dictionary containing summary of all detection methods
        """
        summary = {
            'dataset_info': {
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'numeric_columns': self.data.select_dtypes(include=[np.number]).columns.tolist()
            },
            'methods_applied': list(self.outlier_results.keys()),
            'outlier_counts': {},
            'recommendations': [],
            'timestamp': datetime.now().isoformat()
        }

        for method, results in self.outlier_results.items():
            if 'error' not in results:
                if method in ['iqr', 'zscore']:
                    summary['outlier_counts'][method] = results['summary']['total_outliers']
                else:  # isolation_forest, lof
                    summary['outlier_counts'][method] = results['outliers']['count']

        # Generate recommendations
        if summary['outlier_counts']:
            max_outliers = max(summary['outlier_counts'].values())
            max_method = max(summary['outlier_counts'], key=summary['outlier_counts'].get)

            if max_outliers > len(self.data) * 0.1:
                summary['recommendations'].append(
                    f"High number of outliers detected by {max_method} "
                    f"({max_outliers:,} points). Consider data quality issues."
                )

            # Compare univariate vs multivariate methods
            univariate_count = sum(
                summary['outlier_counts'].get(m, 0)
                for m in ['iqr', 'zscore']
            )
            multivariate_count = sum(
                summary['outlier_counts'].get(m, 0)
                for m in ['isolation_forest', 'lof']
            )

            if multivariate_count > univariate_count * 1.5:
                summary['recommendations'].append(
                    "Multivariate methods detect more outliers. "
                    "Consider relationships between variables."
                )

        return summary

    def handle_outliers(self, method: str = 'remove',
                       method_used: str = 'iqr',
                       columns: Optional[List[str]] = None,
                       replacement_value: Union[str, float] = 'median') -> pd.DataFrame:
        """
        Handle outliers based on detection method and treatment strategy.

        Args:
            method: Treatment method ('remove', 'clip', 'replace')
            method_used: Detection method whose outliers to handle
            columns: Columns to process (default: all with outliers)
            replacement_value: Value to use for replacement ('median', 'mean', or custom value)

        Returns:
            DataFrame with outliers handled
        """
        if method_used not in self.outlier_results:
            raise ValueError(f"Outlier detection method '{method_used}' not applied yet")

        data_clean = self.data.copy()

        if method_used in ['iqr', 'zscore']:
            outlier_info = self.outlier_results[method_used]['outliers_by_column']

            for col in outlier_info:
                if columns and col not in columns:
                    continue

                outlier_indices = outlier_info[col]['indices']

                if method == 'remove':
                    data_clean = data_clean.drop(outlier_indices)
                elif method == 'clip':
                    bounds = outlier_info[col]['bounds']
                    lower = bounds['lower']
                    upper = bounds['upper']
                    data_clean[col] = data_clean[col].clip(lower=lower, upper=upper)
                elif method == 'replace':
                    if replacement_value == 'median':
                        value = data_clean[col].median()
                    elif replacement_value == 'mean':
                        value = data_clean[col].mean()
                    else:
                        value = replacement_value

                    data_clean.loc[outlier_indices, col] = value

        else:  # multivariate methods
            outlier_indices = self.outlier_results[method_used]['outliers']['indices']

            if method == 'remove':
                data_clean = data_clean.drop(outlier_indices)
            elif method == 'replace':
                # For multivariate, replace with column medians
                for col in data_clean.select_dtypes(include=[np.number]).columns:
                    if columns and col not in columns:
                        continue
                    value = data_clean[col].median()
                    data_clean.loc[outlier_indices, col] = value

        # Record treatment
        self.treatment_history.append({
            'method': method,
            'method_used': method_used,
            'columns': columns,
            'replacement_value': replacement_value,
            'original_shape': self.data.shape,
            'new_shape': data_clean.shape,
            'timestamp': datetime.now().isoformat()
        })

        return data_clean

    def visualize_outliers(self, method_used: str = 'iqr',
                          column: Optional[str] = None,
                          save_plot: bool = False) -> Optional[str]:
        """
        Create visualization of outliers.

        Args:
            method_used: Detection method to visualize
            column: Specific column to plot (for univariate methods)
            save_plot: Whether to save the plot as base64 string

        Returns:
            Base64 encoded plot if save_plot=True, None otherwise
        """
        if method_used not in self.outlier_results:
            raise ValueError(f"Outlier detection method '{method_used}' not applied yet")

        if method_used in ['iqr', 'zscore']:
            # For univariate methods, create subplot for each column
            outlier_info = self.outlier_results[method_used]['outliers_by_column']

            if column:
                outlier_info = {column: outlier_info[column]}

            n_cols = len(outlier_info)
            if n_cols == 0:
                plt.close()
                return None

            # Create subplots
            rows = (n_cols + 1) // 2
            fig, axes = plt.subplots(rows, 2, figsize=(15, 5*rows))

            # Handle axes array shape
            # matplotlib returns different shapes based on subplot configuration
            if not isinstance(axes, np.ndarray):
                # Single axes object (when rows=1 and cols=1)
                axes = np.array([[axes]])
            elif axes.ndim == 1:
                # 1D array (when rows=1 or cols=1 but not both)
                if rows == 1:
                    axes = axes.reshape(1, -1)
                else:
                    axes = axes.reshape(-1, 1)
            # axes is already 2D, no reshaping needed

            for idx, (col, info) in enumerate(outlier_info.items()):
                row = idx // 2
                col_idx = idx % 2
                ax = axes[row, col_idx]

                # Plot data
                data_col = self.data[col].dropna()
                ax.hist(data_col.values, bins=50, alpha=0.7, color='skyblue', label='Data')

                # Highlight outliers
                outlier_values = info['values']
                if outlier_values:
                    ax.hist(outlier_values, bins=50, alpha=0.7, color='red', label='Outliers')

                ax.set_title(f'{col} - {method_used.title()} Outliers')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.legend()

                # Add bounds for IQR method
                if method_used == 'iqr' and 'bounds' in info:
                    bounds = info['bounds']
                    ax.axvline(bounds['lower'], color='orange', linestyle='--',
                              label=f'Lower bound: {bounds["lower"]:.2f}')
                    ax.axvline(bounds['upper'], color='orange', linestyle='--',
                              label=f'Upper bound: {bounds["upper"]:.2f}')
                    ax.legend()

            # Hide empty subplots
            for idx in range(n_cols, rows * 2):
                row = idx // 2
                col_idx = idx % 2
                axes[row, col_idx].set_visible(False)

            fig.tight_layout()

            if save_plot:
                buffer = BytesIO()
                fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                encoded = base64.b64encode(buffer.getvalue())
                # Handle both bytes and string return types
                if isinstance(encoded, bytes):
                    plot_data = encoded.decode()
                else:
                    plot_data = encoded
                plt.close(fig)
                return plot_data
            else:
                plt.show()
                plt.close(fig)
                return None

        else:  # multivariate methods
            # Create scatter plot for first two dimensions
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                plt.close()
                return None

            fig, ax = plt.subplots(figsize=(10, 8))
            outlier_indices = set(self.outlier_results[method_used]['outliers']['indices'])

            # Plot normal points
            normal_mask = ~self.data.index.isin(outlier_indices)
            ax.scatter(
                self.data.loc[normal_mask, numeric_cols[0]],
                self.data.loc[normal_mask, numeric_cols[1]],
                c='blue', alpha=0.6, label='Normal', s=50
            )

            # Plot outliers
            if outlier_indices:
                ax.scatter(
                    self.data.loc[list(outlier_indices), numeric_cols[0]],
                    self.data.loc[list(outlier_indices), numeric_cols[1]],
                    c='red', alpha=0.8, label='Outliers', s=50
                )

            ax.set_xlabel(numeric_cols[0])
            ax.set_ylabel(numeric_cols[1])
            ax.set_title(f'{method_used.title()} Outliers')
            ax.legend()

            if save_plot:
                buffer = BytesIO()
                fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                encoded = base64.b64encode(buffer.getvalue())
                # Handle both bytes and string return types
                if isinstance(encoded, bytes):
                    plot_data = encoded.decode()
                else:
                    plot_data = encoded
                plt.close(fig)
                return plot_data
            else:
                plt.show()
                plt.close(fig)
                return None

    def export_results(self, format: str = 'json') -> Union[str, Dict[str, Any]]:
        """
        Export outlier detection results.

        Args:
            format: Export format ('json' or 'dict')

        Returns:
            Results in specified format
        """
        results = {
            'summary': self.get_outlier_summary(),
            'detailed_results': self.outlier_results,
            'treatment_history': self.treatment_history,
            'parameters': {
                'z_threshold': self.z_threshold,
                'iqr_factor': self.iqr_factor,
                'contamination': self.contamination
            }
        }

        if format == 'json':
            import json
            return json.dumps(results, indent=2, default=str)
        else:
            return results