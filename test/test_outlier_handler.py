"""Unit tests for OutlierHandler class."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.outliers.outlier_handler import OutlierHandler


class TestOutlierHandler:
    """Test cases for OutlierHandler class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with outliers."""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)

        # Add some outliers
        outlier_indices = np.random.choice(100, 10, replace=False)
        normal_data[outlier_indices] = np.random.uniform(-5, 5, 10)

        return pd.DataFrame({
            'feature1': normal_data,
            'feature2': np.random.normal(5, 2, 100),
            'feature3': np.random.exponential(1, 100),
            'categorical': np.random.choice(['A', 'B', 'C'], 100)
        })

    @pytest.fixture
    def handler(self, sample_data):
        """Create an OutlierHandler instance for testing."""
        return OutlierHandler(sample_data)

    def test_initialization(self, sample_data):
        """Test OutlierHandler initialization."""
        handler = OutlierHandler(sample_data)

        assert isinstance(handler.data, pd.DataFrame)
        assert len(handler.data) == 100
        assert list(handler.data.columns) == ['feature1', 'feature2', 'feature3', 'categorical']
        assert handler.z_threshold == 3.0
        assert handler.iqr_factor == 1.5
        assert handler.contamination == 'auto'
        assert handler.outlier_results == {}
        assert handler.treatment_history == []

    def test_initialization_with_numpy_array(self):
        """Test initialization with numpy array."""
        data = np.random.normal(0, 1, (50, 3))
        handler = OutlierHandler(data)

        assert isinstance(handler.data, pd.DataFrame)
        assert handler.data.shape == (50, 3)

    def test_initialization_custom_parameters(self, sample_data):
        """Test initialization with custom parameters."""
        handler = OutlierHandler(
            sample_data,
            z_threshold=2.5,
            iqr_factor=2.0,
            contamination=0.1
        )

        assert handler.z_threshold == 2.5
        assert handler.iqr_factor == 2.0
        assert handler.contamination == 0.1

    def test_detect_iqr_outliers(self, handler):
        """Test IQR outlier detection."""
        results = handler.detect_iqr_outliers()

        assert isinstance(results, dict)
        assert results['method'] == 'IQR'
        assert 'parameters' in results
        assert 'outliers_by_column' in results
        assert 'summary' in results
        assert 'timestamp' in results

        # Check structure
        assert results['parameters']['iqr_factor'] == 1.5
        assert isinstance(results['summary']['total_columns'], int)
        assert isinstance(results['summary']['columns_with_outliers'], int)
        assert isinstance(results['summary']['total_outliers'], int)

        # Check results per column
        for col in ['feature1', 'feature2', 'feature3']:
            assert col in results['outliers_by_column']
            col_result = results['outliers_by_column'][col]

            assert 'count' in col_result
            assert 'percentage' in col_result
            assert 'indices' in col_result
            assert 'values' in col_result
            assert 'bounds' in col_result

            bounds = col_result['bounds']
            assert 'lower' in bounds
            assert 'upper' in bounds
            assert 'Q1' in bounds
            assert 'Q3' in bounds
            assert 'IQR' in bounds
            assert bounds['lower'] < bounds['upper']
            assert bounds['Q1'] < bounds['Q3']
            assert bounds['IQR'] == bounds['Q3'] - bounds['Q1']

    def test_detect_iqr_outliers_specific_columns(self, handler):
        """Test IQR detection on specific columns."""
        columns = ['feature1']
        results = handler.detect_iqr_outliers(columns=columns)

        assert len(results['outliers_by_column']) == 1
        assert 'feature1' in results['outliers_by_column']
        assert results['summary']['total_columns'] == 1

    def test_detect_zscore_outliers(self, handler):
        """Test Z-score outlier detection."""
        results = handler.detect_zscore_outliers()

        assert isinstance(results, dict)
        assert results['method'] == 'Z-Score'
        assert 'parameters' in results
        assert 'outliers_by_column' in results
        assert 'summary' in results

        # Check parameters
        assert results['parameters']['threshold'] == 3.0

        # Check results per column
        for col in ['feature1', 'feature2', 'feature3']:
            assert col in results['outliers_by_column']
            col_result = results['outliers_by_column'][col]

            assert 'count' in col_result
            assert 'percentage' in col_result
            assert 'indices' in col_result
            assert 'values' in col_result
            assert 'z_scores' in col_result
            assert 'statistics' in col_result

            # Check statistics
            stats = col_result['statistics']
            assert 'mean' in stats
            assert 'std' in stats
            assert 'min_z' in stats
            assert 'max_z' in stats
            assert stats['std'] > 0
            assert stats['min_z'] >= 0

    def test_detect_zscore_outliers_custom_threshold(self, handler):
        """Test Z-score detection with custom threshold."""
        results = handler.detect_zscore_outliers(z_threshold=2.0)
        assert results['parameters']['threshold'] == 2.0

    def test_detect_isolation_forest_outliers(self, handler):
        """Test Isolation Forest outlier detection."""
        results = handler.detect_isolation_forest_outliers()

        assert isinstance(results, dict)
        assert results['method'] == 'Isolation Forest'
        assert 'parameters' in results
        assert 'outliers' in results
        assert 'model_info' in results
        assert 'timestamp' in results

        # Check parameters
        params = results['parameters']
        assert 'contamination' in params
        assert 'n_estimators' in params
        assert params['n_estimators'] == 100

        # Check outliers
        outliers = results['outliers']
        assert 'count' in outliers
        assert 'percentage' in outliers
        assert 'indices' in outliers
        assert 'scores' in outliers
        assert isinstance(outliers['count'], int)
        assert 0 <= outliers['percentage'] <= 100

    def test_detect_isolation_forest_outliers_custom_contamination(self, handler):
        """Test Isolation Forest with custom contamination."""
        results = handler.detect_isolation_forest_outliers(contamination=0.05)
        assert results['parameters']['contamination'] == 0.05

    def test_detect_lof_outliers(self, handler):
        """Test Local Outlier Factor detection."""
        results = handler.detect_lof_outliers()

        assert isinstance(results, dict)
        assert results['method'] == 'Local Outlier Factor'
        assert 'parameters' in results
        assert 'outliers' in results
        assert 'model_info' in results

        # Check parameters
        params = results['parameters']
        assert 'n_neighbors' in params
        assert params['n_neighbors'] == 20

        # Check outliers
        outliers = results['outliers']
        assert 'count' in outliers
        assert 'percentage' in outliers
        assert 'lof_scores' in outliers
        assert 'average_lof' in outliers

    def test_get_outlier_summary(self, handler):
        """Test outlier summary generation."""
        # Run all detection methods
        handler.detect_iqr_outliers()
        handler.detect_zscore_outliers()
        handler.detect_isolation_forest_outliers()
        handler.detect_lof_outliers()

        summary = handler.get_outlier_summary()

        assert isinstance(summary, dict)
        assert 'dataset_info' in summary
        assert 'methods_applied' in summary
        assert 'outlier_counts' in summary
        assert 'recommendations' in summary
        assert 'timestamp' in summary

        # Check dataset info
        dataset_info = summary['dataset_info']
        assert 'shape' in dataset_info
        assert 'columns' in dataset_info
        assert 'numeric_columns' in dataset_info
        assert dataset_info['shape'] == (100, 4)

        # Check methods applied
        methods = summary['methods_applied']
        assert 'iqr' in methods
        assert 'zscore' in methods
        assert 'isolation_forest' in methods
        assert 'lof' in methods

        # Check outlier counts
        counts = summary['outlier_counts']
        assert 'iqr' in counts
        assert 'zscore' in counts

    def test_handle_outliers_remove(self, handler):
        """Test outlier removal."""
        # First detect outliers
        handler.detect_iqr_outliers()

        # Remove outliers
        clean_data = handler.handle_outliers(method='remove', method_used='iqr')

        assert isinstance(clean_data, pd.DataFrame)
        assert len(clean_data) < len(handler.data)  # Should have fewer rows

        # Check treatment history
        assert len(handler.treatment_history) == 1
        treatment = handler.treatment_history[0]
        assert treatment['method'] == 'remove'
        assert treatment['method_used'] == 'iqr'
        assert treatment['original_shape'] == handler.data.shape
        assert treatment['new_shape'] == clean_data.shape

    def test_handle_outliers_clip(self, handler):
        """Test outlier clipping."""
        handler.detect_iqr_outliers()

        clean_data = handler.handle_outliers(method='clip', method_used='iqr')

        assert isinstance(clean_data, pd.DataFrame)
        assert len(clean_data) == len(handler.data)  # Same number of rows

        # Values should be clipped
        iqr_results = handler.outlier_results['iqr']
        for col in iqr_results['outliers_by_column']:
            bounds = iqr_results['outliers_by_column'][col]['bounds']
            assert clean_data[col].min() >= bounds['lower']
            assert clean_data[col].max() <= bounds['upper']

    def test_handle_outliers_replace_median(self, handler):
        """Test outlier replacement with median."""
        handler.detect_iqr_outliers()

        clean_data = handler.handle_outliers(
            method='replace',
            method_used='iqr',
            replacement_value='median'
        )

        assert isinstance(clean_data, pd.DataFrame)
        assert len(clean_data) == len(handler.data)

    def test_handle_outliers_replace_mean(self, handler):
        """Test outlier replacement with mean."""
        handler.detect_iqr_outliers()

        clean_data = handler.handle_outliers(
            method='replace',
            method_used='iqr',
            replacement_value='mean'
        )

        assert isinstance(clean_data, pd.DataFrame)
        assert len(clean_data) == len(handler.data)

    def test_handle_outliers_replace_custom_value(self, handler):
        """Test outlier replacement with custom value."""
        handler.detect_iqr_outliers()

        clean_data = handler.handle_outliers(
            method='replace',
            method_used='iqr',
            replacement_value=0.0
        )

        assert isinstance(clean_data, pd.DataFrame)
        assert len(clean_data) == len(handler.data)

    def test_handle_outliers_multivariate_remove(self, handler):
        """Test removing multivariate outliers."""
        handler.detect_isolation_forest_outliers()

        clean_data = handler.handle_outliers(
            method='remove',
            method_used='isolation_forest'
        )

        assert isinstance(clean_data, pd.DataFrame)
        assert len(clean_data) <= len(handler.data)

    def test_handle_outliers_error_no_detection(self, handler):
        """Test error when handling outliers without detection."""
        with pytest.raises(ValueError, match="not applied yet"):
            handler.handle_outliers(method='remove', method_used='iqr')

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.savefig')
    def test_visualize_outliers_save_plot(self, mock_savefig, mock_close, mock_subplots, handler):
        """Test outlier visualization with plot saving."""
        handler.detect_iqr_outliers()

        # Mock the figure and axes
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        # For single column with 2 subplot layout (1 row, 2 cols)
        mock_axes_array = np.array([mock_ax1, mock_ax2])  # 1D array with 2 axes
        mock_subplots.return_value = (mock_fig, mock_axes_array)

        # Configure both axes mocks
        for mock_ax in [mock_ax1, mock_ax2]:
            mock_ax.hist.return_value = None
            mock_ax.axvline.return_value = None
            mock_ax.set_title.return_value = None
            mock_ax.set_xlabel.return_value = None
            mock_ax.set_ylabel.return_value = None
            mock_ax.legend.return_value = None
            mock_ax.set_visible.return_value = None

        mock_fig.tight_layout.return_value = None
        mock_fig.savefig.return_value = None

        # Mock the save functionality
        mock_buffer = Mock()
        mock_buffer.getvalue.return_value = b'fake_image_data'
        mock_buffer.seek.return_value = None
        with patch('src.outliers.outlier_handler.BytesIO', return_value=mock_buffer):
            with patch('base64.b64encode') as mock_b64:
                mock_b64.return_value = b'fake_base64_data'.decode()

                plot_data = handler.visualize_outliers(
                    method_used='iqr',
                    column='feature1',
                    save_plot=True
                )

                assert isinstance(plot_data, str)
                assert len(plot_data) > 0

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_visualize_outliers_show_plot(self, mock_close, mock_show, mock_subplots, handler):
        """Test outlier visualization without saving."""
        handler.detect_iqr_outliers()

        # Mock the figure and axes
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        # For single column with 2 subplot layout (1 row, 2 cols)
        mock_axes_array = np.array([mock_ax1, mock_ax2])  # 1D array with 2 axes
        mock_subplots.return_value = (mock_fig, mock_axes_array)

        # Configure both axes mocks
        for mock_ax in [mock_ax1, mock_ax2]:
            mock_ax.hist.return_value = None
            mock_ax.axvline.return_value = None
            mock_ax.set_title.return_value = None
            mock_ax.set_xlabel.return_value = None
            mock_ax.set_ylabel.return_value = None
            mock_ax.legend.return_value = None
            mock_ax.set_visible.return_value = None

        mock_fig.tight_layout.return_value = None
        mock_fig.savefig.return_value = None

        # Should not return anything when not saving
        result = handler.visualize_outliers(
            method_used='iqr',
            column='feature1',
            save_plot=False
        )
        assert result is None

    def test_visualize_outliers_multivariate(self, handler):
        """Test multivariate outlier visualization."""
        handler.detect_isolation_forest_outliers()

        plot_data = handler.visualize_outliers(
            method_used='isolation_forest',
            save_plot=True
        )

        assert isinstance(plot_data, str)
        assert len(plot_data) > 0

    def test_visualize_outliers_error_no_detection(self, handler):
        """Test error when visualizing without detection."""
        with pytest.raises(ValueError, match="not applied yet"):
            handler.visualize_outliers(method_used='iqr')

    def test_export_results_dict(self, handler):
        """Test exporting results as dictionary."""
        handler.detect_iqr_outliers()
        handler.detect_zscore_outliers()

        results = handler.export_results(format='dict')

        assert isinstance(results, dict)
        assert 'summary' in results
        assert 'detailed_results' in results
        assert 'treatment_history' in results
        assert 'parameters' in results

        # Check parameters
        params = results['parameters']
        assert params['z_threshold'] == 3.0
        assert params['iqr_factor'] == 1.5
        assert params['contamination'] == 'auto'

    def test_export_results_json(self, handler):
        """Test exporting results as JSON."""
        handler.detect_iqr_outliers()

        json_str = handler.export_results(format='json')

        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Try to parse JSON
        import json
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_edge_case_empty_data(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()
        handler = OutlierHandler(empty_data)

        results = handler.detect_iqr_outliers()
        assert results['summary']['total_columns'] == 0
        assert results['summary']['columns_with_outliers'] == 0
        assert results['summary']['total_outliers'] == 0

    def test_edge_case_single_column(self):
        """Test handling of single column data."""
        data = pd.DataFrame({'x': [1, 2, 3, 100, 4, 5]})
        handler = OutlierHandler(data)

        results = handler.detect_iqr_outliers()
        assert len(results['outliers_by_column']) == 1
        assert 'x' in results['outliers_by_column']
        assert results['outliers_by_column']['x']['count'] >= 1

    def test_edge_case_all_same_values(self):
        """Test handling of data with no variation."""
        data = pd.DataFrame({'x': [5] * 100})
        handler = OutlierHandler(data)

        results = handler.detect_iqr_outliers()
        # Should handle gracefully without errors
        assert isinstance(results, dict)
        assert results['method'] == 'IQR'

    def test_edge_case_missing_values(self):
        """Test handling of missing values."""
        data = pd.DataFrame({
            'x': [1, 2, np.nan, 100, 4, 5, np.nan],
            'y': [10, np.nan, 30, 40, 50, np.nan, 70]
        })
        handler = OutlierHandler(data)

        results = handler.detect_iqr_outliers()
        # Should handle missing values gracefully
        assert isinstance(results, dict)
        assert len(results['outliers_by_column']) == 2

    def test_isolation_forest_no_data_error(self):
        """Test Isolation Forest error with no data."""
        data = pd.DataFrame({'x': [np.nan, np.nan, np.nan]})
        handler = OutlierHandler(data)

        results = handler.detect_isolation_forest_outliers()
        assert 'error' in results
        assert 'No data available' in results['error']

    def test_lof_no_data_error(self):
        """Test LOF error with no data."""
        data = pd.DataFrame({'x': [np.nan, np.nan, np.nan]})
        handler = OutlierHandler(data)

        results = handler.detect_lof_outliers()
        assert 'error' in results
        assert 'No data available' in results['error']