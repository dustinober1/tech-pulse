"""Unit tests for MissingDataAnalyzer class."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.missing_data.missing_data_analyzer import MissingDataAnalyzer


class TestMissingDataAnalyzer:
    """Test cases for MissingDataAnalyzer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with missing values."""
        np.random.seed(42)

        # Create base data
        n_samples = 200
        data = {
            'complete_var': np.random.normal(0, 1, n_samples),
            'few_missing': np.random.normal(5, 2, n_samples),
            'moderate_missing': np.random.exponential(1, n_samples),
            'high_missing': np.random.normal(10, 3, n_samples),
            'categorical_var': np.random.choice(['A', 'B', 'C'], n_samples)
        }

        df = pd.DataFrame(data)

        # Introduce missing values with different patterns
        # Few missing (5%)
        few_missing_indices = np.random.choice(n_samples, int(0.05 * n_samples), replace=False)
        df.loc[few_missing_indices, 'few_missing'] = np.nan

        # Moderate missing (20%)
        moderate_missing_indices = np.random.choice(n_samples, int(0.2 * n_samples), replace=False)
        df.loc[moderate_missing_indices, 'moderate_missing'] = np.nan

        # High missing (40%)
        high_missing_indices = np.random.choice(n_samples, int(0.4 * n_samples), replace=False)
        df.loc[high_missing_indices, 'high_missing'] = np.nan

        # Some categorical missing (10%)
        cat_missing_indices = np.random.choice(n_samples, int(0.1 * n_samples), replace=False)
        df.loc[cat_missing_indices, 'categorical_var'] = np.nan

        # Add MAR pattern - missingness in one variable depends on another
        mar_condition = df['complete_var'] > 0.5
        mar_indices = df[mar_condition].sample(frac=0.3).index
        df.loc[mar_indices, 'few_missing'] = np.nan

        return df

    @pytest.fixture
    def mcar_data(self):
        """Create data with MCAR missingness."""
        np.random.seed(123)
        n_samples = 100
        data = np.random.normal(0, 1, (n_samples, 3))

        # Random missing values
        mask = np.random.random(data.shape) < 0.1  # 10% missing completely at random
        data[mask] = np.nan

        return pd.DataFrame(data, columns=['var1', 'var2', 'var3'])

    @pytest.fixture
    def monotone_data(self):
        """Create data with monotone missing pattern."""
        np.random.seed(456)
        n_samples = 50

        data = {
            'var1': np.random.normal(0, 1, n_samples),
            'var2': np.random.normal(0, 1, n_samples),
            'var3': np.random.normal(0, 1, n_samples)
        }

        df = pd.DataFrame(data)

        # Create monotone pattern
        # var2 missing after row 30
        df.loc[30:, 'var2'] = np.nan
        # var3 missing after row 20
        df.loc[20:, 'var3'] = np.nan

        return df

    @pytest.fixture
    def analyzer(self, sample_data):
        """Create a MissingDataAnalyzer instance for testing."""
        return MissingDataAnalyzer(sample_data)

    def test_initialization(self, sample_data):
        """Test MissingDataAnalyzer initialization."""
        analyzer = MissingDataAnalyzer(sample_data)

        assert isinstance(analyzer.data, pd.DataFrame)
        assert analyzer.data.shape == sample_data.shape
        assert analyzer.original_shape == sample_data.shape
        assert analyzer.missing_summary == {}
        assert analyzer.missingness_tests == {}
        assert analyzer.imputation_history == []
        assert analyzer.missingness_mechanism == {}

    def test_initialization_with_numpy_array(self):
        """Test initialization with numpy array."""
        data = np.random.normal(0, 1, (50, 3))
        analyzer = MissingDataAnalyzer(data)

        assert isinstance(analyzer.data, pd.DataFrame)
        assert analyzer.data.shape == (50, 3)
        assert list(analyzer.data.columns) == [0, 1, 2]

    def test_analyze_missing_patterns(self, analyzer):
        """Test missing pattern analysis."""
        summary = analyzer.analyze_missing_patterns()

        assert isinstance(summary, dict)
        assert 'overall' in summary
        assert 'by_column' in summary
        assert 'by_row' in summary
        assert 'patterns' in summary
        assert 'correlations' in summary

        # Check overall statistics
        overall = summary['overall']
        assert 'total_cells' in overall
        assert 'total_missing' in overall
        assert 'missing_percentage' in overall
        assert 'complete_cases' in overall
        assert overall['total_missing'] > 0
        assert 0 <= overall['missing_percentage'] <= 100

        # Check column statistics
        by_column = summary['by_column']
        assert len(by_column) == len(analyzer.data.columns)

        for col, stats in by_column.items():
            assert 'missing_count' in stats
            assert 'missing_percentage' in stats
            assert 'data_type' in stats
            assert 'unique_values' in stats
            assert 'has_missing' in stats

        # Check row statistics
        by_row = summary['by_row']
        assert 'mean_missing_per_row' in by_row
        assert 'max_missing_in_row' in by_row
        assert 'rows_with_missing' in by_row

    def test_analyze_missing_patterns_complete_data(self):
        """Test analysis with complete data (no missing values)."""
        complete_data = pd.DataFrame({
            'x': range(100),
            'y': np.random.normal(0, 1, 100)
        })
        analyzer = MissingDataAnalyzer(complete_data)

        summary = analyzer.analyze_missing_patterns()
        assert summary['overall']['total_missing'] == 0
        assert summary['overall']['missing_percentage'] == 0
        assert summary['overall']['complete_cases'] == 100

    def test_test_missingness_mechanism(self, analyzer):
        """Test missingness mechanism testing."""
        results = analyzer.test_missingness_mechanism()

        assert isinstance(results, dict)
        assert 'little_mcar_test' in results
        assert 't_tests' in results
        assert 'correlation_tests' in results
        assert 'pattern_analysis' in results
        assert 'conclusion' in results

        # Check conclusion structure
        conclusion = results['conclusion']
        assert 'likely_mechanism' in conclusion
        assert conclusion['likely_mechanism'] in ['MCAR', 'MAR', 'MNAR']
        assert 'is_mcar' in conclusion
        assert 'is_mar' in conclusion
        assert 'is_mnar' in conclusion
        assert 'confidence' in conclusion

    def test_test_missingness_mechanism_complete_data(self):
        """Test missingness mechanism with complete data."""
        complete_data = pd.DataFrame({'x': range(50)})
        analyzer = MissingDataAnalyzer(complete_data)

        results = analyzer.test_missingness_mechanism()
        assert results['little_mcar_test'] is None  # No missing to test

    def test_test_missingness_mechanism_empty_data(self):
        """Test missingness mechanism with empty data."""
        empty_data = pd.DataFrame()
        analyzer = MissingDataAnalyzer(empty_data)

        results = analyzer.test_missingness_mechanism()
        assert 'error' in results

    def test_detect_monotone_patterns(self, monotone_data):
        """Test monotone pattern detection."""
        analyzer = MissingDataAnalyzer(monotone_data)
        monotone_cols = analyzer._detect_monotone_patterns()

        # var2 and var3 should be detected as monotone
        assert 'var2' in monotone_cols
        assert 'var3' in monotone_cols

    def test_suggest_imputation_methods(self, analyzer):
        """Test imputation method suggestions."""
        suggestions = analyzer.suggest_imputation_methods()

        assert isinstance(suggestions, dict)

        # Should have suggestions for columns with missing values
        missing_cols = analyzer.data.columns[analyzer.data.isnull().any()]

        for col in missing_cols:
            assert col in suggestions
            col_suggestions = suggestions[col]

            assert 'missing_percentage' in col_suggestions
            assert 'data_type' in col_suggestions
            assert 'suggested_methods' in col_suggestions
            assert 'primary_recommendation' in col_suggestions

            # Check that methods are valid
            valid_methods = [
                'mean_median_mode', 'knn', 'iterative', 'multiple_imputation',
                'model_based', 'mean_median', 'mode', 'frequent_category',
                'drop_column', 'flag_missing'
            ]

            for suggestion in col_suggestions['suggested_methods']:
                assert suggestion['method'] in valid_methods
                assert 'confidence' in suggestion
                assert 'reason' in suggestion

    def test_apply_imputation_mean(self, analyzer):
        """Test mean imputation."""
        imputed_data = analyzer.apply_imputation('mean')

        assert isinstance(imputed_data, pd.DataFrame)
        assert imputed_data.shape[1] == analyzer.data.shape[1]

        # Check that numeric columns have no missing values
        numeric_cols = imputed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not imputed_data[col].isnull().any()

        # Check imputation history
        assert len(analyzer.imputation_history) == 1
        record = analyzer.imputation_history[0]
        assert record['method'] == 'mean'

    def test_apply_imputation_median(self, analyzer):
        """Test median imputation."""
        imputed_data = analyzer.apply_imputation('median')

        assert isinstance(imputed_data, pd.DataFrame)

        # Check numeric columns
        numeric_cols = imputed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if analyzer.data[col].isnull().any():
                assert not imputed_data[col].isnull().any()

        assert analyzer.imputation_history[-1]['method'] == 'median'

    def test_apply_imputation_mode(self, analyzer):
        """Test mode imputation."""
        imputed_data = analyzer.apply_imputation('mode')

        assert isinstance(imputed_data, pd.DataFrame)

        # Check that specified columns have no missing values
        for col in imputed_data.columns:
            if analyzer.data[col].isnull().any():
                assert not imputed_data[col].isnull().any()

        assert analyzer.imputation_history[-1]['method'] == 'mode'

    def test_apply_imputation_specific_columns(self, analyzer):
        """Test imputation on specific columns."""
        columns = ['moderate_missing', 'high_missing']
        imputed_data = analyzer.apply_imputation('mean', columns=columns)

        # Check only specified columns were imputed
        for col in columns:
            assert not imputed_data[col].isnull().any()

        # Check other columns still have missing values
        other_cols = [c for c in analyzer.data.columns if c not in columns]
        for col in other_cols:
            if analyzer.data[col].isnull().any():
                assert imputed_data[col].isnull().sum() == analyzer.data[col].isnull().sum()

    def test_apply_imputation_knn(self, analyzer):
        """Test KNN imputation."""
        imputed_data = analyzer.apply_imputation('knn', n_neighbors=3)

        assert isinstance(imputed_data, pd.DataFrame)
        assert analyzer.imputation_history[-1]['method'] == 'knn'
        assert analyzer.imputation_history[-1]['parameters']['n_neighbors'] == 3

    def test_apply_imputation_iterative(self, analyzer):
        """Test iterative imputation."""
        imputed_data = analyzer.apply_imputation('iterative', max_iter=5)

        assert isinstance(imputed_data, pd.DataFrame)
        assert analyzer.imputation_history[-1]['method'] == 'iterative'
        assert analyzer.imputation_history[-1]['parameters']['max_iter'] == 5

    def test_apply_imputation_drop(self, analyzer):
        """Test dropping missing values."""
        imputed_data = analyzer.apply_imputation('drop')

        assert isinstance(imputed_data, pd.DataFrame)
        assert len(imputed_data) <= len(analyzer.data)
        assert not imputed_data.isnull().any().any()  # Should have no missing values

        assert analyzer.imputation_history[-1]['method'] == 'drop'

    def test_apply_imputation_constant(self, analyzer):
        """Test constant value imputation."""
        fill_value = -999
        imputed_data = analyzer.apply_imputation('constant', fill_value=fill_value)

        assert isinstance(imputed_data, pd.DataFrame)

        # Check that missing values were filled with the constant
        for col in analyzer.data.columns:
            if analyzer.data[col].isnull().any():
                missing_mask = analyzer.data[col].isnull()
                filled_values = imputed_data[col][missing_mask]
                assert (filled_values == fill_value).all()

        assert analyzer.imputation_history[-1]['method'] == 'constant'

    def test_apply_imputation_no_missing(self):
        """Test imputation on complete data."""
        complete_data = pd.DataFrame({'x': range(100)})
        analyzer = MissingDataAnalyzer(complete_data)

        imputed_data = analyzer.apply_imputation('mean')

        assert imputed_data.equals(complete_data)
        assert len(analyzer.imputation_history) == 0  # No imputation recorded

    def test_apply_imputation_unknown_method(self, analyzer):
        """Test error with unknown imputation method."""
        with pytest.raises(ValueError, match="Unknown imputation method"):
            analyzer.apply_imputation('unknown_method')

    def test_evaluate_imputation(self, analyzer):
        """Test imputation evaluation."""
        # Apply imputation first
        imputed_data = analyzer.apply_imputation('mean')

        # Evaluate
        evaluation = analyzer.evaluate_imputation(imputed_data=imputed_data)

        assert isinstance(evaluation, dict)
        assert 'missingness_reduction' in evaluation
        assert 'distribution_similarity' in evaluation
        assert 'correlation_preservation' in evaluation

        # Check missingness reduction
        reduction = evaluation['missingness_reduction']
        assert 'original_missing_count' in reduction
        assert 'imputed_missing_count' in reduction
        assert 'missing_reduction_percentage' in reduction
        assert reduction['imputed_missing_count'] <= reduction['original_missing_count']

        # Check distribution similarity
        similarity = evaluation['distribution_similarity']
        for col, stats in similarity.items():
            assert 'ks_statistic' in stats
            assert 'ks_p_value' in stats
            assert 'distributions_similar' in stats
            assert 'original_stats' in stats
            assert 'imputed_stats' in stats

    def test_evaluate_imputation_no_data(self, analyzer):
        """Test evaluation without imputation data."""
        result = analyzer.evaluate_imputation()
        assert 'error' in result

    def test_generate_report(self, analyzer):
        """Test report generation."""
        # Run all analyses
        analyzer.analyze_missing_patterns()
        analyzer.test_missingness_mechanism()

        report = analyzer.generate_report()

        assert isinstance(report, dict)
        assert 'dataset_info' in report
        assert 'missing_summary' in report
        assert 'missingness_analysis' in report
        assert 'imputation_recommendations' in report
        assert 'quality_indicators' in report
        assert 'actionable_insights' in report
        assert 'generated_at' in report

        # Check dataset info
        dataset_info = report['dataset_info']
        assert 'shape' in dataset_info
        assert 'columns' in dataset_info
        assert 'numeric_columns' in dataset_info
        assert 'categorical_columns' in dataset_info

        # Check quality indicators
        quality = report['quality_indicators']
        assert 'completeness' in quality
        assert 'missing_pattern_complexity' in quality
        assert 'imputation_difficulty' in quality
        assert 'overall_quality' in quality

    def test_calculate_quality_indicators(self, analyzer):
        """Test quality indicators calculation."""
        analyzer.analyze_missing_patterns()
        analyzer.test_missingness_mechanism()

        indicators = analyzer._calculate_quality_indicators()

        assert 'completeness' in indicators
        assert 'missing_pattern_complexity' in indicators
        assert 'imputation_difficulty' in indicators
        assert 'overall_quality' in indicators

        assert 0 <= indicators['completeness'] <= 100
        assert indicators['missing_pattern_complexity'] in ['Low', 'Medium', 'High']
        assert indicators['imputation_difficulty'] in ['Easy', 'Moderate', 'Challenging']
        assert indicators['overall_quality'] in ['Excellent', 'Good', 'Fair', 'Poor']

    def test_generate_insights(self, analyzer):
        """Test insights generation."""
        analyzer.analyze_missing_patterns()
        analyzer.test_missingness_mechanism()

        insights = analyzer._generate_insights()

        assert isinstance(insights, list)
        assert len(insights) > 0
        assert all(isinstance(insight, str) for insight in insights)

    def test_export_results_dict(self, analyzer):
        """Test exporting results as dictionary."""
        analyzer.analyze_missing_patterns()

        results = analyzer.export_results(format='dict')

        assert isinstance(results, dict)
        assert 'dataset_info' in results
        assert 'missing_summary' in results

    def test_export_results_json(self, analyzer):
        """Test exporting results as JSON."""
        analyzer.analyze_missing_patterns()

        json_str = analyzer.export_results(format='json')

        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Try to parse JSON
        import json
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_export_results_unknown_format(self, analyzer):
        """Test error with unknown export format."""
        with pytest.raises(ValueError, match="Unknown export format"):
            analyzer.export_results(format='unknown')

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.savefig')
    def test_visualize_missing_patterns_matrix(self, mock_savefig, mock_close, mock_figure, analyzer):
        """Test missing pattern matrix visualization."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_figure.return_value = mock_fig
        mock_fig.tight_layout.return_value = None
        mock_fig.savefig.return_value = None

        # Mock missingno
        with patch('missingno.matrix') as mock_matrix:
            mock_matrix.return_value = None

            plot_data = analyzer.visualize_missing_patterns(
                plot_type='matrix',
                save_plot=True
            )

            assert isinstance(plot_data, str)

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_visualize_missing_patterns_show(self, mock_close, mock_show, mock_figure, analyzer):
        """Test visualization without saving."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_figure.return_value = mock_fig
        mock_fig.tight_layout.return_value = None

        # Mock missingno
        with patch('missingno.bar') as mock_bar:
            mock_bar.return_value = None

            result = analyzer.visualize_missing_patterns(
                plot_type='bar',
                save_plot=False
            )

            assert result is None

    def test_visualize_missing_patterns_unknown_type(self, analyzer):
        """Test error with unknown plot type."""
        with pytest.raises(ValueError, match="Unknown plot type"):
            analyzer.visualize_missing_patterns(plot_type='unknown')

    def test_edge_case_single_column(self):
        """Test handling of single column data."""
        data = pd.DataFrame({'x': [1, 2, np.nan, 4, 5]})
        analyzer = MissingDataAnalyzer(data)

        summary = analyzer.analyze_missing_patterns()
        assert len(summary['by_column']) == 1
        assert summary['by_column']['x']['missing_count'] == 1

    def test_edge_case_all_missing(self):
        """Test handling of completely missing data."""
        data = pd.DataFrame({
            'x': [np.nan] * 100,
            'y': [np.nan] * 100
        })
        analyzer = MissingDataAnalyzer(data)

        summary = analyzer.analyze_missing_patterns()
        assert summary['overall']['total_missing'] == 200
        assert summary['overall']['complete_cases'] == 0

    def test_edge_case_mixed_data_types(self):
        """Test handling of mixed data types."""
        data = pd.DataFrame({
            'numeric': [1, 2, np.nan, 4, 5],
            'categorical': ['A', 'B', None, 'D', 'E'],
            'datetime': pd.date_range('2020-01-01', periods=5),
            'boolean': [True, False, True, None, False]
        })
        analyzer = MissingDataAnalyzer(data)

        summary = analyzer.analyze_missing_patterns()
        assert len(summary['by_column']) == 4
        assert summary['by_column']['numeric']['has_missing']
        assert summary['by_column']['categorical']['has_missing']
        assert summary['by_column']['boolean']['has_missing']