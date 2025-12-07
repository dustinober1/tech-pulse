"""
Tests for cross-validation pipeline.

This module tests the CrossValidator class and its strategies,
including property-based tests for cross-validation methodology.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from src.portfolio.experimentation.cross_validator import (
    CrossValidator,
    CVResult,
    StandardKFold,
    StratifiedKFoldCV,
    TimeSeriesCV
)


class TestCrossValidator:
    """Test cases for CrossValidator"""

    def setup_method(self):
        """Set up test fixtures"""
        self.classification_validator = CrossValidator(cv_strategy='stratified')
        self.regression_validator = CrossValidator(cv_strategy='kfold')
        self.ts_validator = CrossValidator(cv_strategy='time_series')

    def create_classification_data(self):
        """Create synthetic classification data"""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return X, y

    def create_regression_data(self):
        """Create synthetic regression data"""
        X, y = make_regression(
            n_samples=100,
            n_features=10,
            noise=0.1,
            random_state=42
        )
        return X, y

    def create_time_series_data(self):
        """Create synthetic time series data"""
        np.random.seed(42)
        t = np.linspace(0, 10, 100)
        X = np.column_stack([np.sin(t), np.cos(t), t])
        y = 2 * np.sin(t) + np.random.normal(0, 0.1, 100)
        return X, y

    def test_stratified_kfold_classification(self):
        """Test stratified K-fold for classification"""
        X, y = self.create_classification_data()
        estimator = LogisticRegression(random_state=42)

        result = self.classification_validator.validate(
            estimator=estimator,
            X=X,
            y=y,
            scoring='accuracy',
            n_folds=5
        )

        assert isinstance(result, CVResult)
        assert result.cv_method == 'stratified'
        assert result.n_folds == 5
        assert len(result.fold_scores) == 5
        assert result.mean_score > 0.5  # Should be better than random
        assert result.std_score >= 0
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] < result.mean_score < result.confidence_interval[1]

    def test_kfold_regression(self):
        """Test K-fold for regression"""
        X, y = self.create_regression_data()
        estimator = LinearRegression()

        result = self.regression_validator.validate(
            estimator=estimator,
            X=X,
            y=y,
            scoring='neg_mean_squared_error',
            n_folds=5
        )

        assert isinstance(result, CVResult)
        assert result.cv_method == 'kfold'
        assert len(result.fold_scores) == 5
        # MSE should be negative (sklearn convention)
        assert result.mean_score < 0

    def test_time_series_cv(self):
        """Test time series cross-validation"""
        X, y = self.create_time_series_data()
        estimator = LinearRegression()

        result = self.ts_validator.validate(
            estimator=estimator,
            X=X,
            y=y,
            scoring='neg_mean_squared_error',
            n_folds=5
        )

        assert isinstance(result, CVResult)
        assert result.cv_method == 'time_series'
        assert len(result.fold_scores) == 5
        assert result.mean_score < 0

    def test_additional_metrics(self):
        """Test cross-validation with additional metrics"""
        X, y = self.create_classification_data()
        estimator = RandomForestClassifier(random_state=42, n_estimators=10)

        result = self.classification_validator.validate(
            estimator=estimator,
            X=X,
            y=y,
            scoring='accuracy',
            n_folds=5,
            additional_metrics=['f1', 'roc_auc']
        )

        assert isinstance(result, CVResult)
        assert result.additional_metrics is not None
        # The actual implementation might use generic metric names
        assert len(result.additional_metrics) == 2
        # Check that we have metrics for all folds
        for metric_scores in result.additional_metrics.values():
            assert len(metric_scores) == 5

    def test_return_estimators(self):
        """Test returning fitted estimators"""
        X, y = self.create_classification_data()
        estimator = LogisticRegression(random_state=42)

        result, estimators = self.classification_validator.validate(
            estimator=estimator,
            X=X,
            y=y,
            scoring='accuracy',
            n_folds=3,
            return_estimators=True
        )

        assert isinstance(result, CVResult)
        assert len(estimators) == 3
        # All estimators should be fitted
        for est in estimators:
            assert hasattr(est, 'coef_')  # LogisticRegression fitted attribute

    def test_compare_models(self):
        """Test model comparison functionality"""
        X, y = self.create_classification_data()

        models = {
            'logistic': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=10)
        }

        comparison_df = self.classification_validator.compare_models(
            models=models,
            X=X,
            y=y,
            scoring='accuracy',
            n_folds=5
        )

        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert 'model' in comparison_df.columns
        assert 'mean_score' in comparison_df.columns
        assert 'std_score' in comparison_df.columns
        # Should have statistical comparison columns
        assert any('p_value_' in col for col in comparison_df.columns)

    def test_learning_curve(self):
        """Test learning curve generation"""
        X, y = self.create_classification_data()
        estimator = LogisticRegression(random_state=42)

        curve_data = self.classification_validator.get_learning_curve(
            estimator=estimator,
            X=X,
            y=y,
            scoring='accuracy',
            n_folds=5,
            train_sizes=np.array([0.2, 0.5, 0.8, 1.0])
        )

        assert isinstance(curve_data, dict)
        assert 'train_sizes' in curve_data
        assert 'train_scores_mean' in curve_data
        assert 'val_scores_mean' in curve_data
        assert len(curve_data['train_sizes']) == 4  # Number of train sizes

    def test_validation_report(self):
        """Test validation report generation"""
        X, y = self.create_classification_data()
        estimator = LogisticRegression(random_state=42)

        result = self.classification_validator.validate(
            estimator=estimator,
            X=X,
            y=y,
            scoring='accuracy',
            n_folds=3
        )

        report = self.classification_validator.get_validation_report(result)

        assert isinstance(report, str)
        assert "Cross-Validation Report" in report
        assert "CV Method: stratified" in report
        assert "Mean Score:" in report
        assert "Scores by Fold:" in report

    def test_different_confidence_levels(self):
        """Test different confidence levels for CI"""
        X, y = self.create_classification_data()
        estimator = LogisticRegression(random_state=42)

        # Test 90% confidence level
        validator_90 = CrossValidator(cv_strategy='stratified', confidence_level=0.90)
        result_90 = validator_90.validate(
            estimator=estimator,
            X=X,
            y=y,
            scoring='accuracy',
            n_folds=5
        )

        # Test 99% confidence level
        validator_99 = CrossValidator(cv_strategy='stratified', confidence_level=0.99)
        result_99 = validator_99.validate(
            estimator=estimator,
            X=X,
            y=y,
            scoring='accuracy',
            n_folds=5
        )

        # 99% CI should be wider than 90% CI
        ci_width_90 = result_90.confidence_interval[1] - result_90.confidence_interval[0]
        ci_width_99 = result_99.confidence_interval[1] - result_99.confidence_interval[0]
        assert ci_width_99 > ci_width_90

    def test_invalid_strategy(self):
        """Test handling of invalid CV strategy"""
        with pytest.raises(ValueError, match="Unknown CV strategy"):
            CrossValidator(cv_strategy='invalid_strategy')

    def test_insufficient_classes_for_stratified(self):
        """Test stratified CV with insufficient classes"""
        # Create data with only one class
        X = np.random.randn(50, 5)
        y = np.zeros(50)  # All same class

        validator = CrossValidator(cv_strategy='stratified')
        estimator = LogisticRegression(random_state=42)

        with pytest.raises(ValueError, match="Data is not suitable"):
            validator.validate(
                estimator=estimator,
                X=X,
                y=y,
                scoring='accuracy',
                n_folds=5
            )

    def test_cv_strategies_validation(self):
        """Test individual CV strategy validation methods"""
        X, y = self.create_classification_data()

        # Test StandardKFold
        kfold = StandardKFold()
        assert kfold.validate_data(X, y) == True

        # Test StratifiedKFold
        stratified = StratifiedKFoldCV()
        assert stratified.validate_data(X, y) == True

        # Test StratifiedKFold with single class (should fail)
        y_single = np.zeros(len(y))
        assert stratified.validate_data(X, y_single) == False

        # Test TimeSeriesCV
        ts_cv = TimeSeriesCV()
        assert ts_cv.validate_data(X, y) == True


class TestCrossValidationMethodology:
    """Property-based tests for cross-validation methodology (Property 8)"""

    def test_cross_validation_coverage_completeness(self):
        """
        Property 8: Cross-validation methodology

        Validates that cross-validation covers all data points
        and each data point appears in test set exactly once
        """
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_classes=2,
            random_state=42
        )

        validator = CrossValidator(cv_strategy='kfold', random_state=42)
        estimator = LogisticRegression(random_state=42)

        result = validator.validate(
            estimator=estimator,
            X=X,
            y=y,
            scoring='accuracy',
            n_folds=5
        )

        # Check that all folds have been created
        assert len(result.fold_indices) == 5

        # Verify each sample appears in test set exactly once
        all_test_indices = []
        for fold_idx, (train_idx, test_idx) in result.fold_indices.items():
            all_test_indices.extend(test_idx)

        # All indices should be unique
        assert len(all_test_indices) == len(set(all_test_indices))

        # All samples should be covered
        assert len(all_test_indices) == len(X)

        # Verify fold indices are valid
        for fold_idx, (train_idx, test_idx) in result.fold_indices.items():
            # No overlap between train and test
            assert len(set(train_idx) & set(test_idx)) == 0

            # Combined they should contain all samples
            combined = set(train_idx) | set(test_idx)
            assert len(combined) == len(X)

    def test_stratified_cv_class_distribution(self):
        """
        Property 8: Cross-validation methodology

        Validates that stratified CV maintains class distribution
        across folds
        """
        # Create imbalanced dataset
        X, y = make_classification(
            n_samples=200,
            n_features=5,
            n_classes=2,
            weights=[0.3, 0.7],  # 30% class 0, 70% class 1
            random_state=42
        )

        validator = CrossValidator(cv_strategy='stratified', random_state=42)
        estimator = LogisticRegression(random_state=42)

        result = validator.validate(
            estimator=estimator,
            X=X,
            y=y,
            scoring='accuracy',
            n_folds=5
        )

        # Check class distribution in each fold
        overall_class_ratio = np.mean(y)
        fold_class_ratios = []

        for fold_idx, (train_idx, test_idx) in result.fold_indices.items():
            # Check test set class ratio
            test_y = y[test_idx]
            test_ratio = np.mean(test_y)
            fold_class_ratios.append(test_ratio)

            # Stratified folds should maintain similar class ratios
            # Allow small deviation due to rounding
            assert abs(test_ratio - overall_class_ratio) < 0.1

        # All folds should have similar class distributions
        ratios_std = np.std(fold_class_ratios)
        assert ratios_std < 0.05  # Low variation across folds

    def test_time_series_cv_temporal_ordering(self):
        """
        Property 8: Cross-validation methodology

        Validates that time series CV respects temporal ordering
        """
        # Create time series data with clear trend
        np.random.seed(42)
        t = np.arange(100)
        X = t.reshape(-1, 1)  # Simple feature: time
        y = t + np.random.normal(0, 0.1, 100)  # Trend with noise

        validator = CrossValidator(cv_strategy='time_series')
        estimator = LinearRegression()

        result = validator.validate(
            estimator=estimator,
            X=X,
            y=y,
            scoring='neg_mean_squared_error',
            n_folds=5
        )

        # Verify temporal ordering is preserved
        for fold_idx, (train_idx, test_idx) in result.fold_indices.items():
            # All train indices should be less than test indices
            max_train_idx = np.max(train_idx)
            min_test_idx = np.min(test_idx)
            assert max_train_idx < min_test_idx, (
                f"Temporal ordering violation in fold {fold_idx}: "
                f"max_train={max_train_idx}, min_test={min_test_idx}"
            )

            # Test sets should be contiguous blocks in time series
            test_diff = np.diff(np.sort(test_idx))
            assert np.all(test_diff <= 1), (
                f"Test indices not contiguous in fold {fold_idx}"
            )

    def test_confidence_interval_calculation_methods(self):
        """
        Property 8: Cross-validation methodology

        Validates that different CI calculation methods
        produce appropriate intervals
        """
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_classes=2,
            random_state=42
        )

        estimator = LogisticRegression(random_state=42)

        # Test different confidence level
        validator_95 = CrossValidator(
            cv_strategy='kfold',
            confidence_level=0.95,
            random_state=42
        )

        result_95 = validator_95.validate(
            estimator=estimator,
            X=X,
            y=y,
            scoring='accuracy',
            n_folds=10  # More folds for better CI estimation
        )

        # Verify CI properties
        lower, upper = result_95.confidence_interval

        # CI should contain the mean
        assert lower <= result_95.mean_score <= upper

        # CI should be symmetric-ish around mean (approximately)
        mean_to_lower = result_95.mean_score - lower
        mean_to_upper = upper - result_95.mean_score
        # Allow some asymmetry due to t-distribution
        assert abs(mean_to_lower - mean_to_upper) / result_95.mean_score < 0.1

        # CI width should be reasonable (not too wide, not too narrow)
        ci_width = upper - lower
        # Allow zero width if scores are all identical (perfect model)
        if result_95.std_score == 0:
            assert ci_width == 0
        else:
            assert 0.01 < ci_width < 1.0  # Reasonable range for accuracy scores

    def test_fold_score_consistency(self):
        """
        Property 8: Cross-validation methodology

        Validates that fold scores are consistent and reasonable
        """
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=2,
            random_state=42
        )

        validator = CrossValidator(cv_strategy='stratified', random_state=42)
        estimator = RandomForestClassifier(
            random_state=42,
            n_estimators=10,
            max_depth=3  # Limit complexity for consistent scores
        )

        result = validator.validate(
            estimator=estimator,
            X=X,
            y=y,
            scoring='accuracy',
            n_folds=10
        )

        # All fold scores should be within reasonable bounds
        for score in result.fold_scores:
            assert 0 <= score <= 1  # Accuracy scores

        # Scores should not vary too much (assuming stable model)
        cv = np.std(result.fold_scores)
        assert cv < 0.2  # CV should not be too high for stable model

        # Mean score should be within range of fold scores
        assert min(result.fold_scores) <= result.mean_score <= max(result.fold_scores)

        # Verify scores_by_fold matches fold_scores
        for fold_idx in range(result.n_folds):
            assert result.scores_by_fold[fold_idx]['score'] == result.fold_scores[fold_idx]

    def test_multiple_metrics_validation(self):
        """
        Property 8: Cross-validation methodology

        Validates that multiple metrics are computed correctly
        and consistently across folds
        """
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_classes=2,
            random_state=42
        )

        validator = CrossValidator(cv_strategy='stratified', random_state=42)
        estimator = RandomForestClassifier(random_state=42, n_estimators=10)

        result = validator.validate(
            estimator=estimator,
            X=X,
            y=y,
            scoring='accuracy',
            n_folds=5,
            additional_metrics=['precision', 'recall', 'f1']
        )

        # Check that all metrics were computed
        assert result.additional_metrics is not None
        # The implementation uses generic metric names
        assert len(result.additional_metrics) == 3

        # Each metric should have scores for all folds
        for metric_name, scores in result.additional_metrics.items():
            assert len(scores) == result.n_folds
            # All scores should be reasonable
            for score in scores:
                assert 0 <= score <= 1

        # Check that metrics appear in fold scores
        for fold_idx, fold_scores in result.scores_by_fold.items():
            assert 'score' in fold_scores  # Primary metric
            # Additional metrics should also be present
            assert 'metric_0' in fold_scores  # precision
            assert 'metric_1' in fold_scores  # recall
            assert 'metric_2' in fold_scores  # f1