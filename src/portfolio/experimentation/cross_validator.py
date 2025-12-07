"""
Cross-validation pipeline for model evaluation.

This module provides the CrossValidator class with various
cross-validation strategies and confidence interval calculations.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import warnings

from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    cross_validate
)
from sklearn.base import clone, BaseEstimator
from sklearn.metrics import make_scorer
from scipy import stats

from src.portfolio.utils.logging import PortfolioLogger


@dataclass
class CVResult:
    """Results from cross-validation"""
    fold_scores: List[float]
    mean_score: float
    std_score: float
    confidence_interval: Tuple[float, float]
    scores_by_fold: Dict[int, Dict[str, float]]
    fold_indices: Dict[int, Tuple[np.ndarray, np.ndarray]]
    cv_method: str
    n_folds: int
    additional_metrics: Optional[Dict[str, List[float]]] = None


class CVStrategy(ABC):
    """Abstract base class for cross-validation strategies"""

    @abstractmethod
    def get_cv_splitter(self, n_folds: int, **kwargs) -> Any:
        """Get the sklearn CV splitter"""
        pass

    @abstractmethod
    def validate_data(self, X: Any, y: Any) -> bool:
        """Validate if the data is appropriate for this CV method"""
        pass


class StandardKFold(CVStrategy):
    """Standard K-fold cross-validation"""

    def get_cv_splitter(self, n_folds: int, **kwargs) -> KFold:
        """Get KFold splitter"""
        shuffle = kwargs.get('shuffle', True)
        random_state = kwargs.get('random_state', None)
        return KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)

    def validate_data(self, X: Any, y: Any) -> bool:
        """Standard K-fold works with any data"""
        return True


class StratifiedKFoldCV(CVStrategy):
    """Stratified K-fold cross-validation for classification"""

    def get_cv_splitter(self, n_folds: int, **kwargs) -> StratifiedKFold:
        """Get StratifiedKFold splitter"""
        shuffle = kwargs.get('shuffle', True)
        random_state = kwargs.get('random_state', None)
        return StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)

    def validate_data(self, X: Any, y: Any) -> bool:
        """Stratified K-fold requires classification labels"""
        if y is None:
            return False
        # Check if y has discrete values suitable for stratification
        unique_values = np.unique(y)
        if len(unique_values) < 2:
            warnings.warn("Stratified K-fold requires at least 2 classes")
            return False
        # Check if minimum samples per class
        min_samples_per_class = min(np.bincount(y.astype(int)))
        if min_samples_per_class < 2:
            warnings.warn(f"Some classes have fewer than 2 samples: {min_samples_per_class}")
        return True


class TimeSeriesCV(CVStrategy):
    """Time series cross-validation"""

    def get_cv_splitter(self, n_folds: int, **kwargs) -> TimeSeriesSplit:
        """Get TimeSeriesSplit splitter"""
        max_train_size = kwargs.get('max_train_size', None)
        test_size = kwargs.get('test_size', None)
        gap = kwargs.get('gap', 0)
        return TimeSeriesSplit(
            n_splits=n_folds,
            max_train_size=max_train_size,
            test_size=test_size
        )

    def validate_data(self, X: Any, y: Any) -> bool:
        """Time series CV requires temporal ordering"""
        return True  # Time series can work with any data, but user should ensure ordering


class CrossValidator:
    """
    Cross-validation pipeline with multiple strategies and statistical analysis.
    """

    def __init__(
        self,
        cv_strategy: str = 'kfold',
        confidence_level: float = 0.95,
        random_state: Optional[int] = None
    ):
        """
        Initialize cross-validator.

        Args:
            cv_strategy: Cross-validation strategy ('kfold', 'stratified', 'time_series')
            confidence_level: Confidence level for intervals (0-1)
            random_state: Random seed for reproducibility
        """
        self.cv_strategy = cv_strategy
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.logger = PortfolioLogger('cross_validation')

        # Initialize strategies
        self.strategies = {
            'kfold': StandardKFold(),
            'stratified': StratifiedKFoldCV(),
            'time_series': TimeSeriesCV()
        }

        if cv_strategy not in self.strategies:
            raise ValueError(
                f"Unknown CV strategy: {cv_strategy}. "
                f"Available: {list(self.strategies.keys())}"
            )

    def validate(
        self,
        estimator: BaseEstimator,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        scoring: Union[str, Callable] = 'accuracy',
        n_folds: int = 5,
        return_estimators: bool = False,
        additional_metrics: Optional[List[Union[str, Callable]]] = None,
        **cv_kwargs
    ) -> Union[CVResult, Tuple[CVResult, List[BaseEstimator]]]:
        """
        Perform cross-validation with confidence intervals.

        Args:
            estimator: Scikit-learn compatible estimator
            X: Feature matrix
            y: Target vector
            scoring: Scoring metric or callable
            n_folds: Number of CV folds
            return_estimators: Whether to return fitted estimators
            additional_metrics: List of additional metrics to compute
            **cv_kwargs: Additional arguments for CV splitter

        Returns:
            CVResult with detailed validation results
        """
        # Validate inputs
        if not self.strategies[self.cv_strategy].validate_data(X, y):
            raise ValueError(f"Data is not suitable for {self.cv_strategy} CV")

        # Get CV splitter
        cv_splitter = self.strategies[self.cv_strategy].get_cv_splitter(
            n_folds,
            random_state=self.random_state,
            **cv_kwargs
        )

        self.logger.info(
            f"Starting {self.cv_strategy} cross-validation with {n_folds} folds"
        )

        # Perform cross-validation with multiple metrics if requested
        scorers = {'primary': scoring}
        if additional_metrics:
            for i, metric in enumerate(additional_metrics):
                scorers[f'metric_{i}'] = metric

        cv_results = None
        if len(scorers) == 1 and not return_estimators:
            # Single metric case without returning estimators
            scores = cross_val_score(
                estimator,
                X,
                y,
                cv=cv_splitter,
                scoring=scoring,
                n_jobs=-1
            )

            # Get fold indices for detailed analysis
            fold_indices = {}
            scores_by_fold = {}

            for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
                fold_indices[fold] = (train_idx, test_idx)
                scores_by_fold[fold] = {'score': scores[fold]}

            additional_metric_scores = None
        else:
            # Multiple metrics case or returning estimators
            cv_results = cross_validate(
                estimator,
                X,
                y,
                cv=cv_splitter,
                scoring=scorers,
                n_jobs=-1,
                return_estimator=return_estimators
            )

            scores = cv_results['test_primary']

            # Collect additional metric scores
            additional_metric_scores = {}
            for key in cv_results:
                if key.startswith('test_'):
                    metric_name = key[5:]  # Remove 'test_' prefix
                    if metric_name != 'primary':
                        additional_metric_scores[metric_name] = cv_results[key].tolist()

            # Get fold indices
            fold_indices = {}
            scores_by_fold = {}

            cv_splitter_iter = cv_splitter.split(X, y)
            for fold in range(n_folds):
                train_idx, test_idx = next(cv_splitter_iter)
                fold_indices[fold] = (train_idx, test_idx)
                scores_by_fold[fold] = {'score': scores[fold]}

                # Add additional metrics for this fold
                for metric_name, metric_scores in additional_metric_scores.items():
                    scores_by_fold[fold][metric_name] = metric_scores[fold]

        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(scores)

        # Create result object
        result = CVResult(
            fold_scores=scores.tolist(),
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
            confidence_interval=confidence_interval,
            scores_by_fold=scores_by_fold,
            fold_indices=fold_indices,
            cv_method=self.cv_strategy,
            n_folds=n_folds,
            additional_metrics=additional_metric_scores
        )

        self.logger.info(
            f"CV completed. Mean score: {result.mean_score:.4f} "
            f"(±{result.std_score:.4f})"
        )
        self.logger.info(
            f"{self.confidence_level*100:.0f}% CI: [{confidence_interval[0]:.4f}, "
            f"{confidence_interval[1]:.4f}]"
        )

        if return_estimators and 'estimator' in cv_results:
            return result, cv_results['estimator']
        else:
            return result

    def _calculate_confidence_interval(
        self,
        scores: np.ndarray,
        method: str = 't'
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for scores.

        Args:
            scores: Array of scores across folds
            method: Method for CI calculation ('t', 'normal', 'percentile')

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        alpha = 1 - self.confidence_level

        if method == 't':
            # Student's t-distribution (recommended for small samples)
            degrees_of_freedom = len(scores) - 1
            t_value = stats.t.ppf(1 - alpha/2, degrees_of_freedom)
            sem = stats.sem(scores)  # Standard error of the mean
            margin = t_value * sem
            return float(np.mean(scores) - margin), float(np.mean(scores) + margin)

        elif method == 'normal':
            # Normal distribution approximation
            z_value = stats.norm.ppf(1 - alpha/2)
            sem = stats.sem(scores)
            margin = z_value * sem
            return float(np.mean(scores) - margin), float(np.mean(scores) + margin)

        elif method == 'percentile':
            # Percentile bootstrap (non-parametric)
            lower = np.percentile(scores, 100 * alpha/2)
            upper = np.percentile(scores, 100 * (1 - alpha/2))
            return float(lower), float(upper)

        else:
            raise ValueError(f"Unknown CI method: {method}")

    def compare_models(
        self,
        models: Dict[str, BaseEstimator],
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        scoring: Union[str, Callable] = 'accuracy',
        n_folds: int = 5,
        statistical_test: str = 'paired_t',
        **cv_kwargs
    ) -> pd.DataFrame:
        """
        Compare multiple models using cross-validation.

        Args:
            models: Dictionary of model_name -> estimator
            X: Feature matrix
            y: Target vector
            scoring: Scoring metric
            n_folds: Number of CV folds
            statistical_test: Statistical test for comparison
            **cv_kwargs: Additional CV arguments

        Returns:
            DataFrame with model comparison results
        """
        results = {}

        # Get CV results for each model
        for model_name, model in models.items():
            cv_result = self.validate(
                model, X, y, scoring=scoring, n_folds=n_folds, **cv_kwargs
            )
            results[model_name] = cv_result.fold_scores

        # Create comparison dataframe
        comparison_data = []
        for model_name, scores in results.items():
            comparison_data.append({
                'model': model_name,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'scores': scores
            })

        df = pd.DataFrame(comparison_data)

        # Add statistical comparisons if requested
        if statistical_test and len(models) > 1:
            # Perform pairwise comparisons
            model_names = list(models.keys())
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]
                    scores1, scores2 = results[model1], results[model2]

                    if statistical_test == 'paired_t':
                        # Paired t-test
                        t_stat, p_value = stats.ttest_rel(scores1, scores2)
                        test_type = 'Paired t-test'
                    elif statistical_test == 'wilcoxon':
                        # Wilcoxon signed-rank test
                        t_stat, p_value = stats.wilcoxon(scores1, scores2)
                        test_type = 'Wilcoxon signed-rank'
                    else:
                        warnings.warn(f"Unknown test: {statistical_test}")
                        continue

                    # Add to results
                    comparison_name = f"{model1} vs {model2}"
                    p_col = f"p_value_{comparison_name}"
                    test_col = f"test_{comparison_name}"

                    df[p_col] = np.nan
                    df[test_col] = np.nan

                    # Set p-value for both models
                    df.loc[df['model'] == model1, p_col] = p_value
                    df.loc[df['model'] == model2, p_col] = p_value
                    df.loc[df['model'] == model1, test_col] = test_type
                    df.loc[df['model'] == model2, test_col] = test_type

        return df.sort_values('mean_score', ascending=False)

    def get_learning_curve(
        self,
        estimator: BaseEstimator,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        scoring: Union[str, Callable] = 'accuracy',
        train_sizes: Optional[np.ndarray] = None,
        n_folds: int = 5,
        **cv_kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Generate learning curve data.

        Args:
            estimator: Scikit-learn compatible estimator
            X: Feature matrix
            y: Target vector
            scoring: Scoring metric
            train_sizes: Fractions of training set to use
            n_folds: Number of CV folds
            **cv_kwargs: Additional CV arguments

        Returns:
            Dictionary with learning curve data
        """
        from sklearn.model_selection import learning_curve

        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        cv_splitter = self.strategies[self.cv_strategy].get_cv_splitter(
            n_folds,
            random_state=self.random_state,
            **cv_kwargs
        )

        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator,
            X,
            y,
            train_sizes=train_sizes,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=-1,
            random_state=self.random_state
        )

        return {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'val_scores_mean': np.mean(val_scores, axis=1),
            'val_scores_std': np.std(val_scores, axis=1),
            'train_scores': train_scores,
            'val_scores': val_scores
        }

    def get_validation_report(self, result: CVResult) -> str:
        """
        Generate a text report of the validation results.

        Args:
            result: CVResult to report on

        Returns:
            String report
        """
        report = []
        report.append("Cross-Validation Report")
        report.append("=" * 50)
        report.append(f"CV Method: {result.cv_method}")
        report.append(f"Number of Folds: {result.n_folds}")
        report.append(f"Confidence Level: {self.confidence_level*100:.0f}%")
        report.append("")

        report.append("Overall Performance:")
        report.append("-" * 30)
        report.append(f"Mean Score: {result.mean_score:.4f}")
        report.append(f"Std Deviation: {result.std_score:.4f}")
        report.append(f"Confidence Interval: [{result.confidence_interval[0]:.4f}, "
                     f"{result.confidence_interval[1]:.4f}]")
        report.append("")

        report.append("Scores by Fold:")
        report.append("-" * 20)
        for fold, scores in result.scores_by_fold.items():
            report.append(f"Fold {fold + 1}:")
            for metric, score in scores.items():
                report.append(f"  {metric}: {score:.4f}")
        report.append("")

        if result.additional_metrics:
            report.append("Additional Metrics:")
            report.append("-" * 25)
            for metric_name, scores in result.additional_metrics.items():
                report.append(f"{metric_name}:")
                report.append(f"  Mean: {np.mean(scores):.4f}")
                report.append(f"  Std: {np.std(scores):.4f}")

        return "\n".join(report)