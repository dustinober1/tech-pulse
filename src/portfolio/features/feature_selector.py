"""
Feature selection framework for machine learning.

This module provides comprehensive feature selection capabilities including
mutual information, correlation analysis, recursive feature elimination,
and feature importance ranking.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

# Machine learning imports
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
    SelectKBest,
    SelectPercentile,
    RFE,
    RFECV,
    VarianceThreshold,
    f_classif,
    f_regression,
    chi2
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, Lasso, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mutual_info_score
from scipy import stats
from scipy.stats import pearsonr, spearmanr
# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    sns = None

# Suppress warnings
warnings.filterwarnings('ignore')


@dataclass
class SelectionResult:
    """Container for feature selection results."""
    selected_features: List[str]
    feature_scores: Dict[str, float]
    feature_rankings: Dict[str, int]
    selection_method: str
    n_selected: int
    n_original: int
    reduction_ratio: float
    selection_threshold: Optional[float]
    cv_score: Optional[float]
    metadata: Dict[str, Any]


@dataclass
class SelectionConfig:
    """Configuration for feature selection."""
    # Mutual information
    mi_method: str = 'auto'  # 'auto', 'classification', 'regression'
    mi_k_neighbors: int = 3
    mi_discrete_features: str = 'auto'

    # Correlation filtering
    correlation_threshold: float = 0.95
    correlation_method: str = 'pearson'  # 'pearson', 'spearman', 'kendall'

    # Variance threshold
    variance_threshold: float = 0.01

    # Statistical tests
    test_alpha: float = 0.05
    test_method: str = 'auto'  # 'auto', 'f_classif', 'f_regression', 'chi2'

    # Recursive feature elimination
    rfe_step: float = 0.1
    rfe_min_features: int = 5
    rfe_cv_folds: int = 5

    # Model-based selection
    model_n_estimators: int = 100
    model_max_depth: Optional[int] = None
    model_random_state: int = 42

    # Cross-validation
    cv_folds: int = 5
    cv_scoring: str = 'auto'  # 'auto', specific scoring metric

    # Selection thresholds
    percentile_threshold: float = 50.0  # For SelectPercentile
    k_best: Optional[int] = None  # For SelectKBest

    # Stability selection
    stability_n_bootstrap: int = 50
    stability_sample_ratio: float = 0.8

    # Multicollinearity
    vif_threshold: float = 5.0
    max_vif_features: Optional[int] = None


class FeatureSelector:
    """Comprehensive feature selection framework."""

    def __init__(self, config: Optional[SelectionConfig] = None):
        """
        Initialize the feature selector.

        Args:
            config: Configuration for feature selection
        """
        self.config = config or SelectionConfig()
        self.scaler = StandardScaler()
        self.feature_importances_ = {}
        self.selected_features_ = []
        self.selection_history_ = []

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        method: str = 'combined',
        **kwargs
    ) -> pd.DataFrame:
        """
        Fit the selector and transform data.

        Args:
            X: Feature matrix
            y: Target variable
            method: Selection method or combination
            **kwargs: Additional method-specific parameters

        Returns:
            Transformed DataFrame with selected features
        """
        # Store original columns
        self.original_features_ = list(X.columns)

        # Determine if classification or regression
        self.problem_type_ = self._determine_problem_type(y)

        # Apply selected method
        if method == 'combined':
            X_selected = self._combined_selection(X, y, **kwargs)
        elif method == 'mutual_info':
            X_selected = self._mutual_info_selection(X, y, **kwargs)
        elif method == 'correlation':
            X_selected = self._correlation_selection(X, y, **kwargs)
        elif method == 'variance':
            X_selected = self._variance_selection(X, **kwargs)
        elif method == 'statistical':
            X_selected = self._statistical_selection(X, y, **kwargs)
        elif method == 'rfe':
            X_selected = self._rfe_selection(X, y, **kwargs)
        elif method == 'model_based':
            X_selected = self._model_based_selection(X, y, **kwargs)
        elif method == 'stability':
            X_selected = self._stability_selection(X, y, **kwargs)
        elif method == 'vif':
            X_selected = self._vif_selection(X, **kwargs)
        else:
            raise ValueError(f"Unknown selection method: {method}")

        self.selected_features_ = list(X_selected.columns)
        return X_selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using previously fitted selector.

        Args:
            X: Feature matrix

        Returns:
            Transformed DataFrame with selected features
        """
        if not hasattr(self, 'selected_features_'):
            raise ValueError("Selector must be fitted before transform")

        # Return only selected features that exist in X
        existing_features = [f for f in self.selected_features_ if f in X.columns]
        return X[existing_features]

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], method: str = 'combined', **kwargs):
        """
        Fit the selector without transforming data.

        Args:
            X: Feature matrix
            y: Target variable
            method: Selection method
            **kwargs: Additional parameters
        """
        self.fit_transform(X, y, method=method, **kwargs)
        return self

    def _determine_problem_type(self, y: Union[pd.Series, np.ndarray]) -> str:
        """Determine if this is a classification or regression problem."""
        y_array = np.asarray(y)

        # Check if y is numeric and has many unique values
        if np.issubdtype(y_array.dtype, np.number):
            n_unique = len(np.unique(y_array))
            n_samples = len(y_array)

            # Heuristic: regression if many unique values relative to samples
            if n_unique > min(20, n_samples / 10):
                return 'regression'
            else:
                return 'classification'
        else:
            return 'classification'

    def _combined_selection(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        include_methods: Optional[List[str]] = None,
        voting_threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Combine multiple selection methods.

        Args:
            X: Feature matrix
            y: Target variable
            include_methods: Methods to include in combination
            voting_threshold: Minimum vote ratio for selection

        Returns:
            DataFrame with features selected by multiple methods
        """
        if include_methods is None:
            include_methods = ['mutual_info', 'variance', 'statistical', 'model_based']

        # Store votes for each feature
        feature_votes = {feature: 0 for feature in X.columns}

        # Apply each method and collect votes
        for method in include_methods:
            try:
                X_method = getattr(self, f'_{method}_selection')(X, y)
                feature_votes.update({
                    feature: feature_votes[feature] + (1 if feature in X_method.columns else 0)
                    for feature in X.columns
                })
            except Exception as e:
                print(f"Warning: Method {method} failed: {e}")

        # Select features with sufficient votes
        n_methods = len(include_methods)
        min_votes = int(n_methods * voting_threshold)
        selected_features = [
            feature for feature, votes in feature_votes.items()
            if votes >= min_votes
        ]

        # Store selection result
        self.selection_history_.append(SelectionResult(
            selected_features=selected_features,
            feature_scores=feature_votes,
            feature_rankings={f: v for f, v in feature_votes.items()},
            selection_method='combined',
            n_selected=len(selected_features),
            n_original=len(X.columns),
            reduction_ratio=1 - len(selected_features) / len(X.columns),
            selection_threshold=voting_threshold,
            cv_score=None,
            metadata={'methods': include_methods, 'votes': feature_votes}
        ))

        return X[selected_features]

    def _mutual_info_selection(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        k: Optional[int] = None,
        percentile: Optional[float] = None
    ) -> pd.DataFrame:
        """Select features based on mutual information."""
        # Determine problem type if not already set
        if not hasattr(self, 'problem_type_'):
            self.problem_type_ = self._determine_problem_type(y)

        # Determine appropriate function
        if self.config.mi_method == 'auto':
            mi_func = mutual_info_classif if self.problem_type_ == 'classification' else mutual_info_regression
        elif self.config.mi_method == 'classification':
            mi_func = mutual_info_classif
        else:
            mi_func = mutual_info_regression

        # Calculate mutual information scores
        mi_scores = mi_func(
            X,
            y,
            discrete_features=self.config.mi_discrete_features,
            n_neighbors=self.config.mi_k_neighbors
        )

        # Create feature score mapping
        feature_scores = dict(zip(X.columns, mi_scores))

        # Select features
        if k is not None:
            selector = SelectKBest(mi_func, k=k)
        elif percentile is not None:
            selector = SelectPercentile(mi_func, percentile=percentile)
        elif self.config.k_best is not None:
            selector = SelectKBest(mi_func, k=self.config.k_best)
        else:
            selector = SelectPercentile(mi_func, percentile=self.config.percentile_threshold)

        X_selected = pd.DataFrame(
            selector.fit_transform(X, y),
            columns=X.columns[selector.get_support()],
            index=X.index
        )

        # Store selection result
        self.selection_history_.append(SelectionResult(
            selected_features=list(X_selected.columns),
            feature_scores=feature_scores,
            feature_rankings=dict(zip(X.columns, selector.scores_)),
            selection_method='mutual_info',
            n_selected=len(X_selected.columns),
            n_original=len(X.columns),
            reduction_ratio=1 - len(X_selected.columns) / len(X.columns),
            selection_threshold=selector.scores_[selector.get_support()].min() if len(X_selected.columns) > 0 else None,
            cv_score=None,
            metadata={'discrete_features': self.config.mi_discrete_features}
        ))

        return X_selected

    def _correlation_selection(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        keep_correlated_with_y: bool = True
    ) -> pd.DataFrame:
        """Remove highly correlated features."""
        # Determine problem type if not already set
        if not hasattr(self, 'problem_type_'):
            self.problem_type_ = self._determine_problem_type(y)

        # Calculate correlation matrix
        if self.config.correlation_method == 'spearman':
            corr_matrix = X.corr(method='spearman')
        elif self.config.correlation_method == 'kendall':
            corr_matrix = X.corr(method='kendall')
        else:
            corr_matrix = X.corr(method='pearson')

        # Find highly correlated feature pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Identify features to remove
        to_remove = set()
        for col in upper_triangle.columns:
            for row in upper_triangle.index:
                if not pd.isna(upper_triangle.loc[row, col]):
                    if abs(upper_triangle.loc[row, col]) > self.config.correlation_threshold:
                        # Keep the one more correlated with y if specified
                        if keep_correlated_with_y:
                            y_corr_row = abs(X[row].corr(pd.Series(y)))
                            y_corr_col = abs(X[col].corr(pd.Series(y)))
                            if y_corr_row >= y_corr_col:
                                to_remove.add(col)
                            else:
                                to_remove.add(row)
                        else:
                            to_remove.add(col)  # Remove the second one

        # Select features
        selected_features = [f for f in X.columns if f not in to_remove]

        # Store selection result
        self.selection_history_.append(SelectionResult(
            selected_features=selected_features,
            feature_scores={f: 1.0 for f in selected_features},
            feature_rankings={f: i for i, f in enumerate(selected_features)},
            selection_method='correlation',
            n_selected=len(selected_features),
            n_original=len(X.columns),
            reduction_ratio=1 - len(selected_features) / len(X.columns),
            selection_threshold=self.config.correlation_threshold,
            cv_score=None,
            metadata={
                'method': self.config.correlation_method,
                'removed_features': list(to_remove)
            }
        ))

        return X[selected_features]

    def _variance_selection(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """Remove features with low variance."""
        threshold = threshold or self.config.variance_threshold

        # Select only numeric columns for variance threshold
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Apply variance threshold to numeric features
        if numeric_cols:
            X_numeric = X[numeric_cols]
            selector = VarianceThreshold(threshold=threshold)
            X_numeric_selected = pd.DataFrame(
                selector.fit_transform(X_numeric),
                columns=X_numeric.columns[selector.get_support()],
                index=X.index
            )
        else:
            X_numeric_selected = pd.DataFrame(index=X.index)

        # Keep all categorical features (they don't have variance in the same sense)
        if categorical_cols:
            X_categorical = X[categorical_cols]
        else:
            X_categorical = pd.DataFrame(index=X.index)

        # Combine selected features
        X_selected = pd.concat([X_numeric_selected, X_categorical], axis=1)

        # Calculate variances for removed features (only numeric)
        variances = X[numeric_cols].var()
        feature_scores = dict(zip(X.columns, [variances.get(f, np.inf) for f in X.columns]))

        # Store selection result
        self.selection_history_.append(SelectionResult(
            selected_features=list(X_selected.columns),
            feature_scores=feature_scores,
            feature_rankings={f: i for i, f in enumerate(X_selected.columns)},
            selection_method='variance',
            n_selected=len(X_selected.columns),
            n_original=len(X.columns),
            reduction_ratio=1 - len(X_selected.columns) / len(X.columns),
            selection_threshold=threshold,
            cv_score=None,
            metadata={'numeric_cols': numeric_cols, 'categorical_cols': categorical_cols}
        ))

        return X_selected

    def _statistical_selection(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        method: Optional[str] = None,
        alpha: Optional[float] = None
    ) -> pd.DataFrame:
        """Select features using statistical tests."""
        # Determine problem type if not already set
        if not hasattr(self, 'problem_type_'):
            self.problem_type_ = self._determine_problem_type(y)

        method = method or self.config.test_method
        alpha = alpha or self.config.test_alpha

        # Determine appropriate test
        if method == 'auto':
            if self.problem_type_ == 'classification':
                test_func = f_classif
            else:
                test_func = f_regression
        elif method == 'f_classif':
            test_func = f_classif
        elif method == 'f_regression':
            test_func = f_regression
        elif method == 'chi2':
            # Ensure non-negative values for chi2
            X = X - X.min() + 1e-8
            test_func = chi2
        else:
            raise ValueError(f"Unknown statistical test method: {method}")

        # Apply statistical test
        selector = SelectKBest(test_func, k='all')
        selector.fit(X, y)

        # Get p-values and select significant features
        p_values = selector.pvalues_
        selected_mask = p_values < alpha

        X_selected = pd.DataFrame(
            X.iloc[:, selected_mask],
            columns=X.columns[selected_mask],
            index=X.index
        )

        # Store selection result
        self.selection_history_.append(SelectionResult(
            selected_features=list(X_selected.columns),
            feature_scores=dict(zip(X.columns, -np.log10(p_values))),
            feature_rankings=dict(zip(X.columns, selector.scores_)),
            selection_method='statistical',
            n_selected=len(X_selected.columns),
            n_original=len(X.columns),
            reduction_ratio=1 - len(X_selected.columns) / len(X.columns),
            selection_threshold=alpha,
            cv_score=None,
            metadata={
                'test_method': method,
                'p_values': dict(zip(X.columns, p_values))
            }
        ))

        return X_selected

    def _rfe_selection(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        estimator: Optional[Any] = None,
        cv: bool = False,
        min_features: Optional[int] = None
    ) -> pd.DataFrame:
        """Recursive Feature Elimination."""
        # Determine problem type if not already set
        if not hasattr(self, 'problem_type_'):
            self.problem_type_ = self._determine_problem_type(y)

        # Create estimator
        if estimator is None:
            if self.problem_type_ == 'classification':
                estimator = RandomForestClassifier(
                    n_estimators=self.config.model_n_estimators,
                    max_depth=self.config.model_max_depth,
                    random_state=self.config.model_random_state
                )
            else:
                estimator = RandomForestRegressor(
                    n_estimators=self.config.model_n_estimators,
                    max_depth=self.config.model_max_depth,
                    random_state=self.config.model_random_state
                )

        # Use RFECV if cv=True
        if cv:
            selector = RFECV(
                estimator,
                step=self.config.rfe_step,
                cv=self.config.rfe_cv_folds,
                min_features_to_select=min_features or self.config.rfe_min_features,
                scoring='accuracy' if self.problem_type_ == 'classification' else 'neg_mean_squared_error'
            )
        else:
            selector = RFE(
                estimator,
                step=self.config.rfe_step,
                min_features_to_select=min_features or self.config.rfe_min_features
            )

        # Fit and transform
        X_selected = pd.DataFrame(
            selector.fit_transform(X, y),
            columns=X.columns[selector.get_support()],
            index=X.index
        )

        # Get CV score if available
        cv_score = selector.cv_score_ if hasattr(selector, 'cv_score_') else None

        # Store selection result
        self.selection_history_.append(SelectionResult(
            selected_features=list(X_selected.columns),
            feature_scores=dict(zip(X.columns, selector.ranking_)),
            feature_rankings=dict(zip(X.columns, selector.ranking_)),
            selection_method='rfe',
            n_selected=len(X_selected.columns),
            n_original=len(X.columns),
            reduction_ratio=1 - len(X_selected.columns) / len(X.columns),
            selection_threshold=None,
            cv_score=cv_score,
            metadata={
                'cv': cv,
                'n_features_to_select': selector.n_features_ if hasattr(selector, 'n_features_') else None
            }
        ))

        return X_selected

    def _model_based_selection(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        estimator: Optional[Any] = None,
        importance_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """Select features based on model importance."""
        # Determine problem type if not already set
        if not hasattr(self, 'problem_type_'):
            self.problem_type_ = self._determine_problem_type(y)

        # Create estimator
        if estimator is None:
            if self.problem_type_ == 'classification':
                estimator = RandomForestClassifier(
                    n_estimators=self.config.model_n_estimators,
                    max_depth=self.config.model_max_depth,
                    random_state=self.config.model_random_state
                )
            else:
                estimator = RandomForestRegressor(
                    n_estimators=self.config.model_n_estimators,
                    max_depth=self.config.model_max_depth,
                    random_state=self.config.model_random_state
                )

        # Handle categorical features by one-hot encoding
        X_numeric = X.select_dtypes(include=[np.number])
        X_categorical = X.select_dtypes(exclude=[np.number])

        if not X_categorical.empty:
            # One-hot encode categorical features
            X_encoded = pd.get_dummies(X_categorical, drop_first=True)
            X_model = pd.concat([X_numeric, X_encoded], axis=1)
        else:
            X_model = X_numeric

        # Fit model and get feature importances
        estimator.fit(X_model, y)
        importances = estimator.feature_importances_

        # Map importances back to original feature names
        # For one-hot encoded features, take max importance
        feature_importances = {}
        for col in X.columns:
            if col in X_model.columns:
                # Numeric feature
                idx = X_model.columns.get_loc(col)
                feature_importances[col] = importances[idx]
            else:
                # Categorical feature - use max of encoded features
                encoded_cols = [c for c in X_model.columns if c.startswith(f"{col}_")]
                if encoded_cols:
                    max_importance = max(importances[X_model.columns.get_loc(c)] for c in encoded_cols)
                    feature_importances[col] = max_importance
                else:
                    feature_importances[col] = 0.0

        # Store importances
        self.feature_importances_ = feature_importances

        # Select features based on importance threshold
        all_importances = list(feature_importances.values())
        if importance_threshold is None:
            # Use median importance as threshold
            importance_threshold = np.median(all_importances)

        selected_features = [
            feature for feature, importance in feature_importances.items()
            if importance >= importance_threshold
        ]

        # Store selection result
        self.selection_history_.append(SelectionResult(
            selected_features=selected_features,
            feature_scores=feature_importances,
            feature_rankings=dict(zip(
                X.columns,
                sorted(range(len(X.columns)), key=lambda x: feature_importances[X.columns[x]], reverse=True)
            )),
            selection_method='model_based',
            n_selected=len(selected_features),
            n_original=len(X.columns),
            reduction_ratio=1 - len(selected_features) / len(X.columns),
            selection_threshold=importance_threshold,
            cv_score=None,
            metadata={
                'estimator': type(estimator).__name__,
                'importance_threshold': importance_threshold,
                'encoded_features': list(X_model.columns) if not X_categorical.empty else []
            }
        ))

        return X[selected_features]

    def _stability_selection(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        n_bootstrap: Optional[int] = None,
        sample_ratio: Optional[float] = None,
        selection_freq_threshold: float = 0.6
    ) -> pd.DataFrame:
        """Select features based on stability across bootstrap samples."""
        # Determine problem type if not already set
        if not hasattr(self, 'problem_type_'):
            self.problem_type_ = self._determine_problem_type(y)

        n_bootstrap = n_bootstrap or self.config.stability_n_bootstrap
        sample_ratio = sample_ratio or self.config.stability_sample_ratio

        # Track feature selection frequency
        feature_counts = {feature: 0 for feature in X.columns}

        # Bootstrap iterations
        for i in range(n_bootstrap):
            # Sample data
            n_samples = int(len(X) * sample_ratio)
            sample_indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X.iloc[sample_indices]
            y_sample = y[sample_indices] if hasattr(y, '__getitem__') else y[sample_indices]

            # Apply model-based selection
            try:
                X_selected = self._model_based_selection(X_sample, y_sample)
                feature_counts.update({
                    feature: feature_counts[feature] + (1 if feature in X_selected.columns else 0)
                    for feature in X.columns
                })
            except:
                continue

        # Calculate selection frequencies
        feature_freqs = {f: count / n_bootstrap for f, count in feature_counts.items()}

        # Select features with sufficient frequency
        selected_features = [
            feature for feature, freq in feature_freqs.items()
            if freq >= selection_freq_threshold
        ]

        # Store selection result
        self.selection_history_.append(SelectionResult(
            selected_features=selected_features,
            feature_scores=feature_freqs,
            feature_rankings=dict(zip(
                X.columns,
                sorted(range(len(X.columns)), key=lambda x: feature_freqs[X.columns[x]], reverse=True)
            )),
            selection_method='stability',
            n_selected=len(selected_features),
            n_original=len(X.columns),
            reduction_ratio=1 - len(selected_features) / len(X.columns),
            selection_threshold=selection_freq_threshold,
            cv_score=None,
            metadata={
                'n_bootstrap': n_bootstrap,
                'sample_ratio': sample_ratio,
                'feature_counts': feature_counts
            }
        ))

        return X[selected_features]

    def _vif_selection(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None,
        max_features: Optional[int] = None
    ) -> pd.DataFrame:
        """Remove features with high variance inflation factor (VIF)."""
        threshold = threshold or self.config.vif_threshold
        max_features = max_features or self.config.max_vif_features

        def calculate_vif(X_df):
            """Calculate VIF for each feature."""
            vif_data = {}
            for feature in X_df.columns:
                # Skip if only one feature
                if len(X_df.columns) == 1:
                    vif_data[feature] = 1.0
                    continue

                try:
                    # Regress feature against all others
                    other_features = [f for f in X_df.columns if f != feature]
                    X_other = X_df[other_features].values
                    y_feature = X_df[feature].values

                    # Add intercept and calculate R²
                    X_other_with_intercept = np.column_stack([np.ones(len(X_other)), X_other])
                    coeffs = np.linalg.lstsq(X_other_with_intercept, y_feature, rcond=None)[0]
                    y_pred = X_other_with_intercept @ coeffs
                    r_squared = 1 - np.sum((y_feature - y_pred) ** 2) / np.sum((y_feature - np.mean(y_feature)) ** 2)

                    # Calculate VIF
                    vif = 1 / (1 - r_squared) if r_squared < 0.999 else float('inf')
                    vif_data[feature] = vif
                except:
                    vif_data[feature] = float('inf')

            return vif_data

        # Iteratively remove high VIF features
        X_current = X.copy()
        removed_features = []

        while True:
            vif_values = calculate_vif(X_current)
            max_vif = max(vif_values.values())
            max_vif_feature = max(vif_values, key=vif_values.get)

            # Stop conditions
            if (max_vif <= threshold or
                (max_features is not None and len(X_current.columns) <= max_features) or
                len(X_current.columns) == 1):
                break

            # Remove feature with highest VIF
            X_current = X_current.drop(columns=[max_vif_feature])
            removed_features.append(max_vif_feature)

        # Store selection result
        self.selection_history_.append(SelectionResult(
            selected_features=list(X_current.columns),
            feature_scores=vif_values,
            feature_rankings=dict(zip(X_current.columns, sorted(range(len(X_current.columns)), key=lambda x: vif_values[X_current.columns[x]]))),
            selection_method='vif',
            n_selected=len(X_current.columns),
            n_original=len(X.columns),
            reduction_ratio=1 - len(X_current.columns) / len(X.columns),
            selection_threshold=threshold,
            cv_score=None,
            metadata={
                'removed_features': removed_features,
                'final_vif': vif_values
            }
        ))

        return X_current

    def get_feature_importance(self, method: str = 'latest') -> Dict[str, float]:
        """
        Get feature importance scores.

        Args:
            method: Which selection result to use ('latest', 'best', or method name)

        Returns:
            Dictionary of feature importance scores
        """
        if not self.selection_history_:
            raise ValueError("No selection history available")

        if method == 'latest':
            result = self.selection_history_[-1]
        elif method == 'best':
            # Find result with best CV score
            result = max(
                [r for r in self.selection_history_ if r.cv_score is not None],
                key=lambda x: x.cv_score
            )
        else:
            # Find by method name
            results = [r for r in self.selection_history_ if r.selection_method == method]
            if not results:
                raise ValueError(f"No selection result for method: {method}")
            result = results[-1]

        return result.feature_scores

    def plot_feature_importance(
        self,
        top_n: int = 20,
        method: str = 'latest',
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot feature importance scores.

        Args:
            top_n: Number of top features to show
            method: Which selection result to use
            figsize: Figure size
            save_path: Path to save plot
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for plotting. Please install it with: pip install matplotlib")

        importances = self.get_feature_importance(method)

        # Sort and select top features
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, scores = zip(*sorted_features)

        # Create plot
        plt.figure(figsize=figsize)
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importance ({method} selection)')
        plt.gca().invert_yaxis()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def summarize_selection_history(self) -> pd.DataFrame:
        """
        Summarize all selection methods applied.

        Returns:
            DataFrame with selection history summary
        """
        if not self.selection_history_:
            return pd.DataFrame()

        summary_data = []
        for result in self.selection_history_:
            summary_data.append({
                'method': result.selection_method,
                'n_selected': result.n_selected,
                'reduction_ratio': result.reduction_ratio,
                'cv_score': result.cv_score,
                'threshold': result.selection_threshold
            })

        return pd.DataFrame(summary_data)

    def evaluate_feature_set(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        features: List[str],
        estimator: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Evaluate a specific feature set using cross-validation.

        Args:
            X: Full feature matrix
            y: Target variable
            features: Features to evaluate
            estimator: Model to use for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        if estimator is None:
            if self.problem_type_ == 'classification':
                estimator = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.config.model_random_state
                )
                scoring = 'accuracy'
            else:
                estimator = RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.config.model_random_state
                )
                scoring = 'neg_mean_squared_error'
        else:
            scoring = self.config.cv_scoring

        # Subset features
        X_subset = X[features]

        # Cross-validation
        cv_scores = cross_val_score(
            estimator,
            X_subset,
            y,
            cv=self.config.cv_folds,
            scoring=scoring if scoring != 'auto' else (
                'accuracy' if self.problem_type_ == 'classification' else 'neg_mean_squared_error'
            )
        )

        return {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'min_score': cv_scores.min(),
            'max_score': cv_scores.max()
        }