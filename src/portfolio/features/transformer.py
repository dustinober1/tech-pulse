"""
Feature transformation pipeline for machine learning preprocessing.

This module provides a comprehensive feature transformation pipeline
including scaling, encoding, missing value handling, and consistency across train/test splits.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
import pickle
from pathlib import Path
import hashlib
from datetime import datetime

# Scikit-learn imports
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
    PolynomialFeatures
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# Suppress warnings
warnings.filterwarnings('ignore')


class TransformationError(Exception):
    """Exception raised for errors in the transformation pipeline."""
    pass


class TransformationType(Enum):
    """Enumeration of transformation types."""
    SCALING = 'scaling'
    ENCODING = 'encoding'
    IMPUTATION = 'imputation'
    POLYNOMIAL = 'polynomial'
    CUSTOM = 'custom'
    IDENTITY = 'identity'


@dataclass
class TransformationSpec:
    """Specification for a single transformation."""
    name: str
    transformation_type: TransformationType
    transformer_class: Union[str, type]
    parameters: Dict[str, Any] = field(default_factory=dict)
    columns: Optional[List[str]] = None
    apply_to: List[str] = field(default_factory=lambda: ['all'])
    fitted_transformer: Optional[Any] = None
    is_fitted: bool = False
    fit_data: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class TransformationMetadata:
    """Metadata about fitted transformations."""
    fitted_at: str
    feature_mapping: Dict[str, Any] = field(default_factory=dict)
    encoder_mappings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    scaler_params: Dict[str, Any] = field(default_factory=dict)
    input_features: List[str] = field(default_factory=list)
    output_features: List[str] = field(default_factory=list)
    transformation_hash: Optional[str] = None


@dataclass
class PipelineConfig:
    """Configuration for the transformation pipeline."""
    # Missing value handling
    numeric_imputation_strategy: str = 'mean'  # 'mean', 'median', 'mode', 'constant', 'knn'
    categorical_imputation_strategy: str = 'most_frequent'
    imputation_fill_value: Optional[Union[str, float]] = None

    # Scaling
    numeric_scaling_method: str = 'standard'  # 'standard', 'minmax', 'robust', 'quantile', 'none'
    scaling_feature_range: Tuple[float, float] = (0, 1)

    # Encoding
    categorical_encoding_method: str = 'onehot'  # 'onehot', 'ordinal', 'target', 'none'
    handle_unknown_categories: str = 'ignore'  # 'error', 'ignore', 'infrequent_if_exist'
    encoding_drop_first: bool = True

    # Polynomial features
    create_polynomial_features: bool = False
    polynomial_degree: int = 2
    polynomial_include_bias: bool = True
    polynomial_interaction_only: bool = False

    # Pipeline behavior
    handle_missing_in_transform: bool = True
    verbose: bool = True

    # Persistence
    save_transformations: bool = True
    transformations_path: str = "transformations"


class FeatureTransformer:
    """
    Comprehensive feature transformation pipeline.

    Provides consistent feature transformations across train/test splits,
    including scaling, encoding, imputation, and custom transformations.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the feature transformer.

        Args:
            config: Configuration for the pipeline
        """
        self.config = config or PipelineConfig()
        self.pipeline_: Optional[Pipeline] = None
        self.fitted_: bool = False
        self.feature_types_: Dict[str, str] = {}
        self.transformations_: Dict[str, TransformationSpec] = {}
        self.metadata_: Optional[TransformationMetadata] = None
        self.original_columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureTransformer':
        """
        Fit the transformation pipeline to the data.

        Args:
            X: Input features
            y: Target variable (optional, used for target encoding)

        Returns:
            Self for method chaining
        """
        self.original_columns_ = list(X.columns)
        self._detect_feature_types(X)

        # Create a single ColumnTransformer with all steps
        self.pipeline_ = self._create_comprehensive_transformer(X, y)

        # Fit pipeline
        self.pipeline_.fit(X)
        self.fitted_ = True

        # Store metadata
        self._store_metadata(X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted pipeline.

        Args:
            X: Input features to transform

        Returns:
            Transformed features
        """
        if not self.fitted_:
            raise ValueError(
                "Transformer must be fitted before transform. Call fit() first."
            )

        # Handle missing columns
        X_transformed = self._handle_column_mismatch(X)

        # Transform using pipeline
        if self.pipeline_:
            X_transformed = self.pipeline_.transform(X_transformed)

        # Convert back to DataFrame if needed
        if isinstance(X_transformed, np.ndarray):
            X_transformed = pd.DataFrame(
                X_transformed,
                columns=self._get_output_columns(),
                index=X.index
            )

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit the transformer and transform the data.

        Args:
            X: Input features
            y: Target variable (optional)

        Returns:
            Transformed features
        """
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform the data.

        Args:
            X: Transformed features

        Returns:
            Original feature space
        """
        if not self.fitted_:
            raise ValueError(
                "Transformer must be fitted before inverse_transform."
            )

        if not self.pipeline_:
            # No transformations applied
            return X.copy()

        # Inverse transform
        X_original = self.pipeline_.inverse_transform(X)

        # Convert back to DataFrame
        if isinstance(X_original, np.ndarray):
            X_original = pd.DataFrame(
                X_original,
                columns=self.original_columns_,
                index=X.index
            )

        return X_original

    def get_feature_names_out(self) -> List[str]:
        """
        Get the output feature names after transformation.

        Returns:
            List of output feature names
        """
        if not self.fitted_:
            return []

        if not self.pipeline_:
            return self.original_columns_.copy()

        return self._get_output_columns()

    def get_feature_names_in(self) -> List[str]:
        """
        Get the input feature names expected by the transformer.

        Returns:
            List of input feature names
        """
        return self.original_columns_.copy()

    def get_transformations_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all transformations applied.

        Returns:
            Dictionary with transformation details
        """
        summary = {
            'original_features': self.original_columns_,
            'transformed_features': self.get_feature_names_out(),
            'n_transformations': len(self.transformations_),
            'feature_types': self.feature_types_,
            'fitted_at': self.metadata_.fitted_at if self.metadata_ else None,
            'transformations': {}
        }

        # Add details for each transformation
        for name, spec in self.transformations_.items():
            summary['transformations'][name] = {
                'type': spec.transformation_type.value,
                'class': spec.transformer_class,
                'parameters': spec.parameters,
                'columns': spec.columns,
                'is_fitted': spec.is_fitted
            }

        return summary

    def save_transformations(self, path: Optional[str] = None) -> str:
        """
        Save fitted transformations to disk.

        Args:
            path: Path to save transformations

        Returns:
            Path where transformations were saved
        """
        if not self.fitted_:
            raise ValueError("No transformations fitted to save")

        path = path or self.config.transformations_path

        # Create directory if it doesn't exist
        Path(path).mkdir(parents=True, exist_ok=True)

        # Save transformations dictionary
        save_data = {
            'transformations': {},
            'feature_types': self.feature_types_,
            'original_columns': self.original_columns_,
            'metadata': self.metadata_,
            'config': self.config
        }

        # Convert transformations to serializable format
        for name, spec in self.transformations_.items():
            save_data['transformations'][name] = {
                'name': spec.name,
                'transformation_type': spec.transformation_type.value,
                'transformer_class': str(spec.transformer_class),
                'parameters': spec.parameters,
                'columns': spec.columns,
                'is_fitted': spec.is_fitted,
                'fit_data': spec.fit_data
            }

            # Save fitted transformer if possible
            if spec.fitted_transformer and hasattr(spec.fitted_transformer, 'get_params'):
                save_data['transformations'][name]['fitted_params'] = (
                    spec.fitted_transformer.get_params()
                )

        # Save to file
        file_path = Path(path) / "transformations.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(save_data, f)

        return str(file_path)

    def load_transformations(self, path: str) -> 'FeatureTransformer':
        """
        Load fitted transformations from disk.

        Args:
            path: Path to load transformations from

        Returns:
            Self for method chaining
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Transformations file not found: {path}")

        # Load from file
        with open(file_path, 'rb') as f:
            save_data = pickle.load(f)

        # Restore transformations
        self.transformations_ = {}
        for name, data in save_data['transformations'].items():
            spec = TransformationSpec(
                name=data['name'],
                transformation_type=TransformationType(data['transformation_type']),
                transformer_class=eval(data['transformer_class']),
                parameters=data['parameters'],
                columns=data['columns'],
                is_fitted=data['is_fitted'],
                fit_data=data.get('fit_data', {})
            )

            # Note: We can't restore the actual fitted transformer object
            # This would require refitting the pipeline

        self.feature_types_ = save_data['feature_types']
        self.original_columns_ = save_data['original_columns']
        self.metadata_ = save_data['metadata']
        self.config = save_data['config']

        # Mark as not fitted since we can't restore the actual pipeline
        self.fitted_ = False

        return self

    def add_transformation(
        self,
        name: str,
        transformation_type: TransformationType,
        transformer_class: Union[str, type],
        columns: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        apply_to: List[str] = None
    ) -> 'FeatureTransformer':
        """
        Add a custom transformation to the pipeline.

        Args:
            name: Name of the transformation
            transformation_type: Type of transformation
            transformer_class: Transformer class
            columns: Columns to apply to (None for all)
            parameters: Parameters for the transformer
            apply_to: Types to apply to ('numeric', 'categorical', etc.)

        Returns:
            Self for method chaining
        """
        spec = TransformationSpec(
            name=name,
            transformation_type=transformation_type,
            transformer_class=transformer_class,
            parameters=parameters or {},
            columns=columns,
            apply_to=apply_to or ['all']
        )

        self.transformations_[name] = spec

        return self

    def _detect_feature_types(self, X: pd.DataFrame):
        """Detect the type of each feature."""
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                # Check if it's boolean masquerading as numeric
                if set(X[col].dropna().unique()) <= {0, 1}:
                    self.feature_types_[col] = 'boolean'
                else:
                    self.feature_types_[col] = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(X[col]):
                self.feature_types_[col] = 'datetime'
            elif X[col].dtype == 'object':
                # Check if it's text or categorical
                if X[col].str.len().mean() > 50:
                    self.feature_types_[col] = 'text'
                else:
                    self.feature_types_[col] = 'categorical'
            elif pd.api.types.is_bool_dtype(X[col]):
                self.feature_types_[col] = 'boolean'
            else:
                self.feature_types_[col] = 'unknown'

    def _get_columns_by_type(self, feature_type: str) -> List[str]:
        """Get columns of a specific type."""
        return [
            col for col, ftype in self.feature_types_.items()
            if ftype == feature_type or feature_type == 'all'
        ]

    def _create_comprehensive_transformer(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> ColumnTransformer:
        """Create a single ColumnTransformer with all necessary transformations."""
        transformers = []

        numeric_cols = self._get_columns_by_type('numeric')
        categorical_cols = self._get_columns_by_type('categorical')
        boolean_cols = self._get_columns_by_type('boolean')

        # Numeric pipeline
        if numeric_cols:
            numeric_steps = []

            # Imputation
            if self.config.numeric_imputation_strategy == 'knn':
                numeric_steps.append(('imputer', KNNImputer(n_neighbors=5)))
            elif self.config.numeric_imputation_strategy == 'constant':
                fill_value = self.config.imputation_fill_value or 0
                numeric_steps.append(('imputer', SimpleImputer(strategy='constant', fill_value=fill_value)))
            else:
                numeric_steps.append(('imputer', SimpleImputer(strategy=self.config.numeric_imputation_strategy)))

            # Scaling (if not 'none')
            if self.config.numeric_scaling_method != 'none':
                if self.config.numeric_scaling_method == 'standard':
                    numeric_steps.append(('scaler', StandardScaler()))
                elif self.config.numeric_scaling_method == 'minmax':
                    numeric_steps.append(('scaler', MinMaxScaler(feature_range=self.config.scaling_feature_range)))
                elif self.config.numeric_scaling_method == 'robust':
                    numeric_steps.append(('scaler', RobustScaler()))
                elif self.config.numeric_scaling_method == 'quantile':
                    numeric_steps.append(('scaler', QuantileTransformer()))

            # Create numeric pipeline
            from sklearn.pipeline import Pipeline
            numeric_pipeline = Pipeline(numeric_steps)
            transformers.append(('numeric', numeric_pipeline, numeric_cols))

        # Categorical pipeline
        if categorical_cols:
            categorical_steps = []

            # Imputation
            if self.config.categorical_imputation_strategy == 'constant':
                fill_value = self.config.imputation_fill_value or 'missing'
                categorical_steps.append(('imputer', SimpleImputer(
                    strategy='constant',
                    fill_value=fill_value,
                    missing_values=np.nan
                )))
            else:
                categorical_steps.append(('imputer', SimpleImputer(
                    strategy=self.config.categorical_imputation_strategy,
                    missing_values=np.nan
                )))

            # Encoding
            if self.config.categorical_encoding_method != 'none':
                if self.config.categorical_encoding_method == 'onehot':
                    drop_param = 'first' if self.config.encoding_drop_first else None
                    categorical_steps.append(('encoder', OneHotEncoder(
                        drop=drop_param,
                        handle_unknown=self.config.handle_unknown_categories,
                        sparse_output=False
                    )))
                elif self.config.categorical_encoding_method == 'ordinal':
                    handle_unknown = 'use_encoded_value' if self.config.handle_unknown_categories == 'ignore' else 'error'
                    categorical_steps.append(('encoder', OrdinalEncoder(
                        handle_unknown=handle_unknown,
                        unknown_value=-1
                    )))

            # Create categorical pipeline
            from sklearn.pipeline import Pipeline
            categorical_pipeline = Pipeline(categorical_steps)
            transformers.append(('categorical', categorical_pipeline, categorical_cols))

        # Boolean pipeline (treat as categorical but no encoding needed)
        if boolean_cols:
            transformers.append(('boolean', 'passthrough', boolean_cols))

        # Create the ColumnTransformer
        return ColumnTransformer(transformers, remainder='drop')

    def _create_preprocessors(self, X: pd.DataFrame) -> Optional[ColumnTransformer]:
        """Create preprocessing steps for imputation."""
        numeric_cols = self._get_columns_by_type('numeric')
        categorical_cols = self._get_columns_by_type('categorical')

        transformers = []

        # Numeric imputation
        if numeric_cols:
            if self.config.numeric_imputation_strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
            elif self.config.numeric_imputation_strategy == 'constant':
                fill_value = self.config.imputation_fill_value or 0
                imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
            else:
                imputer = SimpleImputer(strategy=self.config.numeric_imputation_strategy)

            transformers.append(
                ('numeric', imputer, numeric_cols)
            )

        # Categorical imputation
        if categorical_cols:
            if self.config.categorical_imputation_strategy == 'constant':
                fill_value = self.config.imputation_fill_value or 'missing'
                imputer = SimpleImputer(
                    strategy='constant',
                    fill_value=fill_value,
                    missing_values=np.nan
                )
            else:
                imputer = SimpleImputer(
                    strategy=self.config.categorical_imputation_strategy,
                    missing_values=np.nan
                )

            transformers.append(
                ('categorical', imputer, categorical_cols)
            )

        if transformers:
            return ColumnTransformer(transformers, remainder='passthrough')

        return None

    def _create_encoders(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Optional[ColumnTransformer]:
        """Create encoding steps for categorical features."""
        if self.config.categorical_encoding_method == 'none':
            return None

        categorical_cols = self._get_columns_by_type('categorical')

        if not categorical_cols:
            return None

        transformers = []

        if self.config.categorical_encoding_method == 'target' and y is not None:
            # Target encoding
            from category_encoders import TargetEncoder

            encoder = TargetEncoder(
                smoothing=1.0,
                handle_unknown=self.config.handle_unknown_categories,
                handle_missing='return_nan'
            )

            transformers.append(
                ('target', encoder, categorical_cols)
            )

        elif self.config.categorical_encoding_method == 'ordinal':
            # Ordinal encoding
            encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )

            transformers.append(
                ('ordinal', encoder, categorical_cols)
            )

        elif self.config.categorical_encoding_method == 'onehot':
            # One-hot encoding
            encoder = OneHotEncoder(
                drop=self.config.encoding_drop_first,
                handle_unknown=self.config.handle_unknown_categories,
                sparse_output=False
            )

            transformers.append(
                ('onehot', encoder, categorical_cols)
            )

        if transformers:
            return ColumnTransformer(transformers, remainder='passthrough')

        return None

    def _create_scalers(self) -> Optional[ColumnTransformer]:
        """Create scaling steps for numeric features."""
        if self.config.numeric_scaling_method == 'none':
            return None

        numeric_cols = self._get_columns_by_type('numeric')

        if not numeric_cols:
            return None

        # Filter out boolean features
        numeric_cols = [
            col for col in numeric_cols
            if self.feature_types_.get(col, 'numeric') == 'numeric'
        ]

        if not numeric_cols:
            return None

        # Choose scaler
        if self.config.numeric_scaling_method == 'standard':
            scaler = StandardScaler()
        elif self.config.numeric_scaling_method == 'minmax':
            scaler = MinMaxScaler(
                feature_range=self.config.scaling_feature_range
            )
        elif self.config.numeric_scaling_method == 'robust':
            scaler = RobustScaler()
        elif self.config.numeric_scaling_method == 'quantile':
            scaler = QuantileTransformer(
                output_distribution='normal'
            )
        else:
            return None

        return ColumnTransformer(
            [('scaler', scaler, numeric_cols)],
            remainder='passthrough'
        )

    def _create_polynomial_features(self) -> Optional[ColumnTransformer]:
        """Create polynomial features if requested."""
        if not self.config.create_polynomial_features:
            return None

        numeric_cols = self._get_columns_by_type('numeric')

        # Filter out boolean features
        numeric_cols = [
            col for col in numeric_cols
            if self.feature_types_.get(col, 'numeric') == 'numeric'
        ]

        if not numeric_cols:
            return None

        poly = PolynomialFeatures(
            degree=self.config.polynomial_degree,
            include_bias=self.config.polynomial_include_bias,
            interaction_only=self.config.polynomial_interaction_only
        )

        return ColumnTransformer(
            [('polynomial', poly, numeric_cols)],
            remainder='passthrough'
        )

    def _handle_column_mismatch(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle mismatch between expected and actual columns.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with correct columns
        """
        expected_cols = self.get_feature_names_in()
        actual_cols = list(X.columns)

        if set(expected_cols) == set(actual_cols):
            return X.copy()

        # Add missing columns with default values
        missing_cols = set(expected_cols) - set(actual_cols)
        extra_cols = set(actual_cols) - set(expected_cols)

        X_result = X.copy()

        # Add missing columns
        for col in missing_cols:
            if self.feature_types_.get(col) in ['numeric']:
                X_result[col] = 0.0
            else:
                X_result[col] = 'missing'

        # Remove extra columns
        for col in extra_cols:
            X_result = X_result.drop(columns=[col])

        # Reorder columns to match expected
        X_result = X_result[expected_cols]

        return X_result

    def _get_output_columns(self) -> List[str]:
        """Get the output column names after all transformations."""
        if not self.pipeline_ or not self.fitted_:
            return self.original_columns_.copy()

        # For ColumnTransformer, use get_feature_names_out directly
        if hasattr(self.pipeline_, 'get_feature_names_out'):
            try:
                feature_names = list(self.pipeline_.get_feature_names_out())
                # Clean up feature names (remove prefixes like "numeric__", "categorical__")
                cleaned_names = []
                for name in feature_names:
                    # Remove transformer name prefix
                    if '__' in name:
                        cleaned_names.append(name.split('__', 1)[1])
                    else:
                        cleaned_names.append(name)
                return cleaned_names
            except:
                pass

        # Fallback: try to infer from transformations
        output_cols = []

        # Get names from each transformer in the ColumnTransformer
        for name, transformer, columns in self.pipeline_.transformers_:
            if hasattr(transformer, 'get_feature_names_out'):
                try:
                    if hasattr(transformer, 'steps_') and transformer.steps_:
                        # For Pipeline transformers
                        last_step = transformer.steps_[-1][1]
                        if hasattr(last_step, 'get_feature_names_out'):
                            names = list(last_step.get_feature_names_out())
                            # Update one-hot encoded feature names
                            if name == 'categorical' and self.config.categorical_encoding_method == 'onehot':
                                for i, col in enumerate(columns):
                                    if i < len(names):
                                        output_cols.append(names[i])
                            else:
                                output_cols.extend(names)
                        else:
                            output_cols.extend(columns)
                    else:
                        # For simple transformers
                        output_cols.extend(columns)
                except:
                    output_cols.extend(columns)
            else:
                output_cols.extend(columns)

        return output_cols or self.original_columns_.copy()

    def _store_metadata(self, X: pd.DataFrame):
        """Store metadata about fitted transformations."""
        self.metadata_ = TransformationMetadata(
            fitted_at=datetime.now().isoformat(),
            input_features=list(X.columns),
            output_features=self.get_feature_names_out()
        )

        # Create hash of transformations for consistency checking
        transform_str = str(sorted(self.transformations_.items()))
        self.metadata_.transformation_hash = hashlib.md5(
            transform_str.encode()
        ).hexdigest()


class CustomTransformer(BaseEstimator, TransformerMixin):
    """Base class for custom transformers."""

    def __init__(self, func: Callable, feature_names: Optional[List[str]] = None):
        """
        Initialize custom transformer.

        Args:
            func: Function to apply to data
            feature_names: Names of output features
        """
        self.func = func
        self.feature_names = feature_names
        self.fitted_ = False

    def fit(self, X, y=None):
        """Fit the transformer (no-op for custom functions)."""
        self.fitted_ = True
        return self

    def transform(self, X):
        """Apply the custom transformation."""
        if not self.fitted_:
            raise ValueError("Transformer must be fitted before transform")

        result = self.func(X)

        # Convert to DataFrame if needed
        if isinstance(result, np.ndarray) and self.feature_names:
            result = pd.DataFrame(
                result,
                columns=self.feature_names,
                index=X.index if hasattr(X, 'index') else range(len(result))
            )

        return result

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names


class LogTransformer(CustomTransformer):
    """Logarithmic transformation for skewed data."""

    def __init__(self, offset: float = 1.0):
        """
        Initialize log transformer.

        Args:
            offset: Value to add before taking log
        """
        self.offset = offset
        super().__init__(self._log_transform, None)

    def _log_transform(self, X):
        """Apply log transformation."""
        if isinstance(X, pd.DataFrame):
            return np.log(X + self.offset)
        else:
            return np.log(X + self.offset)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            return [f"log_{i}" for i in range(10)]
        return [f"log_{col}" for col in input_features]


class BinningTransformer(CustomTransformer):
    """Binning transformer for continuous variables."""

    def __init__(self, n_bins: int = 10, strategy: str = 'uniform'):
        """
        Initialize binning transformer.

        Args:
            n_bins: Number of bins
            strategy: Binning strategy ('uniform', 'quantile', 'kmeans')
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.bin_edges_ = {}
        super().__init__(self._bin_transform, None)

    def _bin_transform(self, X):
        """Apply binning transformation."""
        if isinstance(X, pd.DataFrame):
            result = pd.DataFrame(index=X.index)
            for col in X.columns:
                if self.strategy == 'uniform':
                    edges = np.linspace(
                        X[col].min(),
                        X[col].max(),
                        self.n_bins + 1
                    )
                elif self.strategy == 'quantile':
                    edges = np.quantile(X[col], np.linspace(0, 1, self.n_bins + 1))
                else:
                    edges = np.linspace(
                        X[col].min(),
                        X[col].max(),
                        self.n_bins + 1
                    )

                self.bin_edges_[col] = edges
                # digitize returns 1 to n_bins+1, subtract 1 to get 0 to n_bins
                binned = np.digitize(X[col], edges) - 1
                # Clamp values to be within [0, n_bins-1]
                binned = np.clip(binned, 0, self.n_bins - 1)
                result[col] = binned
            return result
        else:
            # Handle numpy array
            result = []
            for i in range(X.shape[1]):
                col_name = f"col_{i}"
                if self.strategy == 'uniform':
                    edges = np.linspace(
                        X[:, i].min(),
                        X[:, i].max(),
                        self.n_bins + 1
                    )
                elif self.strategy == 'quantile':
                    edges = np.quantile(X[:, i], np.linspace(0, 1, self.n_bins + 1))
                else:
                    edges = np.linspace(
                        X[:, i].min(),
                        X[:, i].max(),
                        self.n_bins + 1
                    )

                self.bin_edges_[col_name] = edges
                # digitize returns 1 to n_bins+1, subtract 1 to get 0 to n_bins
                binned = np.digitize(X[:, i], edges) - 1
                # Clamp values to be within [0, n_bins-1]
                binned = np.clip(binned, 0, self.n_bins - 1)
                result.append(binned)
            return np.array(result).T

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            return [f"bin_{i}" for i in range(len(self.bin_edges_))]
        return input_features


class CyclicalTransformer(CustomTransformer):
    """Cyclical transformer for periodic features."""

    def __init__(self, max_values: Optional[Dict[str, float]] = None):
        """
        Initialize cyclic transformer.

        Args:
            max_values: Maximum values for each feature
        """
        self.max_values = max_values or {}
        super().__init__(self._cyclic_transform, None)

    def _cyclic_transform(self, X):
        """Apply cyclical transformation."""
        if isinstance(X, pd.DataFrame):
            result = pd.DataFrame(index=X.index)
            for col in X.columns:
                max_val = self.max_values.get(col, X[col].max() if X[col].max() > 0 else 2 * np.pi)
                if max_val == 0:
                    max_val = 2 * np.pi
                result[f"{col}_sin"] = np.sin(2 * np.pi * X[col] / max_val)
                result[f"{col}_cos"] = np.cos(2 * np.pi * X[col] / max_val)
            return result
        else:
            # Handle numpy array
            result = []
            for i in range(X.shape[1]):
                col_name = f"col_{i}"
                max_val = self.max_values.get(col_name, 2 * np.pi)
                sin_col = np.sin(2 * np.pi * X[:, i] / max_val)
                cos_col = np.cos(2 * np.pi * X[:, i] / max_val)
                result.extend([sin_col, cos_col])
            return np.array(result).T

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            return []

        names = []
        for col in input_features:
            names.extend([f"{col}_sin", f"{col}_cos"])
        return names