"""SHAP explainer for model interpretability."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Union
import warnings
import base64
import io



class SHAPExplainer:
    """
    SHAP explainer for model interpretability.

    Supports tree-based models and linear models with SHAP value computation,
    visualizations, and global interpretability summaries.
    """

    def __init__(self, model: Any, background_data: Optional[pd.DataFrame] = None):
        """
        Initialize the SHAP explainer.

        Args:
            model: Trained model to explain
            background_data: Background dataset for SHAP value computation
        """
        self.model = model
        self.background_data = background_data
        self.explainer = None
        self.shap_values = None
        self.expected_value = None

        # Initialize SHAP explainer based on model type
        self._initialize_explainer()

    def _initialize_explainer(self) -> None:
        """Initialize the appropriate SHAP explainer based on model type."""
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP library is not installed. Install with: pip install shap")

        # Determine model type and initialize appropriate explainer
        model_type = str(type(self.model).__module__).split('.')[0]

        if model_type in ['sklearn', 'xgboost', 'lightgbm', 'catboost']:
            # For tree-based models
            if hasattr(self.model, 'feature_importances_') or 'forest' in str(type(self.model)).lower():
                self.explainer = shap.TreeExplainer(
                    self.model,
                    data=self.background_data,
                    feature_perturbation="interventional"
                )
            # For linear models
            elif hasattr(self.model, 'coef_'):
                self.explainer = shap.LinearExplainer(
                    self.model,
                    masker=self.background_data,
                    feature_perturbation="independent"
                )
            else:
                # Fall back to generic explainer
                self.explainer = shap.Explainer(self.model)
        else:
            # Use default explainer for unknown model types
            warnings.warn(f"Unknown model type {type(self.model)}. Using default explainer.")
            self.explainer = shap.Explainer(self.model)

    def explain(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Generate SHAP values for the given data.

        Args:
            X: Data to explain

        Returns:
            Dictionary containing SHAP values and metadata
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")

        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            if hasattr(self, 'feature_names'):
                X = pd.DataFrame(X, columns=self.feature_names)
            else:
                X = pd.DataFrame(X)

        # Calculate SHAP values
        self.shap_values = self.explainer.shap_values(X)

        # Get expected value (base value)
        if hasattr(self.explainer, 'expected_value'):
            self.expected_value = self.explainer.expected_value

        # Store feature names
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)

        return {
            'shap_values': self.shap_values,
            'expected_value': self.expected_value,
            'feature_names': getattr(self, 'feature_names', None)
        }

    def explain_instance(self, X_instance: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Generate SHAP values for a single instance.

        Args:
            X_instance: Single instance to explain

        Returns:
            Dictionary containing SHAP values for the instance
        """
        # Ensure 2D array for SHAP
        if isinstance(X_instance, pd.Series):
            feature_names = X_instance.index.tolist()
            X_instance = X_instance.values.reshape(1, -1)
        elif isinstance(X_instance, np.ndarray):
            feature_names = getattr(self, 'feature_names', [f'feature_{i}' for i in range(X_instance.shape[1])])
            X_instance = X_instance.reshape(1, -1)
        else:
            raise ValueError("X_instance must be a pandas Series or numpy array")

        # Convert to DataFrame
        X_df = pd.DataFrame(X_instance, columns=feature_names)

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_df)

        # Handle multi-output models
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Use first class for binary classification

        return {
            'shap_values': shap_values[0],  # Return as 1D array
            'expected_value': self.expected_value,
            'feature_names': feature_names,
            'prediction': self.model.predict(X_instance)[0],
            'features': X_instance[0]
        }

    def get_global_summary(self) -> Dict[str, Any]:
        """
        Get global SHAP summary statistics.

        Returns:
            Dictionary containing global SHAP statistics
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values computed. Call explain() first.")

        # Convert to numpy array if needed
        if isinstance(self.shap_values, list):
            shap_array = np.array(self.shap_values)
        else:
            shap_array = self.shap_values

        # Handle 3D SHAP values (binary/multi-class)
        if len(shap_array.shape) == 3:
            # Use the positive class (last axis) for binary classification
            shap_array = shap_array[:, :, -1] if shap_array.shape[2] == 2 else shap_array.mean(axis=2)

        # Calculate feature importance (mean absolute SHAP value)
        feature_importance = np.mean(np.abs(shap_array), axis=0)

        # Create feature ranking
        feature_names = getattr(self, 'feature_names', [f'feature_{i}' for i in range(len(feature_importance))])
        feature_ranking = sorted(
            zip(feature_names, feature_importance),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'feature_importance': dict(zip(feature_names, feature_importance)),
            'feature_ranking': feature_ranking,
            'mean_shap_values': dict(zip(feature_names, np.mean(shap_array, axis=0))),
            'std_shap_values': dict(zip(feature_names, np.std(shap_array, axis=0)))
        }

    def create_waterfall_plot(self, X_instance: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Create a waterfall plot for a single prediction.

        Args:
            X_instance: Single instance to explain

        Returns:
            Dictionary containing plot data and base64 encoded image
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP library is not installed")

        # Get SHAP values for the instance
        result = self.explain_instance(X_instance)

        # Create the waterfall plot
        plt.figure(figsize=(10, 6))

        # Create Explanation object
        explanation = shap.Explanation(
            values=result['shap_values'],
            base_values=result['expected_value'],
            data=result['features'],
            feature_names=result['feature_names']
        )

        # Generate waterfall plot
        shap.plots.waterfall(explanation, show=False)

        # Save plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return {
            'plot_base64': plot_base64,
            'explanation': result,
            'plot_type': 'waterfall'
        }

    def create_summary_plot(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Create a SHAP summary plot.

        Args:
            X: Data to summarize

        Returns:
            Dictionary containing plot data and base64 encoded image
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP library is not installed")

        # Calculate SHAP values if not already done
        if self.shap_values is None:
            self.explain(X)

        # Create the summary plot
        plt.figure(figsize=(10, 8))

        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            if hasattr(self, 'feature_names'):
                X = pd.DataFrame(X, columns=self.feature_names)
            else:
                X = pd.DataFrame(X)

        # Create Explanation object
        explanation = shap.Explanation(
            values=self.shap_values,
            base_values=self.expected_value,
            data=X.values,
            feature_names=X.columns.tolist()
        )

        # Generate summary plot
        shap.plots.beeswarm(explanation, show=False)

        # Save plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return {
            'plot_base64': plot_base64,
            'plot_type': 'summary',
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }

    def create_force_plot(self, X_instance: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Create a force plot for a single prediction.

        Args:
            X_instance: Single instance to explain

        Returns:
            Dictionary containing plot data and base64 encoded HTML
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP library is not installed")

        # Get SHAP values for the instance
        result = self.explain_instance(X_instance)

        # Create the force plot
        if isinstance(result['expected_value'], (list, np.ndarray)):
            expected_value = result['expected_value'][0] if len(result['expected_value']) > 1 else result['expected_value'][0]
        else:
            expected_value = result['expected_value']

        # Generate force plot
        force_plot = shap.plots.force(
            expected_value,
            result['shap_values'],
            result['features'],
            feature_names=result['feature_names'],
            show=False,
            matplotlib=True
        )

        # Save plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return {
            'plot_base64': plot_base64,
            'explanation': result,
            'plot_type': 'force'
        }

    def create_feature_importance_plot(self) -> Dict[str, Any]:
        """
        Create a feature importance plot based on SHAP values.

        Returns:
            Dictionary containing plot data and base64 encoded image
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values computed. Call explain() first.")

        # Get global summary
        summary = self.get_global_summary()

        # Create the plot
        plt.figure(figsize=(10, 8))

        # Sort features by importance
        features = [item[0] for item in summary['feature_ranking'][:20]]  # Top 20
        importance = [item[1] for item in summary['feature_ranking'][:20]]

        # Create horizontal bar plot
        plt.barh(range(len(features)), importance)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Mean Absolute SHAP Value')
        plt.title('Feature Importance (Mean |SHAP| Value)')
        plt.gca().invert_yaxis()

        # Save plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return {
            'plot_base64': plot_base64,
            'feature_importance': summary['feature_importance'],
            'plot_type': 'feature_importance',
            'n_features_shown': len(features)
        }