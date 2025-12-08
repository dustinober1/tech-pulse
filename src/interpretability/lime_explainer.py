"""LIME explainer for local model interpretability."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Union, Tuple
import warnings
import base64
import io

from config import get_config


class LIMEExplainer:
    """
    LIME explainer for local model interpretability.

    Provides local explanations for individual predictions using LIME
    (Local Interpretable Model-agnostic Explanations) for tabular,
    text, and image data.
    """

    def __init__(
        self,
        mode: str = 'tabular',
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None,
        discretize_continuous: bool = True,
        kernel_width: Optional[float] = None,
        verbose: bool = False
    ):
        """
        Initialize the LIME explainer.

        Args:
            mode: Type of explanation ('tabular', 'text', or 'image')
            feature_names: List of feature names
            categorical_features: List of categorical feature indices
            discretize_continuous: Whether to discretize continuous features
            kernel_width: Width of the exponential kernel
            verbose: Whether to print verbose output
        """
        self.mode = mode
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.discretize_continuous = discretize_continuous
        self.kernel_width = kernel_width
        self.verbose = verbose
        self.explainer = None
        self.training_data = None

    def fit(self, model: Any, training_data: Union[pd.DataFrame, np.ndarray]) -> 'LIMEExplainer':
        """
        Fit the LIME explainer with a model and training data.

        Args:
            model: Trained model to explain
            training_data: Training data for the model

        Returns:
            Self for method chaining
        """
        try:
            import lime
        except ImportError:
            raise ImportError("LIME library is not installed. Install with: pip install lime")

        self.model = model
        self.training_data = training_data

        # Convert training data to numpy if needed
        if isinstance(training_data, pd.DataFrame):
            self.training_data_np = training_data.values
            if self.feature_names is None:
                self.feature_names = list(training_data.columns)
        else:
            self.training_data_np = training_data
            if self.feature_names is None:
                self.feature_names = [f'feature_{i}' for i in range(training_data.shape[1])]

        # Initialize appropriate LIME explainer
        if self.mode == 'tabular':
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                self.training_data_np,
                feature_names=self.feature_names,
                categorical_features=self.categorical_features,
                discretize_continuous=self.discretize_continuous,
                kernel_width=self.kernel_width,
                verbose=self.verbose,
                mode='classification' if self._is_classification() else 'regression'
            )
        elif self.mode == 'text':
            self.explainer = lime.lime_text.LimeTextExplainer(
                verbose=self.verbose,
                class_names=['Not Positive', 'Positive']  # Default class names
            )
        elif self.mode == 'image':
            self.explainer = lime.lime_image.LimeImageExplainer(
                verbose=self.verbose,
                random_state=42
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'tabular', 'text', or 'image'")

        return self

    def explain_instance(
        self,
        data_row: Union[pd.Series, np.ndarray, str],
        predict_fn: Optional[callable] = None,
        num_features: int = 5,
        num_samples: int = 5000,
        distance_metric: str = 'euclidean',
        model_regressor: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a local explanation for a single instance.

        Args:
            data_row: Instance to explain
            predict_fn: Prediction function (uses model.predict_proba if None)
            num_features: Number of features to include in explanation
            num_samples: Number of perturbed samples to generate
            distance_metric: Distance metric for weighting samples
            model_regressor: Model for fitting local explanation

        Returns:
            Dictionary containing explanation results
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call fit() first.")

        # Use model's prediction function if none provided
        if predict_fn is None:
            predict_fn = self._get_predict_function()

        # Generate explanation
        if self.mode == 'tabular':
            # Convert pandas Series to numpy if needed
            if isinstance(data_row, pd.Series):
                data_row = data_row.values

            explanation = self.explainer.explain_instance(
                data_row=data_row,
                predict_fn=predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                distance_metric=distance_metric,
                model_regressor=model_regressor
            )
        elif self.mode == 'text':
            explanation = self.explainer.explain_instance(
                text_instance=data_row,
                classifier_fn=predict_fn,
                num_features=num_features,
                num_samples=num_samples
            )
        elif self.mode == 'image':
            explanation = self.explainer.explain_instance(
                image=data_row,
                classifier_fn=predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=num_samples
            )

        return self._parse_explanation(explanation, data_row)

    def explain_multiple(
        self,
        data: Union[pd.DataFrame, np.ndarray, List[str]],
        predict_fn: Optional[callable] = None,
        num_features: int = 5,
        num_samples: int = 5000
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for multiple instances.

        Args:
            data: Multiple instances to explain
            predict_fn: Prediction function
            num_features: Number of features per explanation
            num_samples: Number of perturbed samples per explanation

        Returns:
            List of explanation dictionaries
        """
        explanations = []

        if isinstance(data, pd.DataFrame):
            data = data.values

        for i, instance in enumerate(data):
            explanation = self.explain_instance(
                instance,
                predict_fn=predict_fn,
                num_features=num_features,
                num_samples=num_samples
            )
            explanation['instance_index'] = i
            explanations.append(explanation)

        return explanations

    def create_explanation_plot(
        self,
        explanation_data: Dict[str, Any],
        max_display: int = 20,
        figsize: Tuple[int, int] = (10, 6)
    ) -> Dict[str, Any]:
        """
        Create a visualization of the LIME explanation.

        Args:
            explanation_data: Explanation dictionary from explain_instance
            max_display: Maximum number of features to display
            figsize: Figure size (width, height)

        Returns:
            Dictionary containing plot data and base64 encoded image
        """
        plt.figure(figsize=figsize)

        # Extract feature contributions
        features = explanation_data['feature_contributions']
        contributions = [abs(contrib) for contrib in explanation_data['feature_values']]
        signs = ['positive' if contrib > 0 else 'negative'
                for contrib in explanation_data['feature_values']]

        # Sort by absolute contribution
        sorted_idx = np.argsort(contributions)[-max_display:]
        sorted_features = [features[i] for i in sorted_idx]
        sorted_contributions = [explanation_data['feature_values'][i] for i in sorted_idx]
        sorted_signs = [signs[i] for i in sorted_idx]

        # Create horizontal bar chart
        colors = ['green' if sign == 'positive' else 'red' for sign in sorted_signs]
        y_pos = np.arange(len(sorted_features))

        plt.barh(y_pos, sorted_contributions, color=colors, alpha=0.7)
        plt.yticks(y_pos, sorted_features)
        plt.xlabel('Contribution to Prediction')
        plt.title(f'LIME Explanation (Predicted: {explanation_data["predicted_value"]:.3f})')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(axis='x', alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Positive contribution'),
            Patch(facecolor='red', alpha=0.7, label='Negative contribution')
        ]
        plt.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()

        # Save plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return {
            'plot_base64': plot_base64,
            'explanation_data': explanation_data,
            'plot_type': 'lime_explanation'
        }

    def get_feature_importance_summary(
        self,
        explanations: List[Dict[str, Any]],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Get a summary of feature importance across multiple explanations.

        Args:
            explanations: List of explanation dictionaries
            top_k: Number of top features to return

        Returns:
            Dictionary containing feature importance summary
        """
        # Aggregate feature contributions
        feature_scores = {}
        feature_counts = {}

        for explanation in explanations:
            for feature, value in zip(
                explanation['feature_contributions'],
                explanation['feature_values']
            ):
                if feature not in feature_scores:
                    feature_scores[feature] = []
                    feature_counts[feature] = 0
                feature_scores[feature].append(abs(value))
                feature_counts[feature] += 1

        # Calculate average importance and ranking
        avg_importance = {}
        for feature in feature_scores:
            avg_importance[feature] = np.mean(feature_scores[feature])

        # Sort by average importance
        sorted_features = sorted(
            avg_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return {
            'top_features': dict(sorted_features),
            'feature_appearance_counts': feature_counts,
            'total_explanations': len(explanations)
        }

    def _is_classification(self) -> bool:
        """Check if the model is a classifier."""
        if hasattr(self.model, 'predict_proba'):
            return True
        elif hasattr(self.model, '_estimator_type'):
            return self.model._estimator_type == 'classifier'
        else:
            # Default to classification for ambiguous cases
            return True

    def _get_predict_function(self) -> callable:
        """Get the appropriate prediction function for the model."""
        if self._is_classification():
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba
            else:
                return self.model.predict
        else:
            return self.model.predict

    def _parse_explanation(self, explanation, data_row) -> Dict[str, Any]:
        """Parse LIME explanation object into a dictionary."""
        if self.mode == 'tabular':
            # Get explanation details
            if hasattr(explanation, 'as_list'):
                exp_list = explanation.as_list()
                features = [item[0] for item in exp_list]
                values = [item[1] for item in exp_list]
            else:
                # Fallback for older LIME versions
                features = []
                values = []
                for feature_idx, value in explanation.local_exp[1]:
                    feature_name = self.feature_names[feature_idx]
                    features.append(f"{feature_name} {'>' if value > 0 else '<='} {explanation.scaled_data[feature_idx]:.3f}")
                    values.append(value)

            # Get prediction
            if hasattr(explanation, 'predicted_proba'):
                predicted_value = explanation.predicted_proba[1]  # Probability of positive class
            else:
                predicted_value = explanation.predicted_value

            # Get intercept
            intercept = explanation.intercept[1] if hasattr(explanation, 'intercept') else 0.0

        elif self.mode == 'text':
            exp_list = explanation.as_list()
            features = [item[0] for item in exp_list]
            values = [item[1] for item in exp_list]
            predicted_value = explanation.score
            intercept = explanation.intercept

        elif self.mode == 'image':
            # For image explanations, return segmentation mask info
            return {
                'type': 'image_explanation',
                'explanation': explanation,
                'data_row': data_row,
                'predicted_value': explanation.top_labels[0] if hasattr(explanation, 'top_labels') else None
            }

        return {
            'type': 'tabular_explanation' if self.mode == 'tabular' else 'text_explanation',
            'feature_contributions': features,
            'feature_values': values,
            'predicted_value': predicted_value,
            'intercept': intercept,
            'local_pred': explanation.local_pred if hasattr(explanation, 'local_pred') else None,
            'score': explanation.score if hasattr(explanation, 'score') else None
        }