"""Partial dependence plot generator for model interpretability."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Union, Tuple
import warnings
import base64
import io
from itertools import product, combinations

from sklearn.inspection import partial_dependence
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.base import is_classifier, is_regressor

from config import get_config


class PartialDependencePlotter:
    """
    Partial dependence plot generator for model interpretability.

    Generates 1D and 2D partial dependence plots and individual
    conditional expectation (ICE) plots to understand model behavior.
    """

    def __init__(self, model: Any, X: pd.DataFrame):
        """
        Initialize the partial dependence plotter.

        Args:
            model: Trained model to analyze
            X: Training data used for the model
        """
        self.model = model
        self.X = X
        self.feature_names = list(X.columns)
        self.n_features = X.shape[1]
        self.is_classification = is_classifier(model)
        self.is_regression = is_regressor(model)

        # Store feature types
        self.categorical_features = []
        self.numerical_features = []
        self._identify_feature_types()

    def _identify_feature_types(self) -> None:
        """Identify categorical and numerical features."""
        for i, col in enumerate(self.X.columns):
            if self.X[col].dtype in ['object', 'category', 'bool']:
                self.categorical_features.append(i)
            elif pd.api.types.is_numeric_dtype(self.X[col]) and self.X[col].nunique() > 10:
                self.numerical_features.append(i)
            else:
                # Treat low-cardinality numeric features as categorical
                self.categorical_features.append(i)

    def plot_1d_partial_dependence(
        self,
        feature: Union[str, int],
        target_class: Optional[int] = None,
        n_points: int = 100,
        percentile_range: Tuple[float, float] = (0.05, 0.95),
        include_ice: bool = False,
        n_ice_samples: int = 50,
        figsize: Tuple[int, int] = (10, 6)
    ) -> Dict[str, Any]:
        """
        Generate 1D partial dependence plot.

        Args:
            feature: Feature name or index to plot
            target_class: Target class for classification (None for regression)
            n_points: Number of points in the grid
            percentile_range: Percentile range for feature values
            include_ice: Whether to include ICE curves
            n_ice_samples: Number of ICE samples to plot
            figsize: Figure size (width, height)

        Returns:
            Dictionary containing plot data and base64 encoded image
        """
        # Convert feature to index if needed
        if isinstance(feature, str):
            feature_idx = self.feature_names.index(feature)
        else:
            feature_idx = feature
            feature = self.feature_names[feature_idx]

        plt.figure(figsize=figsize)

        # Calculate partial dependence
        pdp_result = partial_dependence(
            self.model,
            self.X,
            features=[feature_idx],
            kind='average' if not include_ice else 'both',
            grid_resolution=n_points,
            percentiles=percentile_range,
            method='brute'
        )

        # Extract values and predictions
        feature_values = pdp_result['values'][0].flatten()
        pdp_values = pdp_result['average'][0].flatten()

        # Determine target for classification
        if self.is_classification and target_class is None:
            target_class = 0  # Use first class by default

        # Plot partial dependence
        plt.plot(feature_values, pdp_values, 'b-', linewidth=2, label='Partial Dependence')

        # Add ICE curves if requested
        if include_ice and 'individual' in pdp_result:
            # Sample ICE curves to avoid overcrowding
            ice_curves = pdp_result['individual'][0]
            n_curves = min(len(ice_curves), n_ice_samples)
            sample_indices = np.random.choice(
                len(ice_curves),
                size=n_curves,
                replace=False
            )

            for idx in sample_indices:
                plt.plot(
                    feature_values,
                    ice_curves[idx].flatten(),
                    'gray',
                    alpha=0.1,
                    linewidth=0.5
                )

            # Add legend for ICE
            plt.plot([], [], 'gray', alpha=0.3, label=f'ICE curves (n={n_curves})')

        # Formatting
        plt.xlabel(feature)
        ylabel = 'Partial Dependence'
        if self.is_classification:
            ylabel += f' (Class {target_class})'
        plt.ylabel(ylabel)
        plt.title(f'1D Partial Dependence: {feature}')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return {
            'plot_base64': plot_base64,
            'feature': feature,
            'feature_values': feature_values.tolist(),
            'pdp_values': pdp_values.tolist(),
            'plot_type': '1d_partial_dependence',
            'include_ice': include_ice,
            'target_class': target_class
        }

    def plot_2d_partial_dependence(
        self,
        features: Union[List[str], List[int]],
        target_class: Optional[int] = None,
        n_points: Tuple[int, int] = (50, 50),
        percentile_range: Tuple[float, float] = (0.05, 0.95),
        figsize: Tuple[int, int] = (10, 8),
        plot_type: str = 'contour'
    ) -> Dict[str, Any]:
        """
        Generate 2D partial dependence plot.

        Args:
            features: Two features to plot (names or indices)
            target_class: Target class for classification (None for regression)
            n_points: Number of points in the grid (n_x, n_y)
            percentile_range: Percentile range for feature values
            figsize: Figure size (width, height)
            plot_type: Type of plot ('contour' or 'heatmap')

        Returns:
            Dictionary containing plot data and base64 encoded image
        """
        if len(features) != 2:
            raise ValueError("Exactly two features must be specified for 2D partial dependence")

        # Convert features to indices if needed
        feature_indices = []
        feature_names = []
        for f in features:
            if isinstance(f, str):
                feature_indices.append(self.feature_names.index(f))
                feature_names.append(f)
            else:
                feature_indices.append(f)
                feature_names.append(self.feature_names[f])

        plt.figure(figsize=figsize)

        # Calculate 2D partial dependence
        pdp_result = partial_dependence(
            self.model,
            self.X,
            features=feature_indices,
            grid_resolution=n_points,
            percentiles=percentile_range,
            method='brute'
        )

        # Extract grid values and predictions
        XX, YY = np.meshgrid(pdp_result['values'][0], pdp_result['values'][1])
        Z = pdp_result['average'].T

        # Determine target for classification
        if self.is_classification and target_class is None:
            target_class = 0  # Use first class by default

        # Create plot
        if plot_type == 'contour':
            contour = plt.contour(XX, YY, Z, levels=15, alpha=0.8, linewidths=1)
            plt.clabel(contour, inline=True, fontsize=8)
            csf = plt.contourf(XX, YY, Z, levels=15, alpha=0.3, cmap='viridis')
            plt.colorbar(csf, label='Partial Dependence')
        elif plot_type == 'heatmap':
            im = plt.imshow(
                Z,
                extent=[XX.min(), XX.max(), YY.min(), YY.max()],
                origin='lower',
                aspect='auto',
                cmap='viridis'
            )
            plt.colorbar(im, label='Partial Dependence')
        else:
            raise ValueError(f"Invalid plot_type: {plot_type}")

        # Formatting
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        title = f'2D Partial Dependence: {feature_names[0]} vs {feature_names[1]}'
        if self.is_classification:
            title += f' (Class {target_class})'
        plt.title(title)
        plt.grid(True, alpha=0.3)

        # Save plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return {
            'plot_base64': plot_base64,
            'features': feature_names,
            'grid_x': XX.tolist(),
            'grid_y': YY.tolist(),
            'grid_z': Z.tolist(),
            'plot_type': '2d_partial_dependence',
            'plot_subtype': plot_type,
            'target_class': target_class
        }

    def plot_ice_curves(
        self,
        feature: Union[str, int],
        n_samples: int = 100,
        n_points: int = 100,
        percentile_range: Tuple[float, float] = (0.05, 0.95),
        centered: bool = False,
        figsize: Tuple[int, int] = (10, 6)
    ) -> Dict[str, Any]:
        """
        Generate Individual Conditional Expectation (ICE) curves.

        Args:
            feature: Feature name or index to plot
            n_samples: Number of ICE samples to generate
            n_points: Number of points in the grid
            percentile_range: Percentile range for feature values
            centered: Whether to center the ICE curves
            figsize: Figure size (width, height)

        Returns:
            Dictionary containing plot data and base64 encoded image
        """
        # Convert feature to index if needed
        if isinstance(feature, str):
            feature_idx = self.feature_names.index(feature)
        else:
            feature_idx = feature
            feature = self.feature_names[feature_idx]

        plt.figure(figsize=figsize)

        # Calculate ICE curves
        pdp_result = partial_dependence(
            self.model,
            self.X,
            features=[feature_idx],
            kind='individual',
            grid_resolution=n_points,
            percentiles=percentile_range,
            method='brute'
        )

        # Extract values and ICE curves
        feature_values = pdp_result['values'][0].flatten()
        ice_curves = pdp_result['individual'][0]

        # Sample ICE curves to avoid overcrowding
        n_curves = min(len(ice_curves), n_samples)
        sample_indices = np.random.choice(len(ice_curves), size=n_curves, replace=False)

        # Plot ICE curves
        for idx in sample_indices:
            ice_values = ice_curves[idx].flatten()
            if centered:
                ice_values = ice_values - ice_values[0]  # Center at first value
            plt.plot(
                feature_values,
                ice_values,
                'b-',
                alpha=0.2,
                linewidth=0.8
            )

        # Calculate and plot average curve
        avg_curve = np.mean([ice_curves[i].flatten() for i in range(len(ice_curves))], axis=0)
        if centered:
            avg_curve = avg_curve - avg_curve[0]
        plt.plot(
            feature_values,
            avg_curve,
            'r-',
            linewidth=3,
            label='Average'
        )

        # Formatting
        plt.xlabel(feature)
        plt.ylabel(f'{"Centered " if centered else ""}ICE Values')
        title = f'{"Centered " if centered else ""}ICE Curves: {feature}'
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return {
            'plot_base64': plot_base64,
            'feature': feature,
            'feature_values': feature_values.tolist(),
            'plot_type': 'ice_curves',
            'centered': centered,
            'n_curves_plotted': n_curves
        }

    def plot_multiple_features(
        self,
        features: List[Union[str, int]],
        target_class: Optional[int] = None,
        n_cols: int = 3,
        figsize: Optional[Tuple[int, int]] = None,
        n_points: int = 100,
        percentile_range: Tuple[float, float] = (0.05, 0.95)
    ) -> Dict[str, Any]:
        """
        Generate partial dependence plots for multiple features.

        Args:
            features: List of features to plot (names or indices)
            target_class: Target class for classification
            n_cols: Number of columns in the subplot grid
            figsize: Figure size (width, height)
            n_points: Number of points in the grid
            percentile_range: Percentile range for feature values

        Returns:
            Dictionary containing plot data and base64 encoded image
        """
        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols

        if figsize is None:
            figsize = (5 * n_cols, 4 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        # Flatten axes array for easier indexing
        axes_flat = axes.flatten() if n_features > 1 else axes

        # Determine target for classification
        if self.is_classification and target_class is None:
            target_class = 0

        # Generate plots for each feature
        for i, feature in enumerate(features):
            ax = axes_flat[i] if n_features > 1 else axes

            # Convert feature to index if needed
            if isinstance(feature, str):
                feature_idx = self.feature_names.index(feature)
                feature_name = feature
            else:
                feature_idx = feature
                feature_name = self.feature_names[feature_idx]

            # Calculate partial dependence
            pdp_result = partial_dependence(
                self.model,
                self.X,
                features=[feature_idx],
                grid_resolution=n_points,
                percentiles=percentile_range,
                method='brute'
            )

            # Extract values and predictions
            feature_values = pdp_result['values'][0].flatten()
            pdp_values = pdp_result['average'][0].flatten()

            # Plot
            ax.plot(feature_values, pdp_values, 'b-', linewidth=2)
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Partial Dependence')
            title = f'PD: {feature_name}'
            if self.is_classification:
                title += f' (Class {target_class})'
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_features, len(axes_flat)):
            axes_flat[i].set_visible(False)

        plt.tight_layout()

        # Save plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return {
            'plot_base64': plot_base64,
            'features': [f if isinstance(f, str) else self.feature_names[f] for f in features],
            'plot_type': 'multiple_partial_dependence',
            'n_features': n_features,
            'target_class': target_class
        }

    def get_feature_importance_from_pdp(
        self,
        n_points: int = 100,
        percentile_range: Tuple[float, float] = (0.05, 0.95)
    ) -> Dict[str, Any]:
        """
        Calculate feature importance based on partial dependence range.

        Args:
            n_points: Number of points in the grid
            percentile_range: Percentile range for feature values

        Returns:
            Dictionary containing feature importance scores
        """
        importance_scores = {}
        feature_ranges = {}

        for feature_idx, feature_name in enumerate(self.feature_names):
            # Calculate partial dependence
            pdp_result = partial_dependence(
                self.model,
                self.X,
                features=[feature_idx],
                grid_resolution=n_points,
                percentiles=percentile_range,
                method='brute'
            )

            # Calculate importance as range of PDP values
            pdp_values = pdp_result['average'][0].flatten()
            importance = np.max(pdp_values) - np.min(pdp_values)
            feature_range = np.max(pdp_result['values'][0]) - np.min(pdp_result['values'][0])

            importance_scores[feature_name] = importance
            feature_ranges[feature_name] = feature_range

        # Sort features by importance
        sorted_features = sorted(
            importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'feature_importance': importance_scores,
            'feature_ranges': feature_ranges,
            'feature_ranking': sorted_features
        }

    def generate_interaction_summary(
        self,
        n_interactions: int = 20,
        n_points: Tuple[int, int] = (20, 20),
        percentile_range: Tuple[float, float] = (0.05, 0.95)
    ) -> Dict[str, Any]:
        """
        Generate summary of 2-way feature interactions.

        Args:
            n_interactions: Number of interactions to analyze
            n_points: Grid resolution for 2D PDP
            percentile_range: Percentile range for feature values

        Returns:
            Dictionary containing interaction summary
        """
        if self.n_features < 2:
            return {'error': 'Need at least 2 features for interaction analysis'}

        # Sample feature pairs
        all_pairs = list(combinations(range(self.n_features), 2))
        n_pairs_to_analyze = min(n_interactions, len(all_pairs))
        selected_pairs = np.random.choice(
            len(all_pairs),
            size=n_pairs_to_analyze,
            replace=False
        )

        interaction_strengths = {}
        interaction_details = {}

        for pair_idx in selected_pairs:
            feat1_idx, feat2_idx = all_pairs[pair_idx]
            feat1_name = self.feature_names[feat1_idx]
            feat2_name = self.feature_names[feat2_idx]

            # Calculate 2D partial dependence
            pdp_result = partial_dependence(
                self.model,
                self.X,
                features=[feat1_idx, feat2_idx],
                grid_resolution=n_points,
                percentiles=percentile_range,
                method='brute'
            )

            # Calculate interaction strength as variance of predictions
            Z = pdp_result['average']
            interaction_strength = np.var(Z)
            feature_pair = f"{feat1_name} x {feat2_name}"

            interaction_strengths[feature_pair] = interaction_strength
            interaction_details[feature_pair] = {
                'feature_1': feat1_name,
                'feature_2': feat2_name,
                'interaction_strength': interaction_strength,
                'grid_shape': Z.shape
            }

        # Sort interactions by strength
        sorted_interactions = sorted(
            interaction_strengths.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'interaction_strengths': interaction_strengths,
            'interaction_details': interaction_details,
            'top_interactions': sorted_interactions[:10],
            'n_interactions_analyzed': n_pairs_to_analyze
        }