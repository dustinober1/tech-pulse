"""Interpretability report generator for comprehensive model explanations."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Union
import warnings
import base64
import io
from datetime import datetime
import json

from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer
from .partial_dependence import PartialDependencePlotter


class InterpretabilityReport:
    """
    Comprehensive interpretability report generator.

    Combines SHAP values, LIME explanations, and partial dependence
    plots to create detailed model interpretability reports.
    """

    def __init__(
        self,
        model: Any,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        Initialize the interpretability report generator.

        Args:
            model: Trained model to analyze
            X: Training/test data for the model
            y: Target values (optional)
            model_name: Name of the model
            model_type: Type of model (classification/regression)
            description: Description of the model purpose
        """
        self.model = model
        self.X = X
        self.y = y
        self.model_name = model_name or self._detect_model_name()
        self.model_type = model_type or self._detect_model_type()
        self.description = description or "Machine learning model"

        # Initialize interpreters
        self.shap_explainer = None
        self.lime_explainer = None
        self.pdp_plotter = None

        # Report data storage
        self.report_data = {
            'metadata': {
                'model_name': self.model_name,
                'model_type': self.model_type,
                'description': self.description,
                'generated_at': datetime.now().isoformat(),
                'n_features': X.shape[1],
                'n_samples': X.shape[0]
            },
            'global_interpretability': {},
            'local_interpretability': {},
            'feature_analysis': {},
            'visualizations': {}
        }

    def _detect_model_name(self) -> str:
        """Detect the model name from the model object."""
        return type(self.model).__name__

    def _detect_model_type(self) -> str:
        """Detect if the model is for classification or regression."""
        from sklearn.base import is_classifier, is_regressor

        if is_classifier(self.model):
            return 'classification'
        elif is_regressor(self.model):
            return 'regression'
        else:
            # Fallback: check if target is available and infer from it
            if self.y is not None:
                if pd.api.types.is_numeric_dtype(self.y) and self.y.nunique() > 10:
                    return 'regression'
                else:
                    return 'classification'
            return 'unknown'

    def initialize_explainers(
        self,
        background_data_size: int = 100,
        lime_mode: str = 'tabular'
    ) -> 'InterpretabilityReport':
        """
        Initialize all interpreters.

        Args:
            background_data_size: Size of background data for SHAP
            lime_mode: Mode for LIME explainer

        Returns:
            Self for method chaining
        """
        # Initialize SHAP explainer
        try:
            background_data = self.X.sample(
                min(background_data_size, len(self.X)),
                random_state=42
            )
            self.shap_explainer = SHAPExplainer(self.model, background_data)
            self.report_data['metadata']['shap_available'] = True
        except Exception as e:
            warnings.warn(f"Could not initialize SHAP explainer: {e}")
            self.report_data['metadata']['shap_available'] = False

        # Initialize LIME explainer
        try:
            self.lime_explainer = LIMEExplainer(
                mode=lime_mode,
                feature_names=list(self.X.columns)
            )
            self.lime_explainer.fit(self.model, self.X)
            self.report_data['metadata']['lime_available'] = True
        except Exception as e:
            warnings.warn(f"Could not initialize LIME explainer: {e}")
            self.report_data['metadata']['lime_available'] = False

        # Initialize partial dependence plotter
        try:
            self.pdp_plotter = PartialDependencePlotter(self.model, self.X)
            self.report_data['metadata']['pdp_available'] = True
        except Exception as e:
            warnings.warn(f"Could not initialize PDP plotter: {e}")
            self.report_data['metadata']['pdp_available'] = False

        return self

    def analyze_global_interpretability(
        self,
        n_samples: int = 100,
        top_k_features: int = 10
    ) -> 'InterpretabilityReport':
        """
        Analyze global model interpretability.

        Args:
            n_samples: Number of samples to analyze
            top_k_features: Number of top features to highlight

        Returns:
            Self for method chaining
        """
        if self.shap_explainer:
            # Calculate SHAP values
            sample_data = self.X.sample(min(n_samples, len(self.X)), random_state=42)
            shap_result = self.shap_explainer.explain(sample_data)

            # Get global summary
            global_summary = self.shap_explainer.get_global_summary()

            self.report_data['global_interpretability']['shap'] = {
                'feature_importance': dict(list(global_summary['feature_importance'].items())[:top_k_features]),
                'feature_ranking': global_summary['feature_ranking'][:top_k_features],
                'mean_shap_values': global_summary['mean_shap_values'],
                'std_shap_values': global_summary['std_shap_values']
            }

        if self.pdp_plotter:
            # Get feature importance from PDP
            pdp_importance = self.pdp_plotter.get_feature_importance_from_pdp()
            self.report_data['global_interpretability']['pdp_importance'] = {
                'feature_importance': dict(list(pdp_importance['feature_importance'].items())[:top_k_features]),
                'feature_ranking': pdp_importance['feature_ranking'][:top_k_features]
            }

            # Analyze feature interactions
            if self.X.shape[1] > 1:
                interactions = self.pdp_plotter.generate_interaction_summary(n_interactions=5)
                self.report_data['global_interpretability']['feature_interactions'] = interactions

        # Calculate basic statistics
        if self.y is not None:
            self.report_data['global_interpretability']['target_stats'] = {
                'mean': float(self.y.mean()),
                'std': float(self.y.std()),
                'min': float(self.y.min()),
                'max': float(self.y.max()),
                'unique_values': int(self.y.nunique())
            }

        return self

    def analyze_local_interpretability(
        self,
        n_instances: int = 5,
        instance_indices: Optional[List[int]] = None
    ) -> 'InterpretabilityReport':
        """
        Analyze local interpretability for specific instances.

        Args:
            n_instances: Number of instances to analyze
            instance_indices: Specific indices to analyze (overrides n_instances)

        Returns:
            Self for method chaining
        """
        # Select instances to analyze
        if instance_indices:
            indices = instance_indices
        else:
            indices = np.random.choice(
                len(self.X),
                size=min(n_instances, len(self.X)),
                replace=False
            )

        self.report_data['local_interpretability']['instances'] = {}

        for idx in indices:
            instance_data = self.X.iloc[idx:idx+1]
            instance_info = {
                'index': int(idx),
                'features': instance_data.iloc[0].to_dict()
            }

            # SHAP explanation
            if self.shap_explainer:
                try:
                    shap_instance = self.shap_explainer.explain_instance(instance_data.iloc[0])
                    instance_info['shap'] = {
                        'feature_contributions': dict(zip(
                            shap_instance['feature_names'],
                            shap_instance['shap_values'].tolist()
                        )) if isinstance(shap_instance['shap_values'], np.ndarray) else {},
                        'prediction': float(shap_instance['prediction']),
                        'expected_value': float(shap_instance['expected_value']) if isinstance(shap_instance['expected_value'], (int, float)) else 0.0
                    }
                except Exception as e:
                    warnings.warn(f"Could not generate SHAP explanation for instance {idx}: {e}")

            # LIME explanation
            if self.lime_explainer:
                try:
                    lime_instance = self.lime_explainer.explain_instance(
                        instance_data.iloc[0],
                        num_features=5
                    )
                    instance_info['lime'] = {
                        'feature_contributions': dict(zip(
                            lime_instance['feature_contributions'],
                            lime_instance['feature_values']
                        )),
                        'predicted_value': float(lime_instance['predicted_value']),
                        'score': float(lime_instance['score']) if 'score' in lime_instance else None
                    }
                except Exception as e:
                    warnings.warn(f"Could not generate LIME explanation for instance {idx}: {e}")

            self.report_data['local_interpretability']['instances'][int(idx)] = instance_info

        return self

    def analyze_features(self) -> 'InterpretabilityReport':
        """
        Analyze feature characteristics and relationships.

        Returns:
            Self for method chaining
        """
        feature_analysis = {}

        for feature in self.X.columns:
            feature_info = {
                'dtype': str(self.X[feature].dtype),
                'missing_count': int(self.X[feature].isnull().sum()),
                'missing_percentage': float(self.X[feature].isnull().sum() / len(self.X) * 100),
                'unique_count': int(self.X[feature].nunique()),
                'cardinality': 'high' if self.X[feature].nunique() > len(self.X) * 0.5 else 'low'
            }

            # Numerical features
            if pd.api.types.is_numeric_dtype(self.X[feature]):
                feature_info.update({
                    'mean': float(self.X[feature].mean()),
                    'std': float(self.X[feature].std()),
                    'min': float(self.X[feature].min()),
                    'max': float(self.X[feature].max()),
                    'skewness': float(self.X[feature].skew()),
                    'kurtosis': float(self.X[feature].kurtosis())
                })

            # Categorical features
            else:
                feature_info.update({
                    'most_frequent': str(self.X[feature].mode().iloc[0]) if not self.X[feature].mode().empty else None,
                    'frequency_distribution': self.X[feature].value_counts().head(10).to_dict()
                })

            feature_analysis[feature] = feature_info

        self.report_data['feature_analysis'] = feature_analysis
        return self

    def generate_visualizations(
        self,
        n_plots: int = 10,
        plot_types: Optional[List[str]] = None
    ) -> 'InterpretabilityReport':
        """
        Generate visualizations for the report.

        Args:
            n_plots: Maximum number of plots to generate
            plot_types: Types of plots to generate

        Returns:
            Self for method chaining
        """
        if plot_types is None:
            plot_types = ['shap_summary', 'shap_waterfall', 'pdp_1d', 'pdp_2d', 'lime']

        visualizations = {}

        # SHAP visualizations
        if 'shap_summary' in plot_types and self.shap_explainer:
            try:
                sample_data = self.X.sample(min(50, len(self.X)), random_state=42)
                shap_summary = self.shap_explainer.create_summary_plot(sample_data)
                visualizations['shap_summary'] = shap_summary
            except Exception as e:
                warnings.warn(f"Could not generate SHAP summary plot: {e}")

        if 'shap_waterfall' in plot_types and self.shap_explainer:
            try:
                sample_instance = self.X.sample(1, random_state=42)
                shap_waterfall = self.shap_explainer.create_waterfall_plot(sample_instance.iloc[0])
                visualizations['shap_waterfall'] = shap_waterfall
            except Exception as e:
                warnings.warn(f"Could not generate SHAP waterfall plot: {e}")

        if 'shap_feature_importance' in plot_types and self.shap_explainer:
            try:
                # Calculate SHAP values first
                sample_data = self.X.sample(min(50, len(self.X)), random_state=42)
                self.shap_explainer.explain(sample_data)
                shap_importance = self.shap_explainer.create_feature_importance_plot()
                visualizations['shap_feature_importance'] = shap_importance
            except Exception as e:
                warnings.warn(f"Could not generate SHAP feature importance plot: {e}")

        # Partial dependence plots
        if 'pdp_1d' in plot_types and self.pdp_plotter:
            try:
                # Select top features based on variance
                feature_variances = self.X.var()
                top_features = feature_variances.nlargest(min(3, len(self.X.columns))).index

                for feature in top_features[:min(2, len(top_features))]:
                    pdp_1d = self.pdp_plotter.plot_1d_partial_dependence(feature)
                    visualizations[f'pdp_1d_{feature}'] = pdp_1d
            except Exception as e:
                warnings.warn(f"Could not generate 1D PDP plots: {e}")

        if 'pdp_2d' in plot_types and self.pdp_plotter and self.X.shape[1] > 1:
            try:
                # Select two features with highest variance
                feature_variances = self.X.var()
                top_features = feature_variances.nlargest(2).index
                pdp_2d = self.pdp_plotter.plot_2d_partial_dependence(list(top_features))
                visualizations['pdp_2d'] = pdp_2d
            except Exception as e:
                warnings.warn(f"Could not generate 2D PDP plot: {e}")

        # LIME visualizations
        if 'lime' in plot_types and self.lime_explainer:
            try:
                sample_instance = self.X.sample(1, random_state=42)
                lime_result = self.lime_explainer.explain_instance(sample_instance.iloc[0], num_features=5)
                lime_plot = self.lime_explainer.create_explanation_plot(lime_result)
                visualizations['lime_explanation'] = lime_plot
            except Exception as e:
                warnings.warn(f"Could not generate LIME explanation plot: {e}")

        self.report_data['visualizations'] = visualizations
        return self

    def generate_executive_summary(self) -> Dict[str, Any]:
        """
        Generate executive summary of the interpretability analysis.

        Returns:
            Dictionary containing executive summary
        """
        summary = {
            'model_overview': {
                'name': self.report_data['metadata']['model_name'],
                'type': self.report_data['metadata']['model_type'],
                'description': self.report_data['metadata']['description'],
                'n_features': self.report_data['metadata']['n_features'],
                'n_samples': self.report_data['metadata']['n_samples']
            },
            'key_findings': [],
            'top_features': [],
            'recommendations': []
        }

        # Key findings from SHAP
        if 'shap' in self.report_data['global_interpretability']:
            shap_data = self.report_data['global_interpretability']['shap']
            top_shap_features = shap_data['feature_ranking'][:5]
            summary['key_findings'].append(
                f"Top {len(top_shap_features)} features explain majority of predictions using SHAP analysis"
            )
            summary['top_features'].extend([feature for feature, _ in top_shap_features])

        # Key findings from PDP
        if 'pdp_importance' in self.report_data['global_interpretability']:
            pdp_data = self.report_data['global_interpretability']['pdp_importance']
            top_pdp_features = pdp_data['feature_ranking'][:3]
            if top_pdp_features:
                summary['key_findings'].append(
                    f"Partial dependence analysis identified {len(top_pdp_features)} highly influential features"
                )

        # Feature interactions
        if 'feature_interactions' in self.report_data['global_interpretability']:
            interactions = self.report_data['global_interpretability']['feature_interactions']
            if 'top_interactions' in interactions and interactions['top_interactions']:
                summary['key_findings'].append(
                    f"Strong feature interactions detected between {len(interactions['top_interactions'])} feature pairs"
                )

        # Recommendations
        if len(summary['top_features']) > 0:
            summary['recommendations'].append(
                f"Focus on top {len(summary['top_features'])} features for model improvement and feature engineering"
            )

        if self.report_data['metadata'].get('shap_available', False):
            summary['recommendations'].append(
                "SHAP values provide reliable explanations for individual predictions"
            )

        if self.report_data['metadata'].get('lime_available', False):
            summary['recommendations'].append(
                "LIME explanations are available for local interpretability"
            )

        return summary

    def generate_html_report(
        self,
        output_path: Optional[str] = None,
        include_css: bool = True
    ) -> str:
        """
        Generate comprehensive HTML report.

        Args:
            output_path: Path to save the HTML file
            include_css: Whether to include CSS styling

        Returns:
            HTML content as string
        """
        # Generate executive summary
        executive_summary = self.generate_executive_summary()

        # Build HTML content
        html_content = self._build_html_template(executive_summary, include_css)

        # Save to file if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

        return html_content

    def generate_json_report(self, output_path: Optional[str] = None) -> Union[str, Dict]:
        """
        Generate JSON report.

        Args:
            output_path: Path to save the JSON file

        Returns:
            JSON content as string or dictionary
        """
        # Add executive summary to report data
        self.report_data['executive_summary'] = self.generate_executive_summary()

        json_content = json.dumps(self.report_data, indent=2, default=str)

        # Save to file if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_content)

        return json_content

    def _build_html_template(self, executive_summary: Dict[str, Any], include_css: bool) -> str:
        """Build HTML template for the report."""
        # CSS styles
        css_styles = """
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
            .header { background-color: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
            .section { margin-bottom: 30px; }
            .feature-list { list-style-type: none; padding: 0; }
            .feature-item { background-color: #f9f9f9; margin: 5px 0; padding: 10px; border-radius: 3px; }
            .plot-container { margin: 20px 0; text-align: center; }
            .plot-container img { max-width: 100%; height: auto; }
            table { width: 100%; border-collapse: collapse; margin: 10px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .key-finding { background-color: #e8f4fd; padding: 10px; margin: 5px 0; border-radius: 3px; }
            .recommendation { background-color: #f0f8e8; padding: 10px; margin: 5px 0; border-radius: 3px; }
        </style>
        """ if include_css else ""

        # Build HTML sections
        sections = []

        # Executive Summary
        sections.append(f"""
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="overview">
                <p><strong>Model:</strong> {executive_summary['model_overview']['name']}</p>
                <p><strong>Type:</strong> {executive_summary['model_overview']['type']}</p>
                <p><strong>Features:</strong> {executive_summary['model_overview']['n_features']}</p>
                <p><strong>Samples:</strong> {executive_summary['model_overview']['n_samples']}</p>
            </div>

            <h3>Key Findings</h3>
            {"".join(f'<div class="key-finding">• {finding}</div>' for finding in executive_summary['key_findings'])}

            <h3>Top Features</h3>
            <ul class="feature-list">
                {"".join(f'<li class="feature-item">{feature}</li>' for feature in executive_summary['top_features'][:10])}
            </ul>

            <h3>Recommendations</h3>
            {"".join(f'<div class="recommendation">• {rec}</div>' for rec in executive_summary['recommendations'])}
        </div>
        """)

        # Global Interpretability
        if self.report_data['global_interpretability']:
            global_html = ['<div class="section"><h2>Global Interpretability</h2>']

            if 'shap' in self.report_data['global_interpretability']:
                shap_data = self.report_data['global_interpretability']['shap']
                global_html.append('<h3>SHAP Feature Importance</h3>')
                global_html.append('<table><tr><th>Feature</th><th>Importance</th></tr>')
                for feature, importance in shap_data['feature_ranking'][:10]:
                    global_html.append(f'<tr><td>{feature}</td><td>{importance:.4f}</td></tr>')
                global_html.append('</table>')

            if 'feature_interactions' in self.report_data['global_interpretability']:
                interactions = self.report_data['global_interpretability']['feature_interactions']
                if 'top_interactions' in interactions:
                    global_html.append('<h3>Top Feature Interactions</h3>')
                    global_html.append('<ul>')
                    for interaction, strength in interactions['top_interactions'][:5]:
                        global_html.append(f'<li>{interaction}: {strength:.4f}</li>')
                    global_html.append('</ul>')

            global_html.append('</div>')
            sections.append(''.join(global_html))

        # Local Interpretability
        if self.report_data['local_interpretability']['instances']:
            local_html = ['<div class="section"><h2>Local Interpretability</h2>']

            for idx, instance_data in list(self.report_data['local_interpretability']['instances'].items())[:3]:
                local_html.append(f'<h3>Instance {idx}</h3>')
                if 'shap' in instance_data:
                    local_html.append('<h4>SHAP Explanation</h4>')
                    local_html.append(f'<p>Prediction: {instance_data["shap"]["prediction"]:.4f}</p>')
                    top_contributions = sorted(
                        instance_data['shap']['feature_contributions'].items(),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:5]
                    local_html.append('<ul>')
                    for feature, contribution in top_contributions:
                        local_html.append(f'<li>{feature}: {contribution:.4f}</li>')
                    local_html.append('</ul>')

            local_html.append('</div>')
            sections.append(''.join(local_html))

        # Visualizations
        if self.report_data['visualizations']:
            viz_html = ['<div class="section"><h2>Visualizations</h2>']

            for viz_name, viz_data in self.report_data['visualizations'].items():
                if 'plot_base64' in viz_data:
                    viz_html.append(f'''
                    <div class="plot-container">
                        <h3>{viz_name.replace('_', ' ').title()}</h3>
                        <img src="data:image/png;base64,{viz_data['plot_base64']}" alt="{viz_name}">
                    </div>
                    ''')

            viz_html.append('</div>')
            sections.append(''.join(viz_html))

        # Feature Analysis
        if self.report_data['feature_analysis']:
            feature_html = ['<div class="section"><h2>Feature Analysis</h2>']
            feature_html.append('<table>')
            feature_html.append('<tr><th>Feature</th><th>Type</th><th>Missing %</th><th>Unique Values</th></tr>')

            for feature, info in list(self.report_data['feature_analysis'].items())[:10]:
                feature_html.append(f'''
                <tr>
                    <td>{feature}</td>
                    <td>{info['dtype']}</td>
                    <td>{info['missing_percentage']:.1f}%</td>
                    <td>{info['unique_count']}</td>
                </tr>
                ''')

            feature_html.append('</table></div>')
            sections.append(''.join(feature_html))

        # Combine all sections
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Interpretability Report - {self.model_name}</title>
            {css_styles}
        </head>
        <body>
            <div class="header">
                <h1>Model Interpretability Report</h1>
                <h2>{self.model_name}</h2>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Description:</strong> {self.description}</p>
            </div>
            {"".join(sections)}
        </body>
        </html>
        """

        return full_html