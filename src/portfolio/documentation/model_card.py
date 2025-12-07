"""
Model documentation system for generating model cards.

This module provides the ModelCard class for generating comprehensive
model documentation following model card best practices.
"""

import json
import yaml
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix

from src.portfolio.utils.logging import PortfolioLogger


@dataclass
class ModelMetrics:
    """Metrics for model performance"""
    accuracy: Optional[float] = None
    precision: Optional[Dict[str, float]] = None
    recall: Optional[Dict[str, float]] = None
    f1_score: Optional[Dict[str, float]] = None
    roc_auc: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with only non-None values"""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
        return result


@dataclass
class ModelParameters:
    """Model hyperparameters and configuration"""
    model_type: str
    hyperparameters: Dict[str, Any]
    feature_names: Optional[List[str]] = None
    feature_importance: Optional[Dict[str, float]] = None
    training_data_shape: Optional[tuple] = None
    num_features: Optional[int] = None
    num_classes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ModelConsiderations:
    """Ethical considerations and limitations"""
    limitations: List[str] = field(default_factory=list)
    ethical_considerations: List[str] = field(default_factory=list)
    fairness_analysis: Optional[Dict[str, Any]] = None
    intended_use: str = ""
    prohibited_use: List[str] = field(default_factory=list)
    monitoring_needs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class ModelCard:
    """
    Generates and manages model cards for comprehensive model documentation.

    Follows the Model Card framework proposed by Mitchell et al. (2019)
    and the Model Facts template from Google.
    """

    def __init__(
        self,
        model_name: str,
        model_version: str = "1.0",
        contact: Optional[str] = None
    ):
        """
        Initialize model card.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            contact: Contact information for model owner
        """
        self.model_name = model_name
        self.model_version = model_version
        self.contact = contact
        self.logger = PortfolioLogger('model_card')

        # Initialize sections
        self.model_details: Dict[str, Any] = {}
        self.intended_use: str = ""
        self.factors: List[Dict[str, str]] = []
        self.metrics: Optional[ModelMetrics] = None
        self.evaluation_data: Dict[str, Any] = {}
        self.training_data: Dict[str, Any] = {}
        self.quantitative_analyses: List[Dict[str, Any]] = []
        self.ethical_considerations: Optional[ModelConsiderations] = None
        self.caveats_and_recommendations: List[str] = []

    def set_model_details(
        self,
        model_type: str,
        description: str,
        developers: List[str],
        model_date: Optional[datetime] = None,
        paper_url: Optional[str] = None,
        cite_url: Optional[str] = None,
        license_url: Optional[str] = None
    ) -> None:
        """
        Set basic model details.

        Args:
            model_type: Type of model (e.g., "RandomForestClassifier")
            description: Brief description of the model
            developers: List of developer names
            model_date: Date when model was created
            paper_url: URL to research paper
            cite_url: URL to citation
            license_url: URL to license information
        """
        if model_date is None:
            model_date = datetime.now()

        self.model_details = {
            'name': self.model_name,
            'version': self.model_version,
            'type': model_type,
            'description': description,
            'developers': developers,
            'date': model_date.isoformat(),
            'paper_url': paper_url,
            'cite_url': cite_url,
            'license_url': license_url,
            'contact': self.contact
        }

    def set_intended_use(
        self,
        intended_use: str,
        primary_use: str,
        secondary_use: Optional[str] = None,
        out_of_scope_use: Optional[str] = None
    ) -> None:
        """
        Set intended use cases.

        Args:
            intended_use: General intended use description
            primary_use: Primary intended use case
            secondary_use: Secondary use cases
            out_of_scope_use: Use cases that are out of scope
        """
        self.intended_use = intended_use
        self.factors = [
            {
                'name': 'Primary Use',
                'description': primary_use
            }
        ]

        if secondary_use:
            self.factors.append({
                'name': 'Secondary Use',
                'description': secondary_use
            })

        if out_of_scope_use:
            self.factors.append({
                'name': 'Out of Scope',
                'description': out_of_scope_use
            })

    def set_metrics(
        self,
        metrics: ModelMetrics,
        confidence_interval: Optional[Dict[str, Tuple[float, float]]] = None,
        threshold: Optional[float] = None
    ) -> None:
        """
        Set model performance metrics.

        Args:
            metrics: ModelMetrics object containing performance metrics
            confidence_interval: Confidence intervals for metrics
            threshold: Performance threshold
        """
        self.metrics = metrics
        if confidence_interval:
            self.evaluation_data['confidence_interval'] = confidence_interval
        if threshold:
            self.evaluation_data['threshold'] = threshold

    def set_training_data(
        self,
        dataset_name: str,
        dataset_url: Optional[str] = None,
        dataset_size: int = 0,
        split: Optional[Dict[str, int]] = None,
        preprocessing: Optional[List[str]] = None,
        features: Optional[List[str]] = None
    ) -> None:
        """
        Set training data information.

        Args:
            dataset_name: Name of the dataset
            dataset_url: URL to dataset
            dataset_size: Size of the dataset
            split: Data split (train/validation/test sizes)
            preprocessing: Preprocessing steps applied
            features: List of features used
        """
        self.training_data = {
            'dataset_name': dataset_name,
            'dataset_url': dataset_url,
            'dataset_size': dataset_size,
            'split': split,
            'preprocessing': preprocessing or [],
            'features': features or []
        }

    def add_quantitative_analysis(
        self,
        name: str,
        description: str,
        value: Union[float, str],
        visualization: Optional[str] = None
    ) -> None:
        """
        Add quantitative analysis information.

        Args:
            name: Name of the analysis
            description: Description of the analysis
            value: Analysis result
            visualization: Path to visualization
        """
        analysis = {
            'name': name,
            'description': description,
            'value': value
        }
        if visualization:
            analysis['visualization'] = visualization

        self.quantitative_analyses.append(analysis)

    def set_ethical_considerations(
        self,
        considerations: ModelConsiderations
    ) -> None:
        """
        Set ethical considerations and limitations.

        Args:
            considerations: ModelConsiderations object
        """
        self.ethical_considerations = considerations

    def add_caveat(self, caveat: str) -> None:
        """
        Add a caveat or recommendation.

        Args:
            caveat: Caveat or recommendation text
        """
        self.caveats_and_recommendations.append(caveat)

    def extract_model_parameters(
        self,
        estimator: BaseEstimator,
        feature_names: Optional[List[str]] = None,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None
    ) -> ModelParameters:
        """
        Extract model parameters from fitted estimator.

        Args:
            estimator: Fitted sklearn estimator
            feature_names: Names of features
            X_train: Training features
            y_train: Training targets

        Returns:
            ModelParameters object
        """
        # Get model type and name
        model_type = type(estimator).__name__

        # Extract hyperparameters
        hyperparameters = estimator.get_params()

        # Get feature information
        num_features = None
        if X_train is not None:
            if hasattr(X_train, 'shape'):
                num_features = X_train.shape[1]
                training_data_shape = X_train.shape
            else:
                training_data_shape = (len(X_train), len(X_train[0]) if len(X_train) > 0 else 0)
                num_features = training_data_shape[1]
        else:
            training_data_shape = None

        # Get number of classes for classification models
        num_classes = None
        if y_train is not None and hasattr(estimator, 'classes_'):
            num_classes = len(estimator.classes_)

        # Calculate feature importance if possible
        feature_importance = None
        if hasattr(estimator, 'feature_importances_'):
            if feature_names:
                feature_importance = dict(zip(feature_names, estimator.feature_importances_))
            else:
                feature_importance = {
                    f'feature_{i}': importance
                    for i, importance in enumerate(estimator.feature_importances_)
                }

        return ModelParameters(
            model_type=model_type,
            hyperparameters=hyperparameters,
            feature_names=feature_names,
            feature_importance=feature_importance,
            training_data_shape=training_data_shape,
            num_features=num_features,
            num_classes=num_classes
        )

    def generate_metrics_from_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        task_type: str = 'classification'
    ) -> ModelMetrics:
        """
        Generate metrics from predictions.

        Args:
            y_true: True labels/values
            y_pred: Predicted labels/values
            y_pred_proba: Predicted probabilities (classification only)
            task_type: 'classification' or 'regression'

        Returns:
            ModelMetrics object
        """
        metrics = ModelMetrics()

        if task_type == 'classification':
            # Get classification report
            report = classification_report(y_true, y_pred, output_dict=True)

            # Extract metrics
            if 'accuracy' in report:
                metrics.accuracy = report['accuracy']

            # Extract per-class metrics
            if 'weighted avg' in report:
                metrics.precision = {'weighted_avg': report['weighted avg']['precision']}
                metrics.recall = {'weighted_avg': report['weighted avg']['recall']}
                metrics.f1_score = {'weighted_avg': report['weighted avg']['f1-score']}

            # ROC AUC if probabilities available
            if y_pred_proba is not None:
                from sklearn.metrics import roc_auc_score
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics.roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    # Multi-class classification
                    metrics.roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')

        elif task_type == 'regression':
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            metrics.mse = mean_squared_error(y_true, y_pred)
            metrics.rmse = np.sqrt(metrics.mse)
            metrics.mae = mean_absolute_error(y_true, y_pred)
            metrics.r2_score = r2_score(y_true, y_pred)

        return metrics

    def to_markdown(self) -> str:
        """
        Generate model card as markdown.

        Returns:
            Markdown string of the model card
        """
        sections = []

        # Header
        sections.append(f"# Model Card: {self.model_name}")
        sections.append(f"**Version:** {self.model_version}\n")

        # Model Details
        if self.model_details:
            sections.append("## Model Details\n")
            sections.append(f"**Model Type:** {self.model_details.get('type', 'Unknown')}")
            sections.append(f"**Description:** {self.model_details.get('description', 'No description')}")
            sections.append(f"**Developers:** {', '.join(self.model_details.get('developers', []))}")
            sections.append(f"**Date Created:** {self.model_details.get('date', 'Unknown')}")

            if self.model_details.get('contact'):
                sections.append(f"**Contact:** {self.model_details['contact']}")

            if self.model_details.get('paper_url'):
                sections.append(f"**Paper:** [Link]({self.model_details['paper_url']})")

            if self.model_details.get('cite_url'):
                sections.append(f"**Citation:** [Link]({self.model_details['cite_url']})")

            if self.model_details.get('license_url'):
                sections.append(f"**License:** [Link]({self.model_details['license_url']})")

            sections.append("")

        # Intended Use
        if self.intended_use:
            sections.append("## Intended Use\n")
            sections.append(self.intended_use)
            sections.append("")

            if self.factors:
                sections.append("### Use Cases\n")
                for factor in self.factors:
                    sections.append(f"**{factor['name']}:** {factor['description']}")
                sections.append("")

        # Metrics
        if self.metrics:
            sections.append("## Performance Metrics\n")

            metrics_dict = self.metrics.to_dict()
            for metric, value in metrics_dict.items():
                if isinstance(value, dict):
                    sections.append(f"### {metric.replace('_', ' ').title()}")
                    for k, v in value.items():
                        sections.append(f"- {k}: {v:.4f}")
                else:
                    sections.append(f"- **{metric.replace('_', ' ').title()}:** {value:.4f}")

            # Confidence intervals
            if 'confidence_interval' in self.evaluation_data:
                sections.append("\n### Confidence Intervals")
                for metric, ci in self.evaluation_data['confidence_interval'].items():
                    sections.append(f"- **{metric}:** [{ci[0]:.4f}, {ci[1]:.4f}]")

            sections.append("")

        # Training Data
        if self.training_data:
            sections.append("## Training Data\n")
            sections.append(f"**Dataset:** {self.training_data['dataset_name']}")

            if self.training_data.get('dataset_url'):
                sections.append(f"**URL:** {self.training_data['dataset_url']}")

            sections.append(f"**Size:** {self.training_data['dataset_size']:,} samples")

            if self.training_data.get('split'):
                sections.append("**Data Split:**")
                for split, size in self.training_data['split'].items():
                    sections.append(f"- {split}: {size:,} samples")

            if self.training_data.get('preprocessing'):
                sections.append("\n**Preprocessing Steps:**")
                for step in self.training_data['preprocessing']:
                    sections.append(f"- {step}")

            if self.training_data.get('features'):
                sections.append(f"\n**Number of Features:** {len(self.training_data['features'])}")
                sections.append("**Features:**")
                for feature in self.training_data['features'][:10]:  # Show first 10
                    sections.append(f"- {feature}")
                if len(self.training_data['features']) > 10:
                    sections.append(f"- ... and {len(self.training_data['features']) - 10} more")

            sections.append("")

        # Quantitative Analyses
        if self.quantitative_analyses:
            sections.append("## Quantitative Analyses\n")

            for analysis in self.quantitative_analyses:
                sections.append(f"### {analysis['name']}")
                sections.append(f"{analysis['description']}")
                sections.append(f"**Result:** {analysis['value']}")

                if 'visualization' in analysis:
                    sections.append(f"![{analysis['name']}]({analysis['visualization']})")

                sections.append("")

        # Ethical Considerations
        if self.ethical_considerations:
            sections.append("## Ethical Considerations\n")

            considerations = self.ethical_considerations.to_dict()

            if considerations.get('limitations'):
                sections.append("### Limitations")
                for limitation in considerations['limitations']:
                    sections.append(f"- {limitation}")
                sections.append("")

            if considerations.get('ethical_considerations'):
                sections.append("### Ethical Considerations")
                for consideration in considerations['ethical_considerations']:
                    sections.append(f"- {consideration}")
                sections.append("")

            if considerations.get('intended_use'):
                sections.append(f"### Intended Use\n{considerations['intended_use']}\n")

            if considerations.get('prohibited_use'):
                sections.append("### Prohibited Use")
                for prohibited in considerations['prohibited_use']:
                    sections.append(f"- {prohibited}")
                sections.append("")

            if considerations.get('monitoring_needs'):
                sections.append("### Monitoring Needs")
                for need in considerations['monitoring_needs']:
                    sections.append(f"- {need}")
                sections.append("")

        # Caveats and Recommendations
        if self.caveats_and_recommendations:
            sections.append("## Caveats and Recommendations\n")
            for caveat in self.caveats_and_recommendations:
                sections.append(f"- {caveat}")
            sections.append("")

        # Footer
        sections.append("---")
        sections.append(f"*Model card generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        return "\n".join(sections)

    def to_json(self) -> str:
        """
        Generate model card as JSON.

        Returns:
            JSON string of the model card
        """
        card_dict = {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'model_details': self.model_details,
            'intended_use': self.intended_use,
            'factors': self.factors,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'evaluation_data': self.evaluation_data,
            'training_data': self.training_data,
            'quantitative_analyses': self.quantitative_analyses,
            'ethical_considerations': self.ethical_considerations.to_dict() if self.ethical_considerations else None,
            'caveats_and_recommendations': self.caveats_and_recommendations,
            'generated_date': datetime.now().isoformat()
        }

        return json.dumps(card_dict, indent=2, default=str)

    def to_yaml(self) -> str:
        """
        Generate model card as YAML.

        Returns:
            YAML string of the model card
        """
        card_dict = {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'model_details': self.model_details,
            'intended_use': self.intended_use,
            'factors': self.factors,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'evaluation_data': self.evaluation_data,
            'training_data': self.training_data,
            'quantitative_analyses': self.quantitative_analyses,
            'ethical_considerations': self.ethical_considerations.to_dict() if self.ethical_considerations else None,
            'caveats_and_recommendations': self.caveats_and_recommendations,
            'generated_date': datetime.now().isoformat()
        }

        return yaml.dump(card_dict, default_flow_style=False, sort_keys=False)

    def save(self, filepath: str, format: str = 'markdown') -> None:
        """
        Save model card to file.

        Args:
            filepath: Path to save the model card
            format: Format to save ('markdown', 'json', or 'yaml')
        """
        if format == 'markdown':
            content = self.to_markdown()
            ext = '.md'
        elif format == 'json':
            content = self.to_json()
            ext = '.json'
        elif format == 'yaml':
            content = self.to_yaml()
            ext = '.yaml'
        else:
            raise ValueError(f"Unknown format: {format}")

        # Add extension if not present
        if not filepath.endswith(ext):
            filepath += ext

        with open(filepath, 'w') as f:
            f.write(content)

        self.logger.info(f"Model card saved to: {filepath}")

    @classmethod
    def load_from_estimator(
        cls,
        estimator: BaseEstimator,
        model_name: str,
        model_version: str = "1.0",
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        task_type: str = 'classification',
        **kwargs
    ) -> 'ModelCard':
        """
        Create a model card automatically from a fitted estimator.

        Args:
            estimator: Fitted sklearn estimator
            model_name: Name of the model
            model_version: Version of the model
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            feature_names: Names of features
            task_type: 'classification' or 'regression'
            **kwargs: Additional arguments for model card

        Returns:
            ModelCard object with populated information
        """
        card = cls(model_name=model_name, model_version=model_version)

        # Extract model parameters
        params = card.extract_model_parameters(
            estimator=estimator,
            feature_names=feature_names,
            X_train=X_train,
            y_train=y_train
        )

        # Set model details
        card.set_model_details(
            model_type=params.model_type,
            description=f"{params.model_type} model for {task_type}",
            developers=kwargs.get('developers', ['Data Scientist']),
            **{k: v for k, v in kwargs.items() if k in ['paper_url', 'cite_url', 'license_url']}
        )

        # Add hyperparameters to model details
        card.model_details['hyperparameters'] = params.hyperparameters

        # Generate metrics from test predictions
        if X_test is not None and y_test is not None:
            y_pred = estimator.predict(X_test)
            y_pred_proba = None
            if task_type == 'classification' and hasattr(estimator, 'predict_proba'):
                y_pred_proba = estimator.predict_proba(X_test)

            metrics = card.generate_metrics_from_predictions(
                y_true=y_test,
                y_pred=y_pred,
                y_pred_proba=y_pred_proba,
                task_type=task_type
            )

            card.set_metrics(metrics)

        # Add default ethical considerations
        considerations = ModelConsiderations(
            limitations=[
                "Model performance may degrade with data distribution shifts",
                "Consider retraining regularly with new data"
            ],
            intended_use=kwargs.get('intended_use', f"{task_type} tasks"),
            monitoring_needs=[
                "Monitor prediction drift",
                "Monitor feature distribution changes",
                "Regular performance evaluation"
            ]
        )
        card.set_ethical_considerations(considerations)

        # Also set intended use at the top level for proper markdown generation
        card.intended_use = kwargs.get('intended_use', f"{task_type} tasks")

        return card