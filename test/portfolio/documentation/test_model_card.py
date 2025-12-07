"""
Tests for model documentation system.

This module tests the ModelCard class and its ability to generate
comprehensive model documentation.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import json
from datetime import datetime
from typing import Dict, Any

from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

from src.portfolio.documentation.model_card import (
    ModelCard,
    ModelMetrics,
    ModelParameters,
    ModelConsiderations
)


class TestModelCard:
    """Test cases for ModelCard"""

    def setup_method(self):
        """Set up test fixtures"""
        self.model_card = ModelCard(
            model_name="TestModel",
            model_version="1.0",
            contact="test@example.com"
        )

    def create_classification_data(self):
        """Create synthetic classification data"""
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return X, y

    def create_regression_data(self):
        """Create synthetic regression data"""
        X, y = make_regression(
            n_samples=1000,
            n_features=10,
            noise=0.1,
            random_state=42
        )
        return X, y

    def test_model_card_initialization(self):
        """Test model card initialization"""
        assert self.model_card.model_name == "TestModel"
        assert self.model_card.model_version == "1.0"
        assert self.model_card.contact == "test@example.com"
        assert self.model_card.model_details == {}
        assert self.model_card.metrics is None
        assert self.model_card.training_data == {}
        assert self.model_card.ethical_considerations is None

    def test_set_model_details(self):
        """Test setting model details"""
        self.model_card.set_model_details(
            model_type="RandomForestClassifier",
            description="A random forest classifier for binary classification",
            developers=["Data Scientist", "ML Engineer"],
            paper_url="https://example.com/paper",
            cite_url="https://example.com/cite",
            license_url="https://example.com/license"
        )

        details = self.model_card.model_details
        assert details['type'] == "RandomForestClassifier"
        assert details['description'] == "A random forest classifier for binary classification"
        assert details['developers'] == ["Data Scientist", "ML Engineer"]
        assert details['paper_url'] == "https://example.com/paper"
        assert details['cite_url'] == "https://example.com/cite"
        assert details['license_url'] == "https://example.com/license"
        assert 'date' in details

    def test_set_intended_use(self):
        """Test setting intended use"""
        self.model_card.set_intended_use(
            intended_use="This model is intended for binary classification tasks",
            primary_use="Classifying email spam",
            secondary_use="Email priority scoring",
            out_of_scope_use="Medical diagnosis"
        )

        assert self.model_card.intended_use == "This model is intended for binary classification tasks"
        assert len(self.model_card.factors) == 3

        factor_names = [f['name'] for f in self.model_card.factors]
        assert "Primary Use" in factor_names
        assert "Secondary Use" in factor_names
        assert "Out of Scope" in factor_names

    def test_set_metrics(self):
        """Test setting model metrics"""
        metrics = ModelMetrics(
            accuracy=0.95,
            precision={'weighted_avg': 0.94},
            recall={'weighted_avg': 0.95},
            f1_score={'weighted_avg': 0.94},
            roc_auc=0.97
        )

        confidence_interval = {
            'accuracy': (0.93, 0.97),
            'roc_auc': (0.95, 0.99)
        }

        self.model_card.set_metrics(
            metrics=metrics,
            confidence_interval=confidence_interval,
            threshold=0.90
        )

        assert self.model_card.metrics is not None
        assert self.model_card.metrics.accuracy == 0.95
        assert self.model_card.metrics.roc_auc == 0.97
        assert self.model_card.evaluation_data['confidence_interval'] == confidence_interval
        assert self.model_card.evaluation_data['threshold'] == 0.90

    def test_set_training_data(self):
        """Test setting training data information"""
        self.model_card.set_training_data(
            dataset_name="Test Dataset",
            dataset_url="https://example.com/dataset",
            dataset_size=10000,
            split={'train': 8000, 'validation': 1000, 'test': 1000},
            preprocessing=['Normalization', 'Feature selection'],
            features=['feature1', 'feature2', 'feature3']
        )

        training_data = self.model_card.training_data
        assert training_data['dataset_name'] == "Test Dataset"
        assert training_data['dataset_url'] == "https://example.com/dataset"
        assert training_data['dataset_size'] == 10000
        assert training_data['split'] == {'train': 8000, 'validation': 1000, 'test': 1000}
        assert training_data['preprocessing'] == ['Normalization', 'Feature selection']
        assert training_data['features'] == ['feature1', 'feature2', 'feature3']

    def test_add_quantitative_analysis(self):
        """Test adding quantitative analysis"""
        self.model_card.add_quantitative_analysis(
            name="Feature Importance",
            description="Analysis of feature importance",
            value="Feature X is most important",
            visualization="path/to/plot.png"
        )

        assert len(self.model_card.quantitative_analyses) == 1
        analysis = self.model_card.quantitative_analyses[0]
        assert analysis['name'] == "Feature Importance"
        assert analysis['description'] == "Analysis of feature importance"
        assert analysis['value'] == "Feature X is most important"
        assert analysis['visualization'] == "path/to/plot.png"

    def test_set_ethical_considerations(self):
        """Test setting ethical considerations"""
        considerations = ModelConsiderations(
            limitations=[
                "Model may not generalize to unseen data distributions",
                "Requires regular retraining"
            ],
            ethical_considerations=[
                "Model should not be used for sensitive applications without review"
            ],
            intended_use="Binary classification in controlled environments",
            prohibited_use=["Medical diagnosis", "Legal judgments"],
            monitoring_needs=[
                "Monitor for data drift",
                "Regular bias audits"
            ]
        )

        self.model_card.set_ethical_considerations(considerations)

        assert self.model_card.ethical_considerations is not None
        assert len(self.model_card.ethical_considerations.limitations) == 2
        assert len(self.model_card.ethical_considerations.ethical_considerations) == 1
        assert self.model_card.ethical_considerations.intended_use == "Binary classification in controlled environments"
        assert len(self.model_card.ethical_considerations.prohibited_use) == 2
        assert len(self.model_card.ethical_considerations.monitoring_needs) == 2

    def test_add_caveat(self):
        """Test adding caveats and recommendations"""
        self.model_card.add_caveat("Model should be validated in production before use")
        self.model_card.add_caveat("Regular monitoring is recommended")

        assert len(self.model_card.caveats_and_recommendations) == 2
        assert "Model should be validated in production before use" in self.model_card.caveats_and_recommendations
        assert "Regular monitoring is recommended" in self.model_card.caveats_and_recommendations

    def test_extract_model_parameters(self):
        """Test extracting model parameters from estimator"""
        X, y = self.create_classification_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        estimator.fit(X_train, y_train)

        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        params = self.model_card.extract_model_parameters(
            estimator=estimator,
            feature_names=feature_names,
            X_train=X_train,
            y_train=y_train
        )

        assert isinstance(params, ModelParameters)
        assert params.model_type == "RandomForestClassifier"
        assert params.hyperparameters['n_estimators'] == 10
        assert params.num_features == 10
        assert params.num_classes == 2
        assert len(params.feature_importance) == 10
        assert 'feature_0' in params.feature_importance

    def test_generate_metrics_from_predictions_classification(self):
        """Test generating metrics for classification"""
        y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
        y_pred_proba = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.7, 0.3],
            [0.4, 0.6],
            [0.3, 0.7],
            [0.6, 0.4],
            [0.1, 0.9],
            [0.5, 0.5]
        ])

        metrics = self.model_card.generate_metrics_from_predictions(
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            task_type='classification'
        )

        assert isinstance(metrics, ModelMetrics)
        assert metrics.accuracy is not None
        assert metrics.precision is not None
        assert metrics.recall is not None
        assert metrics.f1_score is not None
        assert metrics.roc_auc is not None
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.roc_auc <= 1

    def test_generate_metrics_from_predictions_regression(self):
        """Test generating metrics for regression"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        metrics = self.model_card.generate_metrics_from_predictions(
            y_true=y_true,
            y_pred=y_pred,
            task_type='regression'
        )

        assert isinstance(metrics, ModelMetrics)
        assert metrics.mse is not None
        assert metrics.rmse is not None
        assert metrics.mae is not None
        assert metrics.r2_score is not None
        assert metrics.mse >= 0
        assert metrics.rmse >= 0
        assert metrics.mae >= 0

    def test_to_markdown(self):
        """Test generating model card as markdown"""
        self.model_card.set_model_details(
            model_type="RandomForestClassifier",
            description="Test model",
            developers=["Test Developer"]
        )

        markdown = self.model_card.to_markdown()

        assert "# Model Card: TestModel" in markdown
        assert "Model Details" in markdown
        assert "RandomForestClassifier" in markdown
        assert "Test model" in markdown
        assert "Test Developer" in markdown

    def test_to_json(self):
        """Test generating model card as JSON"""
        self.model_card.set_model_details(
            model_type="RandomForestClassifier",
            description="Test model",
            developers=["Test Developer"]
        )

        json_str = self.model_card.to_json()
        card_dict = json.loads(json_str)

        assert card_dict['model_name'] == "TestModel"
        assert card_dict['model_version'] == "1.0"
        assert card_dict['model_details']['type'] == "RandomForestClassifier"
        assert card_dict['model_details']['description'] == "Test model"
        assert "generated_date" in card_dict

    def test_to_yaml(self):
        """Test generating model card as YAML"""
        self.model_card.set_model_details(
            model_type="RandomForestClassifier",
            description="Test model",
            developers=["Test Developer"]
        )

        yaml_str = self.model_card.to_yaml()

        assert "model_name: TestModel" in yaml_str
        assert "model_version: '1.0'" in yaml_str
        assert "model_details:" in yaml_str
        assert "type: RandomForestClassifier" in yaml_str

    def test_save_markdown(self):
        """Test saving model card as markdown file"""
        self.model_card.set_model_details(
            model_type="RandomForestClassifier",
            description="Test model",
            developers=["Test Developer"]
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            filepath = f.name

        try:
            self.model_card.save(filepath, format='markdown')

            assert os.path.exists(filepath)
            with open(filepath, 'r') as f:
                content = f.read()
                assert "# Model Card: TestModel" in content
                assert "RandomForestClassifier" in content
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_save_json(self):
        """Test saving model card as JSON file"""
        self.model_card.set_model_details(
            model_type="RandomForestClassifier",
            description="Test model",
            developers=["Test Developer"]
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            self.model_card.save(filepath, format='json')

            assert os.path.exists(filepath)
            with open(filepath, 'r') as f:
                card_dict = json.load(f)
                assert card_dict['model_name'] == "TestModel"
                assert card_dict['model_details']['type'] == "RandomForestClassifier"
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_load_from_estimator_classification(self):
        """Test loading model card from estimator for classification"""
        X, y = self.create_classification_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        estimator.fit(X_train, y_train)

        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        card = ModelCard.load_from_estimator(
            estimator=estimator,
            model_name="AutoGeneratedModel",
            model_version="2.0",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            task_type='classification',
            developers=["AutoML Team"]
        )

        assert card.model_name == "AutoGeneratedModel"
        assert card.model_version == "2.0"
        assert card.model_details['type'] == "RandomForestClassifier"
        assert card.metrics is not None
        assert card.metrics.accuracy is not None
        assert card.metrics.roc_auc is not None
        assert card.ethical_considerations is not None

    def test_load_from_estimator_regression(self):
        """Test loading model card from estimator for regression"""
        X, y = self.create_regression_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        estimator = RandomForestRegressor(n_estimators=10, random_state=42)
        estimator.fit(X_train, y_train)

        card = ModelCard.load_from_estimator(
            estimator=estimator,
            model_name="RegressionModel",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            task_type='regression'
        )

        assert card.model_name == "RegressionModel"
        assert card.model_details['type'] == "RandomForestRegressor"
        assert card.metrics is not None
        assert card.metrics.mse is not None
        assert card.metrics.rmse is not None
        assert card.metrics.r2_score is not None

    def test_model_metrics_to_dict(self):
        """Test ModelMetrics to_dict conversion"""
        metrics = ModelMetrics(
            accuracy=0.95,
            mse=0.01,
            custom_metrics={'custom1': 0.5, 'custom2': 1.0}
        )

        metrics_dict = metrics.to_dict()

        assert metrics_dict['accuracy'] == 0.95
        assert metrics_dict['mse'] == 0.01
        assert metrics_dict['custom_metrics']['custom1'] == 0.5
        assert metrics_dict['custom_metrics']['custom2'] == 1.0
        # None values should be excluded
        assert 'precision' not in metrics_dict
        assert 'recall' not in metrics_dict

    def test_model_parameters_to_dict(self):
        """Test ModelParameters to_dict conversion"""
        params = ModelParameters(
            model_type="RandomForest",
            hyperparameters={'n_estimators': 100, 'max_depth': 10},
            feature_names=['feature1', 'feature2'],
            num_features=2
        )

        params_dict = params.to_dict()

        assert params_dict['model_type'] == "RandomForest"
        assert params_dict['hyperparameters']['n_estimators'] == 100
        assert params_dict['feature_names'] == ['feature1', 'feature2']
        assert params_dict['num_features'] == 2
        # None values should be included as they might be meaningful
        assert 'feature_importance' in params_dict
        assert 'training_data_shape' in params_dict


class TestModelCardCompleteness:
    """Property-based tests for model card generation (Property 1)"""

    def test_model_card_generation_completeness(self):
        """
        Property 1: Model card generation completeness

        Validates that model cards are generated with all required
        sections and comprehensive information
        """
        X, y = make_classification(
            n_samples=1000,
            n_features=5,
            n_classes=2,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        estimator.fit(X_train, y_train)

        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Generate model card with all information
        card = ModelCard.load_from_estimator(
            estimator=estimator,
            model_name="CompleteModel",
            model_version="3.0",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            task_type='classification',
            developers=["Data Science Team"],
            intended_use="Production classification system",
            paper_url="https://example.com/paper",
            license_url="https://example.com/license"
        )

        # Add additional sections
        card.add_quantitative_analysis(
            name="Learning Curve",
            description="Model performance across training sizes",
            value="Improves with more data"
        )

        card.set_training_data(
            dataset_name="Complete Dataset",
            dataset_url="https://example.com/dataset",
            dataset_size=1000,
            split={'train': 800, 'test': 200},
            preprocessing=['Standardization', 'Feature selection']
        )

        # Verify completeness of model details
        assert card.model_details is not None
        required_details = ['name', 'version', 'type', 'description', 'developers']
        for detail in required_details:
            assert detail in card.model_details

        # Verify model type and parameters are documented
        assert card.model_details['type'] == "RandomForestClassifier"
        assert isinstance(card.model_details, dict)

        # Verify evaluation results are documented
        assert card.metrics is not None
        metrics_dict = card.metrics.to_dict()
        assert len(metrics_dict) > 0

        # Check for classification metrics
        if 'accuracy' in metrics_dict:
            assert 0 <= metrics_dict['accuracy'] <= 1

        # Verify training data documentation
        assert card.training_data is not None
        assert 'dataset_name' in card.training_data
        assert 'dataset_size' in card.training_data
        assert card.training_data['dataset_size'] > 0

        # Verify ethical considerations and limitations
        assert card.ethical_considerations is not None
        assert len(card.ethical_considerations.limitations) > 0
        assert len(card.ethical_considerations.monitoring_needs) > 0

        # Check for fairness considerations if applicable
        # (this is documented in the ethical_considerations)

        # Verify versioning information
        assert card.model_version == "3.0"
        assert 'date' in card.model_details

        # Verify reproducibility information
        assert 'hyperparameters' in card.model_details

        # Generate markdown and check structure
        markdown = card.to_markdown()

        # Check that all major sections are present
        required_sections = [
            "Model Card:",
            "Model Details",
            "Intended Use",
            "Performance Metrics",
            "Training Data",
            "Ethical Considerations"
        ]

        for section in required_sections:
            assert section in markdown

        # Verify bias and fairness are addressed
        assert "Ethical Considerations" in markdown

        # Generate JSON and verify structure
        json_str = card.to_json()
        card_dict = json.loads(json_str)

        # Check JSON contains all fields
        required_fields = [
            'model_name',
            'model_version',
            'model_details',
            'metrics',
            'training_data',
            'ethical_considerations',
            'generated_date'
        ]

        for field in required_fields:
            assert field in card_dict

    def test_model_card_documentation_for_multiple_models(self):
        """
        Property 1: Model card generation completeness

        Validates that model cards can be created for different
        model types with appropriate documentation
        """
        # Test classification model
        X_clf, y_clf = make_classification(n_samples=500, n_features=5, random_state=42)
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
            X_clf, y_clf, test_size=0.2, random_state=42
        )

        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        clf.fit(X_train_clf, y_train_clf)

        clf_card = ModelCard.load_from_estimator(
            estimator=clf,
            model_name="ClassificationModel",
            X_train=X_train_clf,
            y_train=y_train_clf,
            X_test=X_test_clf,
            y_test=y_test_clf,
            task_type='classification'
        )

        # Verify classification model card
        assert clf_card.metrics is not None
        assert clf_card.model_details['type'] == "RandomForestClassifier"
        assert clf_card.metrics.accuracy is not None
        if clf_card.metrics.roc_auc is not None:
            assert 0 <= clf_card.metrics.roc_auc <= 1

        # Test regression model
        X_reg, y_reg = make_regression(n_samples=500, n_features=5, noise=0.1, random_state=42)
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )

        reg = RandomForestRegressor(n_estimators=5, random_state=42)
        reg.fit(X_train_reg, y_train_reg)

        reg_card = ModelCard.load_from_estimator(
            estimator=reg,
            model_name="RegressionModel",
            X_train=X_train_reg,
            y_train=y_train_reg,
            X_test=X_test_reg,
            y_test=y_test_reg,
            task_type='regression'
        )

        # Verify regression model card
        assert reg_card.metrics is not None
        assert reg_card.model_details['type'] == "RandomForestRegressor"
        assert reg_card.metrics.mse is not None
        assert reg_card.metrics.rmse is not None
        assert reg_card.metrics.r2_score is not None

        # Both cards should have complete documentation
        for card in [clf_card, reg_card]:
            # Verify architecture documentation
            assert card.model_details is not None
            assert 'type' in card.model_details
            assert 'description' in card.model_details

            # Verify evaluation results
            assert card.metrics is not None
            metrics_dict = card.metrics.to_dict()
            assert len(metrics_dict) > 0

            # Verify ethical considerations
            assert card.ethical_considerations is not None
            assert len(card.ethical_considerations.limitations) > 0

            # Verify both markdown and JSON can be generated
            assert len(card.to_markdown()) > 100
            assert len(card.to_json()) > 100

    def test_model_card_template_completeness(self):
        """
        Property 1: Model card generation completeness

        Validates that the model card template includes all
        required sections from model card standards
        """
        X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_redundant=1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        estimator = LogisticRegression(random_state=42, max_iter=100)
        estimator.fit(X_train, y_train)

        # Create a comprehensive model card
        card = ModelCard.load_from_estimator(
            estimator=estimator,
            model_name="TemplateTestModel",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            task_type='classification'
        )

        # Add comprehensive training data documentation
        card.set_training_data(
            dataset_name="Test Dataset",
            dataset_size=100,
            split={'train': 80, 'test': 20},
            preprocessing=['Feature scaling', 'Missing value imputation'],
            features=['feature_1', 'feature_2', 'feature_3']
        )

        # Generate and verify markdown structure
        markdown = card.to_markdown()

        # Check for all required sections based on Model Card standard
        required_sections = {
            "Model Details": "Basic model information",
            "Intended Use": "Model use cases",
            "Performance Metrics": "Model performance",
            "Training Data": "Data used for training",
            "Ethical Considerations": "Ethical implications"
        }

        for section, description in required_sections.items():
            assert section in markdown, f"Missing required section: {section}"

        # Verify section headers are properly formatted
        lines = markdown.split('\n')
        header_lines = [line for line in lines if line.startswith('##')]

        section_headers = [line.replace('## ', '') for line in header_lines]
        for section in required_sections.keys():
            assert section in section_headers, f"Section header not found: {section}"

        # Verify model documentation includes fairness considerations
        # This should be in ethical considerations
        assert "Ethical Considerations" in markdown

        # Verify model comparison information is available
        # (metrics provide comparison baseline)
        assert "Performance Metrics" in markdown

        # Verify format is consistent and professional
        assert markdown.count("## ") >= 5  # Multiple sections
        assert markdown.count("**") >= 10  # Proper formatting

        # Verify no empty major sections
        sections_content = markdown.split('## ')[1:]  # Split by section headers
        for content in sections_content:
            # Remove empty lines and check if there's meaningful content
            meaningful_lines = [line for line in content.split('\n') if line.strip()]
            assert len(meaningful_lines) > 0, "Empty section found"