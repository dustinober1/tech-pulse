"""Tests for LIME explainer."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_classification, make_regression

from src.interpretability.lime_explainer import LIMEExplainer


@pytest.fixture
def classification_data():
    """Create sample classification data."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, y


@pytest.fixture
def regression_data():
    """Create sample regression data."""
    X, y = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=5,
        noise=0.1,
        random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, y


@pytest.fixture
def classification_model(classification_data):
    """Create a trained classification model."""
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def regression_model(regression_data):
    """Create a trained regression model."""
    X, y = regression_data
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


class TestLIMEExplainer:
    """Test cases for LIMEExplainer class."""

    def test_initialization_tabular(self):
        """Test LIME explainer initialization for tabular data."""
        explainer = LIMEExplainer(
            mode='tabular',
            feature_names=['feature_1', 'feature_2'],
            categorical_features=[0],
            discretize_continuous=True
        )

        assert explainer.mode == 'tabular'
        assert explainer.feature_names == ['feature_1', 'feature_2']
        assert explainer.categorical_features == [0]
        assert explainer.discretize_continuous is True

    def test_initialization_text(self):
        """Test LIME explainer initialization for text data."""
        explainer = LIMEExplainer(mode='text', verbose=True)

        assert explainer.mode == 'text'
        assert explainer.verbose is True

    def test_initialization_image(self):
        """Test LIME explainer initialization for image data."""
        explainer = LIMEExplainer(mode='image')

        assert explainer.mode == 'image'

    def test_initialization_invalid_mode(self):
        """Test error for invalid mode."""
        with pytest.raises(ValueError, match="Invalid mode"):
            LIMEExplainer(mode='invalid_mode')

    @patch('src.interpretability.lime_explainer.lime')
    def test_fit_classification_model(self, mock_lime, classification_model, classification_data):
        """Test fitting LIME explainer with classification model."""
        X, _ = classification_data

        # Mock LIME tabular explainer
        mock_explainer = Mock()
        mock_lime.lime_tabular.LimeTabularExplainer.return_value = mock_explainer

        explainer = LIMEExplainer(mode='tabular')
        explainer.fit(classification_model, X)

        assert explainer.model == classification_model
        assert explainer.training_data is not None
        assert len(explainer.feature_names) == 10
        mock_lime.lime_tabular.LimeTabularExplainer.assert_called_once()

    @patch('src.interpretability.lime_explainer.lime')
    def test_fit_regression_model(self, mock_lime, regression_model, regression_data):
        """Test fitting LIME explainer with regression model."""
        X, _ = regression_data

        # Mock LIME tabular explainer
        mock_explainer = Mock()
        mock_lime.lime_tabular.LimeTabularExplainer.return_value = mock_explainer

        # Mock model attributes
        regression_model._estimator_type = 'regressor'

        explainer = LIMEExplainer(mode='tabular')
        explainer.fit(regression_model, X)

        assert explainer.model == regression_model
        mock_lime.lime_tabular.LimeTabularExplainer.assert_called_once_with(
            X.values,
            feature_names=explainer.feature_names,
            categorical_features=[],
            discretize_continuous=True,
            kernel_width=None,
            verbose=False,
            mode='regression'
        )

    @patch('src.interpretability.lime_explainer.lime')
    def test_fit_text_mode(self, mock_lime):
        """Test fitting LIME explainer in text mode."""
        mock_explainer = Mock()
        mock_lime.lime_text.LimeTextExplainer.return_value = mock_explainer

        model = Mock()
        training_data = ["sample text 1", "sample text 2"]

        explainer = LIMEExplainer(mode='text')
        explainer.fit(model, training_data)

        mock_lime.lime_text.LimeTextExplainer.assert_called_once_with(
            verbose=False,
            class_names=['Not Positive', 'Positive']
        )

    @patch('src.interpretability.lime_explainer.lime')
    def test_fit_image_mode(self, mock_lime):
        """Test fitting LIME explainer in image mode."""
        mock_explainer = Mock()
        mock_lime.lime_image.LimeImageExplainer.return_value = mock_explainer

        model = Mock()
        training_data = np.random.rand(10, 64, 64, 3)

        explainer = LIMEExplainer(mode='image')
        explainer.fit(model, training_data)

        mock_lime.lime_image.LimeImageExplainer.assert_called_once_with(
            verbose=False,
            random_state=42
        )

    @patch('src.interpretability.lime_explainer.lime')
    def test_explain_instance_tabular(self, mock_lime, classification_model, classification_data):
        """Test explaining a single tabular instance."""
        X, _ = classification_data

        # Mock LIME components
        mock_lime_explainer = Mock()
        mock_explanation = Mock()
        mock_explanation.as_list.return_value = [('feature_1 > 0.5', 0.3), ('feature_2 <= 1.0', -0.2)]
        mock_explanation.predicted_proba = np.array([0.2, 0.8])
        mock_explanation.intercept = np.array([0.5, 0.5])

        mock_lime.lime_tabular.LimeTabularExplainer.return_value = mock_lime_explainer
        mock_lime_explainer.explain_instance.return_value = mock_explanation

        # Mock model prediction function
        mock_predict_fn = Mock(return_value=np.array([[0.2, 0.8]]))

        explainer = LIMEExplainer(mode='tabular')
        explainer.fit(classification_model, X)

        result = explainer.explain_instance(
            X.iloc[0],
            predict_fn=mock_predict_fn,
            num_features=5
        )

        assert 'feature_contributions' in result
        assert 'feature_values' in result
        assert 'predicted_value' in result
        assert result['type'] == 'tabular_explanation'
        assert len(result['feature_contributions']) == 2

    @patch('src.interpretability.lime_explainer.lime')
    def test_explain_instance_text(self, mock_lime):
        """Test explaining a text instance."""
        # Mock LIME components
        mock_lime_explainer = Mock()
        mock_explanation = Mock()
        mock_explanation.as_list.return_value = [('good', 0.4), ('bad', -0.3)]
        mock_explanation.score = 0.75
        mock_explanation.intercept = 0.5

        mock_lime.lime_text.LimeTextExplainer.return_value = mock_lime_explainer
        mock_lime_explainer.explain_instance.return_value = mock_explanation

        model = Mock()
        training_data = ["sample text"]

        explainer = LIMEExplainer(mode='text')
        explainer.fit(model, training_data)

        mock_predict_fn = Mock(return_value=np.array([[0.2, 0.8]]))

        result = explainer.explain_instance(
            "This is good text",
            predict_fn=mock_predict_fn
        )

        assert result['type'] == 'text_explanation'
        assert result['score'] == 0.75

    @patch('src.interpretability.lime_explainer.lime')
    def test_explain_instance_image(self, mock_lime):
        """Test explaining an image instance."""
        # Mock LIME components
        mock_lime_explainer = Mock()
        mock_explanation = Mock()
        mock_explanation.top_labels = [1]

        mock_lime.lime_image.LimeImageExplainer.return_value = mock_lime_explainer
        mock_lime_explainer.explain_instance.return_value = mock_explanation

        model = Mock()
        training_data = np.random.rand(10, 64, 64, 3)

        explainer = LIMEExplainer(mode='image')
        explainer.fit(model, training_data)

        mock_predict_fn = Mock(return_value=np.array([[0.2, 0.8]]))

        result = explainer.explain_instance(
            np.random.rand(64, 64, 3),
            predict_fn=mock_predict_fn
        )

        assert result['type'] == 'image_explanation'
        assert 'explanation' in result

    @patch('src.interpretability.lime_explainer.lime')
    def test_explain_multiple(self, mock_lime, classification_model, classification_data):
        """Test explaining multiple instances."""
        X, _ = classification_data

        # Mock LIME components
        mock_lime_explainer = Mock()
        mock_explanation = Mock()
        mock_explanation.as_list.return_value = [('feature_1 > 0.5', 0.3)]
        mock_explanation.predicted_proba = np.array([0.2, 0.8])
        mock_explanation.intercept = np.array([0.5, 0.5])

        mock_lime.lime_tabular.LimeTabularExplainer.return_value = mock_lime_explainer
        mock_lime_explainer.explain_instance.return_value = mock_explanation

        # Mock model prediction function
        mock_predict_fn = Mock(return_value=np.array([[0.2, 0.8]]))

        explainer = LIMEExplainer(mode='tabular')
        explainer.fit(classification_model, X)

        results = explainer.explain_multiple(
            X.iloc[:3],
            predict_fn=mock_predict_fn,
            num_features=2
        )

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result['instance_index'] == i
            assert 'feature_contributions' in result

    def test_explain_multiple_dataframe_input(self, classification_model, classification_data):
        """Test explaining multiple instances with DataFrame input."""
        X, _ = classification_data

        # Create mock explainer that's already fitted
        explainer = LIMEExplainer(mode='tabular')
        explainer.model = classification_model
        explainer.explainer = Mock()

        # Mock explain_instance to return a simple explanation
        mock_explanation = {
            'type': 'tabular_explanation',
            'feature_contributions': ['feature_1 > 0.5'],
            'feature_values': [0.3]
        }
        explainer.explain_instance = Mock(return_value=mock_explanation)

        results = explainer.explain_multiple(X[:3])

        assert len(results) == 3
        assert all(result['instance_index'] == i for i, result in enumerate(results))

    def test_create_explanation_plot(self):
        """Test creating explanation plot."""
        # Sample explanation data
        explanation_data = {
            'feature_contributions': ['feature_1 > 0.5', 'feature_2 <= 1.0', 'feature_3'],
            'feature_values': [0.3, -0.2, 0.1],
            'predicted_value': 0.75
        }

        explainer = LIMEExplainer(mode='tabular')
        result = explainer.create_explanation_plot(explanation_data)

        assert 'plot_base64' in result
        assert 'explanation_data' in result
        assert result['plot_type'] == 'lime_explanation'
        assert isinstance(result['plot_base64'], str)
        assert len(result['plot_base64']) > 0

    def test_get_feature_importance_summary(self):
        """Test getting feature importance summary across explanations."""
        # Create sample explanations
        explanations = [
            {
                'feature_contributions': ['feature_1', 'feature_2'],
                'feature_values': [0.3, -0.2]
            },
            {
                'feature_contributions': ['feature_1', 'feature_3'],
                'feature_values': [0.5, 0.1]
            },
            {
                'feature_contributions': ['feature_2', 'feature_3'],
                'feature_values': [-0.1, 0.4]
            }
        ]

        explainer = LIMEExplainer(mode='tabular')
        summary = explainer.get_feature_importance_summary(explanations, top_k=2)

        assert 'top_features' in summary
        assert 'feature_appearance_counts' in summary
        assert 'total_explanations' in summary
        assert summary['total_explanations'] == 3
        assert len(summary['top_features']) == 2

        # Check appearance counts
        assert summary['feature_appearance_counts']['feature_1'] == 2
        assert summary['feature_appearance_counts']['feature_2'] == 2
        assert summary['feature_appearance_counts']['feature_3'] == 2

    def test_is_classification_classifier(self):
        """Test classification detection for classifier models."""
        model = Mock()
        model.predict_proba = Mock()
        model.predict = Mock()

        explainer = LIMEExplainer(mode='tabular')
        explainer.model = model

        assert explainer._is_classification() is True

    def test_is_classification_regressor(self):
        """Test classification detection for regressor models."""
        model = Mock()
        model._estimator_type = 'regressor'
        model.predict = Mock()

        explainer = LIMEExplainer(mode='tabular')
        explainer.model = model

        assert explainer._is_classification() is False

    def test_get_predict_function_classification(self):
        """Test getting predict function for classification."""
        model = Mock()
        model.predict_proba = Mock()
        model.predict = Mock()

        explainer = LIMEExplainer(mode='tabular')
        explainer.model = model
        explainer._is_classification = Mock(return_value=True)

        predict_fn = explainer._get_predict_function()
        assert predict_fn == model.predict_proba

    def test_get_predict_function_regression(self):
        """Test getting predict function for regression."""
        model = Mock()
        model.predict = Mock()

        explainer = LIMEExplainer(mode='tabular')
        explainer.model = model
        explainer._is_classification = Mock(return_value=False)

        predict_fn = explainer._get_predict_function()
        assert predict_fn == model.predict

    def test_parse_explanation_tabular(self):
        """Test parsing tabular explanation."""
        # Mock explanation object
        mock_explanation = Mock()
        mock_explanation.as_list.return_value = [('feature_1 > 0.5', 0.3), ('feature_2 <= 1.0', -0.2)]
        mock_explanation.predicted_proba = np.array([0.2, 0.8])
        mock_explanation.intercept = np.array([0.5, 0.5])
        mock_explanation.local_pred = 0.75
        mock_explanation.score = 0.9

        explainer = LIMEExplainer(mode='tabular')
        explainer.feature_names = ['feature_1', 'feature_2']

        result = explainer._parse_explanation(mock_explanation, None)

        assert result['type'] == 'tabular_explanation'
        assert len(result['feature_contributions']) == 2
        assert result['predicted_value'] == 0.8
        assert result['intercept'] == 0.5
        assert result['local_pred'] == 0.75
        assert result['score'] == 0.9

    def test_parse_explanation_text(self):
        """Test parsing text explanation."""
        # Mock explanation object
        mock_explanation = Mock()
        mock_explanation.as_list.return_value = [('good', 0.4), ('bad', -0.3)]
        mock_explanation.score = 0.75
        mock_explanation.intercept = 0.5

        explainer = LIMEExplainer(mode='text')

        result = explainer._parse_explanation(mock_explanation, "sample text")

        assert result['type'] == 'text_explanation'
        assert len(result['feature_contributions']) == 2
        assert result['score'] == 0.75
        assert result['intercept'] == 0.5

    def test_parse_explanation_image(self):
        """Test parsing image explanation."""
        # Mock explanation object
        mock_explanation = Mock()
        mock_explanation.top_labels = [1]

        explainer = LIMEExplainer(mode='image')

        result = explainer._parse_explanation(mock_explanation, np.random.rand(64, 64, 3))

        assert result['type'] == 'image_explanation'
        assert 'explanation' in result
        assert result['predicted_value'] == 1

    def test_error_without_lime_library(self, classification_model, classification_data):
        """Test error when LIME library is not installed."""
        with patch.dict('sys.modules', {'lime': None}):
            with patch('src.interpretability.lime_explainer.lime', side_effect=ImportError):
                with pytest.raises(ImportError, match="LIME library is not installed"):
                    explainer = LIMEExplainer(mode='tabular')
                    explainer.fit(classification_model, classification_data[0])

    def test_error_explaining_without_initialization(self):
        """Test error when trying to explain without proper initialization."""
        explainer = LIMEExplainer(mode='tabular')
        # explainer.explainer is None by default

        with pytest.raises(ValueError, match="Explainer not initialized"):
            explainer.explain_instance([1, 2, 3])

    def test_fit_with_numpy_array(self, classification_model, classification_data):
        """Test fitting with numpy array input."""
        X, _ = classification_data

        # Mock LIME components
        with patch('src.interpretability.lime_explainer.lime') as mock_lime:
            mock_explainer = Mock()
            mock_lime.lime_tabular.LimeTabularExplainer.return_value = mock_explainer

            explainer = LIMEExplainer(mode='tabular', feature_names=['f1', 'f2'])
            explainer.fit(classification_model, X.values)

            # Should use provided feature names
            assert explainer.feature_names == ['f1', 'f2']
            mock_lime.lime_tabular.LimeTabularExplainer.assert_called_once()

    def test_explain_instance_with_pandas_series(self, classification_model, classification_data):
        """Test explaining instance with pandas Series input."""
        X, _ = classification_data

        # Create mock explainer that's already fitted
        explainer = LIMEExplainer(mode='tabular')
        explainer.model = classification_model
        explainer.explainer = Mock()

        # Mock explain_instance to track input type
        mock_explain = explainer.explainer.explain_instance
        called_with = []

        def track_call(*args, **kwargs):
            called_with.append((args[0] if args else None, kwargs))
            return Mock()

        explainer.explainer.explain_instance = track_call

        # Explain with Series
        explainer.explain_instance(X.iloc[0])

        # Verify numpy array was passed to LIME
        assert isinstance(called_with[0][0], np.ndarray)
        np.testing.assert_array_equal(called_with[0][0], X.iloc[0].values)

    def test_feature_names_extraction_from_dataframe(self):
        """Test that feature names are extracted from DataFrame when not provided."""
        df = pd.DataFrame({
            'feature_a': [1, 2],
            'feature_b': [3, 4],
            'feature_c': [5, 6]
        })

        model = Mock()

        # Mock LIME to avoid actual library dependency
        with patch('src.interpretability.lime_explainer.lime'):
            explainer = LIMEExplainer(mode='tabular')
            explainer.fit(model, df)

            assert explainer.feature_names == ['feature_a', 'feature_b', 'feature_c']