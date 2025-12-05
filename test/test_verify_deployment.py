"""
Unit tests for the deployment verification script.
"""

import unittest
from unittest.mock import patch, Mock
import sys
import os

# Add scripts directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from verify_deployment import verify_deployment


class TestVerifyDeployment(unittest.TestCase):
    """Test cases for deployment verification functionality."""

    @patch('verify_deployment.requests.get')
    def test_verify_deployment_success(self, mock_get):
        """Test successful deployment verification."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = verify_deployment("https://test-app.streamlit.app")

        self.assertTrue(result)
        mock_get.assert_called_once_with(
            "https://test-app.streamlit.app",
            timeout=30
        )

    @patch('verify_deployment.requests.get')
    def test_verify_deployment_404_error(self, mock_get):
        """Test deployment verification with 404 error."""
        # Mock 404 response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = verify_deployment("https://test-app.streamlit.app")

        self.assertFalse(result)

    @patch('verify_deployment.requests.get')
    def test_verify_deployment_500_error(self, mock_get):
        """Test deployment verification with 500 error."""
        # Mock 500 response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        result = verify_deployment("https://test-app.streamlit.app")

        self.assertFalse(result)

    @patch('verify_deployment.requests.get')
    def test_verify_deployment_connection_error(self, mock_get):
        """Test deployment verification with connection error."""
        # Mock connection error
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Failed to connect")

        result = verify_deployment("https://test-app.streamlit.app")

        self.assertFalse(result)

    @patch('verify_deployment.requests.get')
    def test_verify_deployment_timeout(self, mock_get):
        """Test deployment verification with timeout."""
        # Mock timeout error
        import requests
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        result = verify_deployment("https://test-app.streamlit.app")

        self.assertFalse(result)

    @patch('verify_deployment.requests.get')
    def test_verify_deployment_generic_request_exception(self, mock_get):
        """Test deployment verification with generic request exception."""
        # Mock generic request exception
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("Generic error")

        result = verify_deployment("https://test-app.streamlit.app")

        self.assertFalse(result)

    def test_verify_deployment_invalid_url(self):
        """Test deployment verification with invalid URL."""
        result = verify_deployment("not-a-valid-url")

        self.assertFalse(result)

    def test_verify_deployment_empty_url(self):
        """Test deployment verification with empty URL."""
        result = verify_deployment("")

        self.assertFalse(result)

    @patch('verify_deployment.requests.get')
    def test_verify_deployment_redirect_success(self, mock_get):
        """Test deployment verification with redirect."""
        # Mock redirect that ends successfully
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = verify_deployment("http://test-app.streamlit.app")  # http redirects to https

        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
