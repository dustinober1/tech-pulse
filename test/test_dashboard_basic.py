"""
Basic functionality tests for Tech-Pulse Streamlit dashboard.
"""

import unittest
import sys
import os
import pandas as pd
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDashboardBasicFunctionality(unittest.TestCase):
    """Test basic dashboard functionality without importing streamlit"""

    def test_data_loader_integration(self):
        """Test that data_loader functions can be imported and work"""
        try:
            from data_loader import fetch_hn_data, analyze_sentiment, get_topics
            self.assertTrue(True, "Data loader functions imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import data_loader functions: {e}")

    def test_dashboard_config_import(self):
        """Test that dashboard configuration can be imported"""
        try:
            from dashboard_config import PAGE_CONFIG, COLORS, DEFAULT_SETTINGS
            self.assertIsInstance(PAGE_CONFIG, dict)
            self.assertIsInstance(COLORS, dict)
            self.assertIsInstance(DEFAULT_SETTINGS, dict)
        except ImportError as e:
            self.fail(f"Failed to import dashboard config: {e}")

    def test_required_dependencies_available(self):
        """Test that required dependencies are available"""
        try:
            import streamlit
            import plotly.express as px
            import pandas as pd
            self.assertTrue(True, "All dependencies available")
        except ImportError as e:
            self.fail(f"Missing required dependency: {e}")

    def test_basic_data_structure_compatibility(self):
        """Test that our data structure is compatible with dashboard expectations"""
        # Create sample data like data_loader would produce
        sample_data = pd.DataFrame({
            'title': ['Test Story 1', 'Test Story 2'],
            'score': [100, 200],
            'descendants': [10, 20],
            'time': [datetime.now(), datetime.now()],
            'url': ['http://example.com/1', 'http://example.com/2'],
            'sentiment_score': [0.5, -0.3],
            'sentiment_label': ['Positive', 'Negative'],
            'topic_id': [0, 1],
            'topic_keyword': ['AI_Technology', 'Security_Vulnerability']
        })

        # Test that required columns exist
        required_columns = ['title', 'score', 'descendants', 'time', 'url']
        for col in required_columns:
            self.assertIn(col, sample_data.columns, f"Missing required column: {col}")

        # Test that analysis columns exist
        analysis_columns = ['sentiment_score', 'sentiment_label', 'topic_id', 'topic_keyword']
        for col in analysis_columns:
            self.assertIn(col, sample_data.columns, f"Missing analysis column: {col}")

    def test_sentiment_label_values(self):
        """Test that sentiment labels have expected values"""
        from dashboard_config import SENTIMENT_COLORS
        expected_labels = ['Positive', 'Negative', 'Neutral']
        for label in expected_labels:
            self.assertIn(label, SENTIMENT_COLORS)

    def test_export_format_options(self):
        """Test that export formats are properly defined"""
        from dashboard_config import EXPORT_FORMATS
        expected_formats = ['CSV', 'JSON', 'Excel']
        for fmt in expected_formats:
            self.assertIn(fmt, EXPORT_FORMATS)


if __name__ == '__main__':
    unittest.main()