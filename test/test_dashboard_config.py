"""
Test suite for Tech-Pulse dashboard configuration.
"""

import unittest
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard_config import (
    PAGE_CONFIG, COLORS, SENTIMENT_COLORS, DEFAULT_SETTINGS,
    CHART_CONFIG, HELP_TEXT, ERROR_MESSAGES, SUCCESS_MESSAGES, LOADING_MESSAGES
)


class TestDashboardConfig(unittest.TestCase):
    """Test dashboard configuration"""

    def test_page_config_structure(self):
        """Test that page configuration has required keys"""
        required_keys = ['page_title', 'page_icon', 'layout', 'initial_sidebar_state']
        for key in required_keys:
            self.assertIn(key, PAGE_CONFIG)
            self.assertIsNotNone(PAGE_CONFIG[key])

    def test_page_config_values(self):
        """Test that page config has valid values"""
        self.assertEqual(PAGE_CONFIG['page_title'], "Tech-Pulse Dashboard")
        self.assertEqual(PAGE_CONFIG['page_icon'], "⚡")
        self.assertIn(PAGE_CONFIG['layout'], ['centered', 'wide'])
        self.assertIn(PAGE_CONFIG['initial_sidebar_state'], ['auto', 'expanded', 'collapsed'])

    def test_color_scheme(self):
        """Test color scheme has required colors"""
        required_colors = ['primary', 'secondary', 'accent', 'positive', 'negative', 'neutral']
        for color in required_colors:
            self.assertIn(color, COLORS)
            # Check if color is a valid hex code
            self.assertTrue(COLORS[color].startswith('#'))
            self.assertTrue(len(COLORS[color]) == 7)

    def test_sentiment_colors(self):
        """Test sentiment color mapping"""
        required_sentiments = ['Positive', 'Negative', 'Neutral']
        for sentiment in required_sentiments:
            self.assertIn(sentiment, SENTIMENT_COLORS)
            # Check if mapped to main colors
            self.assertIn(SENTIMENT_COLORS[sentiment], COLORS.values())

    def test_default_settings(self):
        """Test default settings are valid"""
        self.assertIn('default_stories', DEFAULT_SETTINGS)
        self.assertIn('min_stories', DEFAULT_SETTINGS)
        self.assertIn('max_stories', DEFAULT_SETTINGS)

        # Test logical constraints
        self.assertGreater(DEFAULT_SETTINGS['min_stories'], 0)
        self.assertLessEqual(DEFAULT_SETTINGS['min_stories'], DEFAULT_SETTINGS['max_stories'])
        self.assertGreaterEqual(DEFAULT_SETTINGS['default_stories'], DEFAULT_SETTINGS['min_stories'])
        self.assertLessEqual(DEFAULT_SETTINGS['default_stories'], DEFAULT_SETTINGS['max_stories'])
        self.assertGreater(DEFAULT_SETTINGS['refresh_interval'], 0)

    def test_chart_config(self):
        """Test chart configuration"""
        required_keys = ['height', 'theme', 'color_sequence']
        for key in required_keys:
            self.assertIn(key, CHART_CONFIG)

        # Test chart height is reasonable
        self.assertGreater(CHART_CONFIG['height'], 200)
        self.assertLess(CHART_CONFIG['height'], 1000)

        # Test color sequence is not empty
        self.assertGreater(len(CHART_CONFIG['color_sequence']), 0)

    def test_help_text(self):
        """Test help text content"""
        required_help = ['sentiment', 'topics', 'metrics', 'refresh']
        for key in required_help:
            self.assertIn(key, HELP_TEXT)
            self.assertIsInstance(HELP_TEXT[key], str)
            self.assertGreater(len(HELP_TEXT[key]), 10)  # Should have meaningful content

    def test_error_messages(self):
        """Test error messages"""
        required_errors = ['no_data', 'api_error', 'analysis_error', 'connection_error']
        for key in required_errors:
            self.assertIn(key, ERROR_MESSAGES)
            self.assertIsInstance(ERROR_MESSAGES[key], str)
            self.assertGreater(len(ERROR_MESSAGES[key]), 5)

    def test_success_messages(self):
        """Test success messages"""
        required_success = ['data_loaded', 'refresh_complete', 'export_successful']
        for key in required_success:
            self.assertIn(key, SUCCESS_MESSAGES)
            self.assertIsInstance(SUCCESS_MESSAGES[key], str)
            self.assertGreater(len(SUCCESS_MESSAGES[key]), 5)

    def test_loading_messages(self):
        """Test loading messages"""
        required_loading = ['fetching', 'analyzing', 'loading', 'refreshing']
        for key in required_loading:
            self.assertIn(key, LOADING_MESSAGES)
            self.assertIsInstance(LOADING_MESSAGES[key], str)
            self.assertGreater(len(LOADING_MESSAGES[key]), 5)

    def test_config_consistency(self):
        """Test configuration consistency"""
        # Test sentiment colors use main color palette
        for sentiment, color in SENTIMENT_COLORS.items():
            self.assertIn(color, list(COLORS.values()))


if __name__ == '__main__':
    unittest.main()