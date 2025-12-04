"""
Comprehensive test suite for Tech-Pulse Streamlit dashboard.
"""

import unittest
import sys
import os
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock streamlit before importing app
import sys
from types import ModuleType

# Create mock streamlit module
mock_streamlit = ModuleType('streamlit')
mock_streamlit.set_page_config = Mock()
mock_streamlit.markdown = Mock()
mock_streamlit.sidebar = Mock()
mock_streamlit.columns = Mock(return_value=[Mock(), Mock(), Mock()])
mock_streamlit.button = Mock(return_value=False)
mock_streamlit.slider = Mock(return_value=30)
mock_streamlit.checkbox = Mock(return_value=False)
mock_streamlit.selectbox = Mock(return_value="None")
mock_streamlit.multiselect = Mock(return_value=["All"])
mock_streamlit.expander = Mock()
mock_streamlit.spinner = Mock()
mock_streamlit.success = Mock()
mock_streamlit.error = Mock()
mock_streamlit.warning = Mock()
mock_streamlit.info = Mock()
mock_streamlit.plotly_chart = Mock()
mock_streamlit.dataframe = Mock()
mock_streamlit.download_button = Mock()
mock_streamlit.session_state = {}

sys.modules['streamlit'] = mock_streamlit

# Mock plotly
mock_plotly = ModuleType('plotly.express')
mock_plotly.scatter = Mock(return_value=Mock())
mock_plotly.bar = Mock(return_value=Mock())
sys.modules['plotly.express'] = mock_plotly

mock_plotly_go = ModuleType('plotly.graph_objects')
mock_plotly_go.Figure = Mock()
sys.modules['plotly.graph_objects'] = mock_plotly_go

# Now import app after mocking
import app
from dashboard_config import COLORS, DEFAULT_SETTINGS, PAGE_CONFIG


class TestDashboardConfig(unittest.TestCase):
    """Test dashboard configuration"""

    def test_page_config_structure(self):
        """Test that page configuration has required keys"""
        required_keys = ['page_title', 'page_icon', 'layout', 'initial_sidebar_state']
        for key in required_keys:
            self.assertIn(key, PAGE_CONFIG)

    def test_color_scheme(self):
        """Test color scheme has required colors"""
        required_colors = ['primary', 'secondary', 'accent', 'positive', 'negative', 'neutral']
        for color in required_colors:
            self.assertIn(color, COLORS)

    def test_default_settings(self):
        """Test default settings are valid"""
        self.assertIn('default_stories', DEFAULT_SETTINGS)
        self.assertGreater(DEFAULT_SETTINGS['min_stories'], 0)
        self.assertLessEqual(DEFAULT_SETTINGS['min_stories'], DEFAULT_SETTINGS['max_stories'])


class TestDashboardInitialization(unittest.TestCase):
    """Test dashboard initialization functions"""

    def test_initialize_session_state(self):
        """Test session state initialization"""
        # Clear session state
        app.st.session_state.clear()

        # Initialize
        app.initialize_session_state()

        # Check required keys
        required_keys = ['data', 'last_refresh', 'auto_refresh', 'stories_count', 'refresh_countdown']
        for key in required_keys:
            self.assertIn(key, app.st.session_state)

    def test_initialize_session_state_preserves_existing(self):
        """Test that initialization preserves existing session state"""
        # Set some initial state
        app.st.session_state['test_key'] = 'test_value'
        app.st.session_state['data'] = Mock()

        # Initialize
        app.initialize_session_state()

        # Check that existing values are preserved
        self.assertEqual(app.st.session_state['test_key'], 'test_value')


class TestDashboardComponents(unittest.TestCase):
    """Test dashboard component functions"""

    @patch('app.st')
    def test_create_header(self, mock_st):
        """Test header creation"""
        app.create_header()
        mock_st.markdown.assert_called()

        # Check if the call contains expected content
        call_args = mock_st.markdown.call_args
        self.assertIn('Tech-Pulse Dashboard', call_args[0][0])

    @patch('app.st')
    def test_create_sidebar(self, mock_st):
        """Test sidebar creation"""
        # Mock sidebar components
        mock_st.sidebar.slider.return_value = 30
        mock_st.sidebar.button.return_value = False
        mock_st.sidebar.checkbox.return_value = False
        mock_st.sidebar.multiselect.return_value = ['All']
        mock_st.sidebar.selectbox.return_value = 'None'

        # Mock session state
        app.st.session_state.data = pd.DataFrame()
        app.st.session_state.last_refresh = datetime.now()
        app.st.session_state.auto_refresh = False
        app.st.session_state.stories_count = 30

        app.create_sidebar()

        # Verify sidebar components were called
        mock_st.sidebar.slider.assert_called()
        mock_st.sidebar.button.assert_called()
        mock_st.sidebar.checkbox.assert_called()

    def test_create_metrics_row_with_data(self):
        """Test metrics row creation with data"""
        # Create test data
        test_data = pd.DataFrame({
            'sentiment_score': [0.5, -0.3, 0.1],
            'descendants': [10, 20, 5],
            'topic_keyword': ['AI_Technology', 'Security_Vulnerability', 'Cloud_Computing']
        })

        with patch('app.st') as mock_st:
            mock_columns = Mock()
            mock_columns.__enter__ = Mock(return_value=Mock())
            mock_columns.__exit__ = Mock(return_value=None)
            mock_st.columns.return_value = [mock_columns, mock_columns, mock_columns]

            app.create_metrics_row(test_data)

            # Verify metrics were created
            self.assertTrue(mock_st.columns.called)

    def test_create_metrics_row_with_empty_data(self):
        """Test metrics row creation with empty data"""
        with patch('app.st') as mock_st:
            app.create_metrics_row(None)
            # Should not crash with None data
            mock_st.columns.assert_not_called()

    @patch('app.px')
    @patch('app.st')
    def test_create_charts_row_with_data(self, mock_st, mock_px):
        """Test charts row creation with data"""
        # Create test data
        test_data = pd.DataFrame({
            'title': ['Story 1', 'Story 2', 'Story 3'],
            'score': [100, 200, 150],
            'time': [datetime.now()] * 3,
            'sentiment_label': ['Positive', 'Negative', 'Neutral'],
            'topic_keyword': ['AI_Tech', 'Security', 'Cloud']
        })

        mock_fig = Mock()
        mock_px.scatter.return_value = mock_fig
        mock_px.bar.return_value = mock_fig

        with patch('app.st') as mock_st:
            mock_columns = Mock()
            mock_columns.__enter__ = Mock(return_value=Mock())
            mock_columns.__exit__ = Mock(return_value=None)
            mock_st.columns.return_value = [mock_columns, mock_columns]

            app.create_charts_row(test_data)

            # Verify charts were created
            mock_px.scatter.assert_called_once()
            mock_px.bar.assert_called_once()

    @patch('app.st')
    def test_create_data_table_with_data(self, mock_st):
        """Test data table creation with data"""
        # Create test data
        test_data = pd.DataFrame({
            'title': ['Story 1', 'Story 2'],
            'score': [100, 200],
            'sentiment_label': ['Positive', 'Negative']
        })

        mock_expander = Mock()
        mock_st.expander.return_value.__enter__ = Mock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = Mock(return_value=None)

        app.create_data_table(test_data)

        # Verify expander was created
        mock_st.expander.assert_called_once_with("📋 View Raw Data", expanded=False)


class TestDashboardDataOperations(unittest.TestCase):
    """Test dashboard data operations"""

    @patch('app.get_topics')
    @patch('app.analyze_sentiment')
    @patch('app.fetch_hn_data')
    @patch('app.st')
    def test_refresh_data_success(self, mock_st, mock_fetch, mock_analyze, mock_topics):
        """Test successful data refresh"""
        # Mock data
        mock_df = pd.DataFrame({
            'title': ['Test Story'],
            'score': [100]
        })
        mock_fetch.return_value = mock_df
        mock_analyze.return_value = mock_df
        mock_topics.return_value = mock_df

        app.refresh_data()

        # Verify functions were called
        mock_fetch.assert_called_once_with(limit=app.st.session_state.stories_count)
        mock_analyze.assert_called_once_with(mock_df)
        mock_topics.assert_called_once_with(mock_df)

        # Verify success message
        mock_st.success.assert_called()

    @patch('app.fetch_hn_data')
    @patch('app.st')
    def test_refresh_data_empty_response(self, mock_st, mock_fetch):
        """Test data refresh with empty response"""
        mock_fetch.return_value = pd.DataFrame()

        app.refresh_data()

        # Verify error message for empty data
        mock_st.error.assert_called()

    @patch('app.fetch_hn_data')
    @patch('app.st')
    def test_refresh_data_api_error(self, mock_st, mock_fetch):
        """Test data refresh with API error"""
        mock_fetch.side_effect = Exception("API Error")

        app.refresh_data()

        # Verify error message
        mock_st.error.assert_called()

    @patch('app.st')
    def test_export_data_csv(self, mock_st):
        """Test CSV export functionality"""
        # Create test data
        test_data = pd.DataFrame({
            'title': ['Test Story'],
            'score': [100]
        })
        app.st.session_state.data = test_data

        app.export_data("CSV")

        # Verify download button was created
        mock_st.download_button.assert_called()

    @patch('app.st')
    def test_export_data_json(self, mock_st):
        """Test JSON export functionality"""
        # Create test data
        test_data = pd.DataFrame({
            'title': ['Test Story'],
            'score': [100]
        })
        app.st.session_state.data = test_data

        app.export_data("JSON")

        # Verify download button was created
        mock_st.download_button.assert_called()

    @patch('app.st')
    def test_export_data_no_data(self, mock_st):
        """Test export functionality with no data"""
        app.st.session_state.data = None

        app.export_data("CSV")

        # Verify warning message
        mock_st.warning.assert_called()

    def test_check_auto_refresh_disabled(self):
        """Test auto-refresh when disabled"""
        app.st.session_state.auto_refresh = False
        app.st.session_state.last_refresh = datetime.now()

        # Should not trigger refresh
        app.check_auto_refresh()

    @patch('app.refresh_data')
    def test_check_auto_refresh_enabled_not_triggered(self, mock_refresh):
        """Test auto-refresh when enabled but not time yet"""
        app.st.session_state.auto_refresh = True
        app.st.session_state.last_refresh = datetime.now()

        # Should not trigger refresh (not enough time passed)
        app.check_auto_refresh()
        mock_refresh.assert_not_called()

    @patch('app.refresh_data')
    def test_check_auto_refresh_enabled_triggered(self, mock_refresh):
        """Test auto-refresh when enabled and time passed"""
        app.st.session_state.auto_refresh = True
        # Set last refresh to more than refresh interval ago
        old_time = datetime.now() - pd.Timedelta(seconds=DEFAULT_SETTINGS['refresh_interval'] + 60)
        app.st.session_state.last_refresh = old_time

        # Should trigger refresh
        app.check_auto_refresh()
        mock_refresh.assert_called_once()


class TestDashboardMain(unittest.TestCase):
    """Test dashboard main function"""

    @patch('app.create_data_table')
    @patch('app.create_charts_row')
    @patch('app.create_metrics_row')
    @patch('app.create_sidebar')
    @patch('app.create_header')
    @patch('app.initialize_session_state')
    @patch('app.check_auto_refresh')
    @patch('app.st')
    def test_main_with_data(self, mock_st, mock_auto_refresh, mock_init, mock_header,
                           mock_sidebar, mock_metrics, mock_charts, mock_table):
        """Test main function with data present"""
        # Set up session state with data
        app.st.session_state.data = pd.DataFrame({'title': ['Test']})

        app.main()

        # Verify all components were called
        mock_init.assert_called_once()
        mock_header.assert_called_once()
        mock_sidebar.assert_called_once()
        mock_auto_refresh.assert_called_once()
        mock_metrics.assert_called_once()
        mock_charts.assert_called_once()

    @patch('app.create_data_table')
    @patch('app.create_charts_row')
    @patch('app.create_metrics_row')
    @patch('app.create_sidebar')
    @patch('app.create_header')
    @patch('app.initialize_session_state')
    @patch('app.check_auto_refresh')
    @patch('app.refresh_data')
    @patch('app.st')
    def test_main_without_data(self, mock_st, mock_refresh, mock_auto_refresh, mock_init,
                              mock_header, mock_sidebar, mock_metrics, mock_charts, mock_table):
        """Test main function without data"""
        # Set up session state without data and no last refresh
        app.st.session_state.data = None
        app.st.session_state.last_refresh = None

        app.main()

        # Verify welcome message and refresh were triggered
        mock_st.markdown.assert_called()
        mock_refresh.assert_called_once()


class TestDashboardIntegration(unittest.TestCase):
    """Integration tests for dashboard functionality"""

    @patch('app.get_topics')
    @patch('app.analyze_sentiment')
    @patch('app.fetch_hn_data')
    def test_data_flow_integration(self, mock_fetch, mock_analyze, mock_topics):
        """Test complete data flow integration"""
        # Create realistic test data
        test_data = pd.DataFrame({
            'title': [
                'New AI technology breakthrough announced',
                'Major security vulnerability discovered in popular software',
                'Cloud computing reaches new milestone',
                'Tech company announces record earnings'
            ],
            'score': [500, 300, 200, 400],
            'descendants': [100, 50, 30, 80],
            'time': [datetime.now()] * 4,
            'url': ['http://example.com'] * 4
        })

        mock_fetch.return_value = test_data
        mock_analyze.return_value = test_data.copy()
        mock_topics.return_value = test_data.copy()

        # Test complete refresh flow
        app.refresh_data()

        # Verify data pipeline was called correctly
        mock_fetch.assert_called_once()
        mock_analyze.assert_called_once()
        mock_topics.assert_called_once()

        # Verify session state was updated
        self.assertIsNotNone(app.st.session_state.data)
        self.assertIsNotNone(app.st.session_state.last_refresh)


if __name__ == '__main__':
    unittest.main()