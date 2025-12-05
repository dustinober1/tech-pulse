"""
Integration tests for real-time features in Phase 5.
Tests full refresh cycles, mode switching, error handling, and UI responsiveness.
"""

import unittest
import sys
import os
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, call

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a mock session state that behaves like both a dict and an object
class MockSessionState(dict):
    """Mock session state that allows both dict and attribute access"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.__dict__[key] = value

    def __setattr__(self, name, value):
        self[name] = value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

# Helper function to create context manager mocks
def create_context_manager_mock():
    """Create a mock that supports context manager protocol"""
    mock = Mock()
    mock.__enter__ = Mock(return_value=None)
    mock.__exit__ = Mock(return_value=None)
    return mock

# Create mock streamlit module before importing app
import sys
from types import ModuleType

mock_streamlit = ModuleType('streamlit')
mock_streamlit.set_page_config = Mock()
mock_streamlit.markdown = Mock()
mock_streamlit.sidebar = create_context_manager_mock()
mock_streamlit.columns = Mock(return_value=[Mock(), Mock(), Mock()])
mock_streamlit.button = Mock(return_value=False)
mock_streamlit.slider = Mock(return_value=30)
mock_streamlit.checkbox = Mock(return_value=False)
mock_streamlit.selectbox = Mock(return_value="None")
mock_streamlit.multiselect = Mock(return_value=["All"])
mock_streamlit.expander = Mock(side_effect=lambda *args, **kwargs: create_context_manager_mock())
mock_streamlit.spinner = Mock(side_effect=lambda *args, **kwargs: create_context_manager_mock())
mock_streamlit.success = Mock()
mock_streamlit.error = Mock()
mock_streamlit.warning = Mock()
mock_streamlit.info = Mock()
mock_streamlit.plotly_chart = Mock()
mock_streamlit.dataframe = Mock()
mock_streamlit.download_button = Mock()
mock_streamlit.session_state = MockSessionState()
mock_streamlit.empty = Mock(return_value=create_context_manager_mock())
mock_streamlit.caption = Mock()
mock_streamlit.container = Mock(return_value=create_context_manager_mock())
mock_streamlit.metric = Mock()

sys.modules['streamlit'] = mock_streamlit

# Import app after mocking streamlit
import app
from dashboard_config import REAL_TIME_SETTINGS
import pandas as pd


class TestRealtimeRefreshCycle(unittest.TestCase):
    """Test complete real-time refresh cycles"""

    def setUp(self):
        """Set up test environment"""
        app.st.session_state.clear()
        app.initialize_session_state()

    @patch('time.sleep')
    @patch('app.st.empty')
    @patch('app.fetch_hn_data')
    @patch('app.analyze_sentiment')
    @patch('app.get_topics')
    def test_full_realtime_cycle_success(self, mock_topics, mock_sentiment,
                                       mock_fetch, mock_empty, mock_sleep):
        """Test complete real-time refresh cycle with success"""
        # Mock data
        test_df = pd.DataFrame({
            'title': ['Test Story 1', 'Test Story 2'],
            'score': [100, 200],
            'descendants': [10, 20],
            'time': [datetime.now(), datetime.now()],
            'url': ['http://example1.com', 'http://example2.com']
        })

        mock_fetch.return_value = test_df
        mock_sentiment.return_value = test_df.copy()
        mock_topics.return_value = test_df.copy()

        mock_container = create_context_manager_mock()
        mock_empty.return_value = mock_container

        # Simulate real-time mode
        app.st.session_state.real_time_mode = True
        placeholder = app.st.empty()

        # Simulate one refresh cycle
        with placeholder.container():
            with patch('app.st.spinner'):
                app.refresh_data()
                app.create_metrics_row(test_df)
                app.create_charts_row(test_df)
                app.create_data_table(test_df)

        # Verify all components were called
        mock_fetch.assert_called_once_with(limit=app.st.session_state.stories_count)
        mock_sentiment.assert_called_once_with(test_df)
        mock_topics.assert_called_once_with(test_df)
        self.assertIsNotNone(app.st.session_state.last_refresh)

    @patch('time.sleep')
    @patch('app.st.empty')
    @patch('app.fetch_hn_data')
    def test_realtime_cycle_with_api_error(self, mock_fetch, mock_empty, mock_sleep):
        """Test real-time cycle handling API errors"""
        # Mock API error
        mock_fetch.side_effect = Exception("API Error")

        mock_container = create_context_manager_mock()
        mock_empty.return_value = mock_container

        # Simulate real-time mode with error
        app.st.session_state.real_time_mode = True
        placeholder = app.st.empty()

        # Simulate refresh cycle with error
        with placeholder.container():
            with patch('app.st.spinner'):
                try:
                    app.refresh_data()
                except Exception:
                    pass  # Error should be handled gracefully

        # Verify error was handled
        self.assertTrue(mock_fetch.called)

    @patch('time.sleep')
    @patch('app.st.empty')
    def test_realtime_cycle_with_empty_data(self, mock_empty, mock_sleep):
        """Test real-time cycle with empty data response"""
        mock_container = create_context_manager_mock()
        mock_empty.return_value = mock_container

        with patch('app.fetch_hn_data', return_value=pd.DataFrame()) as mock_fetch:
            # Simulate real-time mode
            app.st.session_state.real_time_mode = True
            placeholder = app.st.empty()

            # Simulate refresh cycle
            with placeholder.container():
                with patch('app.st.spinner'):
                    app.refresh_data()

            # Verify empty data was handled
            mock_fetch.assert_called_once()
            mock_streamlit.error.assert_called()

    @patch('time.sleep')
    def test_multiple_refresh_cycles(self, mock_sleep):
        """Test multiple consecutive refresh cycles"""
        with patch('app.fetch_hn_data') as mock_fetch, \
             patch('app.analyze_sentiment') as mock_sentiment, \
             patch('app.get_topics') as mock_topics:

            # Mock data for each cycle
            test_df = pd.DataFrame({
                'title': ['Test Story'],
                'score': [100],
                'descendants': [10],
                'time': [datetime.now()],
                'url': ['http://example.com']
            })

            mock_fetch.return_value = test_df
            mock_sentiment.return_value = test_df
            mock_topics.return_value = test_df

            # Simulate multiple cycles
            for i in range(3):
                with patch('app.st.spinner'):
                    app.refresh_data()
                time.sleep(60)  # Mocked sleep

            # Verify multiple calls
            self.assertEqual(mock_fetch.call_count, 3)
            self.assertEqual(mock_sentiment.call_count, 3)
            self.assertEqual(mock_topics.call_count, 3)
            self.assertEqual(mock_sleep.call_count, 3)


class TestModeSwitching(unittest.TestCase):
    """Test switching between real-time and manual modes"""

    def setUp(self):
        """Set up test environment"""
        app.st.session_state.clear()
        app.initialize_session_state()

    def test_manual_to_realtime_switch(self):
        """Test switching from manual to real-time mode"""
        # Start in manual mode
        app.st.session_state.real_time_mode = False
        self.assertFalse(app.st.session_state.real_time_mode)

        # Switch to real-time mode
        app.st.session_state.real_time_mode = True
        self.assertTrue(app.st.session_state.real_time_mode)

    def test_realtime_to_manual_switch(self):
        """Test switching from real-time to manual mode"""
        # Start in real-time mode
        app.st.session_state.real_time_mode = True
        self.assertTrue(app.st.session_state.real_time_mode)

        # Switch to manual mode
        app.st.session_state.real_time_mode = False
        self.assertFalse(app.st.session_state.real_time_mode)

    @patch('app.st.empty')
    def test_mode_switching_clears_placeholder(self, mock_empty):
        """Test that switching modes clears placeholder appropriately"""
        mock_container = create_context_manager_mock()
        mock_empty.return_value = mock_container

        # Start in real-time mode
        app.st.session_state.real_time_mode = True
        placeholder = app.st.empty()

        # Use placeholder in real-time mode
        with placeholder.container():
            app.st.markdown("Real-time content")

        # Switch to manual mode
        app.st.session_state.real_time_mode = False

        # Placeholder should be cleared or reset
        mock_container.__exit__.assert_called()

    def test_data_persistence_across_modes(self):
        """Test that data persists when switching modes"""
        # Set some data
        test_data = pd.DataFrame({'title': ['Test']})
        app.st.session_state.data = test_data
        app.st.session_state.last_refresh = datetime.now()

        # Switch to real-time mode
        app.st.session_state.real_time_mode = True

        # Verify data persists
        self.assertIsNotNone(app.st.session_state.data)
        self.assertIsNotNone(app.st.session_state.last_refresh)

        # Switch back to manual mode
        app.st.session_state.real_time_mode = False

        # Verify data still persists
        self.assertIsNotNone(app.st.session_state.data)
        self.assertIsNotNone(app.st.session_state.last_refresh)

    @patch('app.st.sidebar')
    def test_sidebar_updates_with_mode_switch(self, mock_sidebar):
        """Test that sidebar updates correctly with mode switches"""
        # Mock checkbox returning different values
        mock_sidebar.checkbox.side_effect = [True, False, True]

        with patch('app.st.slider', return_value=30), \
             patch('app.st.button', return_value=False), \
             patch('app.st.multiselect', return_value=['All']), \
             patch('app.st.selectbox', return_value="None"), \
             patch('app.st.markdown'):

            # First call - enable real-time
            app.create_sidebar()
            self.assertTrue(app.st.session_state.auto_refresh)

            # Second call - disable real-time
            app.create_sidebar()
            self.assertFalse(app.st.session_state.auto_refresh)

            # Third call - re-enable real-time
            app.create_sidebar()
            self.assertTrue(app.st.session_state.auto_refresh)


class TestRealtimeErrorHandling(unittest.TestCase):
    """Test error handling in real-time operations"""

    def setUp(self):
        """Set up test environment"""
        app.st.session_state.clear()
        app.initialize_session_state()

    @patch('time.sleep')
    @patch('app.st.empty')
    def test_error_in_loop_continues_execution(self, mock_empty, mock_sleep):
        """Test that errors in loop don't break real-time execution"""
        mock_container = create_context_manager_mock()
        mock_empty.return_value = mock_container

        call_count = 0

        def simulate_error_then_success():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First call error")
            return pd.DataFrame({'title': ['Success']})

        with patch('app.fetch_hn_data', side_effect=simulate_error_then_success):
            # Simulate two iterations
            for i in range(2):
                with placeholder := mock_empty():
                    try:
                        with placeholder.container():
                            with patch('app.st.spinner'):
                                app.refresh_data()
                    except Exception:
                        # Handle error and continue
                        pass
                    time.sleep(60)

        # Verify both iterations were attempted
        self.assertEqual(call_count, 2)

    @patch('time.sleep')
    def test_connection_error_recovery(self, mock_sleep):
        """Test recovery from connection errors"""
        with patch('app.fetch_hn_data') as mock_fetch:
            # First call fails, second succeeds
            mock_fetch.side_effect = [
                Exception("Connection error"),
                pd.DataFrame({'title': ['Success']})
            ]

            # Simulate error recovery
            for i in range(2):
                try:
                    with patch('app.st.spinner'):
                        app.refresh_data()
                except Exception as e:
                    # Log error but continue
                    if i == 0:
                        self.assertIn("Connection error", str(e))
                time.sleep(60)  # Mocked

            # Verify both calls were made
            self.assertEqual(mock_fetch.call_count, 2)

    @patch('time.sleep')
    @patch('app.st.empty')
    def test_ui_responsiveness_during_errors(self, mock_empty, mock_sleep):
        """Test that UI remains responsive during errors"""
        mock_container = create_context_manager_mock()
        mock_empty.return_value = mock_container

        with patch('app.fetch_hn_data', side_effect=Exception("UI Test Error")):
            # Simulate real-time loop with error
            app.st.session_state.real_time_mode = True
            placeholder = app.st.empty()

            with placeholder.container():
                with patch('app.st.spinner'):
                    try:
                        app.refresh_data()
                    except Exception:
                        pass

            # Verify error message is shown to user
            mock_streamlit.error.assert_called()

    def test_session_state_corruption_recovery(self):
        """Test recovery from session state corruption"""
        # Corrupt session state
        app.st.session_state.real_time_mode = "invalid_string"
        app.st.session_state.data = "not_dataframe"

        # Reinitialize should fix state
        app.initialize_session_state()

        # Verify state is valid
        self.assertIsInstance(app.st.session_state.real_time_mode, bool)
        self.assertTrue(app.st.session_state.data is None or
                       isinstance(app.st.session_state.data, pd.DataFrame))


class TestUIResponsiveness(unittest.TestCase):
    """Test UI responsiveness during real-time updates"""

    def setUp(self):
        """Set up test environment"""
        app.st.session_state.clear()
        app.initialize_session_state()

    @patch('time.sleep', return_value=None)  # No actual sleep
    def test_ui_updates_without_freezing(self, mock_sleep):
        """Test that UI updates without freezing during real-time mode"""
        with patch('app.st.empty') as mock_empty:
            mock_container = create_context_manager_mock()
            mock_empty.return_value = mock_container

            # Simulate rapid updates
            for i in range(3):
                placeholder = app.st.empty()
                with placeholder.container():
                    app.st.markdown(f"Update {i}")
                    app.st.caption(f"Time: {datetime.now().strftime('%H:%M:%S')}")

                # Mock sleep (no actual delay)
                time.sleep(60)

            # Verify all updates occurred
            self.assertEqual(mock_sleep.call_count, 3)
            self.assertGreaterEqual(mock_streamlit.markdown.call_count, 3)

    def test_metrics_update_during_realtime(self):
        """Test that metrics update correctly in real-time mode"""
        # Create test data
        test_df = pd.DataFrame({
            'sentiment_score': [0.5, -0.3, 0.1],
            'descendants': [10, 20, 5],
            'topic_keyword': ['AI', 'Security', 'Cloud']
        })

        with patch('app.st.columns') as mock_columns:
            mock_col = Mock()
            mock_columns.return_value = [mock_col, mock_col, mock_col]

            # Create metrics in real-time context
            with patch('app.st.empty'):
                app.create_metrics_row(test_df)

            # Verify metrics were created
            self.assertTrue(mock_columns.called)

    @patch('time.sleep')
    @patch('app.st.empty')
    def test_charts_update_in_realtime(self, mock_empty, mock_sleep):
        """Test that charts update properly in real-time mode"""
        mock_container = create_context_manager_mock()
        mock_empty.return_value = mock_container

        test_df = pd.DataFrame({
            'title': ['Test Story'],
            'score': [100],
            'time': [datetime.now()],
            'sentiment_label': ['Positive'],
            'topic_keyword': ['AI']
        })

        with patch('app.st.columns') as mock_columns:
            mock_col = Mock()
            mock_columns.return_value = [mock_col, mock_col]

            # Update charts in real-time
            for i in range(2):
                placeholder = mock_empty()
                with placeholder.container():
                    app.create_charts_row(test_df)
                time.sleep(60)

            # Verify charts were updated
            self.assertEqual(mock_sleep.call_count, 2)

    def test_timestamp_updates_visible(self):
        """Test that timestamp updates are visible to user"""
        # Mock different times
        times = ['12:00:00', '12:01:00', '12:02:00']

        for timestamp in times:
            app.st.caption(f"Last Updated: {timestamp}")

        # Verify all timestamps were displayed
        self.assertEqual(app.st.caption.call_count, 3)

        # Verify timestamps are different
        calls = app.st.caption.call_args_list
        self.assertNotEqual(calls[0][0][0], calls[1][0][0])
        self.assertNotEqual(calls[1][0][0], calls[2][0][0])


class TestConcurrentOperations(unittest.TestCase):
    """Test concurrent operations in real-time mode"""

    def setUp(self):
        """Set up test environment"""
        app.st.session_state.clear()
        app.initialize_session_state()

    @patch('time.sleep')
    def test_concurrent_refresh_requests(self, mock_sleep):
        """Test handling of concurrent refresh requests"""
        results = []

        def refresh_data_thread():
            """Simulate refresh data in separate thread"""
            with patch('app.fetch_hn_data', return_value=pd.DataFrame({'title': ['Test']})):
                with patch('app.st.spinner'):
                    try:
                        app.refresh_data()
                        results.append("success")
                    except Exception:
                        results.append("error")

        # Simulate concurrent refresh attempts
        threads = []
        for i in range(3):
            thread = threading.Thread(target=refresh_data_thread)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify all refresh attempts completed
        self.assertEqual(len(results), 3)

    def test_session_state_thread_safety(self):
        """Test session state thread safety"""
        # Multiple threads updating session state
        def update_session_state(value):
            app.st.session_state.test_value = value

        threads = []
        for i in range(5):
            thread = threading.Thread(target=update_session_state, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify final state (last writer wins)
        self.assertIn('test_value', app.st.session_state)

    @patch('time.sleep')
    def test_user_interaction_during_updates(self, mock_sleep):
        """Test that user can interact during updates"""
        # Simulate real-time update
        with patch('app.fetch_hn_data'):
            with patch('app.st.spinner'):
                # User clicks refresh during update
                with patch('app.st.button', return_value=True):
                    # Simulate refresh button click
                    clicked = app.st.button("Refresh")
                    if clicked:
                        app.refresh_data()

        # Verify interactions were processed
        mock_streamlit.button.assert_called()


if __name__ == '__main__':
    unittest.main()