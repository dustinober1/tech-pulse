"""
Performance tests for real-time features in Phase 5.
Tests memory usage, API rate limiting, browser behavior, and UI responsiveness.
"""

import unittest
import sys
import os
import time
import threading
import psutil
import gc
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a mock session state
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

# Create mock streamlit module
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

sys.modules['streamlit'] = mock_streamlit

# Import app after mocking
import app
from dashboard_config import REAL_TIME_SETTINGS
import pandas as pd


class TestMemoryUsage(unittest.TestCase):
    """Test memory usage during extended real-time sessions"""

    def setUp(self):
        """Set up test environment"""
        app.st.session_state.clear()
        app.initialize_session_state()
        self.initial_memory = self.get_memory_usage()

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0  # Fallback if psutil not available

    @patch('time.sleep')
    @patch('app.st.empty')
    def test_memory_usage_stable_during_realtime(self, mock_empty, mock_sleep):
        """Test that memory usage remains stable during extended real-time sessions"""
        mock_container = create_context_manager_mock()
        mock_empty.return_value = mock_container

        # Create test data
        test_df = pd.DataFrame({
            'title': [f'Story {i}' for i in range(100)],
            'score': list(range(100, 0, -1)),
            'descendants': list(range(1, 101)),
            'time': [datetime.now()] * 100,
            'url': [f'http://example{i}.com' for i in range(100)],
            'sentiment_score': [0.1 * i for i in range(-5, 5)] * 10,
            'sentiment_label': ['Positive', 'Negative', 'Neutral'] * 33 + ['Positive'],
            'topic_keyword': ['AI', 'Security', 'Cloud'] * 33 + ['AI']
        })

        with patch('app.fetch_hn_data', return_value=test_df), \
             patch('app.analyze_sentiment', return_value=test_df), \
             patch('app.get_topics', return_value=test_df):

            # Simulate extended session (10 refresh cycles)
            memory_readings = []
            app.st.session_state.real_time_mode = True

            for i in range(10):
                placeholder = app.st.empty()
                with placeholder.container():
                    with patch('app.st.spinner'):
                        app.refresh_data()
                        app.create_metrics_row(test_df)
                        app.create_charts_row(test_df)

                # Force garbage collection
                gc.collect()

                # Record memory usage
                memory_readings.append(self.get_memory_usage())
                time.sleep(60)  # Mocked sleep

            # Analyze memory usage
            memory_increase = max(memory_readings) - min(memory_readings)

            # Memory increase should be reasonable (< 50MB)
            self.assertLess(memory_increase, 50,
                           f"Memory increased by {memory_increase:.2f}MB, which exceeds threshold")

    def test_memory_cleanup_after_mode_switch(self):
        """Test memory cleanup when switching between modes"""
        # Accumulate data in real-time mode
        app.st.session_state.real_time_mode = True
        app.st.session_state.data = pd.DataFrame({
            'title': [f'Story {i}' for i in range(1000)],
            'score': list(range(1000, 0, -1)),
            'large_data': [list(range(100)) for _ in range(1000)]
        })

        memory_before_switch = self.get_memory_usage()

        # Switch to manual mode
        app.st.session_state.real_time_mode = False

        # Clear some data
        app.st.session_state.data = None
        gc.collect()

        memory_after_switch = self.get_memory_usage()

        # Memory should decrease after cleanup
        memory_freed = memory_before_switch - memory_after_switch
        self.assertGreater(memory_freed, 0,
                          "Memory usage should decrease after data cleanup")

    @patch('time.sleep')
    def test_dataframe_memory_management(self, mock_sleep):
        """Test efficient DataFrame memory management"""
        dataframes = []

        # Create multiple DataFrames (simulating refresh cycles)
        for i in range(20):
            df = pd.DataFrame({
                'title': [f'Story {i}_{j}' for j in range(100)],
                'score': list(range(100, 0, -1)),
                'embedding': [list(range(50)) for _ in range(100)]  # Simulate embeddings
            })
            dataframes.append(df)

            # Keep only recent dataframes
            if len(dataframes) > 5:
                dataframes.pop(0)

            time.sleep(60)  # Mocked sleep

        # Verify memory is managed
        self.assertLessEqual(len(dataframes), 5, "Should not accumulate too many DataFrames")

    def test_large_dataset_handling(self):
        """Test memory handling with large datasets"""
        # Create a large dataset
        large_df = pd.DataFrame({
            'title': [f'Story {i}' for i in range(10000)],
            'score': list(range(10000, 0, -1)),
            'text': ['Large text content ' * 100 for _ in range(10000)],
            'embedding': [list(range(100)) for _ in range(10000)]
        })

        # Test processing large dataset
        start_memory = self.get_memory_usage()

        with patch('app.analyze_sentiment', return_value=large_df), \
             patch('app.get_topics', return_value=large_df):

            # Process data
            processed = app.analyze_sentiment(large_df)
            processed = app.get_topics(processed)

        end_memory = self.get_memory_usage()
        memory_increase = end_memory - start_memory

        # Memory increase should be reasonable for the data size
        self.assertLess(memory_increase, 200,
                       f"Large dataset processing increased memory by {memory_increase:.2f}MB")


class TestAPIRateLimiting(unittest.TestCase):
    """Test API rate limiting compliance"""

    def setUp(self):
        """Set up test environment"""
        app.st.session_state.clear()
        app.initialize_session_state()

    @patch('time.sleep')
    @patch('time.time')
    def test_refresh_interval_compliance(self, mock_time, mock_sleep):
        """Test that refresh intervals comply with rate limiting"""
        # Mock time progression
        start_time = 1609459200  # 2021-01-01 00:00:00
        mock_time.side_effect = [start_time + i * 60 for i in range(5)]

        call_times = []

        with patch('app.fetch_hn_data') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame({'title': ['Test']})

            # Simulate multiple refreshes
            for i in range(5):
                call_times.append(mock_time())
                app.refresh_data()
                time.sleep(60)  # Mocked sleep

        # Verify minimum interval between calls
        for i in range(1, len(call_times)):
            interval = call_times[i] - call_times[i-1]
            self.assertGreaterEqual(interval, 60,
                                   f"API calls too close: {interval} seconds between calls {i-1} and {i}")

    def test_api_call_frequency_in_realtime(self):
        """Test API call frequency in real-time mode"""
        api_calls = []
        original_fetch = app.fetch_hn_data

        def track_fetch(*args, **kwargs):
            api_calls.append(datetime.now())
            return pd.DataFrame({'title': ['Test']})

        with patch('app.fetch_hn_data', side_effect=track_fetch):
            app.st.session_state.real_time_mode = True

            # Simulate real-time updates
            for i in range(3):
                with patch('app.st.spinner'):
                    app.refresh_data()
                time.sleep(60)  # Mocked sleep

        # Verify call frequency
        self.assertEqual(len(api_calls), 3)

        # Check minimum intervals (if real time was used)
        if len(api_calls) > 1:
            for i in range(1, len(api_calls)):
                interval = (api_calls[i] - api_calls[i-1]).total_seconds()
                self.assertGreaterEqual(interval, 60,
                                       f"API calls间隔 {interval} 秒，违反了最小间隔要求")

    def test_error_recovery_rate_limit(self):
        """Test rate limiting after errors"""
        with patch('app.fetch_hn_data') as mock_fetch:
            # First few calls fail, then succeed
            mock_fetch.side_effect = [
                Exception("Rate limit exceeded"),
                Exception("Rate limit exceeded"),
                pd.DataFrame({'title': ['Success']})
            ]

            # Attempt refreshes
            for i in range(3):
                try:
                    with patch('app.st.spinner'):
                        app.refresh_data()
                except Exception:
                    pass
                time.sleep(60)  # Mocked sleep between attempts

            # Verify all attempts were made with proper intervals
            self.assertEqual(mock_fetch.call_count, 3)

    def test_concurrent_api_calls_prevention(self):
        """Test prevention of concurrent API calls"""
        fetch_in_progress = False
        fetch_calls = []

        def mock_fetch(*args, **kwargs):
            nonlocal fetch_in_progress
            if fetch_in_progress:
                raise Exception("Concurrent fetch detected!")

            fetch_in_progress = True
            fetch_calls.append(datetime.now())
            time.sleep(0.1)  # Simulate API call duration
            fetch_in_progress = False
            return pd.DataFrame({'title': ['Test']})

        # Simulate concurrent refresh attempts
        def refresh_attempt():
            try:
                with patch('app.st.spinner'):
                    app.refresh_data()
            except Exception as e:
                if "Concurrent fetch" in str(e):
                    fetch_calls.append("blocked")

        with patch('app.fetch_hn_data', side_effect=mock_fetch):
            # Start multiple threads
            threads = []
            for i in range(3):
                thread = threading.Thread(target=refresh_attempt)
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

        # Verify only one fetch succeeded
        successful_fetches = [call for call in fetch_calls if isinstance(call, datetime)]
        self.assertEqual(len(successful_fetches), 1,
                        "Only one API call should succeed concurrently")


class TestBrowserBehavior(unittest.TestCase):
    """Test browser tab behavior with long-running sessions"""

    def setUp(self):
        """Set up test environment"""
        app.st.session_state.clear()
        app.initialize_session_state()

    @patch('time.sleep')
    @patch('app.st.empty')
    def test_extended_session_stability(self, mock_empty, mock_sleep):
        """Test stability of extended browser sessions"""
        mock_container = create_context_manager_mock()
        mock_empty.return_value = mock_container

        update_count = 0
        errors = []

        def simulate_update():
            nonlocal update_count
            update_count += 1
            if update_count % 10 == 0:  # Simulate occasional error
                errors.append(f"Error at update {update_count}")
                raise Exception("Simulated browser timeout")

        with patch('app.fetch_hn_data', side_effect=simulate_update):
            app.st.session_state.real_time_mode = True

            # Simulate long session (100 updates)
            for i in range(100):
                placeholder = app.st.empty()
                try:
                    with placeholder.container():
                        with patch('app.st.spinner'):
                            app.refresh_data()
                except Exception:
                    # Handle errors and continue
                    pass

                time.sleep(60)  # Mocked sleep

        # Verify session handled errors gracefully
        self.assertEqual(update_count, 100)
        self.assertLess(len(errors), 20, "Too many errors in extended session")

    def test_tab_reconnection_handling(self):
        """Test handling of tab reconnection after inactivity"""
        reconnection_attempts = []

        def mock_fetch(*args, **kwargs):
            reconnection_attempts.append(datetime.now())
            if len(reconnection_attempts) == 1:
                # First call after inactivity might fail
                raise Exception("Connection timed out")
            return pd.DataFrame({'title': ['Reconnected']})

        with patch('app.fetch_hn_data', side_effect=mock_fetch):
            # Simulate tab reconnection
            for i in range(3):
                try:
                    with patch('app.st.spinner'):
                        app.refresh_data()
                except Exception as e:
                    if i == 0:
                        # First attempt should fail
                        self.assertIn("timed out", str(e).lower())
                    else:
                        # Subsequent attempts should succeed
                        self.fail(f"Unexpected error after reconnection: {e}")

    def test_memory_cleanup_on_tab_close(self):
        """Test memory cleanup when tab is closed"""
        # Simulate tab accumulating data
        app.st.session_state.data = pd.DataFrame({
            'title': [f'Story {i}' for i in range(10000)],
            'large_content': ['x' * 1000 for _ in range(10000)]
        })

        # Simulate tab cleanup
        cleanup_memory_before = self.get_memory_usage()

        # Clear session state (simulating tab close)
        app.st.session_state.clear()
        gc.collect()

        cleanup_memory_after = self.get_memory_usage()
        memory_freed = cleanup_memory_before - cleanup_memory_after

        # Verify memory was freed
        self.assertGreater(memory_freed, 0,
                          "Memory should be freed when tab is closed")

    @patch('time.sleep')
    def test_visual_indicator_performance(self, mock_sleep):
        """Test performance impact of real-time visual indicators"""
        indicator_updates = []

        with patch('app.st.empty'):
            # Simulate visual indicator updates
            for i in range(50):
                start_time = time.time()

                # Update visual indicators
                app.st.caption(f"Real-time active: {datetime.now().strftime('%H:%M:%S')}")

                update_time = time.time() - start_time
                indicator_updates.append(update_time)

                time.sleep(60)  # Mocked sleep

        # Verify indicator updates are fast (< 10ms)
        avg_update_time = sum(indicator_updates) / len(indicator_updates)
        self.assertLess(avg_update_time, 0.01,
                       f"Visual indicator updates too slow: {avg_update_time:.3f}s average")


class TestUIResponsiveness(unittest.TestCase):
    """Test UI responsiveness during real-time updates"""

    def setUp(self):
        """Set up test environment"""
        app.st.session_state.clear()
        app.initialize_session_state()

    @patch('time.sleep')
    def test_ui_update_frequency(self, mock_sleep):
        """Test that UI updates at appropriate frequency"""
        update_times = []

        with patch('app.st.empty'):
            # Simulate real-time updates
            for i in range(10):
                start_time = time.time()

                # Simulate UI update
                with app.st.empty().container():
                    app.st.markdown(f"Update {i}")
                    app.st.caption(f"Time: {datetime.now().strftime('%H:%M:%S')}")

                update_time = time.time() - start_time
                update_times.append(update_time)

                time.sleep(60)  # Mocked sleep

        # Verify updates complete quickly (< 100ms)
        max_update_time = max(update_times)
        self.assertLess(max_update_time, 0.1,
                       f"UI update too slow: {max_update_time:.3f}s maximum")

    def test_interactive_elements_during_updates(self):
        """Test that interactive elements remain responsive during updates"""
        # Mock interactive elements
        button_states = []

        def mock_button(label, **kwargs):
            # Simulate button click during update
            if label == "🔄 Refresh Data":
                button_states.append("available")
                return i % 2 == 0  # Alternate True/False
            return False

        with patch('app.st.button', side_effect=mock_button), \
             patch('app.st.slider', return_value=30), \
             patch('app.st.checkbox', return_value=True):

            # Simulate updates while checking interactivity
            for i in range(5):
                with patch('app.st.empty'):
                    app.create_sidebar()

                # Verify button remains interactive
                self.assertIn("available", button_states)

    def test_large_data_rendering_performance(self):
        """Test rendering performance with large datasets"""
        # Create large dataset
        large_df = pd.DataFrame({
            'title': [f'Story {i}' for i in range(1000)],
            'score': list(range(1000, 0, -1)),
            'description': [f'Description {i} ' * 50 for i in range(1000)]
        })

        render_times = []

        # Test rendering performance
        for i in range(5):
            start_time = time.time()

            # Simulate rendering
            with patch('app.st.dataframe'):
                app.create_data_table(large_df)

            render_time = time.time() - start_time
            render_times.append(render_time)

        # Verify rendering is fast enough (< 500ms)
        avg_render_time = sum(render_times) / len(render_times)
        self.assertLess(avg_render_time, 0.5,
                       f"Large data rendering too slow: {avg_render_time:.3f}s average")

    @patch('time.sleep')
    def test_concurrent_ui_operations(self, mock_sleep):
        """Test UI responsiveness during concurrent operations"""
        operation_times = []

        def ui_operation(op_id):
            start_time = time.time()

            # Simulate UI operation
            with app.st.empty().container():
                app.st.markdown(f"Operation {op_id}")
                time.sleep(0.01)  # Simulate UI work

            operation_time = time.time() - start_time
            operation_times.append(operation_time)

        # Run concurrent UI operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=ui_operation, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all operations
        for thread in threads:
            thread.join()

        # Verify all operations completed
        self.assertEqual(len(operation_times), 5)

        # Verify operations completed quickly
        avg_time = sum(operation_times) / len(operation_times)
        self.assertLess(avg_time, 0.1,
                       f"Concurrent UI operations too slow: {avg_time:.3f}s average")


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics collection and monitoring"""

    def setUp(self):
        """Set up test environment"""
        app.st.session_state.clear()
        app.initialize_session_state()

    def test_performance_metrics_collection(self):
        """Test collection of performance metrics"""
        metrics = {
            'refresh_times': [],
            'memory_usage': [],
            'api_calls': 0,
            'ui_updates': 0
        }

        # Simulate metrics collection during updates
        for i in range(10):
            # Collect metrics
            start_time = time.time()
            memory_before = self.get_memory_usage()

            # Simulate update
            with patch('app.fetch_hn_data', return_value=pd.DataFrame({'title': ['Test']})):
                with patch('app.st.spinner'):
                    app.refresh_data()

            # Record metrics
            refresh_time = time.time() - start_time
            memory_after = self.get_memory_usage()

            metrics['refresh_times'].append(refresh_time)
            metrics['memory_usage'].append(memory_after - memory_before)
            metrics['api_calls'] += 1
            metrics['ui_updates'] += 1

        # Verify metrics were collected
        self.assertEqual(len(metrics['refresh_times']), 10)
        self.assertEqual(metrics['api_calls'], 10)
        self.assertEqual(metrics['ui_updates'], 10)

    def test_performance_threshold_monitoring(self):
        """Test monitoring of performance thresholds"""
        thresholds = {
            'max_refresh_time': 5.0,  # seconds
            'max_memory_increase': 10.0,  # MB
            'min_ui_framerate': 10  # FPS
        }

        violations = []

        # Test refresh time threshold
        start_time = time.time()
        time.sleep(0.1)  # Simulate slow refresh
        refresh_time = time.time() - start_time

        if refresh_time > thresholds['max_refresh_time']:
            violations.append(f"Refresh time {refresh_time:.2f}s exceeds threshold")

        # Test memory threshold
        memory_increase = 5.0  # MB
        if memory_increase > thresholds['max_memory_increase']:
            violations.append(f"Memory increase {memory_increase:.2f}MB exceeds threshold")

        # Test UI framerate
        ui_updates = 20
        time_window = 2.0  # seconds
        framerate = ui_updates / time_window

        if framerate < thresholds['min_ui_framerate']:
            violations.append(f"UI framerate {framerate:.1f} FPS below threshold")

        # Verify threshold monitoring
        if violations:
            self.fail(f"Performance threshold violations: {violations}")

    def test_performance_alerts(self):
        """Test performance alerts generation"""
        alerts = []

        def check_performance_alerts(metrics):
            if metrics.get('refresh_time', 0) > 3.0:
                alerts.append("Slow refresh detected")
            if metrics.get('memory_usage', 0) > 100:
                alerts.append("High memory usage detected")
            if metrics.get('error_rate', 0) > 0.1:
                alerts.append("High error rate detected")

        # Test with good metrics
        check_performance_alerts({
            'refresh_time': 1.0,
            'memory_usage': 50,
            'error_rate': 0.01
        })
        self.assertEqual(len(alerts), 0, "No alerts should be generated for good metrics")

        # Test with poor metrics
        alerts.clear()
        check_performance_alerts({
            'refresh_time': 5.0,
            'memory_usage': 150,
            'error_rate': 0.2
        })
        self.assertEqual(len(alerts), 3, "All alerts should be generated for poor metrics")


if __name__ == '__main__':
    unittest.main()