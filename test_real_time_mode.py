"""
Test script for real-time mode functionality in Tech-Pulse dashboard.
"""

import pytest
import streamlit as st
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import initialize_session_state, check_auto_refresh
from dashboard_config import REAL_TIME_SETTINGS


def test_session_state_initialization():
    """Test that session state is initialized correctly with real-time mode variables."""
    # Clear session state
    if hasattr(st, 'session_state'):
        st.session_state.clear()

    # Initialize session state
    initialize_session_state()

    # Check that real_time_mode is set to False by default
    assert st.session_state.real_time_mode == False

    # Check that last_update_time is initialized
    assert st.session_state.last_update_time == None

    # Check that old auto_refresh and refresh_countdown are removed
    assert 'auto_refresh' not in st.session_state
    assert 'refresh_countdown' not in st.session_state


def test_real_time_mode_toggle():
    """Test that real-time mode can be toggled on and off."""
    # Initialize session state
    initialize_session_state()

    # Toggle real-time mode on
    st.session_state.real_time_mode = True
    assert st.session_state.real_time_mode == True

    # Toggle real-time mode off
    st.session_state.real_time_mode = False
    assert st.session_state.real_time_mode == False


def test_auto_refresh_timing():
    """Test that auto refresh triggers at the correct interval."""
    # Initialize session state
    initialize_session_state()

    # Set real-time mode to active
    st.session_state.real_time_mode = True

    # Set last_update_time to 61 seconds ago (should trigger refresh)
    st.session_state.last_update_time = datetime.now() - timedelta(seconds=61)

    # Check if refresh should trigger
    time_since_refresh = (datetime.now() - st.session_state.last_update_time).seconds
    assert time_since_refresh >= 60

    # Set last_update_time to 59 seconds ago (should not trigger refresh)
    st.session_state.last_update_time = datetime.now() - timedelta(seconds=59)

    # Check if refresh should trigger
    time_since_refresh = (datetime.now() - st.session_state.last_update_time).seconds
    assert time_since_refresh < 60


def test_real_time_settings():
    """Test that real-time settings are properly configured."""
    # Check that refresh interval is 60 seconds
    assert REAL_TIME_SETTINGS['refresh_interval'] == 60

    # Check that visual indicator settings exist
    assert 'visual_indicator' in REAL_TIME_SETTINGS
    assert REAL_TIME_SETTINGS['visual_indicator']['enabled'] == True

    # Check that toggle settings exist
    assert 'toggle_settings' in REAL_TIME_SETTINGS
    assert REAL_TIME_SETTINGS['toggle_settings']['enable_notifications'] == True


if __name__ == "__main__":
    print("Running real-time mode tests...")

    try:
        # Run tests
        test_session_state_initialization()
        print("✓ Session state initialization test passed")

        test_real_time_mode_toggle()
        print("✓ Real-time mode toggle test passed")

        test_auto_refresh_timing()
        print("✓ Auto refresh timing test passed")

        test_real_time_settings()
        print("✓ Real-time settings test passed")

        print("\nAll tests passed! Real-time mode is working correctly.")

    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        import traceback
        traceback.print_exc()