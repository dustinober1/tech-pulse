"""
Basic test to verify semantic search UI components are properly integrated.
"""

import pytest
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_semantic_search_config_exists():
    """Test that semantic search configuration exists in dashboard_config."""
    from dashboard_config import SEMANTIC_SEARCH_SETTINGS, SEMANTIC_SEARCH_MESSAGES

    # Check that required settings exist
    assert 'model_name' in SEMANTIC_SEARCH_SETTINGS
    assert 'max_results' in SEMANTIC_SEARCH_SETTINGS
    assert 'similarity_threshold' in SEMANTIC_SEARCH_SETTINGS
    assert SEMANTIC_SEARCH_SETTINGS['model_name'] == 'all-MiniLM-L6-v2'

    # Check that required messages exist
    assert 'initializing' in SEMANTIC_SEARCH_MESSAGES
    assert 'searching' in SEMANTIC_SEARCH_MESSAGES
    assert 'no_results' in SEMANTIC_SEARCH_MESSAGES

    print("✅ Semantic search configuration verified")


def test_semantic_search_functions_importable():
    """Test that semantic search functions can be imported from app."""
    from app import (
        create_semantic_search_section,
        display_search_results,
        initialize_vector_db,
        perform_semantic_search
    )

    # Check that functions are callable
    assert callable(create_semantic_search_section)
    assert callable(display_search_results)
    assert callable(initialize_vector_db)
    assert callable(perform_semantic_search)

    print("✅ Semantic search functions importable")


def test_data_loader_has_vector_functions():
    """Test that data_loader has vector search functions."""
    from data_loader import setup_vector_db, semantic_search

    # Check that functions are callable
    assert callable(setup_vector_db)
    assert callable(semantic_search)

    print("✅ Data loader vector functions verified")


def test_app_imports_all_required():
    """Test that app.py imports all required semantic search components."""
    with open('app.py', 'r') as f:
        content = f.read()

    # Check imports
    assert 'from data_loader import' in content
    assert 'setup_vector_db' in content
    assert 'semantic_search' in content
    assert 'SEMANTIC_SEARCH_SETTINGS' in content
    assert 'SEMANTIC_SEARCH_MESSAGES' in content

    # Check function definitions
    assert 'def create_semantic_search_section' in content
    assert 'def display_search_results' in content
    assert 'def initialize_vector_db' in content
    assert 'def perform_semantic_search' in content

    # Check that semantic search is integrated
    assert 'create_semantic_search_section(' in content

    print("✅ App.py semantic search integration verified")


def test_session_state_includes_vector_db():
    """Test that initialize_session_state includes vector DB variables."""
    # We can't easily test session state without Streamlit running,
    # but we can check the code includes the right variables
    with open('app.py', 'r') as f:
        content = f.read()

    # Check that vector DB variables are initialized
    assert "'vector_collection' not in st.session_state:" in content
    assert "'vector_db_initialized' not in st.session_state:" in content
    assert "'search_results' not in st.session_state:" in content

    print("✅ Session state includes vector DB variables")


if __name__ == '__main__':
    # Run all tests
    test_semantic_search_config_exists()
    test_semantic_search_functions_importable()
    test_data_loader_has_vector_functions()
    test_app_imports_all_required()
    test_session_state_includes_vector_db()

    print("\n✅ All semantic search integration tests passed!")