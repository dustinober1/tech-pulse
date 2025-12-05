"""
Test semantic search UI integration in the Streamlit dashboard.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import functions to test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from app import (
    initialize_session_state,
    create_semantic_search_section,
    initialize_vector_db,
    perform_semantic_search,
    display_search_results
)
from dashboard_config import SEMANTIC_SEARCH_SETTINGS, SEMANTIC_SEARCH_MESSAGES


class TestSemanticSearchIntegration:
    """Test semantic search UI integration."""

    @patch('app.st')
    def test_initialize_session_state_includes_vector_db(self, mock_st):
        """Test that session state initialization includes vector DB variables."""
        # Mock session state
        mock_st.session_state = {}

        # Call initialize_session_state
        initialize_session_state()

        # Check that vector DB variables are initialized
        assert 'vector_collection' in mock_st.session_state
        assert 'vector_db_initialized' in mock_st.session_state
        assert 'search_results' in mock_st.session_state
        assert mock_st.session_state['vector_collection'] is None
        assert mock_st.session_state['vector_db_initialized'] is False
        assert mock_st.session_state['search_results'] is None

    @patch('app.st')
    @patch('app.setup_vector_db')
    def test_initialize_vector_db_success(self, mock_setup_vector_db, mock_st):
        """Test successful vector DB initialization."""
        # Setup mocks
        mock_collection = Mock()
        mock_setup_vector_db.return_value = mock_collection
        mock_st.session_state = {
            'data': pd.DataFrame({
                'title': ['Test Story 1', 'Test Story 2'],
                'score': [100, 200]
            })
        }
        mock_st.spinner = Mock()
        mock_st.spinner.return_value.__enter__ = Mock(return_value=None)
        mock_st.spinner.return_value.__exit__ = Mock(return_value=None)
        mock_st.success = Mock()
        mock_st.error = Mock()
        mock_st.rerun = Mock()

        # Call initialize_vector_db
        initialize_vector_db()

        # Verify setup_vector_db was called
        mock_setup_vector_db.assert_called_once_with(mock_st.session_state['data'])

        # Verify session state was updated
        assert mock_st.session_state['vector_collection'] == mock_collection
        assert mock_st.session_state['vector_db_initialized'] is True

        # Verify success message and rerun
        mock_st.success.assert_called_once()
        mock_st.rerun.assert_called_once()

    @patch('app.st')
    @patch('app.semantic_search')
    def test_perform_semantic_search_success(self, mock_semantic_search, mock_st):
        """Test successful semantic search execution."""
        # Setup mocks
        mock_results = [
            {
                'title': 'Test Result',
                'metadata': {'score': 100, 'url': 'http://example.com'},
                'similarity_score': 0.85,
                'distance': 0.15
            }
        ]
        mock_semantic_search.return_value = mock_results

        mock_st.session_state = {
            'vector_collection': Mock()
        }
        mock_st.spinner = Mock()
        mock_st.spinner.return_value.__enter__ = Mock(return_value=None)
        mock_st.spinner.return_value.__exit__ = Mock(return_value=None)
        mock_st.success = Mock()
        mock_st.info = Mock()
        mock_st.error = Mock()

        # Call perform_semantic_search
        perform_semantic_search("test query")

        # Verify semantic_search was called with correct parameters
        mock_semantic_search.assert_called_once_with(
            collection=mock_st.session_state['vector_collection'],
            query="test query",
            max_results=SEMANTIC_SEARCH_SETTINGS['max_results'],
            similarity_threshold=SEMANTIC_SEARCH_SETTINGS['similarity_threshold']
        )

        # Verify session state was updated
        assert mock_st.session_state['search_results'] == mock_results

        # Verify success message
        mock_st.success.assert_called_once_with("Found 1 relevant stories for 'test query'")

    @patch('app.st')
    def test_display_search_results_empty(self, mock_st):
        """Test display_search_results with empty results."""
        # Setup mock
        mock_st.info = Mock()
        mock_st.markdown = Mock()
        mock_st.expander = Mock()

        # Call with empty results
        display_search_results([])

        # Verify info message
        mock_st.info.assert_called_once_with(SEMANTIC_SEARCH_MESSAGES['no_results'])

    @patch('app.st')
    def test_display_search_results_with_data(self, mock_st):
        """Test display_search_results with actual results."""
        # Setup mock data
        results = [
            {
                'title': 'Test Story 1',
                'metadata': {
                    'score': 100,
                    'descendants': 10,
                    'sentiment_label': 'Positive',
                    'topic_keyword': 'ai_ml',
                    'url': 'http://example.com',
                    'time': '2024-01-01T12:00:00',
                    'index': 0
                },
                'similarity_score': 0.85,
                'distance': 0.15,
                'rank': 1,
                'explanation': 'Similarity: 85.0%'
            }
        ]

        # Setup streamlit mocks
        mock_st.markdown = Mock()
        mock_expander = Mock()
        mock_expander.return_value.__enter__ = Mock(return_value=None)
        mock_expander.return_value.__exit__ = Mock(return_value=None)
        mock_st.expander = mock_expander
        mock_st.columns = Mock(return_value=[Mock(), Mock(), Mock()])
        mock_st.metric = Mock()
        mock_st.caption = Mock()
        mock_st.write = Mock()

        # Call display_search_results
        display_search_results(results)

        # Verify markdown with count
        mock_st.markdown.assert_called_with('#### Found 1 relevant stories')

        # Verify expander was called for each result
        assert mock_st.expander.call_count == 2  # One for result, one for details

    @patch('app.st')
    @patch('app.initialize_vector_db')
    def test_create_semantic_search_section_not_initialized(self, mock_initialize_vector_db, mock_st):
        """Test create_semantic_search_section when vector DB is not initialized."""
        # Setup mocks
        mock_st.session_state = {
            'vector_db_initialized': False,
            'vector_collection': None
        }
        mock_st.markdown = Mock()
        mock_st.columns = Mock(return_value=[Mock(), Mock()])
        mock_st.info = Mock()
        mock_st.button = Mock(return_value=False)

        # Create test data
        df = pd.DataFrame({'title': ['Test']})

        # Call create_semantic_search_section
        create_semantic_search_section(df)

        # Verify initialization prompt is shown
        mock_st.info.assert_called_once()
        mock_st.button.assert_called_once()

    @patch('app.st')
    def test_create_semantic_search_section_initialized(self, mock_st):
        """Test create_semantic_search_section when vector DB is initialized."""
        # Setup mocks
        mock_st.session_state = {
            'vector_db_initialized': True,
            'vector_collection': Mock(),
            'search_results': None,
            'last_search_query': None
        }
        mock_st.markdown = Mock()
        mock_st.columns = Mock(return_value=[Mock(), Mock()])
        mock_st.text_input = Mock(return_value="")
        mock_st.button = Mock(return_value=False)
        mock_st.warning = Mock()
        mock_st.write = Mock()

        # Create test data
        df = pd.DataFrame({'title': ['Test']})

        # Call create_semantic_search_section
        create_semantic_search_section(df)

        # Verify search interface is shown
        assert mock_st.text_input.called or mock_st.markdown.called


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])