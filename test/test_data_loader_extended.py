"""Extended tests for data_loader.py to achieve 100% coverage."""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import nltk
from datetime import datetime

from data_loader import (
    fetch_story_ids,
    fetch_story_details,
    extract_story_data,
    process_stories_to_dataframe,
    analyze_sentiment,
    setup_vector_db,
    get_topics,
    semantic_search,
    fetch_hn_data
)


class TestDataLoaderExtended:
    """Extended tests for data_loader functions."""

    @patch('nltk.data.find')
    @patch('nltk.download')
    def test_nltk_download_on_missing(self, mock_download, mock_find):
        """Test NLTK download when lexicon is missing."""
        mock_find.side_effect = LookupError("NLTK resource not found")

        # Re-import to trigger the download check
        import importlib
        import data_loader
        importlib.reload(data_loader)

        mock_download.assert_called_once_with('vader_lexicon')

    def test_fetch_story_details_general_exception(self):
        """Test fetch_story_details with non-request exceptions."""
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = fetch_story_details(123)
            assert result is None

    def test_extract_story_data_with_byline_field(self):
        """Test extract_story_data with byline field."""
        item_data = {
            'id': 123,
            'title': 'Test Story',
            'by': 'author',
            'byline': 'Written by Author',  # Additional field
            'url': 'https://example.com',
            'score': 100,
            'time': 1609459200,
            'descendants': 50
        }

        result = extract_story_data(item_data)
        assert result is not None
        assert result['title'] == 'Test Story'
        assert result['by'] == 'author'

    @patch('data_loader.get_vector_cache_manager')
    @patch('data_loader.get_vector_engine')
    def test_setup_vector_db_with_empty_dataframe(self, mock_get_engine, mock_get_cache):
        """Test setup_vector_db with empty DataFrame."""
        df = pd.DataFrame()

        result = setup_vector_db(df)
        assert result is None
        mock_get_cache.assert_not_called()
        mock_get_engine.assert_not_called()

    @patch('data_loader.get_vector_cache_manager')
    def test_setup_vector_db_missing_title_column(self, mock_get_cache):
        """Test setup_vector_db with DataFrame missing title column."""
        df = pd.DataFrame({'id': [1, 2], 'text': ['a', 'b']})

        result = setup_vector_db(df)
        assert result is None

    @patch('data_loader.get_vector_cache_manager')
    @patch('data_loader.get_vector_engine')
    def test_setup_vector_db_with_valid_cache(self, mock_get_engine, mock_get_cache):
        """Test setup_vector_db when valid cache exists."""
        df = pd.DataFrame({
            'title': ['Test Story 1', 'Test Story 2'],
            'id': [1, 2]
        })

        # Mock cache manager
        mock_cache_manager = MagicMock()
        mock_cache_manager.should_rebuild_collection.return_value = False
        mock_cache_manager.get_cached_collection.return_value = "cached_collection"
        mock_cache_manager.get_cache_info.return_value = {'version': '1.0'}
        mock_get_cache.return_value = mock_cache_manager

        result = setup_vector_db(df)
        assert result == "cached_collection"

    @patch('data_loader.get_vector_cache_manager')
    @patch('data_loader.get_vector_engine')
    @patch('data_loader.chromadb')
    def test_setup_vector_db_force_rebuild(self, mock_chromadb, mock_get_engine, mock_get_cache):
        """Test setup_vector_db with force_rebuild=True."""
        df = pd.DataFrame({
            'title': ['Test Story 1', 'Test Story 2'],
            'id': [1, 2]
        })

        # Mock cache manager
        mock_cache_manager = MagicMock()
        mock_get_cache.return_value = mock_cache_manager

        # Mock vector engine and chromadb
        mock_engine = MagicMock()
        mock_engine.setup_collection.return_value = "new_collection"
        mock_get_engine.return_value = mock_engine

        result = setup_vector_db(df, force_rebuild=True)
        assert result == "new_collection"
        mock_cache_manager.cache_collection.assert_called_once()

    @patch('data_loader.get_vector_engine')
    def test_semantic_search_with_empty_query(self, mock_get_engine):
        """Test semantic_search with empty query."""
        collection = MagicMock()

        result = semantic_search(collection, "")
        assert result == []

    @patch('data_loader.get_vector_engine')
    def test_semantic_search_with_short_query(self, mock_get_engine):
        """Test semantic_search with query too short."""
        collection = MagicMock()

        result = semantic_search(collection, "hi")
        assert result == []

    @patch('data_loader.get_vector_engine')
    def test_semantic_search_with_long_query(self, mock_get_engine):
        """Test semantic_search with query too long."""
        collection = MagicMock()
        long_query = "x" * 1000  # Assuming max is less than 1000

        result = semantic_search(collection, long_query)
        assert result == []

    @patch('data_loader.get_vector_engine')
    def test_semantic_search_with_none_collection(self, mock_get_engine):
        """Test semantic_search with None collection."""
        result = semantic_search(None, "valid query")
        assert result == []

    @patch('data_loader.get_vector_engine')
    def test_semantic_search_with_custom_params(self, mock_get_engine):
        """Test semantic_search with custom parameters."""
        collection = MagicMock()

        # Mock vector engine
        mock_engine = MagicMock()
        mock_engine.search.return_value = [
            {
                'document': 'Test Story',
                'metadata': {'id': 1},
                'similarity': 0.9,
                'distance': 0.1
            }
        ]
        mock_get_engine.return_value = mock_engine

        result = semantic_search(
            collection,
            "test query",
            max_results=5,
            similarity_threshold=0.8
        )

        assert len(result) == 1
        assert result[0]['title'] == 'Test Story'
        assert result[0]['similarity_score'] == 0.9
        assert result[0]['rank'] == 1

    @patch('data_loader.get_vector_engine')
    def test_semantic_search_engine_error(self, mock_get_engine):
        """Test semantic_search when vector engine raises error."""
        collection = MagicMock()

        # Mock vector engine to raise error
        mock_engine = MagicMock()
        mock_engine.search.side_effect = Exception("Engine error")
        mock_get_engine.return_value = mock_engine

        result = semantic_search(collection, "test query")
        assert result == []

    @patch('data_loader.analyze_sentiment')
    def test_fetch_hn_data_with_sentiment_analysis(self, mock_analyze):
        """Test fetch_hn_data includes sentiment analysis."""
        mock_analyze.return_value = pd.DataFrame({'sentiment': ['positive']})

        with patch('data_loader.fetch_story_ids') as mock_ids, \
             patch('data_loader.fetch_story_details') as mock_details:

            mock_ids.return_value = [1, 2]
            mock_details.side_effect = [
                {'id': 1, 'title': 'Story 1'},
                {'id': 2, 'title': 'Story 2'}
            ]

            df = fetch_hn_data(limit=2, analyze_sentiment=True)
            assert not df.empty
            assert 'sentiment' in df.columns

    @patch('data_loader.CacheManager')
    def test_fetch_hn_data_cache_error_handling(self, mock_cache_manager):
        """Test fetch_hn_data handles cache errors gracefully."""
        mock_cache = MagicMock()
        mock_cache.get_cached_data.side_effect = Exception("Cache error")
        mock_cache_manager.return_value = mock_cache

        with patch('data_loader.fetch_story_ids') as mock_ids:
            mock_ids.return_value = [1, 2]

            # Should not raise exception
            df = fetch_hn_data(limit=2, use_cache=True)
            assert isinstance(df, pd.DataFrame)

    def test_process_stories_to_dataframe_with_invalid_dates(self):
        """Test process_stories_to_dataframe with invalid timestamps."""
        stories = [
            {'id': 1, 'title': 'Story 1', 'time': 'invalid_timestamp'},
            {'id': 2, 'title': 'Story 2', 'time': None}
        ]

        df = process_stories_to_dataframe(stories)
        assert len(df) == 2
        # Invalid dates should be handled gracefully

    @patch('data_loader.sentence_transformer')
    def test_get_topics_with_sentence_transformer(self, mock_transformer):
        """Test get_topics using sentence transformer model."""
        # Mock the sentence transformer
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3] for _ in range(5)]
        mock_transformer.SentenceTransformer.return_value = mock_model

        df = pd.DataFrame({
            'title': ['Story 1', 'Story 2', 'Story 3', 'Story 4', 'Story 5'] * 4
        })

        result = get_topics(df, embedding_model='sentence-transformers/test-model')
        assert not result.empty
        assert 'topic' in result.columns