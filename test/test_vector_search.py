"""
Unit tests for vector search functionality.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime
import time

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_search import VectorSearchEngine, get_vector_engine
from data_loader import setup_vector_db, semantic_search
from vector_cache_manager import VectorCacheManager, get_vector_cache_manager


class TestVectorSearchEngine(unittest.TestCase):
    """Test cases for VectorSearchEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = VectorSearchEngine(
            model_name="test-model",
            collection_name="test_collection"
        )

    @patch('vector_search.SentenceTransformer')
    @patch('vector_search.chromadb.Client')
    def test_initialize_success(self, mock_chromadb, mock_transformer):
        """Test successful initialization."""
        # Mock the model
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        # Mock ChromaDB client
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_client.create_collection.return_value = mock_collection
        mock_chromadb.return_value = mock_client

        # Initialize
        self.engine.initialize()

        # Verify
        self.assertTrue(self.engine._initialized)
        mock_transformer.assert_called_once_with("test-model")
        mock_chromadb.assert_called_once()

    def test_generate_embedding_hash(self):
        """Test embedding hash generation."""
        texts = ["Story 1", "Story 2", "Story 3"]
        hash1 = self.engine.generate_embedding_hash(texts)
        hash2 = self.engine.generate_embedding_hash(texts)

        # Same texts should produce same hash
        self.assertEqual(hash1, hash2)

        # Different order should produce same hash
        texts_reordered = ["Story 3", "Story 1", "Story 2"]
        hash3 = self.engine.generate_embedding_hash(texts_reordered)
        self.assertEqual(hash1, hash3)

        # Different texts should produce different hash
        texts_different = ["Story 1", "Story 2", "Different Story"]
        hash4 = self.engine.generate_embedding_hash(texts_different)
        self.assertNotEqual(hash1, hash4)

    @patch.object(VectorSearchEngine, 'initialize')
    def test_generate_embeddings_empty_list(self, mock_init):
        """Test generating embeddings for empty list."""
        mock_init.return_value = None
        self.engine._initialized = True
        self.engine.model = Mock()

        result = self.engine.generate_embeddings([])
        self.assertEqual(result, [])

    @patch.object(VectorSearchEngine, 'initialize')
    def test_generate_embeddings_success(self, mock_init):
        """Test successful embedding generation."""
        mock_init.return_value = None
        self.engine._initialized = True

        # Mock model
        self.engine.model = Mock()
        self.engine.model.get_sentence_embedding_dimension.return_value = 384

        # Mock embeddings
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        self.engine.model.encode.return_value = mock_embeddings

        texts = ["Text 1", "Text 2"]
        result = self.engine.generate_embeddings(texts)

        self.engine.model.encode.assert_called_once()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [0.1, 0.2])
        self.assertEqual(result[1], [0.3, 0.4])

    def test_add_documents_empty(self):
        """Test adding empty documents list."""
        self.engine._initialized = True
        self.engine.collection = Mock()

        result = self.engine.add_documents([])
        self.assertTrue(result)  # Should return True for empty list

    @patch.object(VectorSearchEngine, 'initialize')
    def test_add_documents_success(self, mock_init):
        """Test successful document addition."""
        mock_init.return_value = None
        self.engine._initialized = True

        # Mock collection
        self.engine.collection = Mock()

        # Mock model for dimension
        self.engine.model = Mock()
        self.engine.model.get_sentence_embedding_dimension.return_value = 384

        with patch.object(self.engine, 'generate_embeddings', return_value=[[0.1, 0.2]]):
            documents = [
                {'id': '1', 'text': 'Story 1', 'score': 10},
                {'id': '2', 'text': 'Story 2', 'score': 20}
            ]
            result = self.engine.add_documents(documents)

            self.assertTrue(result)
            self.engine.collection.add.assert_called_once()

    @patch.object(VectorSearchEngine, 'initialize')
    def test_search_short_query(self, mock_init):
        """Test search with too short query."""
        mock_init.return_value = None
        self.engine._initialized = True

        result = self.engine.search("hi")
        self.assertEqual(result, [])

    @patch.object(VectorSearchEngine, 'initialize')
    def test_search_success(self, mock_init):
        """Test successful search."""
        mock_init.return_value = None
        self.engine._initialized = True

        # Mock model
        self.engine.model = Mock()
        self.engine.model.encode.return_value = np.array([[0.1, 0.2]])

        # Mock collection query result
        self.engine.collection = Mock()
        self.engine.collection.query.return_value = {
            'documents': [['Story 1']],
            'metadatas': [[{'score': 10}]],
            'distances': [[0.3]]
        }

        result = self.engine.search("test query")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['document'], 'Story 1')
        self.assertEqual(result[0]['similarity'], 0.7)  # 1 - 0.3

    def test_get_collection_info_not_initialized(self):
        """Test getting collection info when not initialized."""
        info = self.engine.get_collection_info()
        self.assertFalse(info['initialized'])

    @patch.object(VectorSearchEngine, 'initialize')
    def test_clear_collection(self, mock_init):
        """Test clearing collection."""
        mock_init.return_value = None
        self.engine._initialized = True

        # Mock client
        self.engine.client = Mock()
        self.engine.client.create_collection.return_value = Mock()
        self.engine.collection = Mock()

        result = self.engine.clear_collection()
        self.assertTrue(result)
        self.engine.client.delete_collection.assert_called_once()


class TestVectorCacheManager(unittest.TestCase):
    """Test cases for VectorCacheManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = VectorCacheManager(cache_duration=3600)

    @patch('vector_cache_manager.st')
    def test_get_session_state_with_streamlit(self, mock_st):
        """Test getting session state with Streamlit available."""
        result = self.manager.get_session_state()
        self.assertEqual(result, mock_st.session_state)

    @patch('vector_cache_manager.st', side_effect=ImportError)
    def test_get_session_state_without_streamlit(self, mock_st):
        """Test getting session state without Streamlit."""
        result = self.manager.get_session_state()
        self.assertIsInstance(result, dict)

    def test_generate_data_hash(self):
        """Test data hash generation."""
        data = [
            {'title': 'Story 1', 'score': 10},
            {'title': 'Story 2', 'score': 20}
        ]
        hash1 = self.manager.generate_data_hash(data)
        hash2 = self.manager.generate_data_hash(data)
        self.assertEqual(hash1, hash2)

        # Different order should produce same hash
        data_reordered = [
            {'title': 'Story 2', 'score': 20},
            {'title': 'Story 1', 'score': 10}
        ]
        hash3 = self.manager.generate_data_hash(data_reordered)
        self.assertEqual(hash1, hash3)

    @patch.object(VectorCacheManager, 'get_session_state')
    def test_is_cache_valid_no_cache(self, mock_session):
        """Test cache validation when no cache exists."""
        mock_session.return_value = {}
        result = self.manager.is_cache_valid()
        self.assertFalse(result)

    @patch.object(VectorCacheManager, 'get_session_state')
    def test_is_cache_valid_expired(self, mock_session):
        """Test cache validation when expired."""
        mock_session.return_value = {
            'vector_collection': Mock(),
            'vector_cache_timestamp': time.time() - 7200  # 2 hours ago
        }
        result = self.manager.is_cache_valid()
        self.assertFalse(result)

    @patch.object(VectorCacheManager, 'get_session_state')
    def test_should_rebuild_collection(self, mock_session):
        """Test determining if collection should be rebuilt."""
        # Test with no cache
        mock_session.return_value = {}
        data = [{'title': 'Test', 'score': 10}]
        result = self.manager.should_rebuild_collection(data)
        self.assertTrue(result)

        # Test with force rebuild
        result = self.manager.should_rebuild_collection(data, force_rebuild=True)
        self.assertTrue(result)


class TestSetupVectorDB(unittest.TestCase):
    """Test cases for setup_vector_db function."""

    @patch('data_loader.get_vector_cache_manager')
    @patch('data_loader.get_vector_engine')
    def test_setup_vector_db_empty_df(self, mock_engine, mock_cache):
        """Test setup with empty DataFrame."""
        df = pd.DataFrame()
        result = setup_vector_db(df)
        self.assertIsNone(result)

    @patch('data_loader.get_vector_cache_manager')
    def test_setup_vector_db_cached(self, mock_cache):
        """Test setup using cached collection."""
        # Create test DataFrame
        df = pd.DataFrame({
            'title': ['Story 1', 'Story 2'],
            'score': [10, 20]
        })

        # Mock cache manager
        mock_manager = Mock()
        mock_manager.should_rebuild_collection.return_value = False
        mock_manager.get_cached_collection.return_value = Mock()
        mock_manager.get_cache_info.return_value = {'version': 1}
        mock_cache.return_value = mock_manager

        result = setup_vector_db(df)
        self.assertIsNotNone(result)

    @patch('data_loader.get_vector_cache_manager')
    @patch('data_loader.get_vector_engine')
    def test_setup_vector_db_force_rebuild(self, mock_engine, mock_cache):
        """Test setup with force rebuild."""
        # Create test DataFrame
        df = pd.DataFrame({
            'title': ['Story 1', 'Story 2'],
            'score': [10, 20]
        })

        # Mock cache manager
        mock_manager = Mock()
        mock_cache.return_value = mock_manager

        # Mock vector engine
        mock_vec_engine = Mock()
        mock_collection = Mock()
        mock_vec_engine.initialize.return_value = None
        mock_vec_engine.clear_collection.return_value = None
        mock_vec_engine.add_documents.return_value = True
        mock_vec_engine.get_collection_info.return_value = {
            'name': 'test_collection',
            'count': 2
        }
        mock_vec_engine.collection = mock_collection
        mock_engine.return_value = mock_vec_engine

        result = setup_vector_db(df, force_rebuild=True)

        self.assertIsNotNone(result)
        mock_vec_engine.initialize.assert_called_once()
        mock_vec_engine.clear_collection.assert_called_once()


class TestSemanticSearch(unittest.TestCase):
    """Test cases for semantic_search function."""

    @patch('data_loader.get_vector_engine')
    def test_semantic_search_short_query(self, mock_engine):
        """Test semantic search with short query."""
        collection = Mock()
        result = semantic_search(collection, "hi")
        self.assertEqual(result, [])

    @patch('data_loader.get_vector_engine')
    def test_semantic_search_no_collection(self, mock_engine):
        """Test semantic search with no collection."""
        result = semantic_search(None, "test query")
        self.assertEqual(result, [])

    @patch('data_loader.get_vector_engine')
    def test_semantic_search_success(self, mock_engine):
        """Test successful semantic search."""
        # Mock collection
        collection = Mock()

        # Mock vector engine
        mock_vec_engine = Mock()
        mock_vec_engine.search.return_value = [
            {
                'document': 'Test Story',
                'metadata': {'score': 10},
                'similarity': 0.85,
                'distance': 0.15
            }
        ]
        mock_engine.return_value = mock_vec_engine

        result = semantic_search(collection, "test query")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['title'], 'Test Story')
        self.assertEqual(result[0]['similarity_score'], 0.85)


if __name__ == '__main__':
    unittest.main()