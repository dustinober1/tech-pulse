"""
Unit tests for vector search infrastructure.

Tests the VectorSearchManager class and its components including:
- Model initialization (mocked)
- ChromaDB client creation
- Basic embedding generation
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import tempfile
import shutil

# Add the parent directory to the path to import vector_search
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_search import VectorSearchManager, create_vector_search_manager, get_vector_engine


class TestVectorSearchManager(unittest.TestCase):
    """Test cases for VectorSearchManager class"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for ChromaDB persistence
        self.test_dir = tempfile.mkdtemp()

        # Create VectorSearchManager instance with test configuration
        self.manager = VectorSearchManager(
            collection_name="test_collection",
            model_name="test-model",
            persist_directory=self.test_dir,
            device="cpu"
        )

    def tearDown(self):
        """Clean up after each test method."""
        # Remove the temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('vector_search.SentenceTransformer')
    @patch('vector_search.chromadb.PersistentClient')
    def test_initialization(self, mock_chroma_client, mock_sentence_transformer):
        """Test VectorSearchManager initialization"""
        # Create new manager
        manager = VectorSearchManager(
            collection_name="test",
            model_name="test-model",
            persist_directory=self.test_dir
        )

        # Verify attributes
        self.assertEqual(manager.collection_name, "test")
        self.assertEqual(manager.model_name, "test-model")
        self.assertEqual(manager.persist_directory, self.test_dir)
        self.assertEqual(manager.device, "cpu")
        self.assertIsNone(manager._embedding_model)
        self.assertIsNone(manager._chroma_client)
        self.assertIsNone(manager._collection)

    @patch('vector_search.SentenceTransformer')
    @patch('vector_search.tqdm')
    def test_embedding_model_lazy_loading(self, mock_tqdm, mock_sentence_transformer):
        """Test lazy loading of embedding model"""
        # Setup mock
        mock_model_instance = Mock()
        mock_model_instance.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_sentence_transformer.return_value = mock_model_instance

        # Mock tqdm context manager
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_pbar

        # Access embedding_model property for the first time
        model = self.manager.embedding_model

        # Verify SentenceTransformer was called with correct parameters
        mock_sentence_transformer.assert_called_once_with("test-model", device="cpu")

        # Verify model instance is cached
        self.assertEqual(model, mock_model_instance)
        self.assertEqual(self.manager._embedding_model, mock_model_instance)

        # Access again - should not create new instance
        model2 = self.manager.embedding_model
        self.assertEqual(model2, mock_model_instance)
        # Should only be called once due to caching
        mock_sentence_transformer.assert_called_once()

    @patch('vector_search.chromadb.PersistentClient')
    def test_chroma_client_lazy_loading(self, mock_chroma_client):
        """Test lazy loading of ChromaDB client"""
        # Setup mock
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance

        # Access chroma_client property for the first time
        client = self.manager.chroma_client

        # Verify PersistentClient was called with correct path
        mock_chroma_client.assert_called_once_with(path=self.test_dir)

        # Verify client instance is cached
        self.assertEqual(client, mock_client_instance)
        self.assertEqual(self.manager._chroma_client, mock_client_instance)

    @patch('vector_search.chromadb.PersistentClient')
    def test_collection_creation(self, mock_chroma_client):
        """Test ChromaDB collection creation"""
        # Setup mocks
        mock_client_instance = Mock()
        mock_collection_instance = Mock()
        mock_collection_instance.count.return_value = 0

        # Configure client to raise exception on get_collection (collection doesn't exist)
        mock_client_instance.get_collection.side_effect = Exception("Collection not found")
        mock_client_instance.create_collection.return_value = mock_collection_instance
        mock_chroma_client.return_value = mock_client_instance

        # Access collection property
        collection = self.manager.collection

        # Verify collection was created
        mock_client_instance.create_collection.assert_called_once_with(
            name="test_collection",
            metadata={"description": "Tech stories vector embeddings"}
        )

        # Verify collection is cached
        self.assertEqual(collection, mock_collection_instance)

    @patch('vector_search.chromadb.PersistentClient')
    def test_collection_retrieval_existing(self, mock_chroma_client):
        """Test retrieving existing ChromaDB collection"""
        # Setup mocks
        mock_client_instance = Mock()
        mock_collection_instance = Mock()
        mock_collection_instance.count.return_value = 10

        # Configure client to return existing collection
        mock_client_instance.get_collection.return_value = mock_collection_instance
        mock_chroma_client.return_value = mock_client_instance

        # Access collection property
        collection = self.manager.collection

        # Verify existing collection was retrieved
        mock_client_instance.get_collection.assert_called_once_with(name="test_collection")
        mock_client_instance.create_collection.assert_not_called()

        # Verify collection is cached
        self.assertEqual(collection, mock_collection_instance)

    @patch('vector_search.SentenceTransformer')
    @patch('vector_search.chromadb.PersistentClient')
    def test_generate_embeddings(self, mock_chroma_client, mock_sentence_transformer):
        """Test embedding generation"""
        import numpy as np

        # Setup mocks
        mock_model_instance = Mock()
        test_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model_instance.encode.return_value = test_embeddings
        mock_sentence_transformer.return_value = mock_model_instance

        # Set up manager with mocked model
        self.manager._embedding_model = mock_model_instance

        # Test embedding generation
        texts = ["Hello world", "Test document"]
        embeddings = self.manager.generate_embeddings(texts, batch_size=10)

        # Verify encode was called with correct parameters
        mock_model_instance.encode.assert_called_once_with(
            texts,
            batch_size=10,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Verify results (converted to list)
        expected_embeddings = test_embeddings.tolist()
        self.assertEqual(embeddings, expected_embeddings)

    def test_generate_embeddings_empty_list(self):
        """Test embedding generation with empty list"""
        embeddings = self.manager.generate_embeddings([])
        self.assertEqual(embeddings, [])

    @patch('vector_search.SentenceTransformer')
    @patch('vector_search.chromadb.PersistentClient')
    def test_add_documents(self, mock_chroma_client, mock_sentence_transformer):
        """Test adding documents to vector store"""
        import numpy as np

        # Setup mocks
        mock_model_instance = Mock()
        test_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model_instance.encode.return_value = test_embeddings
        mock_sentence_transformer.return_value = mock_model_instance

        mock_collection_instance = Mock()
        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection_instance
        mock_chroma_client.return_value = mock_client_instance

        # Set up manager with mocked components
        self.manager._embedding_model = mock_model_instance
        self.manager._chroma_client = mock_client_instance
        self.manager._collection = mock_collection_instance

        # Test adding documents
        documents = ["Document 1", "Document 2"]
        ids = ["doc1", "doc2"]
        metadata = [{"title": "Doc 1"}, {"title": "Doc 2"}]

        result = self.manager.add_documents(documents, ids, metadata)

        # Verify encode was called
        mock_model_instance.encode.assert_called_once()

        # Verify collection.add was called with correct parameters
        expected_embeddings = test_embeddings.tolist()
        mock_collection_instance.add.assert_called_once_with(
            embeddings=expected_embeddings,
            documents=documents,
            ids=ids,
            metadatas=metadata
        )

        # Verify result
        self.assertTrue(result)

    def test_add_documents_mismatched_lengths(self):
        """Test adding documents with mismatched document and ID lengths"""
        with self.assertRaises(ValueError):
            self.manager.add_documents(
                documents=["Doc 1", "Doc 2"],
                ids=["doc1"]  # Only one ID
            )

    @patch('vector_search.SentenceTransformer')
    @patch('vector_search.chromadb.PersistentClient')
    def test_search(self, mock_chroma_client, mock_sentence_transformer):
        """Test searching for similar documents"""
        import numpy as np

        # Setup mocks
        mock_model_instance = Mock()
        # Return a 2D array with shape (1, 3) since encode is called with [query]
        test_embedding = np.array([[0.1, 0.2, 0.3]])
        mock_model_instance.encode.return_value = test_embedding
        mock_sentence_transformer.return_value = mock_model_instance

        mock_collection_instance = Mock()
        mock_collection_instance.query.return_value = {
            'ids': [['doc1', 'doc2']],
            'documents': [['Document 1', 'Document 2']],
            'metadatas': [[{'title': 'Doc 1'}, {'title': 'Doc 2'}]],
            'distances': [[0.1, 0.2]]
        }
        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection_instance
        mock_chroma_client.return_value = mock_client_instance

        # Set up manager with mocked components
        self.manager._embedding_model = mock_model_instance
        self.manager._chroma_client = mock_client_instance
        self.manager._collection = mock_collection_instance

        # Test search
        query = "test query"
        result = self.manager.search(query, n_results=5)

        # Verify encode was called
        mock_model_instance.encode.assert_called_once_with([query])

        # Verify collection.query was called
        # The code does [0].tolist(), so we need to extract the first row
        expected_embedding = test_embedding[0].tolist()
        mock_collection_instance.query.assert_called_once_with(
            query_embeddings=[expected_embedding],
            n_results=5,
            where=None
        )

        # Verify result format
        self.assertEqual(result['query'], query)
        self.assertEqual(result['count'], 2)
        self.assertEqual(len(result['results']), 2)

        # Check first result
        first_result = result['results'][0]
        self.assertEqual(first_result['id'], 'doc1')
        self.assertEqual(first_result['document'], 'Document 1')
        self.assertEqual(first_result['metadata'], {'title': 'Doc 1'})
        self.assertEqual(first_result['distance'], 0.1)

    @patch('vector_search.chromadb.PersistentClient')
    def test_get_stats(self, mock_chroma_client):
        """Test getting vector store statistics"""
        # Setup mocks
        mock_collection_instance = Mock()
        mock_collection_instance.count.return_value = 42
        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection_instance
        mock_chroma_client.return_value = mock_client_instance

        # Set up manager with mocked collection
        self.manager._chroma_client = mock_client_instance
        self.manager._collection = mock_collection_instance

        # Test get_stats
        stats = self.manager.get_stats()

        # Verify count was queried
        mock_collection_instance.count.assert_called_once()

        # Verify stats
        expected_stats = {
            'collection_name': 'test_collection',
            'document_count': 42,
            'model_name': 'test-model',
            'persist_directory': self.test_dir,
            'device': 'cpu'
        }
        self.assertEqual(stats, expected_stats)

    @patch('vector_search.chromadb.PersistentClient')
    def test_delete_collection(self, mock_chroma_client):
        """Test deleting the collection"""
        # Setup mocks
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance

        # Set up manager with mocked client
        self.manager._chroma_client = mock_client_instance
        self.manager._collection = Mock()  # Non-None collection

        # Test delete_collection
        result = self.manager.delete_collection()

        # Verify delete_collection was called
        mock_client_instance.delete_collection.assert_called_once_with(name="test_collection")

        # Verify collection was reset
        self.assertIsNone(self.manager._collection)

        # Verify result
        self.assertTrue(result)


class TestVectorSearchUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after each test method."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('vector_search.VectorSearchManager')
    def test_create_vector_search_manager(self, mock_manager_class):
        """Test create_vector_search_manager utility function"""
        mock_instance = Mock()
        mock_manager_class.return_value = mock_instance

        # Test with default collection name
        result = create_vector_search_manager()
        mock_manager_class.assert_called_once_with(collection_name="tech_stories")
        self.assertEqual(result, mock_instance)

        # Test with custom collection name
        mock_manager_class.reset_mock()
        result = create_vector_search_manager(collection_name="custom")
        mock_manager_class.assert_called_once_with(collection_name="custom")
        self.assertEqual(result, mock_instance)

    @patch('vector_search.VectorSearchManager')
    def test_get_vector_engine(self, mock_manager_class):
        """Test get_vector_engine global instance function"""
        mock_instance = Mock()
        mock_manager_class.return_value = mock_instance

        # First call should create new instance
        result1 = get_vector_engine(model_name="test-model", collection_name="test-collection")
        mock_manager_class.assert_called_once_with(
            collection_name="test-collection",
            model_name="test-model"
        )
        self.assertEqual(result1, mock_instance)

        # Second call should return same instance
        mock_manager_class.reset_mock()
        result2 = get_vector_engine()
        mock_manager_class.assert_not_called()  # Should not create new instance
        self.assertEqual(result2, mock_instance)
        self.assertEqual(result1, result2)


class TestVectorSearchErrorHandling(unittest.TestCase):
    """Test error handling in VectorSearchManager"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.manager = VectorSearchManager(
            collection_name="test_collection",
            model_name="test-model",
            persist_directory=self.test_dir
        )

    def tearDown(self):
        """Clean up after each test method."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('vector_search.SentenceTransformer')
    def test_embedding_model_load_failure(self, mock_sentence_transformer):
        """Test handling of embedding model load failure"""
        # Setup mock to raise exception
        mock_sentence_transformer.side_effect = Exception("Model load failed")

        # Verify that accessing embedding_model raises RuntimeError
        with self.assertRaises(RuntimeError) as context:
            _ = self.manager.embedding_model

        self.assertIn("Could not load embedding model", str(context.exception))

    @patch('vector_search.chromadb.PersistentClient')
    def test_chroma_client_init_failure(self, mock_chroma_client):
        """Test handling of ChromaDB client initialization failure"""
        # Setup mock to raise exception
        mock_chroma_client.side_effect = Exception("ChromaDB init failed")

        # Verify that accessing chroma_client raises RuntimeError
        with self.assertRaises(RuntimeError) as context:
            _ = self.manager.chroma_client

        self.assertIn("Could not initialize ChromaDB client", str(context.exception))

    @patch('vector_search.SentenceTransformer')
    def test_generate_embeddings_failure(self, mock_sentence_transformer):
        """Test handling of embedding generation failure"""
        # Setup mocks
        mock_model_instance = Mock()
        mock_model_instance.encode.side_effect = Exception("Encoding failed")
        mock_sentence_transformer.return_value = mock_model_instance
        self.manager._embedding_model = mock_model_instance

        # Test error handling
        with self.assertRaises(RuntimeError) as context:
            self.manager.generate_embeddings(["test"])

        self.assertIn("Embedding generation failed", str(context.exception))

    @patch('vector_search.SentenceTransformer')
    @patch('vector_search.chromadb.PersistentClient')
    def test_add_documents_failure(self, mock_chroma_client, mock_sentence_transformer):
        """Test handling of add_documents failure"""
        # Setup mocks
        mock_model_instance = Mock()
        mock_model_instance.encode.side_effect = Exception("Encoding failed")
        mock_sentence_transformer.return_value = mock_model_instance
        self.manager._embedding_model = mock_model_instance

        # Test error handling
        result = self.manager.add_documents(["doc1"], ["id1"])

        # Should return False on failure
        self.assertFalse(result)

    @patch('vector_search.SentenceTransformer')
    @patch('vector_search.chromadb.PersistentClient')
    def test_search_failure(self, mock_chroma_client, mock_sentence_transformer):
        """Test handling of search failure"""
        # Setup mocks
        mock_model_instance = Mock()
        mock_model_instance.encode.side_effect = Exception("Encoding failed")
        mock_sentence_transformer.return_value = mock_model_instance
        self.manager._embedding_model = mock_model_instance

        # Test error handling
        result = self.manager.search("test query")

        # Should return empty results with error
        self.assertEqual(result['results'], [])
        self.assertEqual(result['count'], 0)
        self.assertIn('error', result)


if __name__ == '__main__':
    unittest.main()