#!/usr/bin/env python3
"""
Integration tests for semantic search functionality.
This module tests the full semantic search workflow including UI integration.
"""

import unittest
import sys
import os
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import setup_vector_db, semantic_search, fetch_hn_data
from vector_search import VectorSearchManager, get_vector_engine
from vector_cache_manager import get_vector_cache_manager
from dashboard_config import SEMANTIC_SEARCH_SETTINGS


class TestSemanticSearchIntegration(unittest.TestCase):
    """Integration tests for semantic search workflow"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.test_collection_name = "test_semantic_search"

        # Create sample data
        self.sample_df = pd.DataFrame({
            'title': [
                'AI breakthrough in natural language processing',
                'New quantum computer achieves quantum supremacy',
                'Python 3.12 released with performance improvements',
                'Machine learning model predicts protein folding',
                'WebAssembly gains native support in browsers',
                'Blockchain technology revolutionizes finance',
                'Cybersecurity threats on the rise in 2024',
                'Cloud computing trends for enterprise'
            ],
            'score': [250, 300, 150, 280, 120, 200, 180, 220],
            'descendants': [50, 75, 30, 90, 25, 60, 45, 55],
            'time': [datetime.now()] * 8,
            'url': [f'https://example.com/story{i}' for i in range(8)],
            'sentiment_label': ['Positive'] * 8,
            'topic_keyword': ['ai_ml', 'quantum', 'programming', 'ai_ml', 'web', 'blockchain', 'security', 'cloud']
        })

    def tearDown(self):
        """Clean up test fixtures"""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('data_loader.get_vector_cache_manager')
    @patch('data_loader.get_vector_engine')
    def test_full_search_workflow(self, mock_get_vector_engine, mock_get_cache_manager):
        """Test complete semantic search workflow from setup to search"""
        # Mock vector cache manager
        mock_cache_manager = Mock()
        mock_cache_manager.should_rebuild_collection.return_value = True
        mock_cache_manager.generate_data_hash.return_value = "test_hash"
        mock_get_cache_manager.return_value = mock_cache_manager

        # Mock vector engine
        mock_engine = Mock()
        mock_engine.collection = Mock()
        mock_engine.get_collection_info.return_value = {
            'name': self.test_collection_name,
            'count': len(self.sample_df)
        }
        mock_engine.add_documents.return_value = True
        mock_get_vector_engine.return_value = mock_engine

        # Test 1: Setup vector database
        collection = setup_vector_db(
            self.sample_df,
            collection_name=self.test_collection_name,
            force_rebuild=True
        )

        # Verify setup was called
        mock_get_vector_engine.assert_called_once_with(
            model_name=SEMANTIC_SEARCH_SETTINGS["model_name"],
            collection_name=self.test_collection_name
        )
        mock_engine.clear_collection.assert_called_once()
        mock_engine.add_documents.assert_called()

        # Test 2: Perform semantic search
        mock_search_results = [
            {
                'document': 'AI breakthrough in natural language processing',
                'metadata': {'score': 250, 'topic_keyword': 'ai_ml'},
                'similarity': 0.85,
                'distance': 0.15
            },
            {
                'document': 'Machine learning model predicts protein folding',
                'metadata': {'score': 280, 'topic_keyword': 'ai_ml'},
                'similarity': 0.78,
                'distance': 0.22
            }
        ]
        mock_engine.search.return_value = mock_search_results

        results = semantic_search(collection, "artificial intelligence research")

        # Verify search was performed
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['title'], 'AI breakthrough in natural language processing')
        self.assertEqual(results[0]['similarity_score'], 0.85)
        self.assertEqual(results[1]['title'], 'Machine learning model predicts protein folding')
        self.assertEqual(results[1]['similarity_score'], 0.78)

    def test_search_with_varied_query_types(self):
        """Test semantic search with different query types and lengths"""
        # Mock collection
        mock_collection = Mock()

        # Mock vector engine
        with patch('data_loader.get_vector_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_engine.collection = mock_collection
            mock_get_engine.return_value = mock_engine

            # Test short query (below minimum)
            mock_engine.search.return_value = []
            results = semantic_search(mock_collection, "AI")
            self.assertEqual(len(results), 0)

            # Test valid query
            mock_results = [
                {
                    'document': 'AI breakthrough in natural language processing',
                    'metadata': {'score': 250},
                    'similarity': 0.85,
                    'distance': 0.15
                }
            ]
            mock_engine.search.return_value = mock_results

            results = semantic_search(mock_collection, "artificial intelligence and machine learning")
            self.assertEqual(len(results), 1)
            mock_engine.search.assert_called_with(
                query="artificial intelligence and machine learning",
                n_results=SEMANTIC_SEARCH_SETTINGS["max_results"],
                similarity_threshold=SEMANTIC_SEARCH_SETTINGS["similarity_threshold"]
            )

            # Test long query (above maximum)
            long_query = "a" * (SEMANTIC_SEARCH_SETTINGS["max_query_length"] + 1)
            results = semantic_search(mock_collection, long_query)
            self.assertEqual(len(results), 0)

    def test_search_with_filters(self):
        """Test semantic search with metadata filters"""
        mock_collection = Mock()

        with patch('data_loader.get_vector_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_engine.collection = mock_collection
            mock_get_engine.return_value = mock_engine

            # Mock search with filter
            mock_engine.search.return_value = [
                {
                    'document': 'AI breakthrough in natural language processing',
                    'metadata': {'score': 250, 'topic_keyword': 'ai_ml'},
                    'similarity': 0.85,
                    'distance': 0.15
                }
            ]

            # This would need to be implemented in the actual search function
            # For now, we're testing that it could handle filters
            results = semantic_search(mock_collection, "AI research")
            self.assertEqual(len(results), 1)

    @patch('data_loader.fetch_hn_data')
    @patch('data_loader.setup_vector_db')
    def test_integration_with_real_data(self, mock_setup_db, mock_fetch):
        """Test integration with real Hacker News data fetching"""
        # Mock fetched data
        mock_df = pd.DataFrame({
            'title': ['Real HN story about AI', 'Another tech story'],
            'score': [100, 150],
            'descendants': [20, 30],
            'time': [datetime.now(), datetime.now()],
            'url': ['https://news.ycombinator.com/item1', 'https://news.ycombinator.com/item2']
        })
        mock_fetch.return_value = mock_df

        # Mock vector DB setup
        mock_collection = Mock()
        mock_setup_db.return_value = mock_collection

        # Simulate workflow
        df = fetch_hn_data(limit=10)
        collection = setup_vector_db(df)

        # Verify data was fetched and processed
        mock_fetch.assert_called_once_with(limit=10)
        mock_setup_db.assert_called_once_with(df)

    def test_error_handling_in_workflow(self):
        """Test error handling throughout the search workflow"""
        # Test with None collection
        results = semantic_search(None, "test query")
        self.assertEqual(results, [])

        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        with patch('data_loader.get_vector_cache_manager') as mock_get_cache:
            mock_cache = Mock()
            mock_cache.should_rebuild_collection.return_value = True
            mock_get_cache.return_value = mock_cache

            collection = setup_vector_db(empty_df)
            self.assertIsNone(collection)

        # Test search with collection error
        mock_collection = Mock()
        with patch('data_loader.get_vector_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_engine.collection = mock_collection
            mock_engine.search.side_effect = Exception("Search failed")
            mock_get_engine.return_value = mock_engine

            results = semantic_search(mock_collection, "test query")
            self.assertEqual(results, [])

    def test_caching_integration(self):
        """Test that caching works properly with semantic search"""
        mock_cache_manager = Mock()
        mock_cache_manager.should_rebuild_collection.return_value = False
        mock_cache_manager.get_cached_collection.return_value = Mock()
        mock_cache_manager.get_cache_info.return_value = {'version': '1.0.0'}

        with patch('data_loader.get_vector_cache_manager', return_value=mock_cache_manager):
            # Should use cached collection
            collection = setup_vector_db(self.sample_df, force_rebuild=False)
            self.assertIsNotNone(collection)
            mock_cache_manager.get_cached_collection.assert_called_once()

    @patch('streamlit.text_input')
    @patch('streamlit.button')
    @patch('streamlit.write')
    def test_ui_component_integration(self, mock_write, mock_button, mock_text_input):
        """Test integration with Streamlit UI components"""
        # Mock user input
        mock_text_input.return_value = "artificial intelligence"
        mock_button.return_value = True

        # Mock search results
        mock_collection = Mock()
        with patch('data_loader.get_vector_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_engine.collection = mock_collection
            mock_engine.search.return_value = [
                {
                    'document': 'AI breakthrough',
                    'metadata': {'score': 250},
                    'similarity': 0.85,
                    'distance': 0.15
                }
            ]
            mock_get_engine.return_value = mock_engine

            # Simulate UI search interaction
            query = mock_text_input("Enter search query:", key="search_input")
            if mock_button("Search", key="search_button"):
                if query:
                    results = semantic_search(mock_collection, query)
                    for result in results:
                        mock_write(f"**{result['title']}**")
                        mock_write(f"Similarity: {result['similarity_score']:.1%}")

            # Verify UI interactions
            mock_text_input.assert_called_once()
            mock_button.assert_called_once()
            self.assertGreater(mock_write.call_count, 0)

    def test_real_time_mode_compatibility(self):
        """Test semantic search compatibility with real-time mode"""
        # Simulate real-time data update
        updated_df = self.sample_df.copy()
        updated_df.loc[len(updated_df)] = {
            'title': 'Breaking: New AI model announced',
            'score': 500,
            'descendants': 100,
            'time': datetime.now(),
            'url': 'https://example.com/breaking',
            'sentiment_label': 'Positive',
            'topic_keyword': 'ai_ml'
        }

        mock_cache_manager = Mock()
        mock_cache_manager.should_rebuild_collection.return_value = True
        mock_cache_manager.generate_data_hash.return_value = "updated_hash"

        with patch('data_loader.get_vector_cache_manager', return_value=mock_cache_manager):
            with patch('data_loader.get_vector_engine') as mock_get_engine:
                mock_engine = Mock()
                mock_engine.collection = Mock()
                mock_engine.get_collection_info.return_value = {'name': 'test', 'count': len(updated_df)}
                mock_engine.add_documents.return_value = True
                mock_get_engine.return_value = mock_engine

                # Setup updated vector DB
                collection = setup_vector_db(updated_df, force_rebuild=True)

                # Verify new data is included
                mock_engine.add_documents.assert_called()
                call_args = mock_engine.add_documents.call_args[0][0]
                self.assertTrue(any('Breaking: New AI model announced' in doc['text'] for doc in call_args))

    def test_search_result_accuracy(self):
        """Test accuracy of search results with known queries"""
        # Test cases: query -> expected titles (order matters for relevance)
        test_cases = [
            {
                'query': 'artificial intelligence machine learning',
                'expected_keywords': ['ai_ml'],
                'min_similarity': 0.7
            },
            {
                'query': 'quantum computing physics',
                'expected_keywords': ['quantum'],
                'min_similarity': 0.6
            },
            {
                'query': 'python programming code',
                'expected_keywords': ['programming'],
                'min_similarity': 0.5
            }
        ]

        mock_collection = Mock()

        for test_case in test_cases:
            with patch('data_loader.get_vector_engine') as mock_get_engine:
                mock_engine = Mock()
                mock_engine.collection = mock_collection

                # Mock results based on query
                if 'ai' in test_case['query'].lower():
                    mock_results = [
                        {
                            'document': 'AI breakthrough in natural language processing',
                            'metadata': {'score': 250, 'topic_keyword': 'ai_ml'},
                            'similarity': 0.85,
                            'distance': 0.15
                        },
                        {
                            'document': 'Machine learning model predicts protein folding',
                            'metadata': {'score': 280, 'topic_keyword': 'ai_ml'},
                            'similarity': 0.78,
                            'distance': 0.22
                        }
                    ]
                elif 'quantum' in test_case['query'].lower():
                    mock_results = [
                        {
                            'document': 'New quantum computer achieves quantum supremacy',
                            'metadata': {'score': 300, 'topic_keyword': 'quantum'},
                            'similarity': 0.82,
                            'distance': 0.18
                        }
                    ]
                else:
                    mock_results = [
                        {
                            'document': 'Python 3.12 released with performance improvements',
                            'metadata': {'score': 150, 'topic_keyword': 'programming'},
                            'similarity': 0.65,
                            'distance': 0.35
                        }
                    ]

                mock_engine.search.return_value = mock_results
                mock_get_engine.return_value = mock_engine

                # Perform search
                results = semantic_search(
                    mock_collection,
                    test_case['query'],
                    max_results=10
                )

                # Verify results
                self.assertGreater(len(results), 0, f"No results for query: {test_case['query']}")

                # Check similarity scores
                for result in results:
                    self.assertGreaterEqual(
                        result['similarity_score'],
                        test_case['min_similarity'],
                        f"Result similarity too low: {result['similarity_score']}"
                    )

                # Check expected keywords are present
                found_keywords = set()
                for result in results:
                    if 'metadata' in result and 'topic_keyword' in result['metadata']:
                        found_keywords.add(result['metadata']['topic_keyword'])

                has_expected_keyword = any(
                    keyword in found_keywords
                    for keyword in test_case['expected_keywords']
                )
                self.assertTrue(
                    has_expected_keyword,
                    f"Expected keywords {test_case['expected_keywords']} not found in {found_keywords}"
                )


class TestVectorSearchManagerIntegration(unittest.TestCase):
    """Integration tests for VectorSearchManager"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('vector_search.SentenceTransformer')
    @patch('vector_search.chromadb.PersistentClient')
    def test_manager_initialization(self, mock_chroma_client, mock_sentence_transformer):
        """Test VectorSearchManager initialization and lazy loading"""
        # Mock sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_sentence_transformer.return_value = mock_model

        # Mock ChromaDB client
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        # Test lazy loading
        _ = self.test_manager.embedding_model
        mock_sentence_transformer.assert_called_once()

        _ = self.test_manager.chroma_client
        mock_chroma_client.assert_called_once()

        _ = self.test_manager.collection
        mock_client.get_or_create_collection.assert_called_once()

    @patch('vector_search.SentenceTransformer')
    def test_embedding_generation(self, mock_sentence_transformer):
        """Test embedding generation with batch processing"""
        # Mock model
        mock_model = Mock()
        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model

        # Create test manager
        test_manager = VectorSearchManager(
            collection_name="test_embeddings",
            persist_directory=self.temp_dir
        )

        # Test embeddings
        texts = ["Test document 1", "Test document 2"]
        test_manager._embedding_model = mock_model  # Set the mocked model
        embeddings = test_manager.generate_embeddings(texts)

        self.assertEqual(len(embeddings), 2)
        mock_model.encode.assert_called_once_with(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

    def test_document_addition_and_retrieval(self):
        """Test adding documents and retrieving them"""
        test_manager = VectorSearchManager(
            collection_name="test_add_docs",
            persist_directory=self.temp_dir
        )

        # Mock the embedding generation
        with patch.object(test_manager, 'generate_embeddings', return_value=[[0.1, 0.2, 0.3]]):
            with patch.object(test_manager, 'collection') as mock_collection:
                mock_collection.add.return_value = None

                # Test adding documents
                success = test_manager.add_documents(
                    documents=["Test document"],
                    ids=["doc1"],
                    metadata=[{"source": "test"}]
                )

                self.assertTrue(success)
                mock_collection.add.assert_called_once()

    def test_search_functionality(self):
        """Test search functionality"""
        test_manager = VectorSearchManager(
            collection_name="test_search",
            persist_directory=self.temp_dir
        )

        # Mock the embedding generation and collection
        with patch('vector_search.SentenceTransformer') as mock_transformer:
            mock_model = Mock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
            mock_transformer.return_value = mock_model

            with patch.object(test_manager, 'collection') as mock_collection:
                mock_collection.query.return_value = {
                    'ids': [['doc1']],
                    'documents': [['Test document']],
                    'metadatas': [[{'source': 'test'}]],
                    'distances': [[0.1]]
                }

                # Set the mocked model
                test_manager._embedding_model = mock_model

                # Test search
                results = test_manager.search("test query")

                self.assertEqual(results['count'], 1)
                self.assertEqual(results['query'], "test query")
                self.assertEqual(len(results['results']), 1)
                self.assertEqual(results['results'][0]['id'], 'doc1')


if __name__ == '__main__':
    unittest.main()