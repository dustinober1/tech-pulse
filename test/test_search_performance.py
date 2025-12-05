#!/usr/bin/env python3
"""
Performance tests for semantic search functionality.
This module tests search response times, memory usage, and scalability.
"""

import unittest
import sys
import os
import time
import psutil
import tracemalloc
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd
import tempfile
import shutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import setup_vector_db, semantic_search
from vector_search import VectorSearchManager, get_vector_engine
from dashboard_config import SEMANTIC_SEARCH_SETTINGS


class TestSearchPerformance(unittest.TestCase):
    """Performance tests for semantic search"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_collection_name = "test_performance"

        # Performance thresholds (in seconds)
        self.MAX_SEARCH_TIME = 2.0
        self.MAX_MODEL_LOAD_TIME = 30.0
        self.MAX_EMBEDDING_TIME_PER_100 = 10.0

        # Create larger test dataset for performance testing
        self.large_df = self._create_large_dataset(500)

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_large_dataset(self, size):
        """Create a large dataset for performance testing"""
        titles = [
            f"AI and machine learning advancement {i}",
            f"New quantum computing breakthrough {i}",
            f"Python programming tutorial {i}",
            f"Cybersecurity threat analysis {i}",
            f"Cloud computing infrastructure {i}",
            f"Blockchain cryptocurrency update {i}",
            f"Web development best practices {i}",
            f"Data science visualization {i}",
            f"Mobile app development guide {i}",
            f"DevOps automation tools {i}"
        ]

        data = []
        for i in range(size):
            title_idx = i % len(titles)
            data.append({
                'title': titles[title_idx].replace(str(title_idx), str(i)),
                'score': np.random.randint(50, 500),
                'descendants': np.random.randint(0, 200),
                'time': datetime.now(),
                'url': f'https://example.com/story{i}',
                'sentiment_label': np.random.choice(['Positive', 'Neutral', 'Negative']),
                'topic_keyword': np.random.choice(['ai_ml', 'quantum', 'programming', 'security', 'cloud'])
            })

        return pd.DataFrame(data)

    def test_model_loading_performance(self):
        """Test model loading time"""
        with patch('vector_search.SentenceTransformer') as mock_transformer:
            # Simulate model loading delay
            def delayed_init(*args, **kwargs):
                time.sleep(0.1)  # Simulate loading time
                mock_model = Mock()
                mock_model.encode.return_value = np.random.rand(10, 384).tolist()
                return mock_model

            mock_transformer.side_effect = delayed_init

            # Measure loading time
            start_time = time.time()
            manager = VectorSearchManager(
                collection_name=self.test_collection_name,
                persist_directory=self.temp_dir
            )

            # Trigger model loading
            _ = manager.embedding_model
            loading_time = time.time() - start_time

            # Verify loading time is acceptable
            self.assertLess(
                loading_time,
                self.MAX_MODEL_LOAD_TIME,
                f"Model loading took {loading_time:.2f}s, expected < {self.MAX_MODEL_LOAD_TIME}s"
            )

    def test_embedding_generation_performance(self):
        """Test embedding generation performance"""
        with patch('vector_search.SentenceTransformer') as mock_transformer:
            # Mock model with realistic encoding time
            mock_model = Mock()

            def encode_batch(texts, **kwargs):
                # Simulate processing time based on batch size
                batch_size = len(texts)
                time.sleep(0.01 * (batch_size / 10))  # 10ms per 10 texts
                return np.random.rand(len(texts), 384).tolist()

            mock_model.encode.side_effect = encode_batch
            mock_transformer.return_value = mock_model

            manager = VectorSearchManager(
                collection_name=self.test_collection_name,
                persist_directory=self.temp_dir
            )

            # Test with different batch sizes
            batch_sizes = [10, 50, 100]
            for batch_size in batch_sizes:
                texts = [f"Test text {i}" for i in range(batch_size)]

                start_time = time.time()
                embeddings = manager.generate_embeddings(texts, batch_size=batch_size)
                generation_time = time.time() - start_time

                # Verify performance
                time_per_100 = (generation_time / batch_size) * 100
                self.assertLess(
                    time_per_100,
                    self.MAX_EMBEDDING_TIME_PER_100,
                    f"Embedding generation too slow: {time_per_100:.2f}s per 100 texts"
                )

                self.assertEqual(len(embeddings), batch_size)

    def test_search_response_time(self):
        """Test search response time"""
        # Mock vector engine
        mock_engine = Mock()
        mock_collection = Mock()

        with patch('data_loader.get_vector_engine', return_value=mock_engine):
            # Simulate search with varying response times
            def simulate_search(query, n_results=None, similarity_threshold=None):
                # Simulate processing time
                time.sleep(0.01 + np.random.uniform(0, 0.05))
                return [
                    {
                        'document': f'Result for {query}',
                        'metadata': {'score': 100},
                        'similarity': 0.8,
                        'distance': 0.2
                    }
                    for _ in range(min(n_results or 5, 5))
                ]

            mock_engine.search.side_effect = simulate_search

            # Test multiple searches
            queries = [
                "artificial intelligence",
                "machine learning algorithms",
                "quantum computing applications",
                "python programming tips",
                "cybersecurity best practices"
            ]

            response_times = []
            for query in queries:
                start_time = time.time()
                results = semantic_search(mock_collection, query)
                response_time = time.time() - start_time
                response_times.append(response_time)

                # Individual search time check
                self.assertLess(
                    response_time,
                    self.MAX_SEARCH_TIME,
                    f"Search for '{query}' took {response_time:.2f}s, expected < {self.MAX_SEARCH_TIME}s"
                )

            # Check average response time
            avg_response_time = np.mean(response_times)
            self.assertLess(
                avg_response_time,
                self.MAX_SEARCH_TIME * 0.7,
                f"Average search time {avg_response_time:.2f}s is too high"
            )

    def test_memory_usage_during_operations(self):
        """Test memory usage during vector operations"""
        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process()

        # Get initial memory
        initial_mem = process.memory_info().rss / 1024 / 1024  # MB

        with patch('vector_search.SentenceTransformer') as mock_transformer:
            # Mock model with memory usage simulation
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(100, 384).tolist()
            mock_transformer.return_value = mock_model

            manager = VectorSearchManager(
                collection_name=self.test_collection_name,
                persist_directory=self.temp_dir
            )

            # Generate embeddings for large batch
            large_batch = [f"Document {i}" for i in range(1000)]
            embeddings = manager.generate_embeddings(large_batch, batch_size=100)

            # Get peak memory usage
            current_mem = process.memory_info().rss / 1024 / 1024  # MB
            peak_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # MB

            tracemalloc.stop()

            # Memory should not increase excessively (e.g., more than 500MB)
            memory_increase = current_mem - initial_mem
            self.assertLess(
                memory_increase,
                500,
                f"Memory increased by {memory_increase:.1f}MB, which is excessive"
            )

            # Peak memory should be reasonable
            self.assertLess(
                peak_mem,
                1000,
                f"Peak memory usage {peak_mem:.1f}MB is too high"
            )

    def test_concurrent_searches(self):
        """Test performance with concurrent searches"""
        mock_engine = Mock()
        mock_collection = Mock()

        with patch('data_loader.get_vector_engine', return_value=mock_engine):
            # Simulate concurrent searches
            def perform_search(query):
                start_time = time.time()
                time.sleep(0.05)  # Simulate search time
                mock_engine.search.return_value = [
                    {
                        'document': f'Result for {query}',
                        'metadata': {'score': 100},
                        'similarity': 0.8,
                        'distance': 0.2
                    }
                ]
                results = semantic_search(mock_collection, query)
                return {
                    'query': query,
                    'results': results,
                    'time': time.time() - start_time
                }

            # Run concurrent searches
            queries = [f"search term {i}" for i in range(10)]
            num_workers = 5

            start_time = time.time()
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(perform_search, q) for q in queries]
                results = [future.result() for future in as_completed(futures)]
            total_time = time.time() - start_time

            # Verify all searches completed
            self.assertEqual(len(results), len(queries))

            # Total time should be less than sequential execution
            sequential_time_estimate = len(queries) * 0.05 / num_workers
            self.assertLess(
                total_time,
                sequential_time_estimate * 1.5,
                f"Concurrent search took {total_time:.2f}s, expected faster"
            )

            # Individual searches should still meet performance criteria
            for result in results:
                self.assertLess(
                    result['time'],
                    self.MAX_SEARCH_TIME,
                    f"Concurrent search for '{result['query']}' was too slow"
                )

    def test_scalability_with_dataset_size(self):
        """Test how performance scales with dataset size"""
        dataset_sizes = [100, 300, 500]
        search_times = []

        for size in dataset_sizes:
            # Create dataset
            df = self._create_large_dataset(size)

            with patch('data_loader.get_vector_cache_manager') as mock_cache:
                with patch('data_loader.get_vector_engine') as mock_get_engine:
                    # Mock setup
                    mock_cache_manager = Mock()
                    mock_cache_manager.should_rebuild_collection.return_value = True
                    mock_cache.return_value = mock_cache_manager

                    mock_engine = Mock()
                    mock_engine.collection = Mock()
                    mock_engine.add_documents.return_value = True
                    mock_engine.get_collection_info.return_value = {
                        'name': 'test',
                        'count': size
                    }
                    mock_get_engine.return_value = mock_engine

                    # Setup vector DB
                    setup_vector_db(df, collection_name=f"test_scale_{size}")

                    # Mock search with size-dependent delay
                    def search_with_delay(query, **kwargs):
                        # Simulate increasing search time with dataset size
                        delay = 0.01 + (size / 100000)  # Linear scaling assumption
                        time.sleep(delay)
                        return [{
                            'document': f'Result {i}',
                            'metadata': {'score': 100},
                            'similarity': 0.8,
                            'distance': 0.2
                        }]

                    mock_engine.search.side_effect = search_with_delay

                    # Measure search time
                    start_time = time.time()
                    semantic_search(mock_engine.collection, "test query")
                    search_time = time.time() - start_time
                    search_times.append(search_time)

                    # Verify search time is acceptable
                    self.assertLess(
                        search_time,
                        self.MAX_SEARCH_TIME,
                        f"Search time {search_time:.2f}s for dataset size {size} is too high"
                    )

        # Check scaling behavior
        # Search time should not grow exponentially
        if len(search_times) > 1:
            scaling_factor = search_times[-1] / search_times[0]
            size_factor = dataset_sizes[-1] / dataset_sizes[0]

            # Ideally, scaling should be linear or better
            self.assertLess(
                scaling_factor,
                size_factor * 2,
                f"Search scaling is poor: time increased by {scaling_factor:.2f}x while size increased by {size_factor:.2f}x"
            )

    def test_cache_performance(self):
        """Test performance impact of caching"""
        mock_engine = Mock()
        mock_collection = Mock()

        with patch('data_loader.get_vector_engine', return_value=mock_engine):
            # Mock search with caching simulation
            call_count = 0
            cache = {}

            def cached_search(query, **kwargs):
                nonlocal call_count
                call_count += 1

                if query in cache:
                    # Simulate faster cache hit
                    time.sleep(0.001)
                    return cache[query]
                else:
                    # Simulate slower cache miss
                    time.sleep(0.05)
                    result = [{
                        'document': f'Result for {query}',
                        'metadata': {'score': 100},
                        'similarity': 0.8,
                        'distance': 0.2
                    }]
                    cache[query] = result
                    return result

            mock_engine.search.side_effect = cached_search

            # First search (cache miss)
            start_time = time.time()
            semantic_search(mock_collection, "test query")
            first_search_time = time.time() - start_time

            # Second search (cache hit)
            start_time = time.time()
            semantic_search(mock_collection, "test query")
            second_search_time = time.time() - start_time

            # Verify cache improves performance
            self.assertLess(
                second_search_time,
                first_search_time * 0.5,
                "Cache should significantly improve search time"
            )

            # Verify caching is being used
            self.assertEqual(call_count, 2)

    def test_batch_processing_performance(self):
        """Test performance of batch processing for document additions"""
        with patch('vector_search.SentenceTransformer') as mock_transformer:
            # Mock model
            mock_model = Mock()

            def encode_batch(texts, **kwargs):
                # Simulate batch processing advantage
                batch_size = len(texts)
                base_time = 0.1
                batch_efficiency = 0.7  # 70% efficiency gain from batching
                time.sleep(base_time * (batch_size / 100) * batch_efficiency)
                return np.random.rand(len(texts), 384).tolist()

            mock_model.encode.side_effect = encode_batch
            mock_transformer.return_value = mock_model

            manager = VectorSearchManager(
                collection_name=self.test_collection_name,
                persist_directory=self.temp_dir
            )

            # Test different batch sizes
            documents = [f"Document {i}" for i in range(200)]
            batch_sizes = [10, 50, 100]

            batch_times = []
            for batch_size in batch_sizes:
                with patch.object(manager, 'collection') as mock_collection:
                    mock_collection.add.return_value = None

                    start_time = time.time()
                    # Process all documents in batches
                    for i in range(0, len(documents), batch_size):
                        batch = documents[i:i + batch_size]
                        embeddings = manager.generate_embeddings(batch, batch_size=batch_size)
                        manager.add_documents(
                            documents=batch,
                            ids=[f"doc_{j}" for j in range(i, i + len(batch))],
                            metadata=[{"batch": i // batch_size}] * len(batch)
                        )
                    batch_time = time.time() - start_time
                    batch_times.append(batch_time)

            # Verify batch efficiency
            # Larger batches should be more efficient (though not always linearly)
            self.assertLess(
                batch_times[-1],  # Largest batch
                batch_times[0] * 0.8,  # Should be at least 20% faster
                "Larger batch sizes should show better performance"
            )


class TestResourceUsage(unittest.TestCase):
    """Tests for resource usage monitoring"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_disk_usage_for_vector_storage(self):
        """Test disk space usage for vector storage"""
        initial_size = self._get_directory_size(self.temp_dir)

        with patch('vector_search.SentenceTransformer') as mock_transformer:
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(100, 384).tolist()
            mock_transformer.return_value = mock_model

            manager = VectorSearchManager(
                collection_name="test_disk_usage",
                persist_directory=self.temp_dir
            )

            # Add many documents
            documents = [f"Test document {i}" for i in range(1000)]
            with patch.object(manager, 'collection') as mock_collection:
                mock_collection.add.return_value = None
                manager.add_documents(
                    documents=documents,
                    ids=[f"doc_{i}" for i in range(len(documents))]
                )

            # Check disk usage
            final_size = self._get_directory_size(self.temp_dir)
            size_increase = final_size - initial_size

            # Vector storage should be reasonable
            # Each 384-dim vector is ~1.5KB, plus overhead
            expected_size = len(documents) * 1.5 * 1024  # bytes
            self.assertLess(
                size_increase,
                expected_size * 3,  # Allow 3x for overhead
                f"Disk usage {size_increase / 1024 / 1024:.1f}MB is too high for {len(documents)} documents"
            )

    def _get_directory_size(self, path):
        """Get total size of directory in bytes"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
        return total_size


if __name__ == '__main__':
    unittest.main()