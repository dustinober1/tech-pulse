"""
Unit tests for cache_manager module.
"""

import unittest
import os
import tempfile
import shutil
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Import the module being tested
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cache_manager import CacheManager


class TestCacheManager(unittest.TestCase):
    """Test cases for CacheManager class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test cache
        self.test_cache_dir = tempfile.mkdtemp()
        self.cache_manager = CacheManager(cache_dir=self.test_cache_dir, cache_duration_hours=1)

        # Create sample test data
        self.test_df = pd.DataFrame({
            'title': ['Test Story 1', 'Test Story 2', 'Test Story 3'],
            'score': [100, 200, 150],
            'descendants': [10, 20, 15],
            'time': [datetime.now(), datetime.now(), datetime.now()],
            'url': ['http://test1.com', 'http://test2.com', 'http://test3.com']
        })

    def tearDown(self):
        """Clean up after each test method."""
        # Remove the temporary cache directory
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)

    def test_initialization(self):
        """Test CacheManager initialization."""
        # Check that cache directory was created
        self.assertTrue(os.path.exists(self.test_cache_dir))

    def test_generate_cache_key(self):
        """Test cache key generation."""
        key1 = self.cache_manager._generate_cache_key(30, "basic")
        key2 = self.cache_manager._generate_cache_key(30, "basic")
        key3 = self.cache_manager._generate_cache_key(50, "basic")
        key4 = self.cache_manager._generate_cache_key(30, "advanced")

        # Same parameters should generate same key
        self.assertEqual(key1, key2)

        # Different parameters should generate different keys
        self.assertNotEqual(key1, key3)
        self.assertNotEqual(key1, key4)

        # Keys should be MD5 hashes (32 characters)
        self.assertEqual(len(key1), 32)

    def test_save_and_load_cache(self):
        """Test saving and loading cache data."""
        # Save data to cache
        self.cache_manager.save_to_cache(self.test_df, limit=3, analysis_type="basic")

        # Get cache file paths
        cache_key = self.cache_manager._generate_cache_key(3, "basic")
        metadata_file, data_file = self.cache_manager._get_cache_files(cache_key)

        # Check that files were created
        self.assertTrue(os.path.exists(metadata_file))
        self.assertTrue(os.path.exists(data_file))

        # Load data from cache
        loaded_df = self.cache_manager.get_cached_data(limit=3, analysis_type="basic")

        # Verify data integrity
        self.assertIsNotNone(loaded_df)
        self.assertEqual(len(loaded_df), 3)
        self.assertEqual(list(loaded_df.columns), list(self.test_df.columns))

        # Compare specific values
        pd.testing.assert_frame_equal(loaded_df.reset_index(drop=True), self.test_df.reset_index(drop=True))

    def test_cache_expiry(self):
        """Test cache expiration functionality."""
        # Create a cache manager with 0 hours duration (immediate expiry)
        expired_cache = CacheManager(cache_dir=self.test_cache_dir, cache_duration_hours=0)

        # Save data
        expired_cache.save_to_cache(self.test_df, limit=3)

        # Try to load immediately (should be expired)
        loaded_df = expired_cache.get_cached_data(limit=3)
        self.assertIsNone(loaded_df)

    def test_cache_miss_conditions(self):
        """Test various cache miss scenarios."""
        # 1. Non-existent cache files
        loaded_df = self.cache_manager.get_cached_data(limit=3)
        self.assertIsNone(loaded_df)

        # 2. Mismatched cache key
        self.cache_manager.save_to_cache(self.test_df, limit=3)
        loaded_df = self.cache_manager.get_cached_data(limit=5)  # Different limit
        self.assertIsNone(loaded_df)

        # 3. Corrupted metadata file
        self.cache_manager.save_to_cache(self.test_df, limit=3)
        cache_key = self.cache_manager._generate_cache_key(3, "basic")
        metadata_file, data_file = self.cache_manager._get_cache_files(cache_key)
        with open(metadata_file, 'w') as f:
            f.write("invalid json")
        loaded_df = self.cache_manager.get_cached_data(limit=3)
        self.assertIsNone(loaded_df)

    def test_clear_cache(self):
        """Test clearing cache functionality."""
        # Save some data
        self.cache_manager.save_to_cache(self.test_df, limit=3)

        # Get cache file paths
        cache_key = self.cache_manager._generate_cache_key(3, "basic")
        metadata_file, data_file = self.cache_manager._get_cache_files(cache_key)

        # Verify files exist
        self.assertTrue(os.path.exists(metadata_file))
        self.assertTrue(os.path.exists(data_file))

        # Clear cache
        self.cache_manager.clear_cache()

        # Verify files are deleted
        self.assertFalse(os.path.exists(metadata_file))
        self.assertFalse(os.path.exists(data_file))

    def test_get_cache_info(self):
        """Test cache information retrieval."""
        # Check info with empty cache
        info = self.cache_manager.get_cache_info()
        self.assertEqual(info['cache_dir'], self.test_cache_dir)
        self.assertEqual(info['cache_duration_hours'], 1.0)
        self.assertEqual(len(info['cache_entries']), 0)

        # Save some data
        self.cache_manager.save_to_cache(self.test_df, limit=3)

        # Check info with cached data
        info = self.cache_manager.get_cache_info()
        self.assertEqual(len(info['cache_entries']), 1)

        entry = info['cache_entries'][0]
        self.assertEqual(entry['limit'], 3)
        self.assertEqual(entry['stories_count'], 3)
        self.assertTrue(entry['cache_valid'])
        self.assertIsNotNone(entry['cache_age_hours'])
        self.assertGreaterEqual(entry['cache_age_hours'], 0)

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()

        # Saving empty DataFrame should not create cache files
        self.cache_manager.save_to_cache(empty_df, limit=0)

        # Get potential cache file paths
        cache_key = self.cache_manager._generate_cache_key(0, "basic")
        metadata_file, data_file = self.cache_manager._get_cache_files(cache_key)

        # Files should not exist
        self.assertFalse(os.path.exists(metadata_file))
        self.assertFalse(os.path.exists(data_file))

    def test_cache_with_different_analysis_types(self):
        """Test caching with different analysis types."""
        # Save data with different analysis types
        self.cache_manager.save_to_cache(self.test_df, limit=3, analysis_type="basic")
        self.cache_manager.save_to_cache(self.test_df, limit=3, analysis_type="advanced")

        # Each should be retrievable with its respective analysis type
        basic_df = self.cache_manager.get_cached_data(limit=3, analysis_type="basic")
        advanced_df = self.cache_manager.get_cached_data(limit=3, analysis_type="advanced")

        self.assertIsNotNone(basic_df)
        self.assertIsNotNone(advanced_df)
        self.assertEqual(len(basic_df), 3)
        self.assertEqual(len(advanced_df), 3)

    def test_error_handling(self):
        """Test error handling in cache operations."""
        # Test with read-only directory (simulate error condition)
        readonly_dir = os.path.join(self.test_cache_dir, "readonly")
        os.makedirs(readonly_dir, exist_ok=True)

        # Save to normal cache first
        self.cache_manager.save_to_cache(self.test_df, limit=3)

        # Try loading with non-existent parameters
        loaded_df = self.cache_manager.get_cached_data(limit=999)  # Different limit
        self.assertIsNone(loaded_df)

        # Get cache info should still work
        info = self.cache_manager.get_cache_info()
        self.assertIsInstance(info, dict)

    @patch('cache_manager.logger')
    def test_logging(self, mock_logger):
        """Test that appropriate logging occurs."""
        # Test successful operations
        self.cache_manager.save_to_cache(self.test_df, limit=3)
        loaded_df = self.cache_manager.get_cached_data(limit=3)

        # Check that info messages were logged
        mock_logger.info.assert_called()

        # Test warning for empty DataFrame
        empty_df = pd.DataFrame()
        self.cache_manager.save_to_cache(empty_df, limit=0)

        # Check that warning was logged
        mock_logger.warning.assert_called_with("Attempted to cache empty DataFrame")


if __name__ == '__main__':
    # Create test results directory if it doesn't exist
    test_results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_results')
    os.makedirs(test_results_dir, exist_ok=True)

    # Run tests with verbosity
    unittest.main(verbosity=2)