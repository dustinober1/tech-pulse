"""
Test suite for Tech-Pulse cache manager.
"""

import unittest
import sys
import os
import tempfile
import shutil
import json
import time
from datetime import datetime, timedelta
import pandas as pd

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache_manager import CacheManager, get_cache_manager, clear_cache, get_cache_statistics


class TestCacheManager(unittest.TestCase):
    """Test cache manager functionality"""

    def setUp(self):
        """Set up test environment"""
        # Create temporary cache directory
        self.test_cache_dir = tempfile.mkdtemp()
        self.cache_manager = CacheManager(cache_dir=self.test_cache_dir, expiry_minutes=1)

        # Create test data
        self.test_data = pd.DataFrame({
            'title': ['Test Story 1', 'Test Story 2', 'Test Story 3'],
            'score': [100, 200, 150],
            'descendants': [10, 20, 15],
            'time': [datetime.now()] * 3,
            'url': ['http://test1.com', 'http://test2.com', 'http://test3.com']
        })

    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)

    def test_cache_initialization(self):
        """Test cache manager initialization"""
        self.assertEqual(self.cache_manager.cache_dir, self.test_cache_dir)
        self.assertEqual(self.cache_manager.expiry_minutes, 1)
        self.assertTrue(os.path.exists(self.test_cache_dir))

    def test_cache_key_generation(self):
        """Test cache key generation"""
        key1 = self.cache_manager.get_cache_key(30, "basic")
        key2 = self.cache_manager.get_cache_key(30, "basic")
        key3 = self.cache_manager.get_cache_key(50, "basic")
        key4 = self.cache_manager.get_cache_key(30, "advanced")

        # Same parameters should generate same key
        self.assertEqual(key1, key2)

        # Different parameters should generate different keys
        self.assertNotEqual(key1, key3)
        self.assertNotEqual(key1, key4)

    def test_cache_validity_check(self):
        """Test cache validity checking"""
        # Non-existent file should be invalid
        self.assertFalse(self.cache_manager.is_cache_valid("nonexistent.json"))

        # Create a test cache file
        test_file = os.path.join(self.test_cache_dir, "test.json")
        with open(test_file, 'w') as f:
            f.write("{}")

        # Fresh file should be valid
        self.assertTrue(self.cache_manager.is_cache_valid(test_file))

        # Wait for expiry
        time.sleep(1.1)  # Wait longer than expiry time (1 minute = 1 second for test)

        # Expired file should be invalid
        self.assertFalse(self.cache_manager.is_cache_valid(test_file))

    def test_store_and_load_cache(self):
        """Test storing and loading cached data"""
        limit = 30
        analysis_type = "basic"

        # Store data in cache
        self.cache_manager.cache_stories(self.test_data, limit, analysis_type)

        # Load data from cache
        loaded_data = self.cache_manager.load_cached_stories(limit, analysis_type)

        # Verify data integrity
        self.assertIsNotNone(loaded_data)
        self.assertEqual(len(loaded_data), len(self.test_data))
        pd.testing.assert_frame_equal(loaded_data, self.test_data)

    def test_cache_limit_mismatch(self):
        """Test that cache is not loaded when limit doesn't match"""
        limit = 30
        analysis_type = "basic"

        # Store data with limit=30
        self.cache_manager.cache_stories(self.test_data, limit, analysis_type)

        # Try to load with different limit
        loaded_data = self.cache_manager.load_cached_stories(50, analysis_type)

        # Should return None due to limit mismatch
        self.assertIsNone(loaded_data)

    def test_cache_expiry(self):
        """Test cache expiry functionality"""
        limit = 30
        analysis_type = "basic"

        # Store data in cache
        self.cache_manager.cache_stories(self.test_data, limit, analysis_type)

        # Load immediately should work
        loaded_data = self.cache_manager.load_cached_stories(limit, analysis_type)
        self.assertIsNotNone(loaded_data)

        # Wait for expiry
        time.sleep(1.1)  # Wait longer than expiry time

        # Load after expiry should return None
        loaded_data = self.cache_manager.load_cached_stories(limit, analysis_type)
        self.assertIsNone(loaded_data)

    def test_cache_info_retrieval(self):
        """Test cache information retrieval"""
        limit = 25
        analysis_type = "basic"

        # Initially should show no cache
        cache_info = self.cache_manager.get_story_cache_info()
        self.assertFalse(cache_info["exists"])
        self.assertEqual(cache_info["stories_count"], 0)

        # Store data
        self.cache_manager.cache_stories(self.test_data, limit, analysis_type)

        # Should now show cache info
        cache_info = self.cache_manager.get_story_cache_info()
        self.assertTrue(cache_info["exists"])
        self.assertEqual(cache_info["stories_count"], len(self.test_data))
        self.assertEqual(cache_info["limit"], limit)
        self.assertTrue(cache_info["is_valid"])

    def test_cache_clearing(self):
        """Test cache clearing functionality"""
        limit = 30
        analysis_type = "basic"

        # Store data
        self.cache_manager.cache_stories(self.test_data, limit, analysis_type)

        # Verify cache exists
        cache_info = self.cache_manager.get_story_cache_info()
        self.assertTrue(cache_info["exists"])

        # Clear cache
        self.cache_manager.clear_cache()

        # Verify cache is cleared
        cache_info = self.cache_manager.get_story_cache_info()
        self.assertFalse(cache_info["exists"])

    def test_cache_statistics(self):
        """Test cache statistics retrieval"""
        limit = 30
        analysis_type = "basic"

        # Initially should have minimal stats
        stats = self.cache_manager.get_cache_stats()
        self.assertFalse(stats["cache_info"]["exists"])
        self.assertEqual(stats["cache_size_bytes"], 0)

        # Store data
        self.cache_manager.cache_stories(self.test_data, limit, analysis_type)

        # Should have meaningful stats
        stats = self.cache_manager.get_cache_stats()
        self.assertTrue(stats["cache_info"]["exists"])
        self.assertGreater(stats["cache_size_bytes"], 0)
        self.assertGreater(stats["cache_size_mb"], 0)
        self.assertIn("cache_files", stats)

    def test_force_refresh_needed(self):
        """Test force refresh detection"""
        limit = 30

        # Initially should need refresh
        self.assertTrue(self.cache_manager.force_refresh_needed(limit))

        # Store data
        self.cache_manager.cache_stories(self.test_data, limit, "basic")

        # Should not need refresh with matching limit
        self.assertFalse(self.cache_manager.force_refresh_needed(limit))

        # Should need refresh with different limit
        self.assertTrue(self.cache_manager.force_refresh_needed(limit + 10))

    def test_expiry_update(self):
        """Test updating cache expiry time"""
        # Default expiry should be 1 minute
        self.assertEqual(self.cache_manager.expiry_minutes, 1)

        # Update expiry time
        self.cache_manager.update_expiry(5)
        self.assertEqual(self.cache_manager.expiry_minutes, 5)


class TestCacheManagerGlobalFunctions(unittest.TestCase):
    """Test global cache manager functions"""

    def setUp(self):
        """Set up test environment"""
        # Clear any existing cache
        clear_cache()

    def test_get_cache_manager(self):
        """Test global cache manager instance"""
        manager1 = get_cache_manager()
        manager2 = get_cache_manager()

        # Should return same instance
        self.assertIs(manager1, manager2)
        self.assertIsInstance(manager1, CacheManager)

    def test_get_cache_statistics(self):
        """Test global cache statistics"""
        stats = get_cache_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn("cache_info", stats)
        self.assertIn("cache_size_mb", stats)

    def test_clear_cache(self):
        """Test global cache clearing"""
        # Should not raise an exception
        clear_cache()
        stats = get_cache_statistics()
        self.assertFalse(stats["cache_info"]["exists"])


if __name__ == '__main__':
    unittest.main()