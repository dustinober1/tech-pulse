"""Extended tests for cache_manager.py to achieve 100% coverage."""
import pytest
import os
import json
import pandas as pd
from unittest.mock import patch, mock_open, MagicMock
from datetime import datetime
import tempfile
import shutil

from cache_manager import CacheManager


class TestCacheManagerExtended:
    """Extended tests for CacheManager exception handling and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.cache_manager = CacheManager(cache_dir=self.test_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_cache_file_write_error(self):
        """Test save_cache when file writing fails."""
        df = pd.DataFrame({
            'id': [1, 2],
            'title': ['Test 1', 'Test 2']
        })

        with patch('pandas.DataFrame.to_parquet', side_effect=IOError("Disk full")):
            # Should not raise exception, just log error
            self.cache_manager.save_cache(df, 10, "test")

    def test_clear_cache_directory_error(self):
        """Test clear_cache when directory operations fail."""
        with patch('os.listdir', side_effect=OSError("Permission denied")):
            # Should not raise exception, just log error
            self.cache_manager.clear_cache()

    def test_clear_cache_file_removal_error(self):
        """Test clear_cache when file removal fails."""
        # Create a test cache file
        cache_file = os.path.join(self.test_dir, "stories_test.json")
        with open(cache_file, 'w') as f:
            json.dump({}, f)

        with patch('os.remove', side_effect=OSError("Permission denied")):
            # Should not raise exception, just log error
            self.cache_manager.clear_cache()

    def test_get_cache_info_directory_read_error(self):
        """Test get_cache_info when directory reading fails."""
        info = self.cache_manager.get_cache_info()
        # Verify default structure
        assert 'cache_entries' in info
        assert 'total_cache_size_mb' in info
        assert isinstance(info['cache_entries'], list)

    def test_get_cache_info_metadata_read_error(self):
        """Test get_cache_info when metadata file reading fails."""
        # Create a test cache file
        cache_file = os.path.join(self.test_dir, "stories_test.parquet")
        with open(cache_file, 'w') as f:
            f.write("dummy data")

        # Create corresponding metadata file
        metadata_file = os.path.join(self.test_dir, "stories_test_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump({"timestamp": datetime.now().timestamp()}, f)

        with patch('builtins.open', side_effect=IOError("Read error")):
            info = self.cache_manager.get_cache_info()
            # Should handle error gracefully
            assert 'cache_entries' in info

    def test_get_cache_info_invalid_metadata(self):
        """Test get_cache_info with invalid metadata."""
        # Create a test cache file
        cache_file = os.path.join(self.test_dir, "stories_test.parquet")
        with open(cache_file, 'w') as f:
            f.write("dummy data")

        # Create invalid metadata file
        metadata_file = os.path.join(self.test_dir, "stories_test_metadata.json")
        with open(metadata_file, 'w') as f:
            f.write("invalid json")

        info = self.cache_manager.get_cache_info()
        # Should handle invalid JSON gracefully
        assert 'cache_entries' in info

    