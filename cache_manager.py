"""
Simple cache manager for Tech-Pulse application.
Provides caching functionality for Hacker News stories.
"""

import json
import os
import time
import hashlib
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching of Hacker News stories data."""

    def __init__(self, cache_dir: str = "cache", cache_duration_hours: int = 1):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files
            cache_duration_hours: How long cache entries remain valid
        """
        self.cache_dir = cache_dir
        self.cache_duration = timedelta(hours=cache_duration_hours)

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_files(self, cache_key: str) -> tuple:
        """
        Get cache file paths for a given cache key.

        Args:
            cache_key: The cache key

        Returns:
            Tuple of (metadata_file_path, data_file_path)
        """
        base_filename = f"stories_{cache_key}"
        metadata_file = os.path.join(self.cache_dir, f"{base_filename}.json")
        data_file = os.path.join(self.cache_dir, f"{base_filename}.parquet")
        return metadata_file, data_file

    def _generate_cache_key(self, limit: int, analysis_type: str = "basic") -> str:
        """
        Generate a unique cache key based on parameters.

        Args:
            limit: Number of stories requested
            analysis_type: Type of analysis performed

        Returns:
            MD5 hash as cache key
        """
        # Note: Remove date from cache key to allow same-day caching with different analysis types
        key_data = f"{limit}_{analysis_type}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _is_cache_valid(self, metadata: Dict[str, Any]) -> bool:
        """
        Check if cached data is still valid.

        Args:
            metadata: Cache metadata dictionary

        Returns:
            True if cache is valid, False otherwise
        """
        if not metadata:
            return False

        cache_time = datetime.fromtimestamp(metadata.get('timestamp', 0))
        return datetime.now() - cache_time < self.cache_duration

    def get_cached_data(self, limit: int, analysis_type: str = "basic") -> Optional[pd.DataFrame]:
        """
        Retrieve cached data if available and valid.

        Args:
            limit: Number of stories requested
            analysis_type: Type of analysis performed

        Returns:
            Cached DataFrame or None if not available/invalid
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(limit, analysis_type)
            metadata_file, data_file = self._get_cache_files(cache_key)

            # Check if cache files exist
            if not os.path.exists(metadata_file) or not os.path.exists(data_file):
                logger.info("Cache files not found")
                return None

            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Check if cache is valid
            if not self._is_cache_valid(metadata):
                logger.info("Cache has expired")
                return None

            # Load cached data
            df = pd.read_parquet(data_file)
            logger.info(f"Successfully loaded {len(df)} stories from cache")
            return df

        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return None

    def save_to_cache(self, df: pd.DataFrame, limit: int, analysis_type: str = "basic") -> None:
        """
        Save data to cache.

        Args:
            df: DataFrame to cache
            limit: Number of stories requested
            analysis_type: Type of analysis performed
        """
        try:
            if df.empty:
                logger.warning("Attempted to cache empty DataFrame")
                return

            # Generate cache key and file paths
            cache_key = self._generate_cache_key(limit, analysis_type)
            metadata_file, data_file = self._get_cache_files(cache_key)

            # Create metadata
            metadata = {
                'timestamp': time.time(),
                'limit': limit,
                'analysis_type': analysis_type,
                'stories_count': len(df),
                'columns': list(df.columns),
                'cache_key': cache_key
            }

            # Save DataFrame
            df.to_parquet(data_file, index=False)

            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Successfully cached {len(df)} stories")

        except Exception as e:
            logger.error(f"Error saving to cache: {e}")

    def clear_cache(self) -> None:
        """Clear all cached data."""
        try:
            # Remove all cache files matching our pattern
            for filename in os.listdir(self.cache_dir):
                if filename.startswith("stories_") and (filename.endswith(".json") or filename.endswith(".parquet")):
                    file_path = os.path.join(self.cache_dir, filename)
                    os.remove(file_path)
                    logger.info(f"Removed cache file: {filename}")

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about current cache state.

        Returns:
            Dictionary with cache information
        """
        info = {
            'cache_dir': self.cache_dir,
            'cache_duration_hours': self.cache_duration.total_seconds() / 3600,
            'cache_entries': []
        }

        try:
            # Look for all cache files
            for filename in os.listdir(self.cache_dir):
                if filename.startswith("stories_") and filename.endswith(".json"):
                    metadata_file = os.path.join(self.cache_dir, filename)
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    entry = {
                        'limit': metadata.get('limit', 0),
                        'analysis_type': metadata.get('analysis_type', 'unknown'),
                        'stories_count': metadata.get('stories_count', 0),
                        'cache_valid': self._is_cache_valid(metadata),
                        'cache_age_hours': None,
                        'cache_key': metadata.get('cache_key', 'unknown')
                    }

                    cache_time = datetime.fromtimestamp(metadata.get('timestamp', 0))
                    entry['cache_age_hours'] = (datetime.now() - cache_time).total_seconds() / 3600

                    info['cache_entries'].append(entry)

        except Exception as e:
            logger.error(f"Error getting cache info: {e}")

        return info