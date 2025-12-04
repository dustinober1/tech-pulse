"""
Intelligent caching system for Tech-Pulse data.
Reduces API calls and improves performance.
"""

import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import hashlib

# Cache configuration
CACHE_DIR = "cache"
CACHE_EXPIRY_MINUTES = 5  # Default cache expiry time

def get_cache_files(cache_dir: str = CACHE_DIR) -> tuple:
    """Get cache file paths for given directory"""
    return (
        os.path.join(cache_dir, "stories_cache.json"),
        os.path.join(cache_dir, "stories_data.parquet")
    )

class CacheManager:
    """Manages caching of Hacker News stories and analysis results."""

    def __init__(self, cache_dir: str = CACHE_DIR, expiry_minutes: int = CACHE_EXPIRY_MINUTES):
        self.cache_dir = cache_dir
        self.expiry_minutes = expiry_minutes
        self.ensure_cache_dir()
        self.story_cache_file, self.story_data_file = get_cache_files(cache_dir)

    def ensure_cache_dir(self):
        """Ensure cache directory exists."""
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_key(self, limit: int, analysis_type: str = "basic") -> str:
        """Generate cache key based on parameters."""
        key_data = f"stories_{limit}_{analysis_type}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def is_cache_valid(self, cache_file: str) -> bool:
        """Check if cache file exists and is not expired."""
        if not os.path.exists(cache_file):
            return False

        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        expiry_time = datetime.now() - timedelta(minutes=self.expiry_minutes)

        return file_time > expiry_time

    def get_story_cache_info(self) -> Dict[str, Any]:
        """Get information about cached stories."""
        if not os.path.exists(self.story_cache_file):
            return {
                "exists": False,
                "stories_count": 0,
                "last_updated": None,
                "cache_age_minutes": None,
                "is_valid": False,
                "limit": 0
            }

        try:
            with open(self.story_cache_file, 'r') as f:
                cache_info = json.load(f)

            cache_time = datetime.fromtimestamp(cache_info['timestamp'])
            age_minutes = (datetime.now() - cache_time).total_seconds() / 60
            is_valid = age_minutes < self.expiry_minutes

            return {
                "exists": True,
                "stories_count": cache_info.get('stories_count', 0),
                "last_updated": cache_time,
                "cache_age_minutes": round(age_minutes, 1),
                "is_valid": is_valid,
                "expiry_minutes": self.expiry_minutes,
                "limit": cache_info.get('limit', 0)
            }
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            return {
                "exists": False,
                "stories_count": 0,
                "last_updated": None,
                "cache_age_minutes": None,
                "is_valid": False,
                "limit": 0
            }

    def cache_stories(self, df: pd.DataFrame, limit: int, analysis_type: str = "basic"):
        """Cache stories dataframe to disk."""
        cache_info = {
            "timestamp": time.time(),
            "limit": limit,
            "analysis_type": analysis_type,
            "stories_count": len(df),
            "columns": list(df.columns),
            "cache_key": self.get_cache_key(limit, analysis_type)
        }

        # Save metadata
        with open(self.story_cache_file, 'w') as f:
            json.dump(cache_info, f, indent=2, default=str)

        # Save actual data
        df.to_parquet(self.story_data_file, index=False)

    def load_cached_stories(self, limit: int, analysis_type: str = "basic") -> Optional[pd.DataFrame]:
        """Load cached stories if valid."""
        # Load cache metadata
        if not os.path.exists(self.story_cache_file):
            return None

        with open(self.story_cache_file, 'r') as f:
            cache_info = json.load(f)

        # Check validity
        cache_time = datetime.fromtimestamp(cache_info['timestamp'])
        expiry_time = datetime.now() - timedelta(minutes=self.expiry_minutes)

        if cache_time <= expiry_time:
            return None

        if cache_info.get("limit") != limit:
            return None

        if not os.path.exists(self.story_data_file):
            return None

        try:
            df = pd.read_parquet(self.story_data_file)
            return df
        except Exception as e:
            print(f"Error loading cached data: {e}")
            return None

    def clear_cache(self):
        """Clear all cached data."""
        files_to_remove = [self.story_cache_file, self.story_data_file]

        for file_path in files_to_remove:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        cache_info = self.get_story_cache_info()

        # Get cache directory size
        cache_size = 0
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    cache_size += os.path.getsize(file_path)

        return {
            "cache_info": cache_info,
            "cache_directory": self.cache_dir,
            "cache_size_bytes": cache_size,
            "cache_size_mb": round(cache_size / (1024 * 1024), 2),
            "expiry_minutes": self.expiry_minutes,
            "cache_files": os.listdir(self.cache_dir) if os.path.exists(self.cache_dir) else []
        }

    def update_expiry(self, new_expiry_minutes: int):
        """Update cache expiry time."""
        self.expiry_minutes = new_expiry_minutes

    def force_refresh_needed(self, limit: int) -> bool:
        """Check if a force refresh is needed."""
        cache_info = self.get_story_cache_info()

        if not cache_info["exists"]:
            return True

        if not cache_info["is_valid"]:
            return True

        if cache_info["limit"] != limit:
            return True

        return False

# Global cache manager instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get or create global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

def clear_cache():
    """Clear all cache."""
    get_cache_manager().clear_cache()

def get_cache_statistics() -> Dict[str, Any]:
    """Get cache statistics."""
    return get_cache_manager().get_cache_stats()

def update_cache_expiry(minutes: int):
    """Update cache expiry time."""
    get_cache_manager().update_expiry(minutes)