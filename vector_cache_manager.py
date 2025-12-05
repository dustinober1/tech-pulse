"""
Vector cache management for Streamlit session state integration.
Provides caching layer for ChromaDB collections with version control.
"""

import hashlib
import json
import time
from typing import Optional, Any, Dict, List
import logging

logger = logging.getLogger(__name__)


class VectorCacheManager:
    """
    Manages caching of vector collections in Streamlit session state.
    """

    def __init__(self, cache_duration: int = 3600):
        """
        Initialize the vector cache manager.

        Args:
            cache_duration: Cache duration in seconds (default: 1 hour)
        """
        self.cache_duration = cache_duration
        self.version_key = "vector_cache_version"
        self.collection_key = "vector_collection"
        self.timestamp_key = "vector_cache_timestamp"
        self.data_hash_key = "vector_data_hash"

    def get_session_state(self) -> Dict[str, Any]:
        """
        Get Streamlit session state safely.
        Returns a dict-like object even if Streamlit is not available.
        """
        try:
            import streamlit as st
            return st.session_state
        except ImportError:
            # Return a simple dict for non-Streamlit usage
            return {}

    def generate_data_hash(self, df_data: List[Dict]) -> str:
        """
        Generate a hash for the data to detect changes.

        Args:
            df_data: List of story data dictionaries

        Returns:
            MD5 hash representing the data
        """
        # Create a normalized representation
        normalized_data = []
        for item in df_data:
            # Use only the fields that matter for embeddings
            normalized_item = {
                'title': str(item.get('title', '')).strip(),
                'score': int(item.get('score', 0)),
                'time': str(item.get('time', ''))
            }
            normalized_data.append(normalized_item)

        # Sort to ensure consistent hash regardless of order
        normalized_data.sort(key=lambda x: x['title'])

        # Generate hash
        data_str = json.dumps(normalized_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    def is_cache_valid(self, data_hash: Optional[str] = None) -> bool:
        """
        Check if the cached collection is still valid.

        Args:
            data_hash: Current data hash to compare against

        Returns:
            True if cache is valid, False otherwise
        """
        session_state = self.get_session_state()

        # Check if cache exists
        if self.collection_key not in session_state:
            return False

        # Check timestamp
        if self.timestamp_key not in session_state:
            return False

        cache_age = time.time() - session_state[self.timestamp_key]
        if cache_age > self.cache_duration:
            logger.info("Vector cache expired due to age")
            return False

        # Check data hash if provided
        if data_hash is not None:
            if self.data_hash_key not in session_state:
                return False
            if session_state[self.data_hash_key] != data_hash:
                logger.info("Vector cache expired due to data change")
                return False

        return True

    def get_cached_collection(self) -> Optional[Any]:
        """
        Get cached collection if available and valid.

        Returns:
            Cached ChromaDB collection or None
        """
        session_state = self.get_session_state()

        if not self.is_cache_valid():
            return None

        collection = session_state.get(self.collection_key)
        if collection is not None:
            logger.info("Using cached vector collection")
        return collection

    def cache_collection(self, collection: Any, data_hash: str) -> None:
        """
        Cache a collection in session state.

        Args:
            collection: ChromaDB collection to cache
            data_hash: Hash of the data for version control
        """
        session_state = self.get_session_state()

        try:
            session_state[self.collection_key] = collection
            session_state[self.timestamp_key] = time.time()
            session_state[self.data_hash_key] = data_hash

            # Increment version
            current_version = session_state.get(self.version_key, 0)
            session_state[self.version_key] = current_version + 1

            logger.info(f"Vector collection cached with version {session_state[self.version_key]}")
        except Exception as e:
            logger.error(f"Error caching vector collection: {e}")

    def clear_cache(self) -> None:
        """Clear the vector cache from session state."""
        session_state = self.get_session_state()

        keys_to_remove = [
            self.collection_key,
            self.timestamp_key,
            self.data_hash_key
        ]

        for key in keys_to_remove:
            if key in session_state:
                del session_state[key]

        logger.info("Vector cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the current cache state.

        Returns:
            Dictionary with cache information
        """
        session_state = self.get_session_state()

        info = {
            'has_collection': self.collection_key in session_state,
            'is_valid': self.is_cache_valid(),
            'version': session_state.get(self.version_key, 0)
        }

        if self.timestamp_key in session_state:
            cache_age = time.time() - session_state[self.timestamp_key]
            info['cache_age_seconds'] = cache_age
            info['cache_age_minutes'] = cache_age / 60
            info['cache_expires_in'] = max(0, self.cache_duration - cache_age)

        if self.data_hash_key in session_state:
            info['data_hash'] = session_state[self.data_hash_key]

        return info

    def should_rebuild_collection(self, df_data: List[Dict], force_rebuild: bool = False) -> bool:
        """
        Determine if the vector collection should be rebuilt.

        Args:
            df_data: Current DataFrame data as list of dictionaries
            force_rebuild: Force rebuild regardless of cache state

        Returns:
            True if collection should be rebuilt, False otherwise
        """
        if force_rebuild:
            logger.info("Force rebuild requested")
            return True

        # Check if we have valid cache
        data_hash = self.generate_data_hash(df_data)
        if not self.is_cache_valid(data_hash):
            return True

        return False


# Global instance
_vector_cache_manager = None


def get_vector_cache_manager(cache_duration: Optional[int] = None) -> VectorCacheManager:
    """
    Get or create a global vector cache manager instance.

    Args:
        cache_duration: Cache duration in seconds (optional)

    Returns:
        VectorCacheManager instance
    """
    global _vector_cache_manager

    if _vector_cache_manager is None:
        if cache_duration is None:
            # Use default from dashboard config
            from dashboard_config import SEMANTIC_SEARCH_SETTINGS
            cache_duration = SEMANTIC_SEARCH_SETTINGS.get("cache_duration", 3600)
        _vector_cache_manager = VectorCacheManager(cache_duration)

    return _vector_cache_manager