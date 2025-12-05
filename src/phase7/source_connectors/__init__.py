"""
Source Connectors Module for Phase 7.3: Multi-Source Integration

This module provides connectors for various data sources including:
- Reddit posts and comments
- RSS feeds
- Twitter/X posts (bonus feature)
- Aggregated multi-source content collection
"""

from .reddit_connector import RedditConnector
from .rss_connector import RSSConnector
from .aggregator import MultiSourceAggregator
from .twitter_connector import TwitterConnector

__all__ = [
    'RedditConnector',
    'RSSConnector',
    'MultiSourceAggregator',
    'TwitterConnector'
]