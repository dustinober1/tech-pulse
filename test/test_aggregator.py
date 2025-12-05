"""
Test suite for Multi-source Aggregator functionality.

Tests the parameter validation and fetch_all_sources method
to ensure proper handling of input parameters.
"""

import unittest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.phase7.source_connectors.aggregator import MultiSourceAggregator


class TestMultiSourceAggregator(unittest.TestCase):
    """Test cases for MultiSourceAggregator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.aggregator = MultiSourceAggregator()

    def test_fetch_all_sources_parameters(self):
        """Test that fetch_all_sources handles parameters correctly."""
        # Since fetch_all_sources is async, we need to run it in an event loop
        async def run_test():
            # Mock the logger and internal fetch methods
            with patch('src.phase7.source_connectors.aggregator.logger') as mock_logger:
                self.aggregator._fetch_reddit_content = AsyncMock(return_value=[])
                self.aggregator._fetch_rss_content = AsyncMock(return_value=[])
                self.aggregator._fetch_twitter_content = AsyncMock(return_value=[])

                # Test with valid parameters
                result = await self.aggregator.fetch_all_sources(
                    rss_categories=['tech'],
                    reddit_subreddits=['technology'],
                    twitter_keywords=['#AI']
                )

                # Verify result structure
                self.assertIsInstance(result, dict)
                self.assertIn('rss', result)
                self.assertIn('reddit', result)
                self.assertIn('twitter', result)

                # Verify no NameError occurs
                self.assertFalse(any('error' in str(source).lower() for source in result.keys()))

        # Run the async test
        asyncio.run(run_test())

    def test_fetch_all_sources_default_parameters(self):
        """Test that fetch_all_sources uses default parameters when None is passed."""
        async def run_test():
            # Mock the logger and internal fetch methods
            with patch('src.phase7.source_connectors.aggregator.logger') as mock_logger:
                self.aggregator._fetch_reddit_content = AsyncMock(return_value=[])
                self.aggregator._fetch_rss_content = AsyncMock(return_value=[])
                self.aggregator._fetch_twitter_content = AsyncMock(return_value=[])

                # Test with None parameters (should use defaults)
                result = await self.aggregator.fetch_all_sources(
                    rss_categories=None,
                    reddit_subreddits=None,
                    twitter_keywords=None
                )

                # Verify defaults were used by checking that all sources were called
                self.aggregator._fetch_reddit_content.assert_called_once()
                self.aggregator._fetch_rss_content.assert_called_once()
                self.aggregator._fetch_twitter_content.assert_called_once()

                # Verify result structure
                self.assertIsInstance(result, dict)

        # Run the async test
        asyncio.run(run_test())

    def test_parameter_validation_logs(self):
        """Test that parameter validation logs the correct values."""
        async def run_test():
            # Mock the logger and internal methods
            with patch('src.phase7.source_connectors.aggregator.logger') as mock_logger:
                # Mock the internal methods to avoid actual network calls
                self.aggregator._fetch_reddit_content = AsyncMock(return_value=[])
                self.aggregator._fetch_rss_content = AsyncMock(return_value=[])
                self.aggregator._fetch_twitter_content = AsyncMock(return_value=[])

                # Test with custom parameters
                await self.aggregator.fetch_all_sources(
                    rss_categories=['python', 'ai'],
                    reddit_subreddits=['learnpython'],
                    twitter_keywords=['#python', '#programming']
                )

                # Verify logger was called at least once
                self.assertTrue(mock_logger.info.called)

                # Get the first call (the one with our parameters)
                actual_call = mock_logger.info.call_args_list[0][0][0]

                # Verify it contains our parameters
                self.assertIn("['python', 'ai']", actual_call)
                self.assertIn("['learnpython']", actual_call)
                self.assertIn("['#python', '#programming']", actual_call)

        # Run the async test
        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()