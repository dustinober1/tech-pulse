"""
Unit Tests for Multi-Source Integration (Phase 7.3)

This module contains comprehensive tests for the multi-source connectors
including Reddit, RSS, Twitter, and the aggregator.
"""

import unittest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from phase7.source_connectors import (
    RedditConnector,
    RSSConnector,
    TwitterConnector,
    MultiSourceAggregator
)


class TestRedditConnector(unittest.TestCase):
    """Test cases for RedditConnector"""

    def setUp(self):
        """Set up test fixtures"""
        self.reddit = RedditConnector()

    def test_init(self):
        """Test RedditConnector initialization"""
        self.assertEqual(self.reddit.user_agent, "tech-pulse-v1.0")
        self.assertIsNotNone(self.reddit.tech_subreddits)
        self.assertIn('technology', self.reddit.tech_subreddits)
        self.assertIn('MachineLearning', self.reddit.tech_subreddits)

    async def test_get_mock_reddit_data(self):
        """Test fetching mock Reddit data"""
        data = await self.reddit.get_mock_reddit_data()

        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

        for item in data:
            self.assertIn('id', item)
            self.assertIn('title', item)
            self.assertIn('author', item)
            self.assertIn('source', item)
            self.assertEqual(item['source'], 'reddit')
            self.assertIn('comments', item)
            self.assertIsInstance(item['comments'], list)

    def test_calculate_engagement_rate(self):
        """Test engagement rate calculation"""
        # Create mock post
        post = {
            'score': 1000,
            'num_comments': 100,
            'upvote_ratio': 0.9,
            'created_utc': datetime.now(timezone.utc) - timedelta(hours=1)
        }

        engagement = self.reddit._calculate_engagement_rate(post)
        self.assertGreaterEqual(engagement, 0)
        self.assertLessEqual(engagement, 1)

        # Test older post
        old_post = {
            'score': 1000,
            'num_comments': 100,
            'upvote_ratio': 0.9,
            'created_utc': datetime.now(timezone.utc) - timedelta(hours=10)
        }

        old_engagement = self.reddit._calculate_engagement_rate(old_post)
        self.assertLess(old_engagement, engagement)  # Older post should have lower engagement


class TestRSSConnector(unittest.TestCase):
    """Test cases for RSSConnector"""

    def setUp(self):
        """Set up test fixtures"""
        self.rss = RSSConnector()

    def test_init(self):
        """Test RSSConnector initialization"""
        self.assertEqual(self.rss.user_agent, "tech-pulse-rss-reader/1.0")
        self.assertEqual(self.rss.request_timeout, 30)
        self.assertIsNotNone(self.rss.tech_feeds)
        self.assertGreater(len(self.rss.tech_feeds), 0)

    async def test_get_mock_rss_data(self):
        """Test fetching mock RSS data"""
        data = await self.rss.get_mock_rss_data()

        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

        for item in data:
            self.assertIn('id', item)
            self.assertIn('title', item)
            self.assertIn('url', item)
            self.assertIn('content', item)
            self.assertIn('source', item)
            self.assertIn('published', item)
            self.assertIn('categories', item)
            self.assertIn('read_time', item)
            self.assertIsInstance(item['read_time'], int)
            self.assertGreater(item['read_time'], 0)

    def test_clean_content(self):
        """Test content cleaning"""
        html_content = "<p>This is <b>bold</b> text with <a href='link'>links</a>.</p>"
        cleaned = self.rss._clean_content(html_content)

        self.assertNotIn('<', cleaned)
        self.assertNotIn('>', cleaned)
        self.assertIn('bold', cleaned)
        self.assertIn('text', cleaned)

    def test_extract_summary(self):
        """Test summary extraction"""
        content = "This is a long article with many words. " * 20

        summary = self.rss._extract_summary(content, max_length=100)
        self.assertLessEqual(len(summary), 100 + 10)  # Allow some buffer
        self.assertIn('...', summary or '')

    def test_estimate_read_time(self):
        """Test read time estimation"""
        content = "word " * 400  # 400 words
        read_time = self.rss._estimate_read_time(content)
        self.assertEqual(read_time, 2)  # 400 words / 200 wpm = 2 minutes

    def test_calculate_relevance(self):
        """Test relevance calculation"""
        item = {
            'title': 'Python Programming Tutorial',
            'content': 'This article covers Python programming basics',
            'published': datetime.now(timezone.utc),
            'categories': ['Programming', 'Python']
        }

        relevance = self.rss._calculate_relevance(item, 'python')
        self.assertGreaterEqual(relevance, 0)
        self.assertLessEqual(relevance, 1)


class TestTwitterConnector(unittest.TestCase):
    """Test cases for TwitterConnector"""

    def setUp(self):
        """Set up test fixtures"""
        self.twitter = TwitterConnector()

    def test_init(self):
        """Test TwitterConnector initialization"""
        self.assertIsNotNone(self.twitter.tech_accounts)
        self.assertIsNotNone(self.twitter.tech_hashtags)
        self.assertIn('elonmusk', self.twitter.tech_accounts)
        self.assertIn('#AI', self.twitter.tech_hashtags)

    async def test_get_mock_twitter_data(self):
        """Test fetching mock Twitter data"""
        data = await self.twitter.get_mock_twitter_data()

        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

        for item in data:
            self.assertIn('id', item)
            self.assertIn('text', item)
            self.assertIn('author', item)
            self.assertIn('created_at', item)
            self.assertIn('source', item)
            self.assertEqual(item['source'], 'twitter')
            self.assertIn('metrics', item)
            self.assertIn('hashtags', item)
            self.assertIn('retweet_count', item)
            self.assertIn('like_count', item)

    def test_parse_twitter_date(self):
        """Test Twitter date parsing"""
        date_str = "2024-03-15T10:30:00.000Z"
        parsed = self.twitter._parse_twitter_date(date_str)

        self.assertIsInstance(parsed, datetime)
        self.assertEqual(parsed.year, 2024)
        self.assertEqual(parsed.month, 3)
        self.assertEqual(parsed.day, 15)

    def test_extract_hashtags(self):
        """Test hashtag extraction"""
        entities = {
            'hashtags': [
                {'tag': 'AI'},
                {'tag': 'MachineLearning'}
            ]
        }

        hashtags = self.twitter._extract_hashtags(entities)
        self.assertEqual(len(hashtags), 2)
        self.assertIn('AI', hashtags)
        self.assertIn('MachineLearning', hashtags)

    def test_calculate_tweet_engagement(self):
        """Test tweet engagement calculation"""
        tweet = {
            'metrics': {
                'like_count': 1000,
                'retweet_count': 500,
                'reply_count': 100,
                'quote_count': 50
            },
            'verified': False
        }

        engagement = self.twitter._calculate_tweet_engagement(tweet)
        self.assertGreaterEqual(engagement, 0)
        self.assertLessEqual(engagement, 1)

        # Test verified account
        verified_tweet = {
            'metrics': {
                'like_count': 1000,
                'retweet_count': 500,
                'reply_count': 100,
                'quote_count': 50
            },
            'verified': True
        }

        verified_engagement = self.twitter._calculate_tweet_engagement(verified_tweet)
        self.assertGreater(verified_engagement, engagement)


class TestMultiSourceAggregator(unittest.TestCase):
    """Test cases for MultiSourceAggregator"""

    def setUp(self):
        """Set up test fixtures"""
        self.aggregator = MultiSourceAggregator()

    def test_init(self):
        """Test aggregator initialization"""
        self.assertIsNotNone(self.aggregator.reddit)
        self.assertIsNotNone(self.aggregator.rss)
        self.assertIsNotNone(self.aggregator.twitter)
        self.assertIn('reddit', self.aggregator.source_weights)
        self.assertIn('rss', self.aggregator.source_weights)
        self.assertIn('twitter', self.aggregator.source_weights)

    async def test_fetch_all_sources(self):
        """Test fetching from all sources"""
        # Note: This would use mock data since no API credentials
        results = await self.aggregator.fetch_all_sources(
            reddit_subreddits=['technology'],
            rss_categories=['news'],
            twitter_keywords=['AI']
        )

        self.assertIsInstance(results, dict)
        self.assertIn('reddit', results)
        self.assertIn('rss', results)
        self.assertIn('twitter', results)

    async def test_normalize_content(self):
        """Test content normalization"""
        # Test Reddit content
        reddit_item = {
            'id': 'test123',
            'title': 'Test Reddit Post',
            'selftext': 'This is a test post',
            'author': 'testuser',
            'created_utc': datetime.now(timezone.utc),
            'score': 100,
            'num_comments': 10,
            'subreddit': 'technology'
        }

        normalized = self.aggregator._normalize_content(reddit_item, 'reddit')
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized['title'], 'Test Reddit Post')
        self.assertEqual(normalized['source'], 'reddit')
        self.assertIn('content_hash', normalized)
        self.assertIn('engagement_metrics', normalized)

        # Test RSS content
        rss_item = {
            'id': 'rss123',
            'title': 'Test RSS Article',
            'content': 'This is test RSS content',
            'author': 'RSS Author',
            'published': datetime.now(timezone.utc),
            'source': 'Test RSS Feed',
            'categories': ['Technology']
        }

        normalized = self.aggregator._normalize_content(rss_item, 'rss')
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized['title'], 'Test RSS Article')
        self.assertEqual(normalized['source'], 'rss')

        # Test Twitter content
        twitter_item = {
            'id': 'tweet123',
            'text': 'Test tweet content #AI',
            'author': 'twitteruser',
            'created_at': datetime.now(timezone.utc),
            'public_metrics': {
                'retweet_count': 10,
                'like_count': 50
            },
            'hashtags': [{'tag': 'AI'}]
        }

        normalized = self.aggregator._normalize_content(twitter_item, 'twitter')
        self.assertIsNotNone(normalized)
        self.assertIn('Test tweet', normalized['title'])
        self.assertEqual(normalized['source'], 'twitter')

    def test_generate_content_hash(self):
        """Test content hash generation"""
        item = {
            'title': 'Test Title',
            'content': 'Test content for hashing'
        }

        hash1 = self.aggregator._generate_content_hash(item)
        hash2 = self.aggregator._generate_content_hash(item)

        self.assertEqual(hash1, hash2)
        self.assertIsInstance(hash1, str)
        self.assertEqual(len(hash1), 32)  # MD5 hash length

    def test_extract_topics(self):
        """Test topic extraction"""
        item = {
            'title': 'Python AI and Machine Learning Tutorial',
            'content': 'This tutorial covers Python programming for AI development',
            'source': 'rss',
            'categories': ['Python', 'AI']
        }

        topics = self.aggregator._extract_topics(item)
        self.assertIn('python', topics)
        self.assertIn('ai', topics)
        self.assertIn('machine learning', topics)

    def test_calculate_item_score(self):
        """Test item relevance scoring"""
        item = {
            'source': 'reddit',
            'published': datetime.now(timezone.utc),
            'extracted_topics': ['python', 'ai'],
            'sentiment_score': 0.5,
            'engagement_metrics': {
                'score': 1000,
                'comments': 100
            }
        }

        score = self.aggregator._calculate_item_score(item)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_calculate_recency_score(self):
        """Test recency scoring"""
        # Recent item
        recent_item = {
            'published': datetime.now(timezone.utc) - timedelta(minutes=30)
        }
        recent_score = self.aggregator._calculate_recency_score(recent_item)
        self.assertEqual(recent_score, 1.0)

        # Older item
        old_item = {
            'published': datetime.now(timezone.utc) - timedelta(hours=12)
        }
        old_score = self.aggregator._calculate_recency_score(old_item)
        self.assertLess(old_score, recent_score)

    async def test_aggregate_content(self):
        """Test content aggregation"""
        # Create mock source data
        source_data = {
            'reddit': await self.reddit.get_mock_reddit_data()[:2],
            'rss': await self.rss.get_mock_rss_data()[:2],
            'twitter': await self.twitter.get_mock_twitter_data()[:2]
        }

        aggregated = await self.aggregator.aggregate_content(
            source_data,
            deduplicate=True,
            apply_filters=True,
            calculate_scores=True
        )

        self.assertIsInstance(aggregated, list)
        self.assertGreater(len(aggregated), 0)

        # Check that items are sorted by score
        if len(aggregated) > 1:
            self.assertGreaterEqual(
                aggregated[0].get('relevance_score', 0),
                aggregated[1].get('relevance_score', 0)
            )

    async def test_get_trending_topics(self):
        """Test trending topics detection"""
        # This would fetch real data in production
        # For testing, we'll create mock data
        topics = await self.aggregator.get_trending_topics(
            hours_ago=24,
            min_mentions=3
        )

        self.assertIsInstance(topics, list)

        # Note: With mock data, this may return empty
        # In production with real data, we'd expect topics

    async def test_generate_daily_digest(self):
        """Test daily digest generation"""
        digest = await self.aggregator.generate_daily_digest(
            hours_ago=24,
            top_n=5
        )

        self.assertIsInstance(digest, dict)
        self.assertIn('generated_at', digest)
        self.assertIn('time_window_hours', digest)
        self.assertIn('summary', digest)
        self.assertIn('top_content', digest)
        self.assertIn('trending_topics', digest)
        self.assertIn('highlights', digest)

    def test_generate_highlights(self):
        """Test highlights generation"""
        # Create mock content
        content = [
            {
                'title': 'Top story',
                'source': 'reddit',
                'relevance_score': 0.95
            },
            {
                'title': 'Discussion topic',
                'source': 'reddit',
                'num_comments': 500,
                'relevance_score': 0.8
            },
            {
                'title': 'Viral tweet',
                'source': 'twitter',
                'retweet_count': 1000,
                'relevance_score': 0.85
            }
        ]

        highlights = self.aggregator._generate_highlights(content)
        self.assertIsInstance(highlights, list)
        # Should have at least one highlight
        self.assertGreater(len(highlights), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for multi-source functionality"""

    async def test_full_pipeline(self):
        """Test the complete multi-source pipeline"""
        # Initialize aggregator
        aggregator = MultiSourceAggregator()

        # 1. Fetch from all sources
        source_data = await aggregator.fetch_all_sources(
            reddit_subreddits=['technology', 'programming'],
            rss_categories=['news', 'ai'],
            twitter_keywords=['tech', 'programming'],
            hours_ago=24,
            max_items_per_source=10
        )

        # Verify data was fetched
        self.assertIsInstance(source_data, dict)
        total_items = sum(len(items) for items in source_data.values())
        self.assertGreater(total_items, 0)

        # 2. Aggregate content
        aggregated_content = await aggregator.aggregate_content(
            source_data,
            deduplicate=True,
            apply_filters=True,
            calculate_scores=True
        )

        # Verify aggregation
        self.assertIsInstance(aggregated_content, list)
        self.assertGreater(len(aggregated_content), 0)

        # 3. Check content properties
        for item in aggregated_content[:5]:  # Check first 5 items
            self.assertIn('id', item)
            self.assertIn('title', item)
            self.assertIn('source', item)
            self.assertIn('relevance_score', item)
            self.assertIn('published', item)
            self.assertIn('extracted_topics', item)

            # Check score range
            score = item.get('relevance_score', 0)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

        # 4. Get trending topics
        trending = await aggregator.get_trending_topics(hours_ago=24, min_mentions=2)
        self.assertIsInstance(trending, list)

        # 5. Generate daily digest
        digest = await aggregator.generate_daily_digest(hours_ago=24, top_n=10)
        self.assertIsInstance(digest, dict)
        self.assertIn('top_content', digest)
        self.assertIn('summary', digest)

        # Verify digest structure
        summary = digest.get('summary', {})
        self.assertIn('total_items', summary)
        self.assertIn('source_breakdown', summary)

    def test_concurrent_fetching(self):
        """Test that sources can be fetched concurrently"""
        async def run_concurrent_test():
            aggregator = MultiSourceAggregator()

            # Start timer
            start_time = datetime.now()

            # Fetch from multiple sources
            tasks = [
                aggregator.reddit.get_mock_reddit_data(),
                aggregator.rss.get_mock_rss_data(),
                aggregator.twitter.get_mock_twitter_data()
            ]

            results = await asyncio.gather(*tasks)

            # End timer
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Verify results
            self.assertEqual(len(results), 3)
            self.assertIsInstance(results[0], list)  # Reddit data
            self.assertIsInstance(results[1], list)  # RSS data
            self.assertIsInstance(results[2], list)  # Twitter data

            # Verify concurrent execution (should be fast)
            # In real scenario with network calls, this would show performance benefit
            self.assertLess(duration, 5)  # Should complete quickly with mock data

        # Run the async test
        asyncio.run(run_concurrent_test())


def run_all_tests():
    """Run all test cases"""
    # Create test suite
    test_classes = [
        TestRedditConnector,
        TestRSSConnector,
        TestTwitterConnector,
        TestMultiSourceAggregator,
        TestIntegration
    ]

    suite = unittest.TestSuite()

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    print("Running Multi-Source Integration Tests...")
    print("=" * 50)

    success = run_all_tests()

    if success:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Check the output above.")