"""
Multi-Source Aggregator for Phase 7.3: Multi-Source Integration

This module provides functionality to aggregate content from multiple sources
(Reddit, RSS feeds, Twitter) into a unified stream for analysis.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
import hashlib
import re
import logging
from collections import defaultdict
import json

from .reddit_connector import RedditConnector
from .rss_connector import RSSConnector
from .twitter_connector import TwitterConnector

logger = logging.getLogger(__name__)


class MultiSourceAggregator:
    """
    Aggregates content from multiple sources into a unified stream.

    Features:
    - Fetch content from Reddit, RSS, and Twitter
    - Deduplicate across sources
    - Content scoring and ranking
    - Trending topic detection
    - Custom filtering and weighting
    """

    def __init__(
        self,
        reddit_config: Optional[Dict[str, Any]] = None,
        rss_config: Optional[Dict[str, Any]] = None,
        twitter_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the multi-source aggregator.

        Args:
            reddit_config: Configuration for Reddit connector
            rss_config: Configuration for RSS connector
            twitter_config: Configuration for Twitter connector
        """
        # Initialize connectors
        self.reddit = RedditConnector(**(reddit_config or {}))
        self.rss = RSSConnector(**(rss_config or {}))
        self.twitter = TwitterConnector(**(twitter_config or {}))

        # Aggregation settings
        self.source_weights = {
            'reddit': 1.0,
            'rss': 0.9,
            'twitter': 0.8
        }

        # Content filters
        self.min_content_length = 20
        self.max_content_length = 10000
        self.banned_domains = set()
        self.banned_keywords = set(['spam', 'advertisement'])

        # Cache for deduplication
        self.seen_hashes: Set[str] = set()
        self.seen_urls: Set[str] = set()

        # Trending topics cache
        self.trending_cache: Dict[str, Tuple[List[Dict[str, Any]], datetime]] = {}
        self.trending_cache_ttl = 3600  # 1 hour

    async def fetch_all_sources(
        self,
        reddit_subreddits: Optional[List[str]] = None,
        rss_categories: Optional[List[str]] = None,
        twitter_keywords: Optional[List[str]] = None,
        hours_ago: int = 24,
        max_items_per_source: int = 50
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch content from all configured sources concurrently.

        Args:
            reddit_subreddits: List of subreddits to fetch from
            rss_categories: List of RSS categories to fetch
            twitter_keywords: List of keywords for Twitter search
            hours_ago: How many hours back to look
            max_items_per_source: Maximum items per source

        Returns:
            Dictionary with content from each source
        """
        logger.info("Starting multi-source content fetch...")

        # Create fetch tasks
        tasks = []

        # Reddit fetch task
        if reddit_subreddits:
            task = self._fetch_reddit_content(reddit_subreddits, max_items_per_source, hours_ago)
            tasks.append(('reddit', task))

        # RSS fetch task
        if rss_categories:
            task = self._fetch_rss_content(rss_categories, max_items_per_source, hours_ago)
            tasks.append(('rss', task))

        # Twitter fetch task
        if twitter_keywords:
            task = self._fetch_twitter_content(twitter_keywords, max_items_per_source, hours_ago)
            tasks.append(('twitter', task))

        # Execute all fetch tasks concurrently
        results = {}
        if tasks:
            completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

            for (source, _), result in zip(tasks, completed_tasks):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching from {source}: {str(result)}")
                    results[source] = []
                else:
                    results[source] = result

        logger.info(f"Fetched content from {len(results)} sources")
        return results

    async def _fetch_reddit_content(
        self,
        subreddits: List[str],
        max_items: int,
        hours_ago: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch content from Reddit.

        Args:
            subreddits: List of subreddits to fetch
            max_items: Maximum items to fetch
            hours_ago: Hours back to look

        Returns:
            List of Reddit content items
        """
        all_posts = []
        items_per_subreddit = max_items // len(subreddits)

        for subreddit in subreddits:
            # For now, use mock data since Reddit API requires authentication
            posts = await self.reddit.get_mock_reddit_data()
            all_posts.extend(posts)

        return all_posts[:max_items]

    async def _fetch_rss_content(
        self,
        categories: List[str],
        max_items: int,
        hours_ago: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch content from RSS feeds.

        Args:
            categories: List of RSS categories
            max_items: Maximum items to fetch
            hours_ago: Hours back to look

        Returns:
            List of RSS content items
        """
        # For now, use mock data
        items = await self.rss.get_mock_rss_data()
        return items[:max_items]

    async def _fetch_twitter_content(
        self,
        keywords: List[str],
        max_items: int,
        hours_ago: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch content from Twitter.

        Args:
            keywords: List of keywords to search
            max_items: Maximum items to fetch
            hours_ago: Hours back to look

        Returns:
            List of Twitter content items
        """
        # For now, use mock data
        items = await self.twitter.get_mock_twitter_data()
        return items[:max_items]

    async def aggregate_content(
        self,
        source_data: Dict[str, List[Dict[str, Any]]],
        deduplicate: bool = True,
        apply_filters: bool = True,
        calculate_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Aggregate content from multiple sources into a unified stream.

        Args:
            source_data: Content from different sources
            deduplicate: Whether to remove duplicate content
            apply_filters: Whether to apply content filters
            calculate_scores: Whether to calculate relevance scores

        Returns:
            Aggregated and processed content list
        """
        logger.info("Starting content aggregation...")

        # Normalize content from all sources
        normalized_items = []
        for source, items in source_data.items():
            for item in items:
                normalized = self._normalize_content(item, source)
                if normalized:
                    normalized_items.append(normalized)

        logger.info(f"Normalized {len(normalized_items)} items from all sources")

        # Apply filters
        if apply_filters:
            filtered_items = self._apply_filters(normalized_items)
            logger.info(f"After filtering: {len(filtered_items)} items")
        else:
            filtered_items = normalized_items

        # Deduplicate
        if deduplicate:
            deduplicated_items = self._deduplicate(filtered_items)
            logger.info(f"After deduplication: {len(deduplicated_items)} items")
        else:
            deduplicated_items = filtered_items

        # Calculate scores
        if calculate_scores:
            scored_items = self._calculate_scores(deduplicated_items)
            logger.info("Calculated relevance scores for all items")
        else:
            scored_items = deduplicated_items

        # Sort by score and timestamp
        scored_items.sort(
            key=lambda x: (x.get('relevance_score', 0), x.get('published', datetime.min)),
            reverse=True
        )

        logger.info(f"Final aggregated content: {len(scored_items)} items")
        return scored_items

    def _normalize_content(self, item: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """
        Normalize content from different sources to a common format.

        Args:
            item: Original content item
            source: Source name ('reddit', 'rss', 'twitter')

        Returns:
            Normalized content dictionary or None if invalid
        """
        try:
            # Extract common fields
            if source == 'reddit':
                normalized = {
                    'id': f"reddit_{item.get('id', '')}",
                    'title': item.get('title', ''),
                    'content': item.get('selftext', '') or item.get('title', ''),
                    'url': item.get('permalink', item.get('url', '')),
                    'author': item.get('author', ''),
                    'published': item.get('created_utc', datetime.now(timezone.utc)),
                    'source': 'reddit',
                    'subreddit': item.get('subreddit', ''),
                    'score': item.get('score', 0),
                    'num_comments': item.get('num_comments', 0),
                    'upvote_ratio': item.get('upvote_ratio', 0),
                    'metadata': item.get('metadata', {}),
                    'comments': item.get('comments', [])
                }
            elif source == 'rss':
                normalized = {
                    'id': f"rss_{item.get('id', '')}",
                    'title': item.get('title', ''),
                    'content': item.get('content', item.get('summary', '')),
                    'url': item.get('url', ''),
                    'author': item.get('author', ''),
                    'published': item.get('published', datetime.now(timezone.utc)),
                    'source': 'rss',
                    'source_name': item.get('source', ''),
                    'categories': item.get('categories', []),
                    'read_time': item.get('read_time', 0),
                    'metadata': item.get('metadata', {})
                }
            elif source == 'twitter':
                normalized = {
                    'id': f"twitter_{item.get('id', '')}",
                    'title': item.get('text', '')[:100],  # Truncate for title
                    'content': item.get('text', ''),
                    'url': f"https://twitter.com/{item.get('author', '')}/status/{item.get('id', '')}",
                    'author': item.get('author', ''),
                    'published': item.get('created_at', datetime.now(timezone.utc)),
                    'source': 'twitter',
                    'retweet_count': item.get('retweet_count', 0),
                    'like_count': item.get('like_count', 0),
                    'reply_count': item.get('reply_count', 0),
                    'hashtags': item.get('hashtags', []),
                    'metadata': item.get('metadata', {})
                }
            else:
                logger.warning(f"Unknown source: {source}")
                return None

            # Add common fields
            normalized.update({
                'source_type': source,
                'content_hash': self._generate_content_hash(normalized),
                'engagement_metrics': self._extract_engagement_metrics(normalized),
                'extracted_topics': self._extract_topics(normalized),
                'language': self._detect_language(normalized),
                'sentiment_score': self._calculate_sentiment_score(normalized)
            })

            return normalized

        except Exception as e:
            logger.error(f"Error normalizing content: {str(e)}")
            return None

    def _generate_content_hash(self, item: Dict[str, Any]) -> str:
        """
        Generate a hash for content deduplication.

        Args:
            item: Content item dictionary

        Returns:
            Content hash string
        """
        # Use title + first 200 characters of content
        content_text = item.get('title', '') + item.get('content', '')[:200]
        return hashlib.md5(content_text.encode()).hexdigest()

    def _extract_engagement_metrics(self, item: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract engagement metrics from content.

        Args:
            item: Content item dictionary

        Returns:
            Engagement metrics dictionary
        """
        metrics = {}

        if item.get('source') == 'reddit':
            metrics['score'] = item.get('score', 0)
            metrics['comments'] = item.get('num_comments', 0)
            metrics['upvote_ratio'] = item.get('upvote_ratio', 0)
        elif item.get('source') == 'rss':
            metrics['read_time'] = item.get('read_time', 0)
            metrics['word_count'] = item.get('metadata', {}).get('word_count', 0)
        elif item.get('source') == 'twitter':
            metrics['retweets'] = item.get('retweet_count', 0)
            metrics['likes'] = item.get('like_count', 0)
            metrics['replies'] = item.get('reply_count', 0)

        return metrics

    def _extract_topics(self, item: Dict[str, Any]) -> List[str]:
        """
        Extract topics/keywords from content.

        Args:
            item: Content item dictionary

        Returns:
            List of extracted topics
        """
        topics = []

        # Extract from title and content
        text = (item.get('title', '') + ' ' + item.get('content', '')).lower()

        # Common tech topics
        tech_keywords = [
            'ai', 'machine learning', 'deep learning', 'neural network',
            'python', 'javascript', 'react', 'nodejs', 'docker', 'kubernetes',
            'aws', 'azure', 'gcp', 'cloud', 'devops', 'cybersecurity',
            'blockchain', 'web3', 'cryptocurrency', 'nft', 'metaverse',
            'quantum', 'robotics', 'iot', '5g', 'ar', 'vr', 'api',
            'microservices', 'serverless', 'database', 'sql', 'nosql',
            'mobile', 'ios', 'android', 'flutter', 'react native',
            'testing', 'agile', 'scrum', 'ci/cd', 'git', 'github'
        ]

        for keyword in tech_keywords:
            if keyword in text:
                topics.append(keyword)

        # Extract from hashtags for Twitter
        if item.get('source') == 'twitter':
            hashtags = item.get('hashtags', [])
            topics.extend([tag.lower().lstrip('#') for tag in hashtags])

        # Extract from categories for RSS
        if item.get('source') == 'rss':
            categories = item.get('categories', [])
            topics.extend([cat.lower() for cat in categories])

        return list(set(topics))  # Remove duplicates

    def _detect_language(self, item: Dict[str, Any]) -> str:
        """
        Detect content language.

        Args:
            item: Content item dictionary

        Returns:
            Language code (e.g., 'en')
        """
        # Simple detection based on metadata or default to English
        metadata = item.get('metadata', {})
        return metadata.get('language', 'en')

    def _calculate_sentiment_score(self, item: Dict[str, Any]) -> float:
        """
        Calculate sentiment score (-1 to 1).

        Args:
            item: Content item dictionary

        Returns:
            Sentiment score
        """
        # Simple keyword-based sentiment (would use NLP in production)
        text = (item.get('title', '') + ' ' + item.get('content', '')).lower()

        positive_words = ['good', 'great', 'amazing', 'excellent', 'perfect', 'love', 'best', 'awesome']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disaster', 'fail']

        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        return (pos_count - neg_count) / total

    def _apply_filters(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply content filters.

        Args:
            items: List of content items

        Returns:
            Filtered list of items
        """
        filtered_items = []

        for item in items:
            # Content length filter
            content_length = len(item.get('content', ''))
            if content_length < self.min_content_length or content_length > self.max_content_length:
                continue

            # Domain filter
            url = item.get('url', '')
            if any(domain in url for domain in self.banned_domains):
                continue

            # Keyword filter
            text = (item.get('title', '') + ' ' + item.get('content', '')).lower()
            if any(keyword in text for keyword in self.banned_keywords):
                continue

            filtered_items.append(item)

        return filtered_items

    def _deduplicate(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate content items.

        Args:
            items: List of content items

        Returns:
            Deduplicated list of items
        """
        seen_hashes = set()
        seen_urls = set()
        deduplicated_items = []

        for item in items:
            content_hash = item.get('content_hash', '')
            url = item.get('url', '')

            # Skip if seen
            if content_hash in seen_hashes or url in seen_urls:
                continue

            # Add to seen sets
            seen_hashes.add(content_hash)
            seen_urls.add(url)

            # Add to deduplicated list
            deduplicated_items.append(item)

        return deduplicated_items

    def _calculate_scores(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate relevance scores for content items.

        Args:
            items: List of content items

        Returns:
            Items with calculated scores
        """
        for item in items:
            score = self._calculate_item_score(item)
            item['relevance_score'] = score

        return items

    def _calculate_item_score(self, item: Dict[str, Any]) -> float:
        """
        Calculate relevance score for a single item.

        Args:
            item: Content item dictionary

        Returns:
            Relevance score (0-1)
        """
        score = 0.0

        # Base score from source weight
        source_weight = self.source_weights.get(item.get('source'), 1.0)
        score += source_weight * 0.3

        # Engagement score
        engagement = self._calculate_engagement_score(item)
        score += engagement * 0.4

        # Recency score
        recency = self._calculate_recency_score(item)
        score += recency * 0.2

        # Topic relevance
        topics = item.get('extracted_topics', [])
        topic_score = min(len(topics) * 0.1, 0.1)
        score += topic_score

        # Sentiment boost (positive content gets slight boost)
        sentiment = item.get('sentiment_score', 0)
        sentiment_boost = max(0, sentiment) * 0.1
        score += sentiment_boost

        return min(score, 1.0)

    def _calculate_engagement_score(self, item: Dict[str, Any]) -> float:
        """
        Calculate normalized engagement score.

        Args:
            item: Content item dictionary

        Returns:
            Engagement score (0-1)
        """
        metrics = item.get('engagement_metrics', {})

        if item.get('source') == 'reddit':
            score = min((metrics.get('score', 0) / 1000 + metrics.get('comments', 0) / 100), 1)
        elif item.get('source') == 'twitter':
            score = min((metrics.get('retweets', 0) / 100 + metrics.get('likes', 0) / 1000), 1)
        else:
            score = 0.5  # Default for RSS without explicit metrics

        return score

    def _calculate_recency_score(self, item: Dict[str, Any]) -> float:
        """
        Calculate recency score based on publication time.

        Args:
            item: Content item dictionary

        Returns:
            Recency score (0-1)
        """
        published = item.get('published', datetime.min)
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)

        age_hours = (datetime.now(timezone.utc) - published).total_seconds() / 3600

        # Decay over 24 hours
        if age_hours < 1:
            return 1.0
        elif age_hours < 24:
            return 1.0 - (age_hours / 24) * 0.5
        else:
            return 0.5

    async def get_trending_topics(
        self,
        hours_ago: int = 24,
        min_mentions: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Identify trending topics across all sources.

        Args:
            hours_ago: Time window to analyze
            min_mentions: Minimum mentions to consider trending

        Returns:
            List of trending topics with metrics
        """
        cache_key = f"trending_{hours_ago}_{min_mentions}"

        # Check cache
        if cache_key in self.trending_cache:
            cached_data, cached_time = self.trending_cache[cache_key]
            if (datetime.now(timezone.utc) - cached_time).total_seconds() < self.trending_cache_ttl:
                return cached_data

        # Fetch recent content
        source_data = await self.fetch_all_sources(
            reddit_subreddits=['technology', 'programming', 'MachineLearning'],
            rss_categories=['ai', 'development'],
            twitter_keywords=['AI', 'tech', 'programming'],
            hours_ago=hours_ago,
            max_items_per_source=100
        )

        # Aggregate content
        content = await self.aggregate_content(source_data)

        # Extract topic frequencies
        topic_counts = defaultdict(list)
        for item in content:
            for topic in item.get('extracted_topics', []):
                topic_counts[topic].append({
                    'item_id': item['id'],
                    'source': item['source'],
                    'score': item.get('relevance_score', 0),
                    'published': item['published']
                })

        # Calculate trending metrics
        trending_topics = []
        for topic, mentions in topic_counts.items():
            if len(mentions) >= min_mentions:
                # Calculate metrics
                total_score = sum(m['score'] for m in mentions)
                source_diversity = len(set(m['source'] for m in mentions))
                avg_score = total_score / len(mentions)

                trending_topics.append({
                    'topic': topic,
                    'mention_count': len(mentions),
                    'total_engagement': total_score,
                    'source_diversity': source_diversity,
                    'average_score': avg_score,
                    'recent_mentions': len([m for m in mentions if (datetime.now(timezone.utc) - m['published']).total_seconds() < 3600]),
                    'sources': list(set(m['source'] for m in mentions))
                })

        # Sort by trending score (weighted by mentions, diversity, and recency)
        trending_topics.sort(
            key=lambda x: (
                x['mention_count'] * 0.4 +
                x['source_diversity'] * 0.3 +
                x['average_score'] * 0.2 +
                x['recent_mentions'] * 0.1
            ),
            reverse=True
        )

        # Cache results
        self.trending_cache[cache_key] = (trending_topics, datetime.now(timezone.utc))

        logger.info(f"Identified {len(trending_topics)} trending topics")
        return trending_topics

    async def generate_daily_digest(
        self,
        hours_ago: int = 24,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Generate a daily digest of top content.

        Args:
            hours_ago: Time window for digest
            top_n: Number of top items to include

        Returns:
            Daily digest dictionary
        """
        # Fetch and aggregate content
        source_data = await self.fetch_all_sources(
            reddit_subreddits=['technology', 'programming', 'MachineLearning', 'webdev'],
            rss_categories=['news', 'ai', 'development', 'security'],
            twitter_keywords=['tech', 'AI', 'programming', 'cybersecurity'],
            hours_ago=hours_ago,
            max_items_per_source=50
        )

        content = await self.aggregate_content(source_data)

        # Get top content
        top_content = content[:top_n]

        # Get trending topics
        trending = await self.get_trending_topics(hours_ago=hours_ago, min_mentions=3)

        # Generate summary statistics
        stats = {
            'total_items': len(content),
            'source_breakdown': {
                'reddit': len([c for c in content if c['source'] == 'reddit']),
                'rss': len([c for c in content if c['source'] == 'rss']),
                'twitter': len([c for c in content if c['source'] == 'twitter'])
            },
            'top_topics': [t['topic'] for t in trending[:5]],
            'avg_sentiment': sum(c.get('sentiment_score', 0) for c in content) / len(content) if content else 0
        }

        digest = {
            'generated_at': datetime.now(timezone.utc),
            'time_window_hours': hours_ago,
            'summary': stats,
            'top_content': top_content,
            'trending_topics': trending,
            'highlights': self._generate_highlights(content)
        }

        return digest

    def _generate_highlights(self, content: List[Dict[str, Any]]) -> List[str]:
        """
        Generate highlights from content.

        Args:
            content: List of content items

        Returns:
            List of highlight strings
        """
        highlights = []

        # Highest scored item
        if content:
            top_item = max(content, key=lambda x: x.get('relevance_score', 0))
            highlights.append(f"Top story: {top_item['title']} ({top_item['source']})")

        # Most discussed on Reddit
        reddit_items = [c for c in content if c['source'] == 'reddit']
        if reddit_items:
            most_discussed = max(reddit_items, key=lambda x: x.get('num_comments', 0))
            if most_discussed.get('num_comments', 0) > 100:
                highlights.append(f"Heated discussion: {most_discussed['title']} ({most_discussed['num_comments']} comments)")

        # Most shared on Twitter
        twitter_items = [c for c in content if c['source'] == 'twitter']
        if twitter_items:
            most_shared = max(twitter_items, key=lambda x: x.get('retweet_count', 0))
            if most_shared.get('retweet_count', 0) > 100:
                highlights.append(f"Viral on Twitter: {most_shared['title']} ({most_shared['retweet_count']} retweets)")

        return highlights