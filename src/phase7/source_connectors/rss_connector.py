"""
RSS Connector for Phase 7.3: Multi-Source Integration

This module provides functionality to fetch and parse RSS feeds from
tech news sources, blogs, and publications.
"""

import asyncio
import aiohttp
import feedparser
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Set
import hashlib
import logging
from urllib.parse import urljoin, urlparse
import re

logger = logging.getLogger(__name__)


class RSSConnector:
    """
    RSS feed connector for fetching tech news and blog posts.

    Features:
    - Async RSS feed fetching and parsing
    - Support for multiple feed formats (RSS, Atom)
    - Content extraction and cleaning
    - Duplicate detection across feeds
    - Rate limiting and error handling
    """

    def __init__(
        self,
        user_agent: str = "tech-pulse-rss-reader/1.0",
        request_timeout: int = 30,
        max_concurrent_feeds: int = 10
    ):
        """
        Initialize RSS connector.

        Args:
            user_agent: User agent string for HTTP requests
            request_timeout: Timeout for feed requests in seconds
            max_concurrent_feeds: Maximum number of feeds to fetch concurrently
        """
        self.user_agent = user_agent
        self.request_timeout = request_timeout
        self.max_concurrent_feeds = max_concurrent_feeds
        self.session = None
        self.seen_urls: Set[str] = set()
        self.seen_hashes: Set[str] = set()

        # Predefined list of tech RSS feeds
        self.tech_feeds = [
            # Major Tech News
            {
                'name': 'TechCrunch',
                'url': 'https://techcrunch.com/feed/',
                'category': 'news',
                'update_frequency': 30
            },
            {
                'name': 'Hacker News',
                'url': 'https://hnrss.org/frontpage',
                'category': 'news',
                'update_frequency': 15
            },
            {
                'name': 'Ars Technica',
                'url': 'https://feeds.arstechnica.com/arstechnica/index',
                'category': 'news',
                'update_frequency': 60
            },
            {
                'name': 'The Verge',
                'url': 'https://www.theverge.com/rss/index.xml',
                'category': 'news',
                'update_frequency': 45
            },
            {
                'name': 'Wired',
                'url': 'https://www.wired.com/feed/rss',
                'category': 'news',
                'update_frequency': 60
            },
            # Development Blogs
            {
                'name': 'Martin Fowler',
                'url': 'https://martinfowler.com/feed.atom',
                'category': 'development',
                'update_frequency': 7 * 24 * 60  # Weekly
            },
            {
                'name': 'Joel on Software',
                'url': 'https://www.joelonsoftware.com/feed/',
                'category': 'development',
                'update_frequency': 3 * 24 * 60  # Every 3 days
            },
            {
                'name': 'High Scalability',
                'url': 'http://highscalability.com/feed.xml',
                'category': 'architecture',
                'update_frequency': 24 * 60  # Daily
            },
            # AI/ML Specific
            {
                'name': 'OpenAI Blog',
                'url': 'https://openai.com/blog/rss.xml',
                'category': 'ai',
                'update_frequency': 3 * 24 * 60  # Every 3 days
            },
            {
                'name': 'Google AI Blog',
                'url': 'https://ai.googleblog.com/feeds/posts/default',
                'category': 'ai',
                'update_frequency': 24 * 60  # Daily
            },
            {
                'name': 'DeepMind Blog',
                'url': 'https://deepmind.com/blog/feed/basic/',
                'category': 'ai',
                'update_frequency': 3 * 24 * 60  # Every 3 days
            },
            # Cloud/DevOps
            {
                'name': 'AWS Blog',
                'url': 'https://aws.amazon.com/blogs/aws/feed/',
                'category': 'cloud',
                'update_frequency': 24 * 60  # Daily
            },
            {
                'name': 'Kubernetes Blog',
                'url': 'https://kubernetes.io/feed/',
                'category': 'devops',
                'update_frequency': 24 * 60  # Daily
            },
            {
                'name': 'Docker Blog',
                'url': 'https://www.docker.com/blog/feed/',
                'category': 'devops',
                'update_frequency': 24 * 60  # Daily
            },
            # Security
            {
                'name': 'Krebs on Security',
                'url': 'https://krebsonsecurity.com/feed/',
                'category': 'security',
                'update_frequency': 24 * 60  # Daily
            },
            {
                'name': 'The Hacker News',
                'url': 'https://thehackernews.com/feeds/posts/default',
                'category': 'security',
                'update_frequency': 6 * 60  # Every 6 hours
            }
        ]

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.request_timeout),
            headers={'User-Agent': self.user_agent}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def fetch_feed(
        self,
        feed_url: str,
        max_items: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Fetch and parse a single RSS feed.

        Args:
            feed_url: URL of the RSS feed
            max_items: Maximum number of items to return

        Returns:
            List of feed item dictionaries
        """
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.request_timeout),
                headers={'User-Agent': self.user_agent}
            )

        try:
            async with self.session.get(feed_url) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)

                    if feed.bozo:
                        logger.warning(f"Feed parsing warning for {feed_url}: {feed.bozo_exception}")

                    items = []
                    for entry in feed.entries[:max_items]:
                        item = self._parse_feed_entry(entry, feed.feed)
                        if item and not self._is_duplicate(item):
                            items.append(item)

                    logger.info(f"Fetched {len(items)} items from {feed_url}")
                    return items
                else:
                    logger.error(f"Error fetching feed {feed_url}: HTTP {response.status}")
                    return []

        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching feed {feed_url}")
            return []
        except Exception as e:
            logger.error(f"Error fetching feed {feed_url}: {str(e)}")
            return []

    def _parse_feed_entry(self, entry: Any, feed_info: Any) -> Optional[Dict[str, Any]]:
        """
        Parse a single feed entry into a standardized format.

        Args:
            entry: Feedparser entry object
            feed_info: Feed metadata

        Returns:
            Standardized item dictionary or None if invalid
        """
        try:
            # Extract publication date
            published = None
            for date_field in ['published_parsed', 'updated_parsed']:
                if hasattr(entry, date_field) and getattr(entry, date_field):
                    published = datetime(*getattr(entry, date_field)[:6], tzinfo=timezone.utc)
                    break

            if not published:
                published = datetime.now(timezone.utc)

            # Extract content
            content = ""
            if hasattr(entry, 'content'):
                content = entry.content[0].value if entry.content else ""
            elif hasattr(entry, 'summary'):
                content = entry.summary
            elif hasattr(entry, 'description'):
                content = entry.description

            # Clean content
            content = self._clean_content(content)

            # Extract tags
            tags = []
            if hasattr(entry, 'tags'):
                tags = [tag.term for tag in entry.tags if hasattr(tag, 'term')]

            # Extract media (images, videos)
            media = []
            if hasattr(entry, 'media_content'):
                media = entry.media_content
            elif hasattr(entry, 'enclosures'):
                media = entry.enclosures

            # Extract author
            author = ""
            if hasattr(entry, 'author'):
                author = entry.author
            elif hasattr(feed_info, 'author'):
                author = feed_info.author

            item = {
                'id': self._generate_item_id(entry),
                'title': getattr(entry, 'title', 'Untitled'),
                'url': getattr(entry, 'link', ''),
                'content': content,
                'summary': self._extract_summary(content, 200),
                'author': author,
                'published': published,
                'updated': published,
                'source': getattr(feed_info, 'title', 'Unknown Feed'),
                'source_url': getattr(feed_info, 'link', ''),
                'categories': tags,
                'media': self._extract_media(media),
                'read_time': self._estimate_read_time(content),
                'engagement_score': 0,  # Will be calculated later
                'metadata': {
                    'feed_format': getattr(feed_info, 'version', 'unknown'),
                    'language': getattr(entry, 'language', 'en'),
                    'word_count': len(content.split()),
                    'has_image': bool(media),
                    'is_duplicate': False
                }
            }

            return item

        except Exception as e:
            logger.error(f"Error parsing feed entry: {str(e)}")
            return None

    def _clean_content(self, content: str) -> str:
        """
        Clean HTML content from feed entries.

        Args:
            content: Raw HTML content

        Returns:
            Cleaned text content
        """
        if not content:
            return ""

        # Remove HTML tags
        content = re.sub(r'<[^>]+>', '', content)

        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)

        # Remove common patterns
        content = re.sub(r'Read more\.\.\.$', '', content)
        content = re.sub(r'\[\.\.\.\]$', '', content)

        return content.strip()

    def _extract_summary(self, content: str, max_length: int = 200) -> str:
        """
        Extract a summary from content.

        Args:
            content: Full content text
            max_length: Maximum length of summary

        Returns:
            Content summary
        """
        if not content:
            return ""

        # Find a good breaking point
        if len(content) <= max_length:
            return content

        # Try to break at sentence end
        sentences = content.split('. ')
        summary = ""
        for sentence in sentences:
            if len(summary + sentence) <= max_length:
                summary += sentence + '. '
            else:
                break

        if not summary:
            summary = content[:max_length].rsplit(' ', 1)[0] + '...'

        return summary.strip()

    def _extract_media(self, media: List[Any]) -> List[Dict[str, str]]:
        """
        Extract media information from feed entry.

        Args:
            media: List of media objects

        Returns:
            List of media dictionaries
        """
        extracted_media = []

        for media_item in media:
            if isinstance(media_item, dict):
                media_info = {
                    'url': media_item.get('url', ''),
                    'type': media_item.get('type', ''),
                    'medium': media_item.get('medium', ''),
                    'width': media_item.get('width', 0),
                    'height': media_item.get('height', 0)
                }
                extracted_media.append(media_info)

        return extracted_media

    def _estimate_read_time(self, content: str) -> int:
        """
        Estimate reading time in minutes.

        Args:
            content: Content text

        Returns:
            Estimated read time in minutes
        """
        if not content:
            return 0

        word_count = len(content.split())
        words_per_minute = 200  # Average reading speed
        read_time = max(1, word_count // words_per_minute)

        return read_time

    def _generate_item_id(self, entry: Any) -> str:
        """
        Generate a unique ID for a feed entry.

        Args:
            entry: Feedparser entry object

        Returns:
            Unique ID string
        """
        # Try to use entry ID if available
        if hasattr(entry, 'id') and entry.id:
            return hashlib.md5(entry.id.encode()).hexdigest()

        # Fall back to URL + title hash
        url = getattr(entry, 'link', '')
        title = getattr(entry, 'title', '')
        content = url + title

        return hashlib.md5(content.encode()).hexdigest()

    def _is_duplicate(self, item: Dict[str, Any]) -> bool:
        """
        Check if item is a duplicate.

        Args:
            item: Feed item dictionary

        Returns:
            True if duplicate, False otherwise
        """
        url = item.get('url', '')
        item_hash = item.get('id', '')

        if url in self.seen_urls or item_hash in self.seen_hashes:
            return True

        self.seen_urls.add(url)
        self.seen_hashes.add(item_hash)

        return False

    async def fetch_multiple_feeds(
        self,
        feed_urls: List[str],
        max_items_per_feed: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Fetch and parse multiple RSS feeds concurrently.

        Args:
            feed_urls: List of RSS feed URLs
            max_items_per_feed: Maximum items per feed

        Returns:
            Combined list of feed items
        """
        all_items = []
        semaphore = asyncio.Semaphore(self.max_concurrent_feeds)

        async def fetch_with_semaphore(url: str):
            async with semaphore:
                return await self.fetch_feed(url, max_items_per_feed)

        tasks = [fetch_with_semaphore(url) for url in feed_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_items.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error in feed fetch: {str(result)}")

        # Sort by publication date
        all_items.sort(key=lambda x: x['published'], reverse=True)

        logger.info(f"Fetched total of {len(all_items)} items from {len(feed_urls)} feeds")
        return all_items

    async def get_tech_news(
        self,
        categories: Optional[List[str]] = None,
        hours_ago: int = 24,
        max_items: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent tech news from configured feeds.

        Args:
            categories: List of categories to filter by
            hours_ago: How many hours back to look
            max_items: Maximum total items to return

        Returns:
            List of tech news items
        """
        # Filter feeds by category if specified
        feeds_to_fetch = self.tech_feeds
        if categories:
            feeds_to_fetch = [
                feed for feed in self.tech_feeds
                if feed.get('category', '').lower() in [c.lower() for c in categories]
            ]

        feed_urls = [feed['url'] for feed in feeds_to_fetch]

        # Fetch all feeds
        all_items = await self.fetch_multiple_feeds(feed_urls)

        # Filter by time
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
        recent_items = [
            item for item in all_items
            if item['published'] > cutoff_time
        ]

        # Limit results
        if len(recent_items) > max_items:
            recent_items = recent_items[:max_items]

        logger.info(f"Found {len(recent_items)} tech news items in the last {hours_ago} hours")
        return recent_items

    async def search_feeds(
        self,
        query: str,
        feed_urls: List[str],
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search for items containing query across multiple feeds.

        Args:
            query: Search query
            feed_urls: List of feed URLs to search
            max_results: Maximum results to return

        Returns:
            List of matching items
        """
        # Fetch all feeds
        all_items = await self.fetch_multiple_feeds(feed_urls)

        # Search for matching items
        query_lower = query.lower()
        matching_items = []

        for item in all_items:
            # Search in title, content, and tags
            searchable_text = (
                item['title'].lower() + ' ' +
                item['content'].lower() + ' ' +
                ' '.join([tag.lower() for tag in item['categories']])
            )

            if query_lower in searchable_text:
                # Add search metadata
                item['search_query'] = query
                item['search_relevance'] = self._calculate_relevance(item, query)
                matching_items.append(item)

        # Sort by relevance
        matching_items.sort(key=lambda x: x['search_relevance'], reverse=True)

        # Limit results
        if len(matching_items) > max_results:
            matching_items = matching_items[:max_results]

        logger.info(f"Found {len(matching_items)} items matching query: {query}")
        return matching_items

    def _calculate_relevance(self, item: Dict[str, Any], query: str) -> float:
        """
        Calculate relevance score for search query.

        Args:
            item: Feed item dictionary
            query: Search query

        Returns:
            Relevance score (0-1)
        """
        query_lower = query.lower()
        title_lower = item['title'].lower()
        content_lower = item['content'].lower()

        # Title matches are weighted more heavily
        title_score = 0
        if query_lower in title_lower:
            title_score = 1.0 if title_lower == query_lower else 0.7

        # Content matches
        content_score = 0
        query_words = query_lower.split()
        for word in query_words:
            if word in content_lower:
                content_score += 0.3

        # Recent items get a boost
        age_hours = (datetime.now(timezone.utc) - item['published']).total_seconds() / 3600
        recency_score = max(0, 1 - (age_hours / 168))  # Decay over 1 week

        # Combine scores
        relevance = (title_score * 0.6 + content_score * 0.3 + recency_score * 0.1)

        return min(relevance, 1.0)

    async def get_mock_rss_data(self) -> List[Dict[str, Any]]:
        """
        Generate mock RSS data for testing purposes.

        Returns:
            List of mock RSS feed items
        """
        mock_data = [
            {
                'id': 'mock_rss_1',
                'title': 'Apple Unveils Revolutionary M4 Chip with 20% Performance Boost',
                'url': 'https://techcrunch.com/apple-m4-chip',
                'content': 'Apple today announced its next-generation M4 chip, featuring a revolutionary new architecture...',
                'summary': 'Apple announced the M4 chip with significant performance improvements over its predecessor.',
                'author': 'Sarah Johnson',
                'published': datetime.now(timezone.utc) - timedelta(hours=2),
                'updated': datetime.now(timezone.utc) - timedelta(hours=2),
                'source': 'TechCrunch',
                'source_url': 'https://techcrunch.com',
                'categories': ['Apple', 'Hardware', 'Chips'],
                'media': [{'url': 'https://techcrunch.com/wp-content/uploads/2024/03/m4-chip.jpg', 'type': 'image/jpeg'}],
                'read_time': 3,
                'engagement_score': 0.9,
                'source': 'rss',
                'metadata': {
                    'feed_format': 'rss2.0',
                    'language': 'en',
                    'word_count': 450,
                    'has_image': True,
                    'trending': True
                }
            },
            {
                'id': 'mock_rss_2',
                'title': 'OpenAI Releases GPT-5 with Improved Reasoning Capabilities',
                'url': 'https://openai.com/blog/gpt-5',
                'content': 'OpenAI today announced GPT-5, the latest iteration of its language model family...',
                'summary': 'OpenAI releases GPT-5 with enhanced reasoning and reduced hallucinations.',
                'author': 'Sam Altman',
                'published': datetime.now(timezone.utc) - timedelta(hours=5),
                'updated': datetime.now(timezone.utc) - timedelta(hours=5),
                'source': 'OpenAI Blog',
                'source_url': 'https://openai.com',
                'categories': ['AI', 'Machine Learning', 'GPT'],
                'media': [{'url': 'https://openai.com/images/gpt5-hero.jpg', 'type': 'image/jpeg'}],
                'read_time': 5,
                'engagement_score': 0.95,
                'source': 'rss',
                'metadata': {
                    'feed_format': 'atom',
                    'language': 'en',
                    'word_count': 780,
                    'has_image': True,
                    'trending': True
                }
            },
            {
                'id': 'mock_rss_3',
                'title': 'Critical Kubernetes Vulnerability Affects All Major Cloud Providers',
                'url': 'https://kubernetes.io/blog/security-advisory',
                'content': 'A critical vulnerability has been discovered in Kubernetes that could allow privilege escalation...',
                'summary': 'Critical Kubernetes security vulnerability patched in latest versions.',
                'author': 'Kubernetes Security Team',
                'published': datetime.now(timezone.utc) - timedelta(hours=8),
                'updated': datetime.now(timezone.utc) - timedelta(hours=8),
                'source': 'Kubernetes Blog',
                'source_url': 'https://kubernetes.io',
                'categories': ['Kubernetes', 'Security', 'DevOps', 'Cloud'],
                'media': [],
                'read_time': 4,
                'engagement_score': 0.85,
                'source': 'rss',
                'metadata': {
                    'feed_format': 'rss2.0',
                    'language': 'en',
                    'word_count': 620,
                    'has_image': False,
                    'security_alert': True
                }
            },
            {
                'id': 'mock_rss_4',
                'title': 'Meta Open Sources LLaMA 3: State-of-the-Art Open Language Model',
                'url': 'https://ai.facebook.com/blog/llama-3',
                'content': 'Meta today announced the open source release of LLaMA 3, a 70 billion parameter language model...',
                'summary': 'Meta releases LLaMA 3, democratizing access to large language models.',
                'author': 'Mark Zuckerberg',
                'published': datetime.now(timezone.utc) - timedelta(hours=12),
                'updated': datetime.now(timezone.utc) - timedelta(hours=12),
                'source': 'Meta AI Blog',
                'source_url': 'https://ai.facebook.com',
                'categories': ['AI', 'Open Source', 'LLaMA'],
                'media': [{'url': 'https://ai.facebook.com/images/llama3-banner.png', 'type': 'image/png'}],
                'read_time': 6,
                'engagement_score': 0.92,
                'source': 'rss',
                'metadata': {
                    'feed_format': 'atom',
                    'language': 'en',
                    'word_count': 920,
                    'has_image': True,
                    'trending': True
                }
            },
            {
                'id': 'mock_rss_5',
                'title': 'GitHub Copilot X: AI-Powered Developer Assistant with Voice Support',
                'url': 'https://github.blog/copilot-x',
                'content': 'GitHub announces Copilot X, an evolution of its AI pair programmer...',
                'summary': 'GitHub Copilot X brings voice interaction and enhanced AI capabilities to developers.',
                'author': 'Thomas Dohmke',
                'published': datetime.now(timezone.utc) - timedelta(hours=16),
                'updated': datetime.now(timezone.utc) - timedelta(hours=16),
                'source': 'GitHub Blog',
                'source_url': 'https://github.blog',
                'categories': ['GitHub', 'AI', 'Developer Tools'],
                'media': [{'url': 'https://github.blog/wp-content/uploads/copilot-x-demo.gif', 'type': 'image/gif'}],
                'read_time': 4,
                'engagement_score': 0.88,
                'source': 'rss',
                'metadata': {
                    'feed_format': 'rss2.0',
                    'language': 'en',
                    'word_count': 580,
                    'has_image': True,
                    'developer_tools': True
                }
            }
        ]

        logger.info(f"Generated {len(mock_data)} mock RSS feed items")
        return mock_data