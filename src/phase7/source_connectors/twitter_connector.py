"""
Twitter Connector for Phase 7.3: Multi-Source Integration

This module provides functionality to fetch tweets and trending topics from Twitter/X
for real-time tech trends monitoring and sentiment analysis.

Note: This is a bonus feature that would require Twitter API access for production use.
"""

import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any
import json
import logging
import re

logger = logging.getLogger(__name__)


class TwitterConnector:
    """
    Twitter/X API connector for fetching tech-related content.

    Features:
    - Fetch tweets from specific users or hashtags
    - Track trending topics
    - Real-time stream monitoring (future enhancement)
    - Sentiment analysis integration
    - Rate limiting and error handling

    Note: Requires Twitter API v2 access credentials for production use.
    """

    def __init__(
        self,
        bearer_token: Optional[str] = None,
        consumer_key: Optional[str] = None,
        consumer_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        access_token_secret: Optional[str] = None
    ):
        """
        Initialize Twitter connector.

        Args:
            bearer_token: Twitter API Bearer Token (for app-only authentication)
            consumer_key: Twitter API Consumer Key
            consumer_secret: Twitter API Consumer Secret
            access_token: Twitter API Access Token
            access_token_secret: Twitter API Access Token Secret
        """
        self.bearer_token = bearer_token
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.base_url = "https://api.twitter.com/2"
        self.rate_limit_remaining = 300
        self.rate_limit_reset = datetime.now()

        # Tech-focused accounts to monitor
        self.tech_accounts = [
            'elonmusk',
            'sundarpichai',
            'satyanadella',
            'tim_cook',
            'vitalikbuterin',
            'sama',
            'ylecun',
            'karpathy',
            'fchollet',
            'hardmaru',
            'ylecun',
            'simonw',
            'gdb',
            'peterlevi',
            'swyx',
            'levelsio',
            'patio11',
            'naval',
            'paulg',
            'djacobs',
            'jasonfried',
            'dhh',
            'dbanabout'
        ]

        # Tech hashtags to track
        self.tech_hashtags = [
            '#AI',
            '#MachineLearning',
            '#Python',
            '#JavaScript',
            '#React',
            '#NodeJS',
            '#Docker',
            '#Kubernetes',
            '#DevOps',
            '#Cybersecurity',
            '#Web3',
            '#Blockchain',
            '#CloudComputing',
            '#AWS',
            '#Azure',
            '#GCP',
            '#OpenSource',
            '#Tech',
            '#Programming',
            '#Coding',
            '#SoftwareDevelopment'
        ]

    async def authenticate(self) -> bool:
        """
        Authenticate with Twitter API.

        Returns:
            bool: True if authentication successful
        """
        if not self.bearer_token:
            logger.warning("No Twitter Bearer Token provided, using mock data")
            return False

        # Validate authentication with a simple API call
        headers = {
            'Authorization': f'Bearer {self.bearer_token}'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/users/me",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        logger.info("Twitter authentication successful")
                        return True
                    else:
                        logger.error(f"Twitter authentication failed: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Error during Twitter authentication: {str(e)}")
            return False

    async def search_tweets(
        self,
        query: str,
        max_results: int = 100,
        sort_by: str = 'relevance',
        tweet_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for tweets matching a query.

        Args:
            query: Search query (supports Twitter search syntax)
            max_results: Maximum number of tweets to return
            sort_by: Sort method (relevance, recent)
            tweet_fields: Additional tweet fields to request

        Returns:
            List of tweet dictionaries
        """
        if not self.bearer_token:
            logger.warning("No Twitter credentials, returning mock data")
            return await self.get_mock_twitter_data()

        # Build request parameters
        params = {
            'query': query,
            'max_results': min(max_results, 100),
            'sort_order': sort_by,
            'tweet.fields': tweet_fields or [
                'created_at',
                'author_id',
                'public_metrics',
                'context_annotations',
                'entities',
                'attachments'
            ],
            'expansions': [
                'author_id',
                'attachments.media_keys',
                'referenced_tweets.id'
            ],
            'user.fields': [
                'username',
                'name',
                'verified',
                'public_metrics'
            ]
        }

        headers = {
            'Authorization': f'Bearer {self.bearer_token}'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/tweets/search/recent",
                    params=params,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        tweets = self._process_search_response(data)
                        logger.info(f"Found {len(tweets)} tweets for query: {query}")
                        return tweets
                    else:
                        error_text = await response.text()
                        logger.error(f"Error searching tweets: {response.status} - {error_text}")
                        return []

        except Exception as e:
            logger.error(f"Error searching Twitter: {str(e)}")
            return []

    def _process_search_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process Twitter API search response.

        Args:
            data: Raw API response

        Returns:
            List of processed tweet dictionaries
        """
        tweets = []
        users = {user['id']: user for user in data.get('includes', {}).get('users', [])}
        media = {media['media_key']: media for media in data.get('includes', {}).get('media', [])}

        for tweet_data in data.get('data', []):
            tweet_info = {
                'id': tweet_data['id'],
                'text': tweet_data['text'],
                'created_at': self._parse_twitter_date(tweet_data['created_at']),
                'author_id': tweet_data['author_id'],
                'author': users.get(tweet_data['author_id'], {}).get('username', 'unknown'),
                'author_name': users.get(tweet_data['author_id'], {}).get('name', 'Unknown'),
                'verified': users.get(tweet_data['author_id'], {}).get('verified', False),
                'metrics': tweet_data.get('public_metrics', {}),
                'reply_count': tweet_data.get('public_metrics', {}).get('reply_count', 0),
                'retweet_count': tweet_data.get('public_metrics', {}).get('retweet_count', 0),
                'like_count': tweet_data.get('public_metrics', {}).get('like_count', 0),
                'quote_count': tweet_data.get('public_metrics', {}).get('quote_count', 0),
                'url': f"https://twitter.com/{users.get(tweet_data['author_id'], {}).get('username', 'unknown')}/status/{tweet_data['id']}",
                'source': 'twitter',
                'hashtags': self._extract_hashtags(tweet_data.get('entities', {})),
                'mentions': self._extract_mentions(tweet_data.get('entities', {})),
                'urls': self._extract_urls(tweet_data.get('entities', {})),
                'media': self._extract_media(tweet_data, media),
                'context_annotations': tweet_data.get('context_annotations', []),
                'metadata': {
                    'lang': tweet_data.get('lang', 'en'),
                    'possibly_sensitive': tweet_data.get('possibly_sensitive', False),
                    'reply_settings': tweet_data.get('reply_settings', 'everyone')
                }
            }

            # Calculate engagement score
            tweet_info['engagement_score'] = self._calculate_tweet_engagement(tweet_info)

            tweets.append(tweet_info)

        return tweets

    def _parse_twitter_date(self, date_str: str) -> datetime:
        """
        Parse Twitter date string to datetime object.

        Args:
            date_str: Twitter date string

        Returns:
            datetime object
        """
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)

    def _extract_hashtags(self, entities: Dict[str, Any]) -> List[str]:
        """
        Extract hashtags from tweet entities.

        Args:
            entities: Tweet entities

        Returns:
            List of hashtags
        """
        return [tag['tag'] for tag in entities.get('hashtags', [])]

    def _extract_mentions(self, entities: Dict[str, Any]) -> List[str]:
        """
        Extract mentions from tweet entities.

        Args:
            entities: Tweet entities

        Returns:
            List of mentioned usernames
        """
        return [mention['username'] for mention in entities.get('mentions', [])]

    def _extract_urls(self, entities: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract URLs from tweet entities.

        Args:
            entities: Tweet entities

        Returns:
            List of URL dictionaries
        """
        return [
            {
                'url': url['url'],
                'expanded_url': url.get('expanded_url', ''),
                'display_url': url.get('display_url', '')
            }
            for url in entities.get('urls', [])
        ]

    def _extract_media(
        self,
        tweet_data: Dict[str, Any],
        media_lookup: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract media information from tweet.

        Args:
            tweet_data: Tweet data
            media_lookup: Media lookup dictionary

        Returns:
            List of media dictionaries
        """
        media_items = []
        attachments = tweet_data.get('attachments', {}).get('media_keys', [])

        for media_key in attachments:
            if media_key in media_lookup:
                media_info = media_lookup[media_key]
                media_items.append({
                    'media_key': media_key,
                    'type': media_info.get('type', ''),
                    'url': media_info.get('url', ''),
                    'width': media_info.get('width', 0),
                    'height': media_info.get('height', 0),
                    'preview_image_url': media_info.get('preview_image_url', '')
                })

        return media_items

    def _calculate_tweet_engagement(self, tweet: Dict[str, Any]) -> float:
        """
        Calculate engagement score for a tweet.

        Args:
            tweet: Tweet dictionary

        Returns:
            Engagement score (0-1)
        """
        metrics = tweet.get('metrics', {})

        # Weight different engagement types
        likes = metrics.get('like_count', 0)
        retweets = metrics.get('retweet_count', 0) * 2  # Retweets weighted higher
        replies = metrics.get('reply_count', 0) * 1.5
        quotes = metrics.get('quote_count', 0) * 1.5

        # Author verification boost
        if tweet.get('verified', False):
            verified_boost = 0.1
        else:
            verified_boost = 0

        # Calculate normalized engagement
        total_engagement = likes + retweets + replies + quotes
        normalized_engagement = min(total_engagement / 1000, 0.9)  # Cap at 0.9

        return normalized_engagement + verified_boost

    async def get_user_tweets(
        self,
        username: str,
        max_results: int = 50,
        exclude_replies: bool = True,
        include_retweets: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get recent tweets from a specific user.

        Args:
            username: Twitter username
            max_results: Maximum number of tweets
            exclude_replies: Whether to exclude replies
            include_retweets: Whether to include retweets

        Returns:
            List of tweet dictionaries
        """
        if not self.bearer_token:
            return await self.get_mock_twitter_data()

        # First get user ID
        user_id = await self._get_user_id(username)
        if not user_id:
            logger.error(f"Could not find user: {username}")
            return []

        # Build request parameters
        params = {
            'max_results': min(max_results, 100),
            'tweet.fields': [
                'created_at',
                'public_metrics',
                'context_annotations',
                'entities',
                'in_reply_to_user_id',
                'referenced_tweets'
            ],
            'expansions': ['referenced_tweets.id', 'attachments.media_keys']
        }

        # Add exclusions
        exclusions = []
        if exclude_replies:
            exclusions.append('replies')
        if not include_retweets:
            exclusions.append('retweets')

        if exclusions:
            params['exclude'] = ','.join(exclusions)

        headers = {
            'Authorization': f'Bearer {self.bearer_token}'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/users/{user_id}/tweets",
                    params=params,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        tweets = self._process_user_tweets_response(data, username)
                        logger.info(f"Fetched {len(tweets)} tweets from @{username}")
                        return tweets
                    else:
                        error_text = await response.text()
                        logger.error(f"Error fetching user tweets: {response.status} - {error_text}")
                        return []

        except Exception as e:
            logger.error(f"Error fetching user tweets: {str(e)}")
            return []

    async def _get_user_id(self, username: str) -> Optional[str]:
        """
        Get Twitter user ID from username.

        Args:
            username: Twitter username

        Returns:
            User ID or None if not found
        """
        if not self.bearer_token:
            return None

        params = {
            'user.fields': ['id'],
            'usernames': username
        }

        headers = {
            'Authorization': f'Bearer {self.bearer_token}'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/users/by",
                    params=params,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        users = data.get('data', [])
                        if users:
                            return users[0]['id']
                    return None

        except Exception as e:
            logger.error(f"Error getting user ID: {str(e)}")
            return None

    def _process_user_tweets_response(
        self,
        data: Dict[str, Any],
        username: str
    ) -> List[Dict[str, Any]]:
        """
        Process user tweets API response.

        Args:
            data: Raw API response
            username: Username

        Returns:
            List of processed tweet dictionaries
        """
        tweets = []

        for tweet_data in data.get('data', []):
            tweet_info = {
                'id': tweet_data['id'],
                'text': tweet_data['text'],
                'created_at': self._parse_twitter_date(tweet_data['created_at']),
                'author': username,
                'author_name': username,  # Would need separate user lookup
                'verified': False,  # Would need separate user lookup
                'metrics': tweet_data.get('public_metrics', {}),
                'reply_count': tweet_data.get('public_metrics', {}).get('reply_count', 0),
                'retweet_count': tweet_data.get('public_metrics', {}).get('retweet_count', 0),
                'like_count': tweet_data.get('public_metrics', {}).get('like_count', 0),
                'quote_count': tweet_data.get('public_metrics', {}).get('quote_count', 0),
                'url': f"https://twitter.com/{username}/status/{tweet_data['id']}",
                'source': 'twitter',
                'hashtags': self._extract_hashtags(tweet_data.get('entities', {})),
                'mentions': self._extract_mentions(tweet_data.get('entities', {})),
                'urls': self._extract_urls(tweet_data.get('entities', {})),
                'context_annotations': tweet_data.get('context_annotations', []),
                'metadata': {
                    'lang': tweet_data.get('lang', 'en'),
                    'possibly_sensitive': tweet_data.get('possibly_sensitive', False)
                }
            }

            # Calculate engagement score
            tweet_info['engagement_score'] = self._calculate_tweet_engagement(tweet_info)

            tweets.append(tweet_info)

        return tweets

    async def get_trending_topics(
        self,
        woeid: int = 1,  # Worldwide
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get trending topics.

        Args:
            woeid: Where On Earth ID (1 for worldwide)
            limit: Maximum number of trends to return

        Returns:
            List of trending topics
        """
        # This would use Twitter's trends endpoint in production
        # For now, return mock trending topics
        mock_trends = [
            {'name': '#AI', 'tweet_volume': 125000},
            {'name': '#MachineLearning', 'tweet_volume': 89000},
            {'name': '#OpenAI', 'tweet_volume': 76000},
            {'name': '#Python', 'tweet_volume': 67000},
            {'name': '#Cybersecurity', 'tweet_volume': 54000},
            {'name': '#Web3', 'tweet_volume': 48000},
            {'name': '#TechNews', 'tweet_volume': 42000},
            {'name': '#CloudComputing', 'tweet_volume': 38000},
            {'name': '#DevOps', 'tweet_volume': 35000},
            {'name': '#JavaScript', 'tweet_volume': 32000}
        ]

        return mock_trends[:limit]

    async def monitor_hashtag_stream(
        self,
        hashtags: List[str],
        callback_func: callable,
        duration_minutes: int = 60
    ):
        """
        Monitor hashtags in real-time (future enhancement).

        Args:
            hashtags: List of hashtags to monitor
            callback_func: Function to call for each matching tweet
            duration_minutes: How long to monitor
        """
        # This would implement Twitter's filtered stream endpoint
        # for real-time monitoring in a production version
        logger.info(f"Would monitor hashtags {hashtags} for {duration_minutes} minutes")

    async def get_mock_twitter_data(self) -> List[Dict[str, Any]]:
        """
        Generate mock Twitter data for testing purposes.

        Returns:
            List of mock tweet dictionaries
        """
        mock_data = [
            {
                'id': '1234567890123456789',
                'text': 'Just pushed a major update to our AI framework! 🚀 The new features include: 1) 40% faster inference 2) Support for multimodal inputs 3) Better memory efficiency #AI #MachineLearning #OpenSource',
                'created_at': datetime.now(timezone.utc) - timedelta(minutes=30),
                'author': 'AI_Researcher',
                'author_name': 'Dr. AI Research',
                'verified': True,
                'metrics': {
                    'retweet_count': 234,
                    'like_count': 1567,
                    'reply_count': 89,
                    'quote_count': 45
                },
                'reply_count': 89,
                'retweet_count': 234,
                'like_count': 1567,
                'quote_count': 45,
                'url': 'https://twitter.com/AI_Researcher/status/1234567890123456789',
                'source': 'twitter',
                'hashtags': ['AI', 'MachineLearning', 'OpenSource'],
                'mentions': [],
                'urls': [],
                'media': [],
                'context_annotations': [
                    {
                        'domain': {
                            'id': '65',
                            'name': 'Interests and Hobbies Vertical',
                            'description': 'A vertical category.'
                        },
                        'entity': {
                            'id': '848920371311001600',
                            'name': 'Technology',
                            'description': 'Technology and computing'
                        }
                    }
                ],
                'metadata': {
                    'lang': 'en',
                    'possibly_sensitive': False
                }
            },
            {
                'id': '1234567890123456790',
                'text': 'BREAKING: Major security vulnerability discovered in popular cloud provider\'s API. Immediate patch required! Details: 🧵 #Cybersecurity #CloudSecurity #InfoSec',
                'created_at': datetime.now(timezone.utc) - timedelta(hours=2),
                'author': 'SecurityExpert',
                'author_name': 'Cyber Security Pro',
                'verified': True,
                'metrics': {
                    'retweet_count': 890,
                    'like_count': 3456,
                    'reply_count': 234,
                    'quote_count': 123
                },
                'reply_count': 234,
                'retweet_count': 890,
                'like_count': 3456,
                'quote_count': 123,
                'url': 'https://twitter.com/SecurityExpert/status/1234567890123456790',
                'source': 'twitter',
                'hashtags': ['Cybersecurity', 'CloudSecurity', 'InfoSec'],
                'mentions': [],
                'urls': [],
                'media': [],
                'context_annotations': [
                    {
                        'domain': {
                            'id': '46',
                            'name': 'Category',
                            'description': 'A category or vertical of content.'
                        },
                        'entity': {
                            'id': '848919033124003840',
                            'name': 'Computer security',
                            'description': 'Computer security'
                        }
                    }
                ],
                'metadata': {
                    'lang': 'en',
                    'possibly_sensitive': False,
                    'security_alert': True
                }
            },
            {
                'id': '1234567890123456791',
                'text': 'Just shipped a new Python package for async web scraping! Features: - Rate limiting - Proxy rotation - JavaScript rendering - Distributed scraping Check it out: https://github.com/example/async-scraper #Python #WebScraping #AsyncIO',
                'created_at': datetime.now(timezone.utc) - timedelta(hours=4),
                'author': 'PythonDev',
                'author_name': 'Python Developer',
                'verified': False,
                'metrics': {
                    'retweet_count': 45,
                    'like_count': 234,
                    'reply_count': 23,
                    'quote_count': 12
                },
                'reply_count': 23,
                'retweet_count': 45,
                'like_count': 234,
                'quote_count': 12,
                'url': 'https://twitter.com/PythonDev/status/1234567890123456791',
                'source': 'twitter',
                'hashtags': ['Python', 'WebScraping', 'AsyncIO'],
                'mentions': [],
                'urls': [
                    {
                        'url': 'https://github.com/example/async-scraper',
                        'expanded_url': 'https://github.com/example/async-scraper',
                        'display_url': 'github.com/example/async-scraper'
                    }
                ],
                'media': [],
                'context_annotations': [
                    {
                        'domain': {
                            'id': '65',
                            'name': 'Interests and Hobbies Vertical',
                            'description': 'A vertical category.'
                        },
                        'entity': {
                            'id': '848920371311001600',
                            'name': 'Technology',
                            'description': 'Technology and computing'
                        }
                    }
                ],
                'metadata': {
                    'lang': 'en',
                    'possibly_sensitive': False,
                    'open_source': True
                }
            },
            {
                'id': '1234567890123456792',
                'text': 'My thoughts on the future of cloud computing: 1️⃣ Serverless will dominate 2️⃣ Edge computing adoption will surge 3️⃣ Multi-cloud becomes the norm 4️⃣ AI-driven ops will be standard What do you think? #CloudComputing #Serverless #DevOps',
                'created_at': datetime.now(timezone.utc) - timedelta(hours=6),
                'author': 'CloudArchitect',
                'author_name': 'Cloud Solutions Architect',
                'verified': False,
                'metrics': {
                    'retweet_count': 123,
                    'like_count': 789,
                    'reply_count': 156,
                    'quote_count': 34
                },
                'reply_count': 156,
                'retweet_count': 123,
                'like_count': 789,
                'quote_count': 34,
                'url': 'https://twitter.com/CloudArchitect/status/1234567890123456792',
                'source': 'twitter',
                'hashtags': ['CloudComputing', 'Serverless', 'DevOps'],
                'mentions': [],
                'urls': [],
                'media': [],
                'context_annotations': [
                    {
                        'domain': {
                            'id': '65',
                            'name': 'Interests and Hobbies Vertical',
                            'description': 'A vertical category.'
                        },
                        'entity': {
                            'id': '848920371311001600',
                            'name': 'Technology',
                            'description': 'Technology and computing'
                        }
                    }
                ],
                'metadata': {
                    'lang': 'en',
                    'possibly_sensitive': False,
                    'opinion': True
                }
            },
            {
                'id': '1234567890123456793',
                'text': '🚀 Excited to announce that our startup just raised $50M Series B! We\'re building the future of decentralized AI with our blockchain-based platform. Thank you to all our supporters! #Web3 #Blockchain #AI #Startup #Funding',
                'created_at': datetime.now(timezone.utc) - timedelta(hours=8),
                'author': 'StartupCEO',
                'author_name': 'Founder & CEO',
                'verified': True,
                'metrics': {
                    'retweet_count': 567,
                    'like_count': 2345,
                    'reply_count': 345,
                    'quote_count': 89
                },
                'reply_count': 345,
                'retweet_count': 567,
                'like_count': 2345,
                'quote_count': 89,
                'url': 'https://twitter.com/StartupCEO/status/1234567890123456793',
                'source': 'twitter',
                'hashtags': ['Web3', 'Blockchain', 'AI', 'Startup', 'Funding'],
                'mentions': [],
                'urls': [],
                'media': [],
                'context_annotations': [
                    {
                        'domain': {
                            'id': '65',
                            'name': 'Interests and Hobbies Vertical',
                            'description': 'A vertical category.'
                        },
                        'entity': {
                            'id': '848920371311001600',
                            'name': 'Technology',
                            'description': 'Technology and computing'
                        }
                    }
                ],
                'metadata': {
                    'lang': 'en',
                    'possibly_sensitive': False,
                    'funding_announcement': True
                }
            }
        ]

        logger.info(f"Generated {len(mock_data)} mock Twitter tweets")
        return mock_data