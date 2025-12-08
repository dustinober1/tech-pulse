"""
Reddit Connector for Phase 7.3: Multi-Source Integration

This module provides functionality to fetch posts and comments from Reddit
for tech trends analysis and sentiment tracking.
"""

import os
import asyncio
import aiohttp
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any
import json
import logging

logger = logging.getLogger(__name__)


class RedditConnector:
    """
    Reddit API connector for fetching tech-related content.

    Features:
    - Fetch posts from specified subreddits
    - Extract comments and discussions
    - Rate limiting and error handling
    - Support for authenticated and anonymous access
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: str = "tech-pulse-v1.0"
    ):
        """
        Initialize Reddit connector.

        Args:
            client_id: Reddit API client ID (optional for read-only access)
            client_secret: Reddit API client secret
            user_agent: User agent string for API requests
        """
        self.client_id = client_id or os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = user_agent
        self.base_url = "https://www.reddit.com"
        self.access_token = None
        self.rate_limit_remaining = 100
        self.rate_limit_reset = datetime.now()

        # Tech-focused subreddits for monitoring
        self.tech_subreddits = [
            'technology',
            'programming',
            'MachineLearning',
            'webdev',
            'python',
            'javascript',
            'devops',
            'cybersecurity',
            'DataScience',
            'TechNewsToday'
        ]

    async def authenticate(self) -> bool:
        """
        Authenticate with Reddit API using OAuth2.

        Returns:
            bool: True if authentication successful
        """
        if not self.client_id or not self.client_secret:
            logger.warning("No Reddit credentials provided, using read-only access")
            return False

        auth_url = "https://www.reddit.com/api/v1/access_token"
        auth = aiohttp.BasicAuth(login=self.client_id, password=self.client_secret)

        data = {
            'grant_type': 'client_credentials'
        }

        headers = {
            'User-Agent': self.user_agent
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(auth_url, auth=auth, data=data, headers=headers) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        self.access_token = token_data.get('access_token')
                        logger.info("Reddit authentication successful")
                        return True
                    else:
                        logger.error(f"Reddit authentication failed: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Error during Reddit authentication: {str(e)}")
            return False

    async def get_posts(
        self,
        subreddit: str,
        limit: int = 25,
        sort: str = 'hot',
        time_filter: str = 'day'
    ) -> List[Dict[str, Any]]:
        """
        Fetch posts from a subreddit.

        Args:
            subreddit: Name of the subreddit
            limit: Number of posts to fetch (max 100)
            sort: Sort method (hot, new, top, rising)
            time_filter: Time period for top sort (hour, day, week, month, year, all)

        Returns:
            List of post dictionaries
        """
        url = f"{self.base_url}/r/{subreddit}/{sort}.json"
        params = {
            'limit': min(limit, 100),
            't': time_filter
        }

        headers = {
            'User-Agent': self.user_agent
        }

        if self.access_token:
            headers['Authorization'] = f"Bearer {self.access_token}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        posts = []

                        for post in data['data']['children']:
                            post_data = post['data']

                            # Extract relevant fields
                            post_info = {
                                'id': post_data['id'],
                                'title': post_data['title'],
                                'author': post_data.get('author', '[deleted]'),
                                'created_utc': datetime.fromtimestamp(post_data['created_utc']),
                                'score': post_data['score'],
                                'num_comments': post_data['num_comments'],
                                'url': post_data['url'],
                                'selftext': post_data.get('selftext', ''),
                                'subreddit': post_data['subreddit'],
                                'permalink': f"https://reddit.com{post_data['permalink']}",
                                'upvote_ratio': post_data.get('upvote_ratio', 0),
                                'gilded': post_data.get('gilded', 0),
                                'stickied': post_data.get('stickied', False),
                                'source': 'reddit',
                                'metadata': {
                                    'post_hint': post_data.get('post_hint'),
                                    'domain': post_data.get('domain'),
                                    'flair_text': post_data.get('link_flair_text'),
                                    'is_crosspost': post_data.get('crosspost_parent') is not None
                                }
                            }

                            posts.append(post_info)

                        logger.info(f"Fetched {len(posts)} posts from r/{subreddit}")
                        return posts
                    else:
                        logger.error(f"Error fetching posts from r/{subreddit}: {response.status}")
                        return []

        except Exception as e:
            logger.error(f"Error fetching Reddit posts: {str(e)}")
            return []

    async def get_comments(
        self,
        post_id: str,
        subreddit: str,
        limit: int = 100,
        sort: str = 'top'
    ) -> List[Dict[str, Any]]:
        """
        Fetch comments for a specific post.

        Args:
            post_id: Reddit post ID
            subreddit: Subreddit name
            limit: Number of comments to fetch
            sort: Sort method (top, new, controversial)

        Returns:
            List of comment dictionaries
        """
        url = f"{self.base_url}/r/{subreddit}/comments/{post_id}.json"
        params = {
            'limit': min(limit, 500),
            'sort': sort
        }

        headers = {
            'User-Agent': self.user_agent
        }

        if self.access_token:
            headers['Authorization'] = f"Bearer {self.access_token}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        comments = []

                        # Comments are in the second item of the response
                        if len(data) > 1:
                            self._extract_comments(data[1]['data']['children'], comments)

                        logger.info(f"Fetched {len(comments)} comments for post {post_id}")
                        return comments
                    else:
                        logger.error(f"Error fetching comments for post {post_id}: {response.status}")
                        return []

        except Exception as e:
            logger.error(f"Error fetching Reddit comments: {str(e)}")
            return []

    def _extract_comments(self, comments_data: List[Dict], comments_list: List[Dict], depth: int = 0):
        """
        Recursively extract comments from Reddit API response.

        Args:
            comments_data: Raw comment data from API
            comments_list: List to populate with processed comments
            depth: Comment thread depth
        """
        for comment in comments_data:
            if comment['kind'] == 't1':
                comment_data = comment['data']

                comment_info = {
                    'id': comment_data['id'],
                    'author': comment_data.get('author', '[deleted]'),
                    'created_utc': datetime.fromtimestamp(comment_data['created_utc']),
                    'score': comment_data['score'],
                    'body': comment_data.get('body', ''),
                    'depth': depth,
                    'replies': comment_data.get('replies', ''),
                    'gilded': comment_data.get('gilded', 0),
                    'stickied': comment_data.get('stickied', False),
                    'is_submitter': comment_data.get('is_submitter', False),
                    'source': 'reddit_comment',
                    'metadata': {
                        'parent_id': comment_data.get('parent_id'),
                        'edited': comment_data.get('edited', False),
                        'controversiality': comment_data.get('controversiality', 0)
                    }
                }

                comments_list.append(comment_info)

                # Extract replies if they exist
                if comment_data.get('replies') and comment_data['replies'].get('data'):
                    self._extract_comments(
                        comment_data['replies']['data']['children'],
                        comments_list,
                        depth + 1
                    )

    async def search_posts(
        self,
        query: str,
        subreddit: Optional[str] = None,
        limit: int = 25,
        sort: str = 'relevance'
    ) -> List[Dict[str, Any]]:
        """
        Search for posts matching a query.

        Args:
            query: Search query string
            subreddit: Optional subreddit to limit search to
            limit: Number of results to fetch
            sort: Sort method (relevance, hot, top, new)

        Returns:
            List of matching post dictionaries
        """
        if subreddit:
            url = f"{self.base_url}/r/{subreddit}/search.json"
            params['restrict_sr'] = 'true'
        else:
            url = f"{self.base_url}/search.json"

        params = {
            'q': query,
            'limit': min(limit, 100),
            'sort': sort,
            'type': 'link'
        }

        headers = {
            'User-Agent': self.user_agent
        }

        if self.access_token:
            headers['Authorization'] = f"Bearer {self.access_token}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        posts = []

                        for post in data['data']['children']:
                            post_data = post['data']

                            post_info = {
                                'id': post_data['id'],
                                'title': post_data['title'],
                                'author': post_data.get('author', '[deleted]'),
                                'created_utc': datetime.fromtimestamp(post_data['created_utc']),
                                'score': post_data['score'],
                                'num_comments': post_data['num_comments'],
                                'url': post_data['url'],
                                'selftext': post_data.get('selftext', ''),
                                'subreddit': post_data['subreddit'],
                                'permalink': f"https://reddit.com{post_data['permalink']}",
                                'source': 'reddit',
                                'search_query': query,
                                'metadata': {
                                    'domain': post_data.get('domain'),
                                    'flair_text': post_data.get('link_flair_text')
                                }
                            }

                            posts.append(post_info)

                        logger.info(f"Found {len(posts)} posts matching query: {query}")
                        return posts
                    else:
                        logger.error(f"Error searching Reddit posts: {response.status}")
                        return []

        except Exception as e:
            logger.error(f"Error searching Reddit: {str(e)}")
            return []

    async def get_trending_tech_topics(
        self,
        time_window: str = 'day',
        min_score: int = 100,
        subreddits: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get trending tech topics from multiple subreddits.

        Args:
            time_window: Time period to consider (hour, day, week, month)
            min_score: Minimum score threshold
            subreddits: List of subreddits to check (defaults to tech_subreddits)

        Returns:
            List of trending tech topics with engagement metrics
        """
        target_subreddits = subreddits or self.tech_subreddits
        trending_topics = []

        # Fetch posts from all subreddits concurrently
        tasks = []
        for subreddit in target_subreddits:
            task = self.get_posts(subreddit, limit=50, sort='top', time_filter=time_window)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and filter by minimum score
        for i, posts in enumerate(results):
            if isinstance(posts, list):
                for post in posts:
                    if post['score'] >= min_score:
                        topic = {
                            'title': post['title'],
                            'subreddit': post['subreddit'],
                            'score': post['score'],
                            'num_comments': post['num_comments'],
                            'url': post['url'],
                            'created_utc': post['created_utc'],
                            'engagement_rate': self._calculate_engagement_rate(post),
                            'source': 'reddit_trending',
                            'metadata': {
                                'upvote_ratio': post.get('upvote_ratio', 0),
                                'gilded': post.get('gilded', 0),
                                'flair': post.get('metadata', {}).get('flair_text')
                            }
                        }
                        trending_topics.append(topic)

        # Sort by engagement rate
        trending_topics.sort(key=lambda x: x['engagement_rate'], reverse=True)

        logger.info(f"Identified {len(trending_topics)} trending tech topics")
        return trending_topics

    def _calculate_engagement_rate(self, post: Dict[str, Any]) -> float:
        """
        Calculate engagement rate for a post.

        Args:
            post: Post dictionary

        Returns:
            Engagement rate (0-1)
        """
        upvotes = post['score']
        comments = post['num_comments']
        upvote_ratio = post.get('upvote_ratio', 0.5)

        # Adjust for post age (newer posts get higher engagement rate)
        age_hours = (datetime.now(timezone.utc) - post['created_utc']).total_seconds() / 3600

        if age_hours < 1:
            age_factor = 1.0
        else:
            age_factor = 1.0 / (age_hours ** 0.5)

        # Engagement rate formula
        engagement = ((upvotes * upvote_ratio + comments * 2) /
                     (upvotes + comments + 1)) * age_factor

        return min(engagement, 1.0)

    async def get_mock_reddit_data(self) -> List[Dict[str, Any]]:
        """
        Generate mock Reddit data for testing purposes.

        Returns:
            List of mock Reddit posts and discussions
        """
        mock_data = [
            {
                'id': 'mock_post_1',
                'title': 'New AI Framework Achieves 99% Accuracy on Benchmark Tests',
                'author': 'tech_enthusiast_123',
                'created_utc': datetime.now() - timedelta(hours=6),
                'score': 5432,
                'num_comments': 342,
                'url': 'https://github.com/example/ai-framework',
                'selftext': 'Just released a new machine learning framework that significantly improves performance...',
                'subreddit': 'MachineLearning',
                'permalink': 'https://reddit.com/r/MachineLearning/comments/mock_post_1',
                'upvote_ratio': 0.95,
                'source': 'reddit',
                'comments': [
                    {
                        'id': 'comment_1',
                        'author': 'ml_researcher',
                        'body': 'This is groundbreaking! The architecture looks very promising.',
                        'score': 234,
                        'depth': 0,
                        'source': 'reddit_comment'
                    },
                    {
                        'id': 'comment_2',
                        'author': 'data_scientist',
                        'body': 'How does it compare to existing frameworks like TensorFlow and PyTorch?',
                        'score': 156,
                        'depth': 0,
                        'source': 'reddit_comment'
                    }
                ],
                'metadata': {
                    'domain': 'github.com',
                    'flair_text': '[Research]',
                    'trending': True
                }
            },
            {
                'id': 'mock_post_2',
                'title': 'Breaking: Major Cloud Provider Announces 50% Price Reduction',
                'author': 'cloud_observer',
                'created_utc': datetime.now() - timedelta(hours=12),
                'score': 8765,
                'num_comments': 567,
                'url': 'https://technews.com/cloud-pricing',
                'selftext': 'In a surprising move, AWS announced significant price cuts across all services...',
                'subreddit': 'technology',
                'permalink': 'https://reddit.com/r/technology/comments/mock_post_2',
                'upvote_ratio': 0.92,
                'source': 'reddit',
                'comments': [
                    {
                        'id': 'comment_3',
                        'author': 'startup_founder',
                        'body': 'This will dramatically reduce our operational costs!',
                        'score': 445,
                        'depth': 0,
                        'source': 'reddit_comment'
                    }
                ],
                'metadata': {
                    'domain': 'technews.com',
                    'flair_text': '[Breaking News]',
                    'trending': True
                }
            },
            {
                'id': 'mock_post_3',
                'title': 'Ask HN: What programming language should I learn in 2024?',
                'author': 'coding_newbie',
                'created_utc': datetime.now() - timedelta(hours=18),
                'score': 2345,
                'num_comments': 890,
                'url': 'https://self.programming/learn-2024',
                'selftext': 'I\'m looking to start my programming journey and wondering which language to focus on...',
                'subreddit': 'programming',
                'permalink': 'https://reddit.com/r/programming/comments/mock_post_3',
                'upvote_ratio': 0.88,
                'source': 'reddit',
                'comments': [
                    {
                        'id': 'comment_4',
                        'author': 'senior_dev',
                        'body': 'Python is great for beginners, but also consider JavaScript for web development.',
                        'score': 567,
                        'depth': 0,
                        'source': 'reddit_comment'
                    },
                    {
                        'id': 'comment_5',
                        'author': 'rust_evangelist',
                        'body': 'Rust is the future! Memory safety without garbage collection.',
                        'score': 234,
                        'depth': 0,
                        'source': 'reddit_comment'
                    }
                ],
                'metadata': {
                    'domain': 'self.programming',
                    'flair_text': '[Discussion]',
                    'trending': False
                }
            }
        ]

        logger.info(f"Generated {len(mock_data)} mock Reddit posts")
        return mock_data