"""
Hacker News API Client Module.
Handles fetching data from Hacker News API with concurrency support.
"""

import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class HNClient:
    """
    Client for interacting with the Hacker News API.
    """

    BASE_URL = "https://hacker-news.firebaseio.com/v0"

    def __init__(self, base_url: Optional[str] = None, max_workers: int = 10):
        """
        Initialize the HN Client.

        Args:
            base_url: Optional custom base URL for the API
            max_workers: Maximum number of threads for concurrent fetching
        """
        self.base_url = base_url or self.BASE_URL
        self.max_workers = max_workers

    def fetch_story_ids(self) -> Optional[List[int]]:
        """
        Fetch top story IDs from Hacker News.

        Returns:
            List of story IDs or None if fetch fails
        """
        try:
            response = requests.get(f"{self.base_url}/topstories.json", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching story IDs: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching story IDs: {e}")
            return None

    def fetch_story_details(self, story_id: int) -> Optional[Dict]:
        """
        Fetch details for a single story.

        Args:
            story_id: The story ID to fetch

        Returns:
            Dictionary of story details or None if fetch fails
        """
        try:
            response = requests.get(f"{self.base_url}/item/{story_id}.json", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch story {story_id}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error processing story {story_id}: {e}")
            return None

    def extract_story_data(self, item_data: Dict) -> Optional[Dict]:
        """
        Extract relevant fields from story item data.

        Args:
            item_data: Raw story data from API

        Returns:
            Dictionary with extracted fields or None if invalid
        """
        if not item_data or 'title' not in item_data:
            return None

        try:
            return {
                'title': item_data.get('title', ''),
                'score': item_data.get('score', 0),
                'descendants': item_data.get('descendants', 0),
                'time': datetime.fromtimestamp(item_data.get('time', 0)),
                'url': item_data.get('url', ''),
            }
        except Exception as e:
            logger.error(f"Error extracting data from story: {e}")
            return None

    def fetch_top_stories(self, limit: int = 30) -> List[Dict]:
        """
        Fetch top stories with details concurrently.

        Args:
            limit: Number of stories to fetch

        Returns:
            List of processed story dictionaries
        """
        story_ids = self.fetch_story_ids()
        if not story_ids:
            return []

        # Limit IDs
        target_ids = story_ids[:limit]
        results = []

        logger.info(f"Fetching {len(target_ids)} stories using {self.max_workers} workers...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create a map of future to story_id
            future_to_id = {
                executor.submit(self.fetch_story_details, sid): sid
                for sid in target_ids
            }

            for future in as_completed(future_to_id):
                story_id = future_to_id[future]
                try:
                    data = future.result()
                    if data:
                        processed = self.extract_story_data(data)
                        if processed:
                            results.append(processed)
                except Exception as e:
                    logger.error(f"Error fetching story {story_id}: {e}")

        return results
