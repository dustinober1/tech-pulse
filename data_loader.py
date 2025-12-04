import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import unittest.mock as mock


def fetch_story_ids(base_url: str = "https://hacker-news.firebaseio.com/v0") -> Optional[List[int]]:
    """
    Fetch top story IDs from Hacker News.

    Returns:
        List of story IDs or None if fetch fails
    """
    try:
        response = requests.get(f"{base_url}/topstories.json")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching story IDs: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def fetch_story_details(story_id: int, base_url: str = "https://hacker-news.firebaseio.com/v0") -> Optional[Dict]:
    """
    Fetch details for a single story.

    Args:
        story_id: The story ID to fetch

    Returns:
        Dictionary of story details or None if fetch fails
    """
    try:
        response = requests.get(f"{base_url}/item/{story_id}.json")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Warning: Failed to fetch story {story_id}: {e}")
        return None
    except Exception as e:
        print(f"Warning: Error processing story {story_id}: {e}")
        return None


def extract_story_data(item_data: Dict) -> Optional[Dict]:
    """
    Extract relevant fields from story item data.

    Args:
        item_data: Raw story data from API

    Returns:
        Dictionary with extracted fields or None if invalid
    """
    if not item_data or 'title' not in item_data:
        return None

    return {
        'title': item_data.get('title', ''),
        'score': item_data.get('score', 0),
        'descendants': item_data.get('descendants', 0),
        'time': datetime.fromtimestamp(item_data.get('time', 0)),
        'url': item_data.get('url', ''),
    }


def process_stories_to_dataframe(stories_data: List[Dict]) -> pd.DataFrame:
    """
    Convert list of story dictionaries to a sorted DataFrame.

    Args:
        stories_data: List of story dictionaries

    Returns:
        Sorted DataFrame or empty DataFrame if no data
    """
    if not stories_data:
        return pd.DataFrame()

    df = pd.DataFrame(stories_data)
    df = df.sort_values('score', ascending=False)
    df = df.reset_index(drop=True)

    return df


def fetch_hn_data(limit: int = 30) -> pd.DataFrame:
    """
    Fetch top stories from Hacker News and return structured data.

    Args:
        limit: Number of stories to fetch (default: 30)

    Returns:
        Pandas DataFrame containing story data with columns:
        - title (str): Story title
        - score (int): Number of upvotes
        - descendants (int): Comment count
        - time (datetime): Story creation time
        - url (str): Link to the article
    """
    base_url = "https://hacker-news.firebaseio.com/v0"
    stories_data = []

    # Fetch top story IDs
    print("Fetching top story IDs...")
    story_ids = fetch_story_ids(base_url)

    if story_ids is None:
        return pd.DataFrame()

    # Limit the number of stories to fetch
    story_ids = story_ids[:limit]
    print(f"Fetching details for {len(story_ids)} stories...")

    # Fetch details for each story
    for i, story_id in enumerate(story_ids, 1):
        print(f"Fetching story {i}/{len(story_ids)} (ID: {story_id})")

        item_data = fetch_story_details(story_id, base_url)
        if item_data is None:
            continue

        story_dict = extract_story_data(item_data)
        if story_dict is not None:
            stories_data.append(story_dict)
        else:
            print(f"Warning: Story {story_id} has no title, skipping...")

    # Convert to DataFrame
    df = process_stories_to_dataframe(stories_data)

    if not df.empty:
        print(f"Successfully fetched {len(df)} stories.")
    else:
        print("No stories were successfully fetched.")

    return df


if __name__ == '__main__':
    print('Fetching data...')
    df = fetch_hn_data(limit=5)
    print(df.head())
    print(f'Successfully fetched {len(df)} stories.')