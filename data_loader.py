import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional


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

    try:
        # Fetch top story IDs
        print("Fetching top story IDs...")
        response = requests.get(f"{base_url}/topstories.json")
        response.raise_for_status()
        story_ids = response.json()

        # Limit the number of stories to fetch
        story_ids = story_ids[:limit]
        print(f"Fetching details for {len(story_ids)} stories...")

        # Fetch details for each story
        for i, story_id in enumerate(story_ids, 1):
            try:
                print(f"Fetching story {i}/{len(story_ids)} (ID: {story_id})")
                item_response = requests.get(f"{base_url}/item/{story_id}.json")
                item_response.raise_for_status()
                item_data = item_response.json()

                # Skip if item is None or doesn't have required fields
                if not item_data or 'title' not in item_data:
                    print(f"Warning: Story {story_id} has no title, skipping...")
                    continue

                # Extract story data
                story_dict = {
                    'title': item_data.get('title', ''),
                    'score': item_data.get('score', 0),
                    'descendants': item_data.get('descendants', 0),
                    'time': datetime.fromtimestamp(item_data.get('time', 0)),
                    'url': item_data.get('url', ''),
                }

                stories_data.append(story_dict)

            except requests.exceptions.RequestException as e:
                print(f"Warning: Failed to fetch story {story_id}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing story {story_id}: {e}")
                continue

    except requests.exceptions.RequestException as e:
        print(f"Error fetching story IDs: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error: {e}")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(stories_data)

    if not df.empty:
        # Sort by score (highest first)
        df = df.sort_values('score', ascending=False)
        df = df.reset_index(drop=True)

        print(f"Successfully fetched {len(df)} stories.")
    else:
        print("No stories were successfully fetched.")

    return df


if __name__ == '__main__':
    print('Fetching data...')
    df = fetch_hn_data(limit=5)
    print(df.head())
    print(f'Successfully fetched {len(df)} stories.')