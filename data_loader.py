import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import unittest.mock as mock

# Analysis imports
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bertopic import BERTopic

# Cache imports
from cache_manager import CacheManager

# Download NLTK data if needed
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading VADER lexicon for sentiment analysis...")
    nltk.download('vader_lexicon')


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


def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze sentiment of story titles using VADER.

    Args:
        df: DataFrame containing story data with 'title' column

    Returns:
        DataFrame with added sentiment_score and sentiment_label columns
    """
    if df.empty or 'title' not in df.columns:
        print("Warning: DataFrame is empty or missing 'title' column")
        return df

    print("Analyzing sentiment using VADER...")

    # Initialize the VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Apply sentiment analysis to the title column
    sentiment_scores = []
    for title in df['title']:
        if pd.isna(title) or title == '':
            sentiment_scores.append({'compound': 0.0})
        else:
            scores = sia.polarity_scores(str(title))
            sentiment_scores.append(scores)

    # Extract compound scores
    df['sentiment_score'] = [score['compound'] for score in sentiment_scores]

    # Create sentiment labels based on compound score
    df['sentiment_label'] = df['sentiment_score'].apply(
        lambda score: 'Positive' if score > 0.05
                     else 'Negative' if score < -0.05
                     else 'Neutral'
    )

    # Print sentiment summary
    sentiment_counts = df['sentiment_label'].value_counts()
    print(f"Sentiment Analysis Results:")
    for label, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {label}: {count} stories ({percentage:.1f}%)")

    return df


def get_topics(df: pd.DataFrame, embedding_model: str = 'all-MiniLM-L6-v2') -> pd.DataFrame:
    """
    Extract topics from story titles using BERTopic.

    Args:
        df: DataFrame containing story data with 'title' column
        embedding_model: Name of the sentence transformer model to use

    Returns:
        DataFrame with added topic_id and topic_keyword columns
    """
    if df.empty or 'title' not in df.columns:
        print("Warning: DataFrame is empty or missing 'title' column")
        return df

    print("Extracting topics using BERTopic...")

    # Filter out empty or NaN titles
    valid_titles = df['title'].dropna().astype(str).tolist()
    valid_indices = df.index[df['title'].notna()].tolist()

    if len(valid_titles) < 2:
        print("Warning: Need at least 2 valid titles for topic modeling")
        df['topic_id'] = -1
        df['topic_keyword'] = 'Insufficient Data'
        return df

    try:
        # Initialize BERTopic model
        topic_model = BERTopic(
            embedding_model=embedding_model,
            min_topic_size=2,
            verbose=True
        )

        # Fit the model and transform the documents
        topics, probs = topic_model.fit_transform(valid_titles)

        # Get topic info for keywords
        topic_info = topic_model.get_topic_info()

        # Create a mapping from topic ID to keywords
        topic_keywords = {}
        for _, row in topic_info.iterrows():
            if row['Topic'] == -1:
                topic_keywords[row['Topic']] = 'Outlier/No Topic'
            else:
                # Extract top keywords (join with underscores)
                keywords = '_'.join(row['Representation'][:3])
                topic_keywords[row['Topic']] = keywords

        # Add topic information to DataFrame
        df['topic_id'] = -1
        df['topic_keyword'] = 'No Data'

        for idx, topic_id in zip(valid_indices, topics):
            df.at[idx, 'topic_id'] = topic_id
            df.at[idx, 'topic_keyword'] = topic_keywords.get(topic_id, f'Topic_{topic_id}')

        # Print topic summary
        topic_counts = df['topic_id'].value_counts()
        print(f"\nTopic Modeling Results:")
        print(f"  Total Topics Found: {len(topic_counts) - 1 if -1 in topic_counts else len(topic_counts)}")

        # Show top topics
        for topic_id, count in topic_counts.head(5).items():
            if topic_id == -1:
                print(f"  Outliers: {count} stories")
            else:
                keyword = topic_keywords.get(topic_id, f'Topic_{topic_id}')
                percentage = (count / len(df)) * 100
                print(f"  Topic {topic_id} ({keyword}): {count} stories ({percentage:.1f}%)")

        return df

    except Exception as e:
        print(f"Error in topic modeling: {e}")
        df['topic_id'] = -1
        df['topic_keyword'] = 'Error'
        return df


def fetch_hn_data(limit: int = 30, use_cache: bool = True, cache_duration_hours: int = 1) -> pd.DataFrame:
    """
    Fetch top stories from Hacker News and return structured data.
    Uses cache to avoid re-fetching the same data repeatedly.

    Args:
        limit: Number of stories to fetch (default: 30)
        use_cache: Whether to use cached data if available (default: True)
        cache_duration_hours: How long cached data remains valid (default: 1)

    Returns:
        Pandas DataFrame containing story data with columns:
        - title (str): Story title
        - score (int): Number of upvotes
        - descendants (int): Comment count
        - time (datetime): Story creation time
        - url (str): Link to the article
    """
    # Initialize cache manager
    cache_manager = CacheManager(cache_duration_hours=cache_duration_hours)

    # Try to get data from cache first
    if use_cache:
        cached_df = cache_manager.get_cached_data(limit)
        if cached_df is not None and not cached_df.empty:
            print(f"Using cached data: {len(cached_df)} stories")
            return cached_df
        else:
            print("No valid cache found, fetching fresh data...")

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

        # Save to cache for future use
        if use_cache:
            cache_manager.save_to_cache(df, limit)
            print("Data saved to cache")
    else:
        print("No stories were successfully fetched.")

    return df


if __name__ == '__main__':
    print('1. Fetching data...')
    df = fetch_hn_data(limit=20)  # Fetch 20 for a good sample

    print('\n2. Analyzing Sentiment...')
    df = analyze_sentiment(df)

    print('\n3. Extracting Topics...')
    df = get_topics(df)

    print('\nTop 5 Rows with Analysis:')
    # Display relevant columns
    display_columns = ['title', 'sentiment_label', 'topic_keyword', 'score']

    # Make sure all columns exist before selecting
    available_columns = [col for col in display_columns if col in df.columns]
    if available_columns:
        print(df[available_columns].head())
    else:
        print(df.head())

    print(f'\nSuccessfully analyzed {len(df)} stories with sentiment and topics.')

    # Show summary statistics
    if 'sentiment_label' in df.columns:
        print(f"\nSentiment Distribution:")
        print(df['sentiment_label'].value_counts())

    if 'topic_keyword' in df.columns:
        print(f"\nTop Topics:")
        print(df['topic_keyword'].value_counts().head(10))