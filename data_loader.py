import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Any
import unittest.mock as mock

# Analysis imports
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bertopic import BERTopic

# Cache imports
from cache_manager import CacheManager

# Vector search imports
from vector_search import get_vector_engine
from dashboard_config import SEMANTIC_SEARCH_SETTINGS

# Vector cache imports
from vector_cache_manager import get_vector_cache_manager

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


def setup_vector_db(df: pd.DataFrame, collection_name: str = "hn_stories",
                    force_rebuild: bool = False) -> Optional[Any]:
    """
    Set up vector database with story titles for semantic search.
    Uses caching to avoid rebuilding when data hasn't changed.

    Args:
        df: DataFrame containing story data with 'title' column
        collection_name: Name for the ChromaDB collection
        force_rebuild: Force rebuild regardless of cache

    Returns:
        ChromaDB collection object or None if setup fails
    """
    if df.empty or 'title' not in df.columns:
        print("Warning: DataFrame is empty or missing 'title' column")
        return None

    # Get vector cache manager
    cache_manager = get_vector_cache_manager()

    # Convert DataFrame to list of dictionaries for hashing
    df_data = df.to_dict('records')

    # Check if we should use cached collection
    if not force_rebuild and not cache_manager.should_rebuild_collection(df_data):
        print("Using cached vector collection...")
        cached_collection = cache_manager.get_cached_collection()
        if cached_collection is not None:
            cache_info = cache_manager.get_cache_info()
            print(f"Cache version: {cache_info.get('version', 'unknown')}")
            return cached_collection

    print("Setting up vector database for semantic search...")

    # Get vector engine
    vector_engine = get_vector_engine(
        model_name=SEMANTIC_SEARCH_SETTINGS["model_name"],
        collection_name=collection_name
    )

    try:
        # Initialize the engine
        vector_engine.initialize()

        # Clear existing collection to ensure fresh data
        vector_engine.clear_collection()

        # Prepare documents with metadata
        documents = []
        for idx, row in df.iterrows():
            doc = {
                'id': f"story_{idx}_{hash(row['title']) % 1000000}",  # Generate unique ID
                'text': row['title'],
                'score': row.get('score', 0),
                'time': row.get('time', datetime.now()).isoformat(),
                'url': row.get('url', ''),
                'sentiment_label': row.get('sentiment_label', ''),
                'topic_keyword': row.get('topic_keyword', ''),
                'index': idx  # Keep original index for reference
            }
            documents.append(doc)

        # Process in batches
        batch_size = SEMANTIC_SEARCH_SETTINGS["batch_size"]
        total_batches = (len(documents) + batch_size - 1) // batch_size

        print(f"Processing {len(documents)} stories in {total_batches} batches...")

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1

            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} stories)...")

            # Add batch to vector database
            success = vector_engine.add_documents(batch)
            if not success:
                print(f"Warning: Failed to add batch {batch_num} to vector database")

        # Get collection info
        info = vector_engine.get_collection_info()
        print(f"Vector database setup complete! Collection '{info['name']}' contains {info['count']} stories")

        # Cache the collection
        data_hash = cache_manager.generate_data_hash(df_data)
        cache_manager.cache_collection(vector_engine.collection, data_hash)

        return vector_engine.collection

    except Exception as e:
        print(f"Error setting up vector database: {e}")
        return None


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


def semantic_search(collection, query: str, max_results: Optional[int] = None,
                   similarity_threshold: Optional[float] = None) -> List[Dict[str, any]]:
    """
    Perform semantic search on the vector collection.

    Args:
        collection: ChromaDB collection object
        query: Search query string
        max_results: Maximum number of results to return
        similarity_threshold: Minimum similarity score threshold

    Returns:
        List of search results with document metadata and similarity scores
    """
    # Use defaults from config if not provided
    if max_results is None:
        max_results = SEMANTIC_SEARCH_SETTINGS["max_results"]
    if similarity_threshold is None:
        similarity_threshold = SEMANTIC_SEARCH_SETTINGS["similarity_threshold"]

    # Validate query
    if not query or len(query.strip()) < SEMANTIC_SEARCH_SETTINGS["min_query_length"]:
        print(f"Query too short. Minimum length: {SEMANTIC_SEARCH_SETTINGS['min_query_length']} characters")
        return []

    if len(query) > SEMANTIC_SEARCH_SETTINGS["max_query_length"]:
        print(f"Query too long. Maximum length: {SEMANTIC_SEARCH_SETTINGS['max_query_length']} characters")
        return []

    if collection is None:
        print("Error: Vector collection not initialized")
        return []

    try:
        # Get vector engine and perform search
        vector_engine = get_vector_engine(
            model_name=SEMANTIC_SEARCH_SETTINGS["model_name"]
        )
        vector_engine.collection = collection  # Use the provided collection

        print(f"Performing semantic search for: '{query}'...")
        results = vector_engine.search(
            query=query,
            n_results=max_results,
            similarity_threshold=similarity_threshold
        )

        # Format results with additional context
        formatted_results = []
        for result in results:
            formatted_result = {
                'title': result['document'],
                'metadata': result['metadata'],
                'similarity_score': round(result['similarity'], 3),
                'distance': round(result['distance'], 3),
                'rank': len(formatted_results) + 1,
                'explanation': f"Similarity: {result['similarity']:.1%}"
            }
            formatted_results.append(formatted_result)

        print(f"Found {len(formatted_results)} relevant stories")
        return formatted_results

    except Exception as e:
        print(f"Error during semantic search: {e}")
        return []


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

    print('\n4. Setting up Vector Database...')
    vector_collection = setup_vector_db(df)

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

    # Test semantic search if vector database was set up
    if vector_collection:
        print('\n5. Testing Semantic Search...')
        test_queries = [
            "artificial intelligence",
            "python programming",
            "startup funding"
        ]

        for query in test_queries:
            print(f"\nSearching for: '{query}'")
            results = semantic_search(vector_collection, query)
            if results:
                for i, result in enumerate(results[:3], 1):
                    print(f"  {i}. {result['title'][:80]}...")
                    print(f"     Similarity: {result['similarity_score']:.3f}")
            else:
                print("  No results found")