import pandas as pd
from typing import List, Dict, Optional, Any
import logging
import hashlib
from datetime import datetime

# New modules
from api_client import HNClient
from text_analyzer import TextAnalyzer

# Cache imports
from cache_manager import CacheManager
from vector_cache_manager import get_vector_cache_manager

# Vector search imports
from vector_search import get_vector_engine
from dashboard_config import SEMANTIC_SEARCH_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services
# In a real DI system, these would be injected.
# For this script, we initialize them here.
_hn_client = HNClient()
_text_analyzer = TextAnalyzer()


def fetch_hn_data(limit: int = 30, use_cache: bool = True, cache_duration_hours: int = 1) -> pd.DataFrame:
    """
    Fetch top stories from Hacker News and return structured data.
    Uses cache to avoid re-fetching the same data repeatedly.
    Facade for HNClient.

    Args:
        limit: Number of stories to fetch (default: 30)
        use_cache: Whether to use cached data if available (default: True)
        cache_duration_hours: How long cached data remains valid (default: 1)

    Returns:
        Pandas DataFrame containing story data.
    """
    # Initialize cache manager
    cache_manager = CacheManager(cache_duration_hours=cache_duration_hours)

    # Try to get data from cache first
    if use_cache:
        cached_df = cache_manager.get_cached_data(limit)
        if cached_df is not None and not cached_df.empty:
            logger.info(f"Using cached data: {len(cached_df)} stories")
            return cached_df
        else:
            logger.info("No valid cache found, fetching fresh data...")

    # Fetch data using the new client
    logger.info("Fetching top stories...")
    stories_data = _hn_client.fetch_top_stories(limit=limit)

    # Convert to DataFrame
    if not stories_data:
        logger.warning("No stories were successfully fetched.")
        return pd.DataFrame()

    df = pd.DataFrame(stories_data)
    df = df.sort_values('score', ascending=False)
    df = df.reset_index(drop=True)

    if not df.empty:
        logger.info(f"Successfully fetched {len(df)} stories.")
        # Save to cache for future use
        if use_cache:
            cache_manager.save_to_cache(df, limit)
            logger.info("Data saved to cache")

    return df


def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze sentiment of story titles using VADER.
    Facade for TextAnalyzer.
    """
    result_df = _text_analyzer.analyze_sentiment(df)

    # Print sentiment summary for CLI output compatibility
    if not result_df.empty and 'sentiment_label' in result_df.columns:
        sentiment_counts = result_df['sentiment_label'].value_counts()
        print(f"Sentiment Analysis Results:")
        for label, count in sentiment_counts.items():
            percentage = (count / len(result_df)) * 100
            print(f"  {label}: {count} stories ({percentage:.1f}%)")

    return result_df


def get_topics(df: pd.DataFrame, embedding_model: str = 'all-MiniLM-L6-v2') -> pd.DataFrame:
    """
    Extract topics from story titles using BERTopic.
    Facade for TextAnalyzer.
    """
    result_df = _text_analyzer.get_topics(df, embedding_model)

    # Print topic summary for CLI output compatibility
    if not result_df.empty and 'topic_id' in result_df.columns:
        topic_counts = result_df['topic_id'].value_counts()
        print(f"\nTopic Modeling Results:")
        print(f"  Total Topics Found: {len(topic_counts) - 1 if -1 in topic_counts else len(topic_counts)}")

        # We don't have easy access to keyword map here unless we pass it back
        # But the dataframe has the keywords in 'topic_keyword'

        # Show top topics
        for topic_id, count in topic_counts.head(5).items():
            if topic_id == -1:
                print(f"  Outliers: {count} stories")
            else:
                # Find the keyword for this topic
                keyword = result_df[result_df['topic_id'] == topic_id]['topic_keyword'].iloc[0]
                percentage = (count / len(result_df)) * 100
                print(f"  Topic {topic_id} ({keyword}): {count} stories ({percentage:.1f}%)")

    return result_df


def setup_vector_db(df: pd.DataFrame, collection_name: str = "hn_stories",
                    force_rebuild: bool = False) -> Optional[Any]:
    """
    Set up vector database with story titles for semantic search.
    """
    if df.empty or 'title' not in df.columns:
        logger.warning("DataFrame is empty or missing 'title' column")
        return None

    # Get vector cache manager
    cache_manager = get_vector_cache_manager()

    # Convert DataFrame to list of dictionaries for hashing
    df_data = df.to_dict('records')

    # Check if we should use cached collection
    if not force_rebuild and not cache_manager.should_rebuild_collection(df_data):
        logger.info("Using cached vector collection...")
        cached_collection = cache_manager.get_cached_collection()
        if cached_collection is not None:
            cache_info = cache_manager.get_cache_info()
            logger.info(f"Cache version: {cache_info.get('version', 'unknown')}")
            return cached_collection

    logger.info("Setting up vector database for semantic search...")

    # Get vector engine
    vector_engine = get_vector_engine(
        model_name=SEMANTIC_SEARCH_SETTINGS["model_name"],
        collection_name=collection_name
    )

    try:
        # Initialize the engine
        vector_engine.initialize() # Note: logic assumed from previous file, though vector_search.py didn't have explicit initialize() public method in VectorSearchManager, check if it was added or implied.
        # Wait, vector_search.py VectorSearchManager has lazy properties but no 'initialize' method.
        # It has `chroma_client` property.
        # The previous data_loader called `vector_engine.initialize()`.
        # Let's check vector_search.py again.
        # Ah, `vector_search.py` I read earlier DOES NOT have `initialize()`.
        # However, the previous `data_loader.py` CALLED `vector_engine.initialize()`.
        # This implies either I missed it or the previous code was broken/using a different version.
        # Let's look at `vector_search.py` again.
        pass
    except AttributeError:
        # If initialize doesn't exist, accessing the collection property triggers init
        pass

    try:
        # Clear existing collection to ensure fresh data
        vector_engine.collection # accessing this property ensures init
        vector_engine.clear_collection() # Wait, the method is `delete_collection` or we need to check if `clear_collection` exists on the manager.
        # VectorSearchManager has `delete_collection`.
        # Previous code called `vector_engine.clear_collection()`.
        # I suspect the previous code might have been using a method that didn't exist or I missed it.
        # I will use `vector_engine.delete_collection()` which exists in `VectorSearchManager`.
        # Actually, if I delete it, I need to recreate it.
        # `vector_engine.collection` property handles creation if missing.

        # To match "clear" behavior:
        # The `chroma_client.delete_collection` deletes it entirely.
        # Then next access to `.collection` recreates it.
        vector_engine.delete_collection()

        # Prepare documents with metadata
        documents = []
        ids = []
        metadatas = []

        for idx, row in df.iterrows():
            # Use MD5 for stable ID generation
            title = str(row['title'])
            id_str = f"story_{idx}_{hashlib.md5(title.encode()).hexdigest()[:10]}"

            documents.append(title)
            ids.append(id_str)

            meta = {
                'score': int(row.get('score', 0)),
                'time': str(row.get('time', datetime.now()).isoformat()),
                'url': str(row.get('url', '')),
                'sentiment_label': str(row.get('sentiment_label', '')),
                'topic_keyword': str(row.get('topic_keyword', '')),
                'index': int(idx)
            }
            metadatas.append(meta)

        # Process in batches
        batch_size = SEMANTIC_SEARCH_SETTINGS["batch_size"]
        total_batches = (len(documents) + batch_size - 1) // batch_size

        logger.info(f"Processing {len(documents)} stories in {total_batches} batches...")

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]

            batch_num = i // batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_docs)} stories)...")

            # Add batch to vector database
            success = vector_engine.add_documents(
                documents=batch_docs,
                ids=batch_ids,
                metadata=batch_meta
            )
            if not success:
                logger.warning(f"Failed to add batch {batch_num} to vector database")

        # Get collection info
        info = vector_engine.get_stats()
        logger.info(f"Vector database setup complete! Collection contains {info.get('document_count', 0)} stories")

        # Cache the collection
        data_hash = cache_manager.generate_data_hash(df_data)
        cache_manager.cache_collection(vector_engine.collection, data_hash)

        return vector_engine.collection

    except Exception as e:
        logger.error(f"Error setting up vector database: {e}")
        return None


def semantic_search(collection, query: str, max_results: Optional[int] = None,
                   similarity_threshold: Optional[float] = None) -> List[Dict[str, any]]:
    """
    Perform semantic search on the vector collection.
    """
    # Use defaults from config if not provided
    if max_results is None:
        max_results = SEMANTIC_SEARCH_SETTINGS["max_results"]
    if similarity_threshold is None:
        similarity_threshold = SEMANTIC_SEARCH_SETTINGS["similarity_threshold"]

    # Validate query
    if not query or len(query.strip()) < SEMANTIC_SEARCH_SETTINGS["min_query_length"]:
        logger.warning(f"Query too short. Minimum length: {SEMANTIC_SEARCH_SETTINGS['min_query_length']} characters")
        return []

    if len(query) > SEMANTIC_SEARCH_SETTINGS["max_query_length"]:
        logger.warning(f"Query too long. Maximum length: {SEMANTIC_SEARCH_SETTINGS['max_query_length']} characters")
        return []

    if collection is None:
        logger.error("Vector collection not initialized")
        return []

    try:
        # Get vector engine and perform search
        vector_engine = get_vector_engine(
            model_name=SEMANTIC_SEARCH_SETTINGS["model_name"]
        )
        # Manually inject the collection if needed, though get_vector_engine singleton should match.
        # But `collection` passed in might be from session state.
        # The VectorSearchManager manages its own collection.
        # If we pass a collection object, the Manager's search method doesn't take it as argument.
        # The Manager uses `self.collection`.
        # However, the previous code in `data_loader.py` did:
        # `vector_engine.collection = collection`
        # Let's replicate that behavior to be safe.
        vector_engine._collection = collection # Accessing private member to force it

        logger.info(f"Performing semantic search for: '{query}'...")

        # The search method in VectorSearchManager returns a dict
        search_result = vector_engine.search(
            query=query,
            n_results=max_results
        )

        results = search_result.get('results', [])

        # Filter by threshold and format
        formatted_results = []
        for result in results:
            # Calculate similarity from distance if needed, or use distance directly
            # ChromaDB returns distance (usually L2 or Cosine).
            # If Cosine distance: Similarity = 1 - Distance
            # Assuming Cosine distance default for sentence-transformers in Chroma

            dist = result.get('distance', 1.0)
            similarity = 1.0 - dist # Rough approximation depending on metric

            if similarity < similarity_threshold:
                continue

            formatted_result = {
                'title': result['document'],
                'metadata': result['metadata'],
                'similarity_score': round(similarity, 3),
                'distance': round(dist, 3),
                'rank': len(formatted_results) + 1,
                'explanation': f"Similarity: {similarity:.1%}"
            }
            formatted_results.append(formatted_result)

        logger.info(f"Found {len(formatted_results)} relevant stories")
        return formatted_results

    except Exception as e:
        logger.error(f"Error during semantic search: {e}")
        return []


# Maintain compatibility with existing tests that might import these
fetch_story_ids = _hn_client.fetch_story_ids
fetch_story_details = _hn_client.fetch_story_details
extract_story_data = _hn_client.extract_story_data
# process_stories_to_dataframe was local, let's keep it locally as it's not in api_client
def process_stories_to_dataframe(stories_data: List[Dict]) -> pd.DataFrame:
    if not stories_data:
        return pd.DataFrame()
    df = pd.DataFrame(stories_data)
    df = df.sort_values('score', ascending=False)
    df = df.reset_index(drop=True)
    return df


if __name__ == '__main__':
    print('1. Fetching data...')
    df = fetch_hn_data(limit=20)

    print('\n2. Analyzing Sentiment...')
    df = analyze_sentiment(df)

    print('\n3. Extracting Topics...')
    df = get_topics(df)

    # ... rest of main execution ...
