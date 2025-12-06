"""
Text Analysis Module.
Handles Sentiment Analysis (VADER) and Topic Modeling (BERTopic).
"""

import logging
import pandas as pd
from typing import Optional, List, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)


class TextAnalyzer:
    """
    Analyzer for text data using VADER and BERTopic.
    Uses lazy loading for heavy libraries.
    """

    def __init__(self):
        self._sia = None
        self._bertopic_model = None

    @property
    def sia(self):
        """Lazy load VADER SentimentIntensityAnalyzer"""
        if self._sia is None:
            import nltk
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            try:
                nltk.data.find('sentiment/vader_lexicon.zip')
            except LookupError:
                logger.info("Downloading VADER lexicon for sentiment analysis...")
                nltk.download('vader_lexicon', quiet=True)

            self._sia = SentimentIntensityAnalyzer()
        return self._sia

    def analyze_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment of story titles using VADER.

        Args:
            df: DataFrame containing story data with 'title' column

        Returns:
            DataFrame with added sentiment_score and sentiment_label columns
        """
        if df.empty or 'title' not in df.columns:
            logger.warning("DataFrame is empty or missing 'title' column")
            return df

        logger.info("Analyzing sentiment using VADER...")

        # Apply sentiment analysis to the title column
        sentiment_scores = []
        for title in df['title']:
            if pd.isna(title) or title == '':
                sentiment_scores.append({'compound': 0.0})
            else:
                scores = self.sia.polarity_scores(str(title))
                sentiment_scores.append(scores)

        # Extract compound scores
        df['sentiment_score'] = [score['compound'] for score in sentiment_scores]

        # Create sentiment labels based on compound score
        df['sentiment_label'] = df['sentiment_score'].apply(
            lambda score: 'Positive' if score > 0.05
                         else 'Negative' if score < -0.05
                         else 'Neutral'
        )

        return df

    def get_topics(self, df: pd.DataFrame, embedding_model: str = 'all-MiniLM-L6-v2') -> pd.DataFrame:
        """
        Extract topics from story titles using BERTopic.

        Args:
            df: DataFrame containing story data with 'title' column
            embedding_model: Name of the sentence transformer model to use

        Returns:
            DataFrame with added topic_id and topic_keyword columns
        """
        if df.empty or 'title' not in df.columns:
            logger.warning("DataFrame is empty or missing 'title' column")
            return df

        logger.info("Extracting topics using BERTopic...")

        # Lazy import BERTopic here to avoid overhead if not used
        from bertopic import BERTopic

        # Filter out empty or NaN titles
        valid_titles = df['title'].dropna().astype(str).tolist()
        valid_indices = df.index[df['title'].notna()].tolist()

        if len(valid_titles) < 2:
            logger.warning("Need at least 2 valid titles for topic modeling")
            df['topic_id'] = -1
            df['topic_keyword'] = 'Insufficient Data'
            return df

        try:
            # Initialize BERTopic model
            # Note: Re-initializing per call is expensive but necessary if parameters change.
            # For a long-running service, we might want to cache the model if parameters are static.
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

            return df

        except Exception as e:
            logger.error(f"Error in topic modeling: {e}")
            df['topic_id'] = -1
            df['topic_keyword'] = 'Error'
            return df
