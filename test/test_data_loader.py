import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd
import requests.exceptions

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_client import HNClient
from text_analyzer import TextAnalyzer
from data_loader import fetch_hn_data, analyze_sentiment, get_topics


class TestHNClient(unittest.TestCase):
    """Test cases for HNClient"""

    def setUp(self):
        self.client = HNClient()

    @patch('requests.get')
    def test_fetch_story_ids_success(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = [1, 2, 3]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        ids = self.client.fetch_story_ids()
        self.assertEqual(ids, [1, 2, 3])

    @patch('requests.get')
    def test_fetch_story_details_success(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {'id': 1, 'title': 'Test'}
        mock_get.return_value = mock_response

        details = self.client.fetch_story_details(1)
        self.assertEqual(details['title'], 'Test')

    def test_extract_story_data(self):
        raw = {
            'title': 'Test Story',
            'score': 100,
            'descendants': 10,
            'time': 1600000000,
            'url': 'http://test.com'
        }
        extracted = self.client.extract_story_data(raw)
        self.assertEqual(extracted['title'], 'Test Story')
        self.assertEqual(extracted['score'], 100)
        self.assertIsInstance(extracted['time'], datetime)

    @patch('api_client.HNClient.fetch_story_ids')
    @patch('api_client.HNClient.fetch_story_details')
    def test_fetch_top_stories_concurrent(self, mock_details, mock_ids):
        """Test that fetch_top_stories calls details concurrently"""
        mock_ids.return_value = list(range(10))

        # Simulate details fetch
        def get_details(sid):
            return {'id': sid, 'title': f'Story {sid}', 'score': sid*10, 'time': 1600000000}

        mock_details.side_effect = get_details

        stories = self.client.fetch_top_stories(limit=5)

        self.assertEqual(len(stories), 5)
        self.assertEqual(mock_details.call_count, 5)


class TestTextAnalyzer(unittest.TestCase):
    """Test cases for TextAnalyzer"""

    def setUp(self):
        self.analyzer = TextAnalyzer()

    @patch('nltk.sentiment.vader.SentimentIntensityAnalyzer')
    @patch('nltk.data.find')
    def test_analyze_sentiment(self, mock_find, mock_sia_cls):
        # Setup mocks
        mock_sia_instance = Mock()
        mock_sia_instance.polarity_scores.return_value = {'compound': 0.8}
        mock_sia_cls.return_value = mock_sia_instance

        df = pd.DataFrame({'title': ['Good story']})

        result_df = self.analyzer.analyze_sentiment(df)

        self.assertIn('sentiment_score', result_df.columns)
        self.assertEqual(result_df['sentiment_score'].iloc[0], 0.8)
        self.assertEqual(result_df['sentiment_label'].iloc[0], 'Positive')

    @patch('bertopic.BERTopic')
    def test_get_topics(self, mock_bertopic_cls):
        # Setup mocks
        mock_model = Mock()
        mock_model.fit_transform.return_value = ([0, 1], None)
        mock_model.get_topic_info.return_value = pd.DataFrame({
            'Topic': [0, 1],
            'Representation': [['ai', 'ml', 'tech'], ['web', 'dev', 'code']]
        })
        mock_bertopic_cls.return_value = mock_model

        df = pd.DataFrame({'title': ['AI Story', 'Web Story']})

        result_df = self.analyzer.get_topics(df)

        self.assertIn('topic_id', result_df.columns)
        self.assertIn('topic_keyword', result_df.columns)
        self.assertEqual(result_df['topic_keyword'].iloc[0], 'ai_ml_tech')


class TestDataLoaderFacade(unittest.TestCase):
    """Test that data_loader correctly delegates to new modules"""

    @patch('data_loader._hn_client')
    @patch('data_loader.CacheManager')
    def test_fetch_hn_data_delegation(self, mock_cache_cls, mock_client):
        # Mock cache to miss
        mock_cache = Mock()
        mock_cache.get_cached_data.return_value = None
        mock_cache_cls.return_value = mock_cache

        # Mock client response
        mock_client.fetch_top_stories.return_value = [
            {'title': 'Story 1', 'score': 100},
            {'title': 'Story 2', 'score': 200}
        ]

        df = fetch_hn_data(limit=2)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        mock_client.fetch_top_stories.assert_called_with(limit=2)

    @patch('data_loader._text_analyzer')
    def test_analyze_sentiment_delegation(self, mock_analyzer):
        df = pd.DataFrame({'title': ['Test']})
        mock_analyzer.analyze_sentiment.return_value = df

        fetch_result = analyze_sentiment(df)

        mock_analyzer.analyze_sentiment.assert_called_with(df)
        pd.testing.assert_frame_equal(fetch_result, df)


if __name__ == '__main__':
    unittest.main()
