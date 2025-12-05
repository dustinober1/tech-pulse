import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd
import requests.exceptions

# Add the parent directory to the path to import data_loader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import (
    fetch_story_ids,
    fetch_story_details,
    extract_story_data,
    process_stories_to_dataframe,
    fetch_hn_data,
    analyze_sentiment,
    get_topics
)


class TestFetchStoryIds(unittest.TestCase):
    """Test cases for fetch_story_ids function"""

    @patch('data_loader.requests.get')
    def test_fetch_story_ids_success(self, mock_get):
        """Test successful fetch of story IDs"""
        mock_response = Mock()
        mock_response.json.return_value = [1, 2, 3, 4, 5]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = fetch_story_ids()

        self.assertEqual(result, [1, 2, 3, 4, 5])
        mock_get.assert_called_once_with("https://hacker-news.firebaseio.com/v0/topstories.json")

    @patch('data_loader.requests.get')
    @patch('builtins.print')
    def test_fetch_story_ids_connection_error(self, mock_print, mock_get):
        """Test handling of connection errors"""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")

        result = fetch_story_ids()

        self.assertIsNone(result)
        mock_print.assert_called_with("Error fetching story IDs: Connection failed")

    @patch('data_loader.requests.get')
    @patch('builtins.print')
    def test_fetch_story_ids_json_error(self, mock_print, mock_get):
        """Test handling of JSON decoding errors"""
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = fetch_story_ids()

        self.assertIsNone(result)
        mock_print.assert_called_with("Unexpected error: Invalid JSON")

    @patch('data_loader.requests.get')
    def test_fetch_story_ids_custom_url(self, mock_get):
        """Test fetching from custom base URL"""
        mock_response = Mock()
        mock_response.json.return_value = [10, 20, 30]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = fetch_story_ids("https://custom-api.example.com/v0")

        self.assertEqual(result, [10, 20, 30])
        mock_get.assert_called_once_with("https://custom-api.example.com/v0/topstories.json")


class TestFetchStoryDetails(unittest.TestCase):
    """Test cases for fetch_story_details function"""

    @patch('data_loader.requests.get')
    def test_fetch_story_details_success(self, mock_get):
        """Test successful fetch of story details"""
        mock_response = Mock()
        mock_response.json.return_value = {
            'id': 123,
            'title': 'Test Story',
            'score': 100,
            'descendants': 50,
            'time': 1700000000,
            'url': 'https://example.com'
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = fetch_story_details(123)

        self.assertEqual(result['id'], 123)
        self.assertEqual(result['title'], 'Test Story')
        mock_get.assert_called_once_with("https://hacker-news.firebaseio.com/v0/item/123.json")

    @patch('data_loader.requests.get')
    @patch('builtins.print')
    def test_fetch_story_details_not_found(self, mock_print, mock_get):
        """Test handling of 404 errors"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        result = fetch_story_details(999)

        self.assertIsNone(result)
        mock_print.assert_called_with("Warning: Failed to fetch story 999: 404 Not Found")

    @patch('data_loader.requests.get')
    def test_fetch_story_details_custom_url(self, mock_get):
        """Test fetching from custom base URL"""
        mock_response = Mock()
        mock_response.json.return_value = {'id': 456, 'title': 'Custom Story'}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = fetch_story_details(456, "https://custom-api.example.com/v0")

        self.assertEqual(result['id'], 456)
        mock_get.assert_called_once_with("https://custom-api.example.com/v0/item/456.json")


class TestExtractStoryData(unittest.TestCase):
    """Test cases for extract_story_data function"""

    def test_extract_story_data_complete(self):
        """Test extraction with complete story data"""
        item_data = {
            'id': 123,
            'title': 'Test Story',
            'score': 100,
            'descendants': 50,
            'time': 1700000000,
            'url': 'https://example.com',
            'by': 'testuser',
            'text': 'Test content'
        }

        result = extract_story_data(item_data)

        expected = {
            'title': 'Test Story',
            'score': 100,
            'descendants': 50,
            'time': datetime.fromtimestamp(1700000000),
            'url': 'https://example.com'
        }

        self.assertEqual(result, expected)

    def test_extract_story_data_minimal(self):
        """Test extraction with minimal story data"""
        item_data = {'title': 'Minimal Story'}

        result = extract_story_data(item_data)

        expected = {
            'title': 'Minimal Story',
            'score': 0,
            'descendants': 0,
            'time': datetime.fromtimestamp(0),
            'url': ''
        }

        self.assertEqual(result, expected)

    def test_extract_story_data_no_title(self):
        """Test extraction with missing title"""
        item_data = {'id': 123, 'score': 100}

        result = extract_story_data(item_data)

        self.assertIsNone(result)

    def test_extract_story_data_none_input(self):
        """Test extraction with None input"""
        result = extract_story_data(None)

        self.assertIsNone(result)

    def test_extract_story_data_empty_dict(self):
        """Test extraction with empty dictionary"""
        result = extract_story_data({})

        self.assertIsNone(result)

    def test_extract_story_data_missing_optional_fields(self):
        """Test extraction with missing optional fields"""
        item_data = {'title': 'Story with missing fields'}

        result = extract_story_data(item_data)

        self.assertEqual(result['score'], 0)
        self.assertEqual(result['descendants'], 0)
        self.assertEqual(result['url'], '')


class TestProcessStoriesToDataframe(unittest.TestCase):
    """Test cases for process_stories_to_dataframe function"""

    def test_process_stories_to_dataframe_success(self):
        """Test processing valid stories list"""
        stories_data = [
            {
                'title': 'Story 1',
                'score': 100,
                'descendants': 50,
                'time': datetime.fromtimestamp(1700000000),
                'url': 'https://example1.com'
            },
            {
                'title': 'Story 2',
                'score': 200,
                'descendants': 25,
                'time': datetime.fromtimestamp(1700001000),
                'url': 'https://example2.com'
            }
        ]

        result = process_stories_to_dataframe(stories_data)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]['score'], 200)  # Should be sorted by score desc
        self.assertEqual(result.iloc[1]['score'], 100)
        self.assertEqual(result.index.tolist(), [0, 1])  # Should be reset index

    def test_process_stories_to_dataframe_empty(self):
        """Test processing empty stories list"""
        result = process_stories_to_dataframe([])

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)

    def test_process_stories_to_dataframe_none(self):
        """Test processing None input"""
        result = process_stories_to_dataframe(None)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)

    def test_process_stories_to_dataframe_single_story(self):
        """Test processing single story"""
        stories_data = [{
            'title': 'Single Story',
            'score': 150,
            'descendants': 75,
            'time': datetime.fromtimestamp(1700000000),
            'url': 'https://single.com'
        }]

        result = process_stories_to_dataframe(stories_data)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['title'], 'Single Story')
        self.assertEqual(result.index.tolist(), [0])


class TestFetchHnData(unittest.TestCase):
    """Test cases for fetch_hn_data function"""

    @patch('data_loader.process_stories_to_dataframe')
    @patch('data_loader.extract_story_data')
    @patch('data_loader.fetch_story_details')
    @patch('data_loader.fetch_story_ids')
    @patch('data_loader.CacheManager')
    @patch('builtins.print')
    def test_fetch_hn_data_success(self, mock_print, mock_cache_manager, mock_fetch_ids,
                                 mock_fetch_details, mock_extract, mock_process):
        """Test successful fetch of HN data"""
        # Mock cache manager to return None (no cached data)
        mock_cache_manager.return_value.get_cached_data.return_value = None

        # Setup mocks
        mock_fetch_ids.return_value = [1, 2, 3]
        mock_fetch_details.side_effect = [
            {'id': 1, 'title': 'Story 1', 'score': 100},
            {'id': 2, 'title': 'Story 2', 'score': 200},
            {'id': 3, 'title': 'Story 3', 'score': 150}
        ]
        mock_extract.side_effect = [
            {'title': 'Story 1', 'score': 100, 'descendants': 50, 'time': datetime.now(), 'url': 'url1'},
            {'title': 'Story 2', 'score': 200, 'descendants': 25, 'time': datetime.now(), 'url': 'url2'},
            {'title': 'Story 3', 'score': 150, 'descendants': 10, 'time': datetime.now(), 'url': 'url3'}
        ]

        mock_df = pd.DataFrame([{'title': 'Story 2', 'score': 200},
                               {'title': 'Story 3', 'score': 150},
                               {'title': 'Story 1', 'score': 100}])
        mock_process.return_value = mock_df

        result = fetch_hn_data(limit=3)

        self.assertIsInstance(result, pd.DataFrame)
        mock_fetch_ids.assert_called_once()
        self.assertEqual(mock_fetch_details.call_count, 3)
        self.assertEqual(mock_extract.call_count, 3)
        mock_process.assert_called_once()

    @patch('data_loader.fetch_story_ids')
    @patch('builtins.print')
    def test_fetch_hn_data_ids_fetch_fails(self, mock_print, mock_fetch_ids):
        """Test when initial ID fetch fails"""
        mock_fetch_ids.return_value = None

        result = fetch_hn_data(limit=5)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)

    @patch('data_loader.process_stories_to_dataframe')
    @patch('data_loader.extract_story_data')
    @patch('data_loader.fetch_story_details')
    @patch('data_loader.fetch_story_ids')
    @patch('builtins.print')
    def test_fetch_hn_data_skip_invalid_stories(self, mock_print, mock_fetch_ids,
                                              mock_fetch_details, mock_extract, mock_process):
        """Test skipping stories without titles"""
        mock_fetch_ids.return_value = [1, 2, 3]
        mock_fetch_details.side_effect = [
            {'id': 1, 'title': 'Valid Story', 'score': 100},
            {'id': 2, 'score': 200},  # Missing title
            {'id': 3, 'title': 'Another Valid Story', 'score': 150}
        ]
        mock_extract.side_effect = [
            {'title': 'Valid Story', 'score': 100, 'descendants': 50, 'time': datetime.now(), 'url': 'url1'},
            None,  # Story 2 has no title
            {'title': 'Another Valid Story', 'score': 150, 'descendants': 25, 'time': datetime.now(), 'url': 'url3'}
        ]

        mock_df = pd.DataFrame([
            {'title': 'Another Valid Story', 'score': 150},
            {'title': 'Valid Story', 'score': 100}
        ])
        mock_process.return_value = mock_df

        result = fetch_hn_data(limit=3)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

    @patch('data_loader.process_stories_to_dataframe')
    @patch('data_loader.fetch_story_details')
    @patch('data_loader.fetch_story_ids')
    @patch('builtins.print')
    def test_fetch_hn_data_story_details_fails(self, mock_print, mock_fetch_ids,
                                             mock_fetch_details, mock_process):
        """Test when individual story detail fetch fails"""
        mock_fetch_ids.return_value = [1, 2]
        mock_fetch_details.side_effect = [None, {'id': 2, 'title': 'Story 2', 'score': 100}]

        mock_df = pd.DataFrame([{'title': 'Story 2', 'score': 100}])
        mock_process.return_value = mock_df

        result = fetch_hn_data(limit=2)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)

    @patch('data_loader.CacheManager')
    @patch('data_loader.fetch_story_ids')
    @patch('builtins.print')
    def test_fetch_hn_data_custom_limit(self, mock_print, mock_fetch_ids, mock_cache_manager):
        """Test fetching with custom limit"""
        # Mock cache manager to return None (no cached data)
        mock_cache_manager.return_value.get_cached_data.return_value = None
        mock_fetch_ids.return_value = list(range(100))

        with patch('data_loader.process_stories_to_dataframe') as mock_process, \
             patch('data_loader.fetch_story_details', return_value={'title': 'Test', 'score': 100}) as mock_details, \
             patch('data_loader.extract_story_data', return_value={'title': 'Test', 'score': 100}) as mock_extract:

            mock_process.return_value = pd.DataFrame([{'title': 'Test', 'score': 100}])

            fetch_hn_data(limit=10)

            self.assertEqual(mock_details.call_count, 10)
            self.assertEqual(mock_extract.call_count, 10)


class TestAnalyzeSentiment(unittest.TestCase):
    """Test cases for analyze_sentiment function"""

    @patch('data_loader.SentimentIntensityAnalyzer')
    @patch('builtins.print')
    def test_analyze_sentiment_normal_case(self, mock_print, mock_sia):
        """Test sentiment analysis with normal data"""
        # Mock VADER analyzer
        mock_analyzer = Mock()
        mock_sia.return_value = mock_analyzer

        # Mock sentiment scores for different titles
        mock_analyzer.polarity_scores.side_effect = [
            {'compound': 0.5, 'pos': 0.6, 'neg': 0.1, 'neu': 0.3},  # Positive
            {'compound': -0.3, 'pos': 0.1, 'neg': 0.5, 'neu': 0.4},  # Negative
            {'compound': 0.02, 'pos': 0.2, 'neg': 0.2, 'neu': 0.6},  # Neutral
            {'compound': 0.8, 'pos': 0.7, 'neg': 0.0, 'neu': 0.3},  # Strong Positive
        ]

        # Create test DataFrame
        df = pd.DataFrame({
            'title': [
                'Great new AI technology released',
                'Terrible security vulnerability discovered',
                'New software update available',
                'Amazing breakthrough in quantum computing'
            ],
            'score': [100, 50, 75, 200]
        })

        result = analyze_sentiment(df)

        # Check that new columns were added
        self.assertIn('sentiment_score', result.columns)
        self.assertIn('sentiment_label', result.columns)

        # Check sentiment scores
        self.assertEqual(result['sentiment_score'].tolist(), [0.5, -0.3, 0.02, 0.8])

        # Check sentiment labels
        expected_labels = ['Positive', 'Negative', 'Neutral', 'Positive']
        self.assertEqual(result['sentiment_label'].tolist(), expected_labels)

    @patch('builtins.print')
    def test_analyze_sentiment_empty_dataframe(self, mock_print):
        """Test sentiment analysis with empty DataFrame"""
        df = pd.DataFrame()

        result = analyze_sentiment(df)

        self.assertTrue(result.empty)
        mock_print.assert_called_with("Warning: DataFrame is empty or missing 'title' column")

    @patch('builtins.print')
    def test_analyze_sentiment_missing_title_column(self, mock_print):
        """Test sentiment analysis with missing title column"""
        df = pd.DataFrame({
            'score': [100, 200],
            'url': ['http://example.com', 'http://test.com']
        })

        result = analyze_sentiment(df)

        self.assertEqual(len(result), 2)  # DataFrame unchanged
        mock_print.assert_called_with("Warning: DataFrame is empty or missing 'title' column")

    @patch('data_loader.SentimentIntensityAnalyzer')
    def test_analyze_sentiment_with_empty_titles(self, mock_sia):
        """Test sentiment analysis with empty and NaN titles"""
        mock_analyzer = Mock()
        mock_sia.return_value = mock_analyzer

        # Mock sentiment score for valid title
        mock_analyzer.polarity_scores.return_value = {'compound': 0.5, 'pos': 0.6, 'neg': 0.1, 'neu': 0.3}

        # Create test DataFrame with empty and NaN titles
        df = pd.DataFrame({
            'title': ['Valid title', '', None, 'Another valid title'],
            'score': [100, 200, 300, 400]
        })

        result = analyze_sentiment(df)

        # Check that empty and None titles get compound score of 0.0
        self.assertEqual(result['sentiment_score'].tolist(), [0.5, 0.0, 0.0, 0.5])

        # Check sentiment labels
        expected_labels = ['Positive', 'Neutral', 'Neutral', 'Positive']
        self.assertEqual(result['sentiment_label'].tolist(), expected_labels)


class TestGetTopics(unittest.TestCase):
    """Test cases for get_topics function"""

    @patch('data_loader.BERTopic')
    @patch('builtins.print')
    def test_get_topics_normal_case(self, mock_print, mock_bertopic):
        """Test topic modeling with normal data"""
        # Mock BERTopic model
        mock_model = Mock()
        mock_bertopic.return_value = mock_model

        # Mock topic model behavior
        mock_model.fit_transform.return_value = ([0, 1, 0, 2, 1], [0.8, 0.7, 0.6, 0.9, 0.7])

        # Mock topic info
        mock_topic_info = pd.DataFrame({
            'Topic': [0, 1, 2, -1],
            'Representation': [
                ['ai', 'machine', 'learning'],
                ['security', 'vulnerability', 'cyber'],
                ['cloud', 'computing', 'aws'],
                ['outlier', 'noise']
            ]
        })
        mock_model.get_topic_info.return_value = mock_topic_info

        # Create test DataFrame
        df = pd.DataFrame({
            'title': [
                'New AI machine learning breakthrough',
                'Security vulnerability discovered',
                'AI and machine learning advances',
                'Cloud computing with AWS',
                'Cyber security best practices'
            ],
            'score': [100, 200, 150, 175, 125]
        })

        result = get_topics(df)

        # Check that new columns were added
        self.assertIn('topic_id', result.columns)
        self.assertIn('topic_keyword', result.columns)

        # Check topic assignments
        expected_topics = [0, 1, 0, 2, 1]
        self.assertEqual(result['topic_id'].tolist(), expected_topics)

        # Check topic keywords
        expected_keywords = ['ai_machine_learning', 'security_vulnerability_cyber',
                            'ai_machine_learning', 'cloud_computing_aws', 'security_vulnerability_cyber']
        self.assertEqual(result['topic_keyword'].tolist(), expected_keywords)

    @patch('builtins.print')
    def test_get_topics_empty_dataframe(self, mock_print):
        """Test topic modeling with empty DataFrame"""
        df = pd.DataFrame()

        result = get_topics(df)

        self.assertTrue(result.empty)
        mock_print.assert_called_with("Warning: DataFrame is empty or missing 'title' column")

    @patch('builtins.print')
    def test_get_topics_missing_title_column(self, mock_print):
        """Test topic modeling with missing title column"""
        df = pd.DataFrame({
            'score': [100, 200],
            'url': ['http://example.com', 'http://test.com']
        })

        result = get_topics(df)

        self.assertEqual(len(result), 2)  # DataFrame unchanged
        mock_print.assert_called_with("Warning: DataFrame is empty or missing 'title' column")

    @patch('builtins.print')
    def test_get_topics_insufficient_data(self, mock_print):
        """Test topic modeling with insufficient data (less than 2 valid titles)"""
        # Create DataFrame with only one valid title
        df = pd.DataFrame({
            'title': ['Single title'],
            'score': [100]
        })

        result = get_topics(df)

        # Check that default values are assigned
        self.assertEqual(result['topic_id'].tolist(), [-1])
        self.assertEqual(result['topic_keyword'].tolist(), ['Insufficient Data'])
        mock_print.assert_called_with("Warning: Need at least 2 valid titles for topic modeling")

    @patch('data_loader.BERTopic')
    @patch('builtins.print')
    def test_get_topics_with_none_titles(self, mock_print, mock_bertopic):
        """Test topic modeling with None and NaN titles"""
        mock_model = Mock()
        mock_bertopic.return_value = mock_model
        mock_model.fit_transform.return_value = ([0, 1], [0.8, 0.7])

        mock_topic_info = pd.DataFrame({
            'Topic': [0, 1],
            'Representation': [['ai', 'learning'], ['security', 'vulnerability']]
        })
        mock_model.get_topic_info.return_value = mock_topic_info

        # Create DataFrame with None and NaN titles
        df = pd.DataFrame({
            'title': ['Valid AI title', None, 'Valid security title', ''],
            'score': [100, 200, 150, 250]
        })

        result = get_topics(df)

        # Check that None and empty titles are handled
        self.assertEqual(result['topic_id'].iloc[0], 0)
        self.assertEqual(result['topic_id'].iloc[2], 1)
        # None and empty titles should be -1
        self.assertEqual(result['topic_id'].iloc[1], -1)
        self.assertEqual(result['topic_id'].iloc[3], -1)

    @patch('data_loader.BERTopic')
    @patch('builtins.print')
    def test_get_topics_model_error(self, mock_print, mock_bertopic):
        """Test topic modeling with model error"""
        mock_bertopic.side_effect = Exception("Model error")

        # Create test DataFrame
        df = pd.DataFrame({
            'title': ['Title 1', 'Title 2', 'Title 3'],
            'score': [100, 200, 300]
        })

        result = get_topics(df)

        # Check that error values are assigned
        self.assertEqual(result['topic_id'].tolist(), [-1, -1, -1])
        self.assertEqual(result['topic_keyword'].tolist(), ['Error', 'Error', 'Error'])
        mock_print.assert_called_with("Error in topic modeling: Model error")

    @patch('data_loader.BERTopic')
    @patch('builtins.print')
    def test_get_topics_custom_embedding_model(self, mock_print, mock_bertopic):
        """Test topic modeling with custom embedding model"""
        mock_model = Mock()
        mock_bertopic.return_value = mock_model
        mock_model.fit_transform.return_value = ([0, 1], [0.8, 0.7])

        mock_topic_info = pd.DataFrame({
            'Topic': [0, 1],
            'Representation': [['custom', 'topic'], ['another', 'topic']]
        })
        mock_model.get_topic_info.return_value = mock_topic_info

        # Use two titles to avoid "insufficient data" warning
        df = pd.DataFrame({
            'title': ['Test title 1', 'Test title 2'],
            'score': [100, 200]
        })

        # Test with custom embedding model
        result = get_topics(df, embedding_model='custom-model-name')

        # Check that BERTopic was called with custom model
        mock_bertopic.assert_called_once_with(
            embedding_model='custom-model-name',
            min_topic_size=2,
            verbose=True
        )


if __name__ == '__main__':
    unittest.main()