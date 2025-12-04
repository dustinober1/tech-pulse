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
    fetch_hn_data
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
    @patch('builtins.print')
    def test_fetch_hn_data_success(self, mock_print, mock_fetch_ids,
                                 mock_fetch_details, mock_extract, mock_process):
        """Test successful fetch of HN data"""
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

    @patch('data_loader.fetch_story_ids')
    @patch('builtins.print')
    def test_fetch_hn_data_custom_limit(self, mock_print, mock_fetch_ids):
        """Test fetching with custom limit"""
        mock_fetch_ids.return_value = list(range(100))

        with patch('data_loader.process_stories_to_dataframe') as mock_process, \
             patch('data_loader.fetch_story_details', return_value={'title': 'Test', 'score': 100}) as mock_details, \
             patch('data_loader.extract_story_data', return_value={'title': 'Test', 'score': 100}) as mock_extract:

            mock_process.return_value = pd.DataFrame([{'title': 'Test', 'score': 100}])

            fetch_hn_data(limit=10)

            self.assertEqual(mock_details.call_count, 10)
            self.assertEqual(mock_extract.call_count, 10)


if __name__ == '__main__':
    unittest.main()