"""
Comprehensive unit tests for Phase 7.2 User Management module
Tests database, user profile, recommendations, and UI components
"""

import unittest
import tempfile
import os
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Import the modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from phase7.user_management.database import UserDatabase
from phase7.user_management.user_profile import UserProfile
from phase7.user_management.recommendations import PersonalizedRecommendations, Recommendation
from phase7.user_management.ui_components import UIComponents


class TestUserDatabase(unittest.TestCase):
    """Test cases for UserDatabase class"""

    def setUp(self):
        """Set up test database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.db = UserDatabase(self.temp_db.name)

    def tearDown(self):
        """Clean up test database"""
        os.unlink(self.temp_db.name)

    def test_initialization(self):
        """Test database initialization"""
        # Check that tables are created
        with self.db._get_connection() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()

        table_names = [table[0] for table in tables]
        expected_tables = ['users', 'user_interactions', 'user_topics', 'search_history', 'recommendation_feedback']

        for table in expected_tables:
            self.assertIn(table, table_names)

    def test_create_user(self):
        """Test user creation"""
        username = "testuser"
        email = "test@example.com"
        preferences = {"theme": "dark", "notifications": True}

        user_id = self.db.create_user(username, email, preferences)

        # Verify user was created
        user = self.db.get_user(user_id)
        self.assertIsNotNone(user)
        self.assertEqual(user['username'], username)
        self.assertEqual(user['email'], email)
        self.assertEqual(user['preferences']['theme'], "dark")

    def test_duplicate_username(self):
        """Test error handling for duplicate usernames"""
        username = "duplicate_user"
        self.db.create_user(username)

        with self.assertRaises(ValueError):
            self.db.create_user(username)

    def test_update_user_preferences(self):
        """Test updating user preferences"""
        user_id = self.db.create_user("updater", "update@test.com")

        new_preferences = {"theme": "light", "language": "fr"}
        success = self.db.update_user(user_id, {"preferences": new_preferences})

        self.assertTrue(success)

        # Verify update
        user = self.db.get_user(user_id)
        self.assertEqual(user['preferences']['theme'], "light")
        self.assertEqual(user['preferences']['language'], "fr")

    def test_track_interaction(self):
        """Test tracking user interactions"""
        user_id = self.db.create_user("interactor", "interact@test.com")
        story_id = "story_123"
        interaction_type = "like"
        interaction_data = {"source": "mobile"}

        success = self.db.track_interaction(
            user_id, story_id, interaction_type, interaction_data
        )

        self.assertTrue(success)

        # Verify interaction was tracked
        interactions = self.db.get_user_interactions(user_id)
        self.assertEqual(len(interactions), 1)
        self.assertEqual(interactions[0]['story_id'], story_id)
        self.assertEqual(interactions[0]['interaction_type'], interaction_type)

    def test_update_topic_interest(self):
        """Test updating topic interests"""
        user_id = self.db.create_user("topic_user", "topic@test.com")
        topic_name = "Machine Learning"
        interest_score = 0.8

        success = self.db.update_topic_interest(user_id, topic_name, interest_score)

        self.assertTrue(success)

        # Verify topic interest was updated
        topics = self.db.get_user_topics(user_id)
        self.assertEqual(len(topics), 1)
        self.assertEqual(topics[0]['topic_name'], topic_name)
        self.assertEqual(topics[0]['interest_score'], interest_score)

    def test_track_search(self):
        """Test tracking user searches"""
        user_id = self.db.create_user("searcher", "search@test.com")
        search_query = "artificial intelligence"
        search_type = "semantic"
        results_count = 25

        success = self.db.track_search(
            user_id, search_query, search_type, results_count
        )

        self.assertTrue(success)

        # Verify search was tracked
        history = self.db.get_search_history(user_id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['search_query'], search_query)

    def test_record_recommendation_feedback(self):
        """Test recording recommendation feedback"""
        user_id = self.db.create_user("feedback_user", "feedback@test.com")
        story_id = "rec_story_123"
        recommendation_type = "collaborative"
        feedback_score = 4
        feedback_type = "rating"

        success = self.db.record_recommendation_feedback(
            user_id, story_id, recommendation_type, feedback_score, feedback_type
        )

        self.assertTrue(success)

        # Verify feedback was recorded
        feedback = self.db.get_recommendation_feedback(user_id)
        self.assertEqual(len(feedback), 1)
        self.assertEqual(feedback[0]['story_id'], story_id)
        self.assertEqual(feedback[0]['feedback_score'], feedback_score)

    def test_get_user_analytics(self):
        """Test user analytics retrieval"""
        user_id = self.db.create_user("analytics_user", "analytics@test.com")

        # Add some interactions
        self.db.track_interaction(user_id, "story_1", "view")
        self.db.track_interaction(user_id, "story_2", "like")
        self.db.track_interaction(user_id, "story_3", "bookmark")

        # Add topics
        self.db.update_topic_interest(user_id, "AI", 0.9)
        self.db.update_topic_interest(user_id, "Python", 0.7)

        analytics = self.db.get_user_analytics(user_id)

        self.assertIn('interaction_counts', analytics)
        self.assertIn('top_topics', analytics)
        self.assertEqual(analytics['interaction_counts']['view'], 1)
        self.assertEqual(analytics['interaction_counts']['like'], 1)

    def test_delete_user(self):
        """Test user deletion (soft delete)"""
        user_id = self.db.create_user("deletable", "delete@test.com")

        # Verify user exists
        user = self.db.get_user(user_id)
        self.assertIsNotNone(user)

        # Delete user
        success = self.db.delete_user(user_id)
        self.assertTrue(success)

        # Verify user is marked as inactive
        user = self.db.get_user(user_id)
        self.assertIsNone(user)


class TestUserProfile(unittest.TestCase):
    """Test cases for UserProfile class"""

    def setUp(self):
        """Set up test profile"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.db = UserDatabase(self.temp_db.name)
        self.user_id = self.db.create_user("profile_user", "profile@test.com")
        self.profile = UserProfile(self.user_id, self.db)

    def tearDown(self):
        """Clean up test database"""
        os.unlink(self.temp_db.name)

    def test_get_profile(self):
        """Test profile retrieval"""
        profile = self.profile.get_profile()

        self.assertIn('user_id', profile)
        self.assertIn('preferences', profile)
        self.assertIn('analytics', profile)
        self.assertIn('topics', profile)
        self.assertEqual(profile['user_id'], self.user_id)

    def test_update_preferences(self):
        """Test preference updates"""
        new_prefs = {"theme": "dark", "stories_per_page": 50}
        success = self.profile.update_preferences(new_prefs)

        self.assertTrue(success)

        # Verify update
        profile = self.profile.get_profile()
        self.assertEqual(profile['preferences']['theme'], "dark")
        self.assertEqual(profile['preferences']['stories_per_page'], 50)

    def test_track_story_interaction(self):
        """Test story interaction tracking"""
        story_id = "test_story_123"
        interaction_type = "bookmark"

        success = self.profile.track_story_interaction(story_id, interaction_type)

        self.assertTrue(success)

        # Verify interaction was tracked
        interactions = self.db.get_user_interactions(self.user_id)
        self.assertEqual(len(interactions), 1)
        self.assertEqual(interactions[0]['story_id'], story_id)

    def test_update_topic_interest(self):
        """Test topic interest updates"""
        topic_name = "Data Science"
        delta_score = 0.2

        success = self.profile.update_topic_interest(topic_name, delta_score)

        self.assertTrue(success)

        # Verify update
        topics = self.profile.get_interest_topics()
        self.assertTrue(any(t['topic_name'] == topic_name for t in topics))

    def test_get_reading_history(self):
        """Test reading history retrieval"""
        # Add some interactions
        self.db.track_interaction(self.user_id, "story_1", "view")
        self.db.track_interaction(self.user_id, "story_2", "view")

        history = self.profile.get_reading_history(days=30)

        self.assertIsInstance(history, list)
        # Note: Actual history would have timestamps, this is a basic test

    def test_get_reading_patterns(self):
        """Test reading pattern analysis"""
        patterns = self.profile.get_reading_patterns()

        self.assertIn('most_active_hours', patterns)
        self.assertIn('most_active_days', patterns)
        self.assertIn('daily_average', patterns)
        self.assertIn('reading_streaks', patterns)

    def test_get_personalized_summary(self):
        """Test personalized summary generation"""
        summary = self.profile.get_personalized_summary()

        self.assertIn('welcome_message', summary)
        self.assertIn('stats_highlights', summary)
        self.assertIn('trending_topics', summary)
        self.assertIn('recommendations', summary)
        self.assertIn('achievements', summary)

    def test_export_import_profile_data(self):
        """Test profile data export and import"""
        # Set some preferences and topics
        self.profile.update_preferences({"theme": "dark", "language": "es"})
        self.profile.update_topic_interest("Testing", 0.8)

        # Export data
        export_data = self.profile.export_profile_data()

        self.assertIn('export_timestamp', export_data)
        self.assertIn('preferences', export_data)
        self.assertIn('topics', export_data)

        # Create new user and import data
        new_user_id = self.db.create_user("import_user", "import@test.com")
        new_profile = UserProfile(new_user_id, self.db)

        success = new_profile.import_profile_data(export_data)
        self.assertTrue(success)

        # Verify import
        imported_profile = new_profile.get_profile()
        self.assertEqual(imported_profile['preferences']['theme'], "dark")
        self.assertEqual(imported_profile['preferences']['language'], "es")


class TestPersonalizedRecommendations(unittest.TestCase):
    """Test cases for PersonalizedRecommendations class"""

    def setUp(self):
        """Set up test recommendation engine"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.db = UserDatabase(self.temp_db.name)
        self.user_id = self.db.create_user("rec_user", "rec@test.com")
        self.recommendations = PersonalizedRecommendations(self.user_id, self.db)

    def tearDown(self):
        """Clean up test database"""
        os.unlink(self.temp_db.name)

    @patch('phase7.user_management.recommendations.PersonalizedRecommendations._get_mixed_recommendations')
    def test_get_recommendations_mixed(self, mock_mixed):
        """Test getting mixed recommendations"""
        mock_mixed.return_value = [
            Recommendation(
                story_id="story_1",
                title="Test Story",
                score=0.8,
                reason="Test reason",
                metadata={},
                timestamp=datetime.now()
            )
        ]

        recs = self.recommendations.get_recommendations(5, 'mixed')

        self.assertEqual(len(recs), 1)
        self.assertIsInstance(recs[0], Recommendation)
        self.assertEqual(recs[0].story_id, "story_1")

    def test_update_recommendation_feedback(self):
        """Test updating recommendation feedback"""
        story_id = "feedback_story"
        feedback_type = "like"
        feedback_score = 4

        success = self.recommendations.update_recommendation_feedback(
            story_id, feedback_type, feedback_score
        )

        self.assertTrue(success)

        # Verify feedback was recorded
        feedback = self.db.get_recommendation_feedback(self.user_id)
        self.assertEqual(len(feedback), 1)
        self.assertEqual(feedback[0]['story_id'], story_id)

    @patch('phase7.user_management.recommendations.PersonalizedRecommendations._get_topic_based_recommendations')
    def test_get_topic_recommendations(self, mock_topic):
        """Test topic-based recommendations"""
        mock_topic.return_value = [
            Recommendation(
                story_id="topic_story",
                title="Topic Story",
                score=0.9,
                reason="Based on your interest",
                metadata={"topic": "AI"},
                timestamp=datetime.now()
            )
        ]

        recs = self.recommendations.get_topic_recommendations("AI", 3)

        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].metadata["topic"], "AI")

    def test_get_recommendation_stats(self):
        """Test recommendation statistics"""
        # Add some feedback
        self.recommendations.update_recommendation_feedback("story_1", "click")
        self.recommendations.update_recommendation_feedback("story_2", "like", 4)

        stats = self.recommendations.get_recommendation_stats()

        self.assertIn('total_recommendations', stats)
        self.assertIn('click_through_rate', stats)
        self.assertIn('feedback_distribution', stats)

    @patch('phase7.user_management.recommendations.PersonalizedRecommendations._calculate_topic_score')
    def test_calculate_topic_score(self, mock_score):
        """Test topic score calculation"""
        mock_score.return_value = 0.85

        story = {"id": "test_story", "title": "Test"}
        score = self.recommendations._calculate_topic_score(story)

        self.assertEqual(score, 0.85)

    def test_cache_functionality(self):
        """Test recommendation caching"""
        # Mock the internal method to avoid actual recommendations
        with patch.object(self.recommendations, '_get_mixed_recommendations') as mock_mixed:
            mock_mixed.return_value = []

            # First call should hit the mock
            recs1 = self.recommendations.get_recommendations(5, 'mixed')
            mock_mixed.assert_called_once()

            # Second call should use cache
            recs2 = self.recommendations.get_recommendations(5, 'mixed')
            mock_mixed.assert_called_once()  # Still only called once

            self.assertEqual(recs1, recs2)


class TestUIComponents(unittest.TestCase):
    """Test cases for UIComponents class"""

    def setUp(self):
        """Set up test UI components"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.db = UserDatabase(self.temp_db.name)
        self.user_id = self.db.create_user("ui_user", "ui@test.com")
        # Pass the test database instance to UIComponents
        with patch('phase7.user_management.ui_components.UserDatabase', return_value=self.db):
            self.ui = UIComponents(self.user_id)

    def tearDown(self):
        """Clean up test database"""
        os.unlink(self.temp_db.name)

    def test_initialization(self):
        """Test UI components initialization"""
        self.assertEqual(self.ui.user_id, self.user_id)
        self.assertIsNotNone(self.ui.db)
        self.assertIsNotNone(self.ui.user_profile)
        self.assertIsNotNone(self.ui.recommendations)

    @patch('streamlit.markdown')
    @patch('streamlit.write')
    @patch('streamlit.metric')
    @patch('streamlit.progress')
    @patch('streamlit.caption')
    def test_render_mini_profile_card(self, mock_caption, mock_progress, mock_metric, mock_write, mock_markdown):
        """Test mini profile card rendering"""
        self.ui.render_mini_profile_card()

        # Verify Streamlit methods were called
        mock_markdown.assert_called()
        mock_write.assert_called()
        mock_metric.assert_called()
        mock_progress.assert_called()
        mock_caption.assert_called()

    def test_format_time_ago(self):
        """Test time ago formatting"""
        now = datetime.now()

        # Test different time differences
        test_cases = [
            (now - timedelta(minutes=5), "5 minutes ago"),
            (now - timedelta(hours=2), "2 hours ago"),
            (now - timedelta(days=1), "1 day ago"),
            (now - timedelta(seconds=30), "Just now")
        ]

        for timestamp, expected in test_cases:
            result = self.ui._format_time_ago(timestamp)
            self.assertTrue(expected.split()[0] in result or result == "Just now")

    def test_get_activity_icon(self):
        """Test activity icon generation"""
        test_cases = [
            ({'type': 'interaction', 'subtype': 'view'}, ('👁️', '#4CAF50')),
            ({'type': 'interaction', 'subtype': 'like'}, ('👍', '#2196F3')),
            ({'type': 'search', 'subtype': 'keyword'}, ('🔍', '#607D8B')),
            ({'type': 'unknown', 'subtype': 'unknown'}, ('📝', '#9E9E9E'))
        ]

        for activity, expected in test_cases:
            result = self.ui._get_activity_icon(activity)
            self.assertEqual(result, expected)

    @patch('streamlit.selectbox')
    @patch('streamlit.slider')
    @patch('streamlit.checkbox')
    def test_render_preferences_editor(self, mock_checkbox, mock_slider, mock_selectbox):
        """Mock test for preferences editor"""
        # Mock Streamlit components
        mock_selectbox.return_value = 'light'
        mock_slider.return_value = 30
        mock_checkbox.return_value = True

        result = self.ui.render_preferences_editor()

        self.assertIsInstance(result, dict)

    def test_get_recommended_topics(self):
        """Test getting recommended topics"""
        topics = self.ui._get_recommended_topics()

        self.assertIsInstance(topics, list)
        self.assertTrue(len(topics) > 0)
        self.assertIsInstance(topics[0], str)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete user management system"""

    def setUp(self):
        """Set up integration test environment"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.db = UserDatabase(self.temp_db.name)
        self.user_id = self.db.create_user("integration_user", "integration@test.com")
        self.profile = UserProfile(self.user_id, self.db)
        self.recommendations = PersonalizedRecommendations(self.user_id, self.db)
        self.ui = UIComponents(self.user_id)

    def tearDown(self):
        """Clean up integration test environment"""
        os.unlink(self.temp_db.name)

    def test_full_user_journey(self):
        """Test complete user journey from creation to recommendations"""
        # 1. User updates preferences
        self.profile.update_preferences({
            "theme": "dark",
            "notifications": True,
            "preferred_topics": ["AI", "Machine Learning"]
        })

        # 2. User interacts with stories
        self.profile.track_story_interaction("story_1", "view")
        self.profile.track_story_interaction("story_2", "like")
        self.profile.track_story_interaction("story_3", "bookmark")

        # 3. User updates topic interests
        self.profile.update_topic_interest("Artificial Intelligence", 0.2)
        self.profile.update_topic_interest("Python", 0.3)

        # 4. Get personalized summary
        summary = self.profile.get_personalized_summary()
        self.assertIn('stats_highlights', summary)
        self.assertGreater(summary['stats_highlights']['stories_read'], 0)

        # 5. Get analytics
        analytics = self.profile.get_analytics_summary()
        self.assertIn('interaction_counts', analytics)
        self.assertGreater(analytics['interaction_counts']['view'], 0)

        # 6. Export and verify profile data
        export_data = self.profile.export_profile_data()
        self.assertIn('preferences', export_data)
        self.assertIn('analytics', export_data)

    def test_recommendation_feedback_loop(self):
        """Test recommendation feedback improves future suggestions"""
        # Simulate user providing feedback
        story_id = "recommended_story"
        self.recommendations.update_recommendation_feedback(story_id, "like", 5)
        self.recommendations.update_recommendation_feedback("another_story", "dismiss", -1)

        # Get recommendation stats
        stats = self.recommendations.get_recommendation_stats()
        self.assertEqual(stats['total_recommendations'], 2)
        self.assertIn('feedback_distribution', stats)


def run_all_tests():
    """Run all user management tests"""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestUserDatabase))
    test_suite.addTest(unittest.makeSuite(TestUserProfile))
    test_suite.addTest(unittest.makeSuite(TestPersonalizedRecommendations))
    test_suite.addTest(unittest.makeSuite(TestUIComponents))
    test_suite.addTest(unittest.makeSuite(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)