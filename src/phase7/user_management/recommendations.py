"""
Personalized Recommendations Module for Phase 7.2
Handles intelligent content recommendation based on user behavior and preferences
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
import math
from dataclasses import dataclass

from .database import UserDatabase
from .user_profile import UserProfile


@dataclass
class Recommendation:
    """Data class for recommendation items."""
    story_id: str
    title: str
    score: float
    reason: str
    metadata: Dict[str, Any]
    timestamp: datetime


class PersonalizedRecommendations:
    """
    Intelligent content recommendation engine that uses multiple factors
    to provide personalized story recommendations.
    """

    def __init__(self, user_id: str, database: Optional[UserDatabase] = None):
        """
        Initialize recommendation engine for a specific user.

        Args:
            user_id (str): Unique user identifier
            database (UserDatabase, optional): Database instance
        """
        self.user_id = user_id
        self.db = database or UserDatabase()
        self.user_profile = UserProfile(user_id, self.db)
        self._recommendation_cache = {}
        self._cache_expiry = {}

        # Recommendation weights
        self.weights = {
            'topic_interest': 0.35,
            'collaborative': 0.25,
            'content_similarity': 0.20,
            'time_decay': 0.10,
            'diversity': 0.10
        }

    def get_recommendations(self, count: int = 10,
                          recommendation_type: str = 'mixed') -> List[Recommendation]:
        """
        Get personalized story recommendations.

        Args:
            count (int): Number of recommendations to return
            recommendation_type (str): Type of recommendations
                ('mixed', 'topics', 'collaborative', 'trending', 'diverse')

        Returns:
            list: List of recommendation objects
        """
        # Check cache first
        cache_key = f"{recommendation_type}_{count}"
        if self._is_cache_valid(cache_key):
            return self._recommendation_cache[cache_key]

        recommendations = []

        if recommendation_type == 'mixed':
            recommendations = self._get_mixed_recommendations(count)
        elif recommendation_type == 'topics':
            recommendations = self._get_topic_based_recommendations(count)
        elif recommendation_type == 'collaborative':
            recommendations = self._get_collaborative_recommendations(count)
        elif recommendation_type == 'trending':
            recommendations = self._get_trending_recommendations(count)
        elif recommendation_type == 'diverse':
            recommendations = self._get_diverse_recommendations(count)
        else:
            recommendations = self._get_mixed_recommendations(count)

        # Cache the results
        self._recommendation_cache[cache_key] = recommendations
        self._cache_expiry[cache_key] = datetime.now() + timedelta(minutes=15)

        return recommendations

    def get_topic_recommendations(self, topic_name: str, count: int = 5) -> List[Recommendation]:
        """
        Get recommendations for a specific topic.

        Args:
            topic_name (str): Topic to get recommendations for
            count (int): Number of recommendations

        Returns:
            list: List of recommendations for the topic
        """
        # Get stories related to the topic
        topic_stories = self._get_stories_by_topic(topic_name, count * 2)

        recommendations = []
        for story in topic_stories:
            score = self._calculate_topic_score(story, topic_name)
            reason = f"Based on your interest in {topic_name}"

            recommendations.append(Recommendation(
                story_id=story['id'],
                title=story['title'],
                score=score,
                reason=reason,
                metadata={
                    'topic': topic_name,
                    'topic_score': score,
                    'story_metadata': story
                },
                timestamp=datetime.now()
            ))

        return sorted(recommendations, key=lambda x: x.score, reverse=True)[:count]

    def get_similar_users_recommendations(self, count: int = 5) -> List[Recommendation]:
        """
        Get recommendations based on similar users' behavior.

        Args:
            count (int): Number of recommendations

        Returns:
            list: Recommendations from similar users
        """
        similar_users = self._find_similar_users(limit=5)

        recommendations = []
        for similar_user in similar_users:
            user_recommendations = self._get_user_top_stories(similar_user['user_id'])
            for story in user_recommendations:
                score = self._calculate_collaborative_score(story, similar_user)
                reason = f"Users like you also read this"

                recommendations.append(Recommendation(
                    story_id=story['story_id'],
                    title=story.get('title', 'Unknown'),
                    score=score,
                    reason=reason,
                    metadata={
                        'similar_user': similar_user['user_id'],
                        'similarity_score': similar_user['similarity'],
                        'story_metadata': story
                    },
                    timestamp=datetime.now()
                ))

        return self._deduplicate_and_rank(recommendations, count)

    def update_recommendation_feedback(self, story_id: str, feedback_type: str,
                                     feedback_score: Optional[int] = None) -> bool:
        """
        Update recommendation feedback to improve future suggestions.

        Args:
            story_id (str): Story identifier
            feedback_type (str): Type of feedback ('click', 'like', 'dismiss', 'rating')
            feedback_score (int, optional): Numeric feedback score (1-5)

        Returns:
            bool: True if feedback recorded successfully
        """
        # Determine feedback score based on type
        if feedback_score is None:
            feedback_scores = {
                'click': 1,
                'like': 3,
                'dismiss': -1,
                'share': 4,
                'bookmark': 5
            }
            feedback_score = feedback_scores.get(feedback_type, 0)

        success = self.db.record_recommendation_feedback(
            self.user_id, story_id, 'personalized', feedback_score, feedback_type
        )

        # Clear cache to force refresh
        if success:
            self._recommendation_cache.clear()
            self._cache_expiry.clear()

        return success

    def get_recommendation_explanation(self, story_id: str) -> Dict[str, Any]:
        """
        Get explanation for why a story was recommended.

        Args:
            story_id (str): Story identifier

        Returns:
            dict: Explanation of recommendation factors
        """
        # Get story information
        story = self._get_story_by_id(story_id)
        if not story:
            return {'error': 'Story not found'}

        explanation = {
            'story_id': story_id,
            'title': story.get('title', 'Unknown'),
            'factors': {},
            'overall_score': 0.0,
            'primary_reason': ''
        }

        # Calculate factor contributions
        topic_score = self._calculate_topic_score(story)
        explanation['factors']['topic_interest'] = {
            'score': topic_score,
            'weight': self.weights['topic_interest'],
            'contribution': topic_score * self.weights['topic_interest'],
            'details': self._get_topic_details(story)
        }

        collab_score = self._calculate_collaborative_score(story)
        explanation['factors']['collaborative'] = {
            'score': collab_score,
            'weight': self.weights['collaborative'],
            'contribution': collab_score * self.weights['collaborative'],
            'details': self._get_collaborative_details(story)
        }

        similarity_score = self._calculate_content_similarity(story)
        explanation['factors']['content_similarity'] = {
            'score': similarity_score,
            'weight': self.weights['content_similarity'],
            'contribution': similarity_score * self.weights['content_similarity'],
            'details': self._get_similarity_details(story)
        }

        # Calculate overall score
        explanation['overall_score'] = sum(
            factor['contribution'] for factor in explanation['factors'].values()
        )

        # Determine primary reason
        max_factor = max(
            explanation['factors'].items(),
            key=lambda x: x[1]['contribution']
        )
        explanation['primary_reason'] = self._format_primary_reason(max_factor)

        return explanation

    def get_recommendation_stats(self) -> Dict[str, Any]:
        """
        Get recommendation statistics and performance metrics.

        Returns:
            dict: Recommendation statistics
        """
        feedback = self.db.get_recommendation_feedback(self.user_id)

        if not feedback:
            return {
                'total_recommendations': 0,
                'average_rating': 0.0,
                'click_through_rate': 0.0,
                'feedback_distribution': {},
                'popular_categories': [],
                'recommendation_accuracy': 0.0
            }

        df = pd.DataFrame(feedback)

        stats = {
            'total_recommendations': len(feedback),
            'average_rating': df[df['feedback_score'] > 0]['feedback_score'].mean(),
            'click_through_rate': len(df[df['feedback_type'] == 'click']) / len(df),
            'feedback_distribution': df['feedback_type'].value_counts().to_dict(),
            'popular_categories': self._get_popular_recommended_categories(),
            'recommendation_accuracy': self._calculate_recommendation_accuracy()
        }

        return stats

    # Private implementation methods

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached recommendations are still valid."""
        if cache_key not in self._recommendation_cache:
            return False

        expiry_time = self._cache_expiry.get(cache_key)
        if not expiry_time:
            return False

        return datetime.now() < expiry_time

    def _get_mixed_recommendations(self, count: int) -> List[Recommendation]:
        """Get mixed recommendations using multiple strategies."""
        recommendations = []

        # Get recommendations from different strategies
        topic_recs = self._get_topic_based_recommendations(count // 3)
        collab_recs = self._get_collaborative_recommendations(count // 3)
        trending_recs = self._get_trending_recommendations(count // 3)
        diverse_recs = self._get_diverse_recommendations(count // 4)

        all_recommendations = topic_recs + collab_recs + trending_recs + diverse_recs

        # Deduplicate and rank
        return self._deduplicate_and_rank(all_recommendations, count)

    def _get_topic_based_recommendations(self, count: int) -> List[Recommendation]:
        """Get recommendations based on user's topic interests."""
        user_topics = self.user_profile.get_interest_topics(min_score=0.3)

        if not user_topics:
            return self._get_fallback_recommendations(count)

        recommendations = []
        stories_per_topic = max(1, count // len(user_topics))

        for topic in user_topics[:5]:  # Limit to top 5 topics
            topic_recs = self.get_topic_recommendations(
                topic['topic_name'], stories_per_topic
            )
            recommendations.extend(topic_recs)

        return recommendations[:count]

    def _get_collaborative_recommendations(self, count: int) -> List[Recommendation]:
        """Get recommendations based on similar users' behavior."""
        similar_users = self._find_similar_users(limit=10)
        recommendations = []

        for similar_user in similar_users[:5]:  # Use top 5 similar users
            user_stories = self._get_user_liked_stories(similar_user['user_id'])

            for story in user_stories:
                # Check if current user hasn't read this story
                if not self._user_has_read_story(story['story_id']):
                    score = self._calculate_collaborative_score(story, similar_user)
                    reason = f"Similar users enjoyed this story"

                    recommendations.append(Recommendation(
                        story_id=story['story_id'],
                        title=story.get('title', 'Unknown'),
                        score=score,
                        reason=reason,
                        metadata={
                            'similar_user': similar_user['user_id'],
                            'similarity': similar_user['similarity'],
                            'method': 'collaborative'
                        },
                        timestamp=datetime.now()
                    ))

        return sorted(recommendations, key=lambda x: x.score, reverse=True)[:count]

    def _get_trending_recommendations(self, count: int) -> List[Recommendation]:
        """Get recommendations based on trending stories."""
        # Get trending stories from the last 24 hours
        trending_stories = self._get_trending_stories(count * 2)

        recommendations = []
        for story in trending_stories:
            score = self._calculate_trending_score(story)
            reason = "Trending in the community"

            recommendations.append(Recommendation(
                story_id=story['id'],
                title=story['title'],
                score=score,
                reason=reason,
                metadata={
                    'trending_score': score,
                    'engagement': story.get('engagement', 0),
                    'method': 'trending'
                },
                timestamp=datetime.now()
            ))

        return recommendations[:count]

    def _get_diverse_recommendations(self, count: int) -> List[Recommendation]:
        """Get diverse recommendations across different topics."""
        # Get stories from various topics the user hasn't explored much
        user_topics = set(topic['topic_name'] for topic in self.user_profile.get_interest_topics())

        diverse_stories = self._get_stories_from_new_topics(
            exclude_topics=user_topics, count=count * 2
        )

        recommendations = []
        for story in diverse_stories:
            score = self._calculate_diversity_score(story, user_topics)
            reason = f"Explore {story.get('primary_topic', 'new topics')}"

            recommendations.append(Recommendation(
                story_id=story['id'],
                title=story['title'],
                score=score,
                reason=reason,
                metadata={
                    'diversity_score': score,
                    'new_topic': story.get('primary_topic'),
                    'method': 'diversity'
                },
                timestamp=datetime.now()
            ))

        return recommendations[:count]

    def _get_fallback_recommendations(self, count: int) -> List[Recommendation]:
        """Get fallback recommendations when no personal data is available."""
        # Return popular stories from the last week
        popular_stories = self._get_popular_stories(count)

        recommendations = []
        for story in popular_stories:
            reason = "Popular this week"

            recommendations.append(Recommendation(
                story_id=story['id'],
                title=story['title'],
                score=0.7,  # Moderate score for fallback
                reason=reason,
                metadata={'method': 'fallback', 'popularity': story.get('score', 0)},
                timestamp=datetime.now()
            ))

        return recommendations

    def _find_similar_users(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Find users with similar preferences and behavior."""
        # Get current user's topics
        user_topics = self.user_profile.get_interest_topics()
        user_topic_set = set(topic['topic_name'] for topic in user_topics)

        # Get all users and compare
        all_users = self.db.get_all_active_users()
        similarities = []

        for user in all_users:
            if user['user_id'] == self.user_id:
                continue

            # Get other user's topics
            other_topics = self.db.get_user_topics(user['user_id'])
            other_topic_set = set(topic['topic_name'] for topic in other_topics)

            # Calculate Jaccard similarity
            intersection = len(user_topic_set & other_topic_set)
            union = len(user_topic_set | other_topic_set)

            if union > 0:
                similarity = intersection / union
                if similarity > 0.1:  # Minimum similarity threshold
                    similarities.append({
                        'user_id': user['user_id'],
                        'similarity': similarity,
                        'common_topics': list(user_topic_set & other_topic_set)
                    })

        # Sort by similarity and return top matches
        return sorted(similarities, key=lambda x: x['similarity'], reverse=True)[:limit]

    def _user_has_read_story(self, story_id: str) -> bool:
        """Check if user has already interacted with a story."""
        interactions = self.db.get_user_interactions(self.user_id, limit=1000)
        return any(interaction['story_id'] == story_id for interaction in interactions)

    def _calculate_topic_score(self, story: Dict[str, Any], topic_name: Optional[str] = None) -> float:
        """Calculate topic-based recommendation score."""
        if not topic_name:
            # Extract topics from story
            story_topics = self._extract_story_topics(story)
            if not story_topics:
                return 0.0

            # Use highest scoring topic
            max_score = 0.0
            for topic in story_topics:
                user_topics = self.user_profile.get_interest_topics()
                for user_topic in user_topics:
                    if user_topic['topic_name'] == topic:
                        max_score = max(max_score, user_topic['interest_score'])
                        break

            return max_score
        else:
            # Get score for specific topic
            user_topics = self.user_profile.get_interest_topics()
            for topic in user_topics:
                if topic['topic_name'] == topic_name:
                    return topic['interest_score']

            return 0.0

    def _calculate_collaborative_score(self, story: Dict[str, Any],
                                     similar_user: Dict[str, Any]) -> float:
        """Calculate collaborative filtering score."""
        similarity = similar_user['similarity']

        # Get story's average rating from all users
        avg_rating = self._get_story_average_rating(story['story_id'])

        # Combine similarity and rating
        return similarity * (avg_rating / 5.0)  # Normalize rating to 0-1

    def _calculate_content_similarity(self, story: Dict[str, Any]) -> float:
        """Calculate content similarity score based on user's reading history."""
        # Get user's recently read stories
        recent_stories = self.user_profile.get_reading_history(days=30, limit=50)

        if not recent_stories:
            return 0.5  # Default score

        # Calculate similarity to recent stories (simplified)
        similarities = []
        for past_story in recent_stories:
            similarity = self._calculate_story_similarity(story, past_story)
            similarities.append(similarity)

        # Return average similarity
        return np.mean(similarities) if similarities else 0.5

    def _calculate_trending_score(self, story: Dict[str, Any]) -> float:
        """Calculate trending score based on recent popularity."""
        # Factor in engagement metrics
        comments = story.get('comments', 0)
        points = story.get('points', 0)
        recency = self._calculate_recency_score(story.get('created_at', ''))

        # Normalize and combine
        normalized_comments = min(1.0, comments / 100.0)
        normalized_points = min(1.0, points / 500.0)

        return (normalized_comments * 0.4 +
                normalized_points * 0.4 +
                recency * 0.2)

    def _calculate_diversity_score(self, story: Dict[str, Any],
                                 user_topics: set) -> float:
        """Calculate diversity score for exploring new topics."""
        story_topics = self._extract_story_topics(story)

        # Score based on how many topics are new to the user
        new_topics = [topic for topic in story_topics if topic not in user_topics]

        if not story_topics:
            return 0.0

        new_topic_ratio = len(new_topics) / len(story_topics)
        return new_topic_ratio

    def _deduplicate_and_rank(self, recommendations: List[Recommendation],
                            count: int) -> List[Recommendation]:
        """Remove duplicates and rank recommendations."""
        # Track seen story IDs
        seen_stories = set()
        unique_recommendations = []

        for rec in recommendations:
            if rec.story_id not in seen_stories:
                unique_recommendations.append(rec)
                seen_stories.add(rec.story_id)

        # Sort by score
        unique_recommendations.sort(key=lambda x: x.score, reverse=True)

        return unique_recommendations[:count]

    def _extract_story_topics(self, story: Dict[str, Any]) -> List[str]:
        """Extract topics from story (placeholder implementation)."""
        # This would integrate with the content analysis system
        # For now, return empty list
        return []

    def _get_stories_by_topic(self, topic_name: str, count: int) -> List[Dict[str, Any]]:
        """Get stories related to a specific topic."""
        # Placeholder implementation - would integrate with data loader
        return []

    def _get_story_by_id(self, story_id: str) -> Optional[Dict[str, Any]]:
        """Get story details by ID."""
        # Placeholder implementation
        return {'id': story_id, 'title': 'Sample Story'}

    def _get_user_top_stories(self, user_id: str) -> List[Dict[str, Any]]:
        """Get top stories for a user."""
        interactions = self.db.get_user_interactions(user_id, 'like', limit=50)
        return [{'story_id': interaction['story_id']} for interaction in interactions]

    def _get_user_liked_stories(self, user_id: str) -> List[Dict[str, Any]]:
        """Get stories liked by a user."""
        return self._get_user_top_stories(user_id)

    def _get_trending_stories(self, count: int) -> List[Dict[str, Any]]:
        """Get currently trending stories."""
        # Placeholder implementation
        return [{'id': f'trending_{i}', 'title': f'Trending Story {i}'}
                for i in range(count)]

    def _get_stories_from_new_topics(self, exclude_topics: set,
                                   count: int) -> List[Dict[str, Any]]:
        """Get stories from topics the user hasn't explored."""
        # Placeholder implementation
        return [{'id': f'new_topic_{i}', 'title': f'New Topic Story {i}'}
                for i in range(count)]

    def _get_popular_stories(self, count: int) -> List[Dict[str, Any]]:
        """Get popular stories from the last week."""
        # Placeholder implementation
        return [{'id': f'popular_{i}', 'title': f'Popular Story {i}'}
                for i in range(count)]

    def _get_story_average_rating(self, story_id: str) -> float:
        """Get average rating for a story."""
        # Placeholder implementation
        return 3.5

    def _calculate_story_similarity(self, story1: Dict[str, Any],
                                  story2: Dict[str, Any]) -> float:
        """Calculate similarity between two stories."""
        # Placeholder implementation
        return 0.5

    def _calculate_recency_score(self, created_at: str) -> float:
        """Calculate recency score based on creation time."""
        try:
            if not created_at:
                return 0.0

            # Parse timestamp and calculate hours ago
            story_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            hours_ago = (datetime.now() - story_time).total_seconds() / 3600

            # Exponential decay
            return math.exp(-hours_ago / 24)  # 24-hour half-life
        except:
            return 0.0

    def _get_topic_details(self, story: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed topic information for explanation."""
        return {
            'matched_topics': self._extract_story_topics(story),
            'topic_scores': {}  # Would contain individual topic scores
        }

    def _get_collaborative_details(self, story: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed collaborative filtering information."""
        return {
            'similar_users_count': 5,
            'average_rating': self._get_story_average_rating(story['story_id']),
            'popularity_rank': 10
        }

    def _get_similarity_details(self, story: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed content similarity information."""
        return {
            'similar_stories_count': 3,
            'similarity_factors': ['topic', 'sentiment', 'source'],
            'average_similarity': 0.7
        }

    def _format_primary_reason(self, factor: Tuple[str, Dict[str, Any]]) -> str:
        """Format the primary reason for recommendation."""
        factor_name, factor_data = factor

        reasons = {
            'topic_interest': "Based on your interest in related topics",
            'collaborative': "Similar users found this interesting",
            'content_similarity': "Similar to stories you've read",
            'time_decay': "Fresh and relevant content",
            'diversity': "Explore new perspectives"
        }

        return reasons.get(factor_name, "Personalized for you")

    def _get_popular_recommended_categories(self) -> List[str]:
        """Get most popular categories from user's recommended stories."""
        # Placeholder implementation
        return ['Technology', 'Science', 'Business']

    def _calculate_recommendation_accuracy(self) -> float:
        """Calculate recommendation accuracy based on feedback."""
        # Placeholder implementation
        return 0.75