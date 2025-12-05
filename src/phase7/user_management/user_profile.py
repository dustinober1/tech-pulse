"""
User Profile Management Module for Phase 7.2
Handles user profile operations, preferences, and personalization features
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from collections import defaultdict, Counter

from .database import UserDatabase


class UserProfile:
    """
    User Profile management class that handles user preferences,
    behavior tracking, and personalization features.
    """

    def __init__(self, user_id: str, database: Optional[UserDatabase] = None):
        """
        Initialize UserProfile for a specific user.

        Args:
            user_id (str): Unique user identifier
            database (UserDatabase, optional): Database instance
        """
        self.user_id = user_id
        self.db = database or UserDatabase()
        self._profile_cache = None
        self._last_cache_update = None

    def get_profile(self) -> Dict[str, Any]:
        """
        Get complete user profile with cached optimization.

        Returns:
            dict: User profile data
        """
        # Use cache if recent (within 5 minutes)
        if (self._profile_cache and self._last_cache_update and
            (datetime.now() - self._last_cache_update).seconds < 300):
            return self._profile_cache

        profile = self.db.get_user(self.user_id)
        if not profile:
            raise ValueError(f"User profile not found: {self.user_id}")

        # Enhance profile with additional computed data
        profile['analytics'] = self.get_analytics_summary()
        profile['preferences'] = self.get_enhanced_preferences()
        profile['topics'] = self.get_interest_topics()
        profile['recent_activity'] = self.get_recent_activity()

        self._profile_cache = profile
        self._last_cache_update = datetime.now()

        return profile

    def update_preferences(self, preferences: Dict[str, Any]) -> bool:
        """
        Update user preferences with validation.

        Args:
            preferences (dict): Preferences to update

        Returns:
            bool: True if successful
        """
        # Validate preferences
        validated_prefs = self._validate_preferences(preferences)

        success = self.db.update_user(self.user_id, {'preferences': validated_prefs})

        # Clear cache on update
        if success:
            self._profile_cache = None

        return success

    def get_enhanced_preferences(self) -> Dict[str, Any]:
        """
        Get user preferences with enhanced defaults and computed values.

        Returns:
            dict: Enhanced user preferences
        """
        profile = self.db.get_user(self.user_id)
        if not profile:
            return self._get_default_preferences()

        base_prefs = profile.get('preferences', {})
        enhanced = self._get_default_preferences()
        enhanced.update(base_prefs)

        # Add computed preferences based on behavior
        enhanced['computed'] = {
            'optimal_stories_count': self._compute_optimal_stories_count(),
            'preferred_time_ranges': self._compute_preferred_time_ranges(),
            'reading_speed': self._estimate_reading_speed(),
            'engagement_level': self._calculate_engagement_level()
        }

        return enhanced

    def track_story_interaction(self, story_id: str, interaction_type: str,
                              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Track user interaction with a story and update preferences accordingly.

        Args:
            story_id (str): Story identifier
            interaction_type (str): Type of interaction
            metadata (dict, optional): Additional interaction metadata

        Returns:
            bool: True if tracking successful
        """
        # Track the interaction
        success = self.db.track_interaction(
            self.user_id, story_id, interaction_type, metadata
        )

        if success:
            # Update topic interests based on interaction
            self._update_topic_interests_from_interaction(story_id, interaction_type)

            # Clear cache
            self._profile_cache = None

        return success

    def get_interest_topics(self, min_score: float = 0.1) -> List[Dict[str, Any]]:
        """
        Get user's interested topics with scores.

        Args:
            min_score (float): Minimum interest score threshold

        Returns:
            list: Topics with interest scores
        """
        topics = self.db.get_user_topics(self.user_id, min_score)

        # Enhance with additional metrics
        for topic in topics:
            topic['trend'] = self._calculate_topic_trend(topic['topic_name'])
            topic['last_interaction'] = self._get_last_topic_interaction(topic['topic_name'])
            topic['related_topics'] = self._get_related_topics(topic['topic_name'])

        return topics

    def update_topic_interest(self, topic_name: str, delta_score: float) -> bool:
        """
        Update interest score for a topic using delta adjustment.

        Args:
            topic_name (str): Topic name
            delta_score (float): Score adjustment (-1.0 to 1.0)

        Returns:
            bool: True if successful
        """
        # Get current score
        current_topics = self.db.get_user_topics(self.user_id)
        current_score = 0.5  # Default score for new topics

        for topic in current_topics:
            if topic['topic_name'] == topic_name:
                current_score = topic['interest_score']
                break

        # Calculate new score with bounds checking
        new_score = max(0.0, min(1.0, current_score + delta_score))

        success = self.db.update_topic_interest(self.user_id, topic_name, new_score)

        # Clear cache on update
        if success:
            self._profile_cache = None

        return success

    def get_reading_history(self, days: int = 30, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get user's reading history with analysis.

        Args:
            days (int): Number of days to look back
            limit (int): Maximum number of stories

        Returns:
            list: Reading history with metadata
        """
        interactions = self.db.get_user_interactions(
            self.user_id, 'view', limit
        )

        # Filter by date and enhance with analysis
        cutoff_date = datetime.now() - timedelta(days=days)
        history = []

        for interaction in interactions:
            interaction_date = datetime.fromisoformat(interaction['timestamp'])
            if interaction_date >= cutoff_date:
                enhanced = dict(interaction)
                enhanced['reading_time'] = self._estimate_reading_time(interaction)
                enhanced['engagement_score'] = self._calculate_story_engagement(
                    interaction['story_id']
                )
                history.append(enhanced)

        return history

    def get_reading_patterns(self) -> Dict[str, Any]:
        """
        Analyze user's reading patterns and habits.

        Returns:
            dict: Reading pattern analysis
        """
        interactions = self.db.get_user_interactions(self.user_id, 'view', 1000)

        if not interactions:
            return self._get_default_reading_patterns()

        # Convert to DataFrame for analysis
        df = pd.DataFrame(interactions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['date'] = df['timestamp'].dt.date

        patterns = {
            'most_active_hours': df['hour'].value_counts().head(3).to_dict(),
            'most_active_days': df['day_of_week'].value_counts().head(3).to_dict(),
            'daily_average': len(df) / max(1, df['date'].nunique()),
            'reading_streaks': self._calculate_reading_streaks(df),
            'peak_productivity': self._identify_peak_productivity_times(df),
            'session_patterns': self._analyze_reading_sessions(df)
        }

        return patterns

    def get_personalized_summary(self) -> Dict[str, Any]:
        """
        Generate personalized user summary for dashboard display.

        Returns:
            dict: Personalized summary
        """
        profile = self.get_profile()
        analytics = profile.get('analytics', {})

        summary = {
            'welcome_message': self._generate_welcome_message(),
            'stats_highlights': {
                'stories_read': analytics.get('interaction_counts', {}).get('view', 0),
                'bookmarks_saved': analytics.get('interaction_counts', {}).get('bookmark', 0),
                'topics_following': len(profile.get('topics', [])),
                'searches_performed': analytics.get('search_stats', {}).get('total_searches', 0)
            },
            'trending_topics': self._get_trending_user_topics(),
            'recommendations': self._get_personalized_insights(),
            'achievements': self._check_user_achievements(),
            'suggestions': self._generate_improvement_suggestions()
        }

        return summary

    def get_analytics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive user analytics summary.

        Returns:
            dict: Analytics summary
        """
        return self.db.get_user_analytics(self.user_id)

    def export_profile_data(self) -> Dict[str, Any]:
        """
        Export complete user profile data for backup or analysis.

        Returns:
            dict: Complete profile data
        """
        profile = self.get_profile()

        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'user_id': self.user_id,
            'profile_info': {
                'username': profile.get('username'),
                'created_at': profile.get('created_at'),
                'last_active': profile.get('last_active')
            },
            'preferences': profile.get('preferences', {}),
            'topics': profile.get('topics', []),
            'analytics': profile.get('analytics', {}),
            'reading_history': self.get_reading_history(days=90),
            'search_history': self.db.get_search_history(self.user_id, limit=100),
            'recommendation_feedback': self.db.get_recommendation_feedback(self.user_id)
        }

        return export_data

    def import_profile_data(self, data: Dict[str, Any]) -> bool:
        """
        Import user profile data from export.

        Args:
            data (dict): Exported profile data

        Returns:
            bool: True if import successful
        """
        try:
            # Update preferences
            if 'preferences' in data:
                self.update_preferences(data['preferences'])

            # Update topic interests
            if 'topics' in data:
                for topic in data['topics']:
                    self.db.update_topic_interest(
                        self.user_id,
                        topic['topic_name'],
                        topic['interest_score']
                    )

            # Clear cache after import
            self._profile_cache = None

            return True
        except Exception as e:
            print(f"Error importing profile data: {e}")
            return False

    # Private helper methods

    def _validate_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize user preferences."""
        valid_keys = {
            'theme', 'language', 'notifications', 'preferred_topics',
            'email_alerts', 'auto_refresh', 'stories_per_page',
            'content_filter', 'reading_mode', 'timezone'
        }

        validated = {}
        for key, value in preferences.items():
            if key in valid_keys:
                # Additional validation based on key
                if key == 'theme' and value in ['light', 'dark', 'auto']:
                    validated[key] = value
                elif key == 'language' and isinstance(value, str):
                    validated[key] = value.lower()
                elif key == 'stories_per_page' and isinstance(value, int):
                    validated[key] = max(10, min(100, value))
                elif key == 'notifications' and isinstance(value, bool):
                    validated[key] = value
                else:
                    validated[key] = value

        return validated

    def _get_default_preferences(self) -> Dict[str, Any]:
        """Get default user preferences."""
        return {
            'theme': 'light',
            'language': 'en',
            'notifications': True,
            'preferred_topics': [],
            'email_alerts': False,
            'auto_refresh': False,
            'stories_per_page': 30,
            'content_filter': 'moderate',
            'reading_mode': 'standard',
            'timezone': 'UTC'
        }

    def _update_topic_interests_from_interaction(self, story_id: str, interaction_type: str):
        """Update topic interests based on story interaction."""
        # This would typically extract topics from the story and adjust scores
        # For now, we'll use a simplified approach
        topic_delta = {
            'view': 0.01,
            'like': 0.05,
            'bookmark': 0.1,
            'share': 0.08
        }

        if interaction_type in topic_delta:
            # Extract topics from story (placeholder implementation)
            extracted_topics = self._extract_topics_from_story(story_id)
            for topic in extracted_topics:
                self.update_topic_interest(topic, topic_delta[interaction_type])

    def _extract_topics_from_story(self, story_id: str) -> List[str]:
        """Extract topics from a story (placeholder implementation)."""
        # This would integrate with the content analysis system
        # For now, return empty list
        return []

    def _compute_optimal_stories_count(self) -> int:
        """Compute optimal number of stories based on user behavior."""
        interactions = self.db.get_user_interactions(self.user_id, 'view', 100)
        if not interactions:
            return 30

        # Analyze viewing patterns to determine optimal count
        daily_views = Counter([
            interaction['timestamp'][:10]  # Extract date
            for interaction in interactions
        ])

        avg_daily_views = sum(daily_views.values()) / max(1, len(daily_views))
        return min(100, max(10, int(avg_daily_views * 1.5)))

    def _compute_preferred_time_ranges(self) -> List[Dict[str, str]]:
        """Compute user's preferred reading time ranges."""
        interactions = self.db.get_user_interactions(self.user_id, 'view', 500)

        if not interactions:
            return [{'start': '09:00', 'end': '17:00'}]

        # Analyze interaction times
        hours = [int(interaction['timestamp'][11:13]) for interaction in interactions]
        hour_counts = Counter(hours)

        # Find peak hours
        peak_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        time_ranges = []
        for hour, _ in peak_hours:
            time_ranges.append({
                'start': f"{hour:02d}:00",
                'end': f"{(hour + 2) % 24:02d}:00"
            })

        return time_ranges

    def _estimate_reading_speed(self) -> str:
        """Estimate user's reading speed based on interactions."""
        # Placeholder implementation
        return 'medium'

    def _calculate_engagement_level(self) -> float:
        """Calculate overall engagement level (0.0 to 1.0)."""
        analytics = self.db.get_user_analytics(self.user_id)
        interaction_counts = analytics.get('interaction_counts', {})

        total_interactions = sum(interaction_counts.values())
        if total_interactions == 0:
            return 0.0

        # Weight different interaction types
        weighted_score = (
            interaction_counts.get('view', 0) * 1 +
            interaction_counts.get('like', 0) * 2 +
            interaction_counts.get('bookmark', 0) * 3 +
            interaction_counts.get('share', 0) * 4
        )

        return min(1.0, weighted_score / (total_interactions * 2))

    def _calculate_topic_trend(self, topic_name: str) -> str:
        """Calculate trend for a topic (increasing/decreasing/stable)."""
        # Placeholder implementation
        return 'stable'

    def _get_last_topic_interaction(self, topic_name: str) -> Optional[str]:
        """Get last interaction timestamp for a topic."""
        # Placeholder implementation
        return None

    def _get_related_topics(self, topic_name: str) -> List[str]:
        """Get topics related to the given topic."""
        # Placeholder implementation
        return []

    def _estimate_reading_time(self, interaction: Dict[str, Any]) -> int:
        """Estimate reading time for a story in minutes."""
        # Placeholder implementation
        return 5

    def _calculate_story_engagement(self, story_id: str) -> float:
        """Calculate engagement score for a specific story."""
        # Placeholder implementation
        return 0.5

    def _get_default_reading_patterns(self) -> Dict[str, Any]:
        """Get default reading patterns for new users."""
        return {
            'most_active_hours': {9: 1, 14: 1, 19: 1},
            'most_active_days': {'Monday': 1, 'Wednesday': 1, 'Friday': 1},
            'daily_average': 5.0,
            'reading_streaks': {'current': 0, 'longest': 0},
            'peak_productivity': ['09:00-11:00', '14:00-16:00'],
            'session_patterns': {'avg_session_length': 15, 'sessions_per_day': 3}
        }

    def _calculate_reading_streaks(self, df: pd.DataFrame) -> Dict[str, int]:
        """Calculate reading streaks from interaction data."""
        dates = sorted(df['date'].unique())

        if not dates:
            return {'current': 0, 'longest': 0}

        current_streak = 0
        longest_streak = 0
        temp_streak = 0

        for i in range(len(dates)):
            if i == 0:
                temp_streak = 1
            elif (dates[i] - dates[i-1]).days == 1:
                temp_streak += 1
            else:
                longest_streak = max(longest_streak, temp_streak)
                temp_streak = 1

            # Check if this is the most recent streak
            if dates[i] == datetime.now().date():
                current_streak = temp_streak

        longest_streak = max(longest_streak, temp_streak)

        return {'current': current_streak, 'longest': longest_streak}

    def _identify_peak_productivity_times(self, df: pd.DataFrame) -> List[str]:
        """Identify peak productivity time ranges."""
        hourly_activity = df['hour'].value_counts()
        peak_hours = hourly_activity.nlargest(3).index.tolist()

        return [f"{h:02d}:00-{(h+2)%24:02d}:00" for h in peak_hours]

    def _analyze_reading_sessions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze reading session patterns."""
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp')

        # Calculate time differences between consecutive reads
        df_sorted['time_diff'] = df_sorted['timestamp'].diff().dt.total_seconds() / 60

        # Define sessions (reads within 30 minutes are same session)
        session_threshold = 30
        df_sorted['new_session'] = df_sorted['time_diff'] > session_threshold

        # Calculate session metrics
        session_starts = df_sorted[df_sorted['new_session'] | df_sorted['time_diff'].isna()]
        session_lengths = df_sorted.groupby(df_sorted['new_session'].cumsum()).size()

        return {
            'avg_session_length': session_lengths.mean() if not session_lengths.empty else 1,
            'sessions_per_day': len(session_starts) / max(1, df_sorted['date'].nunique()),
            'longest_session': session_lengths.max() if not session_lengths.empty else 1
        }

    def _generate_welcome_message(self) -> str:
        """Generate personalized welcome message."""
        profile = self.get_profile()
        username = profile.get('username', 'User')

        hour = datetime.now().hour
        if 5 <= hour < 12:
            time_greeting = "Good morning"
        elif 12 <= hour < 17:
            time_greeting = "Good afternoon"
        else:
            time_greeting = "Good evening"

        engagement = self._calculate_engagement_level()
        if engagement > 0.7:
            engagement_msg = "Great to see you staying engaged!"
        elif engagement > 0.3:
            engagement_msg = "Keep up the good work!"
        else:
            engagement_msg = "Discover more interesting stories today!"

        return f"{time_greeting}, {username}! {engagement_msg}"

    def _get_trending_user_topics(self) -> List[Dict[str, Any]]:
        """Get user's trending topics."""
        topics = self.get_interest_topics()
        return topics[:5]  # Return top 5 trending topics

    def _get_personalized_insights(self) -> List[str]:
        """Generate personalized insights for the user."""
        insights = []

        patterns = self.get_reading_patterns()
        if patterns['daily_average'] > 10:
            insights.append("You're a voracious reader! Consider diversifying your topics.")

        engagement = self._calculate_engagement_level()
        if engagement < 0.3:
            insights.append("Try bookmarking or sharing stories you find interesting.")

        return insights

    def _check_user_achievements(self) -> List[Dict[str, str]]:
        """Check and return user achievements."""
        achievements = []
        analytics = self.get_analytics_summary()

        interaction_counts = analytics.get('interaction_counts', {})

        if interaction_counts.get('view', 0) > 100:
            achievements.append({'name': 'Avid Reader', 'icon': '📚'})

        if interaction_counts.get('bookmark', 0) > 50:
            achievements.append({'name': 'Bookmark Master', 'icon': '🔖'})

        return achievements

    def _generate_improvement_suggestions(self) -> List[str]:
        """Generate personalized improvement suggestions."""
        suggestions = []

        topics = self.get_interest_topics()
        if len(topics) < 3:
            suggestions.append("Explore more topics to diversify your feed.")

        patterns = self.get_reading_patterns()
        if patterns['daily_average'] < 3:
            suggestions.append("Set a daily reading goal to stay updated.")

        return suggestions

    def get_recent_activity(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get user's recent activity across all interactions.

        Args:
            limit (int): Maximum number of activities to return

        Returns:
            list: Recent activities
        """
        activities = []

        # Get recent interactions
        interactions = self.db.get_user_interactions(self.user_id, limit=limit)

        # Get recent searches
        searches = self.db.get_search_history(self.user_id, limit=limit)

        # Combine and sort by timestamp
        for interaction in interactions:
            activities.append({
                'type': 'interaction',
                'subtype': interaction['interaction_type'],
                'target': interaction['story_id'],
                'timestamp': interaction['timestamp'],
                'data': interaction.get('interaction_data', {})
            })

        for search in searches:
            activities.append({
                'type': 'search',
                'subtype': search['search_type'],
                'target': search['search_query'],
                'timestamp': search['timestamp'],
                'data': {'results_count': search['results_count']}
            })

        # Sort by timestamp (most recent first)
        activities.sort(key=lambda x: x['timestamp'], reverse=True)

        return activities[:limit]

    def clear_cache(self):
        """Clear the internal profile cache."""
        self._profile_cache = None
        self._last_cache_update = None