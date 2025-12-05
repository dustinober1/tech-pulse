"""
User Management Database Module for Phase 7.2
Handles database operations for user profiles and preferences
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import os
from contextlib import contextmanager

class UserDatabase:
    """
    SQLite database manager for user profiles and preferences.
    Handles all database operations including user creation, updates, and analytics tracking.
    """

    def __init__(self, db_path: str = "data/user_profiles.db"):
        """
        Initialize UserDatabase with specified database path.

        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_data_directory()
        self._initialize_database()

    def _ensure_data_directory(self):
        """Ensure the data directory exists for the database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
        try:
            yield conn
        finally:
            conn.close()

    def _initialize_database(self):
        """Initialize database tables if they don't exist."""
        with self._get_connection() as conn:
            # Users table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    preferences TEXT,  -- JSON string
                    is_active BOOLEAN DEFAULT 1
                )
            """)

            # User interactions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    story_id TEXT NOT NULL,
                    interaction_type TEXT NOT NULL,  -- 'view', 'like', 'bookmark', 'share'
                    interaction_data TEXT,  -- JSON string for additional data
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    UNIQUE(user_id, story_id, interaction_type)
                )
            """)

            # User topics table for personalized interests
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    topic_name TEXT NOT NULL,
                    interest_score REAL DEFAULT 1.0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    UNIQUE(user_id, topic_name)
                )
            """)

            # User search history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    search_query TEXT NOT NULL,
                    search_type TEXT NOT NULL,  -- 'keyword', 'semantic', 'topic'
                    results_count INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)

            # User recommendations tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recommendation_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    story_id TEXT NOT NULL,
                    recommendation_type TEXT NOT NULL,
                    feedback_score INTEGER,  -- 1-5 rating or -1 for negative
                    feedback_type TEXT,  -- 'rating', 'click', 'ignore'
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)

            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_user_story ON user_interactions(user_id, story_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_type ON user_interactions(interaction_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_topics_user ON user_topics(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_search_user ON search_history(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_user ON recommendation_feedback(user_id)")

            conn.commit()

    def create_user(self, username: str, email: Optional[str] = None,
                   preferences: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new user profile.

        Args:
            username (str): Unique username
            email (str, optional): User email address
            preferences (dict, optional): User preferences

        Returns:
            str: Generated user ID
        """
        user_id = self._generate_user_id(username)

        default_preferences = {
            "theme": "light",
            "language": "en",
            "notifications": True,
            "preferred_topics": [],
            "email_alerts": False,
            "auto_refresh": False,
            "stories_per_page": 30
        }

        if preferences:
            default_preferences.update(preferences)

        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO users (user_id, username, email, preferences)
                    VALUES (?, ?, ?, ?)
                """, (user_id, username, email, json.dumps(default_preferences)))
                conn.commit()
                return user_id
            except sqlite3.IntegrityError as e:
                raise ValueError(f"Username or email already exists: {str(e)}")

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve user profile by ID.

        Args:
            user_id (str): User ID to retrieve

        Returns:
            dict: User profile data or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM users WHERE user_id = ? AND is_active = 1
            """, (user_id,))
            row = cursor.fetchone()

            if row:
                user_data = dict(row)
                user_data['preferences'] = json.loads(user_data['preferences'] or '{}')
                return user_data
            return None

    def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update user profile.

        Args:
            user_id (str): User ID to update
            updates (dict): Fields to update

        Returns:
            bool: True if successful, False otherwise
        """
        if not updates:
            return False

        # Handle preferences separately if present
        preferences_update = None
        if 'preferences' in updates:
            preferences_update = json.dumps(updates.pop('preferences'))

        with self._get_connection() as conn:
            try:
                # Update preferences if provided
                if preferences_update is not None:
                    conn.execute("""
                        UPDATE users SET preferences = ? WHERE user_id = ?
                    """, (preferences_update, user_id))

                # Update other fields
                if updates:
                    set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
                    values = list(updates.values()) + [user_id]
                    conn.execute(f"""
                        UPDATE users SET {set_clause} WHERE user_id = ?
                    """, values)

                # Update last_active timestamp
                conn.execute("""
                    UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE user_id = ?
                """, (user_id,))

                conn.commit()
                return True
            except sqlite3.Error:
                return False

    def track_interaction(self, user_id: str, story_id: str,
                         interaction_type: str, interaction_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Track user interaction with a story.

        Args:
            user_id (str): User ID
            story_id (str): Story ID
            interaction_type (str): Type of interaction ('view', 'like', 'bookmark', 'share')
            interaction_data (dict, optional): Additional interaction data

        Returns:
            bool: True if successful, False otherwise
        """
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO user_interactions
                    (user_id, story_id, interaction_type, interaction_data)
                    VALUES (?, ?, ?, ?)
                """, (user_id, story_id, interaction_type, json.dumps(interaction_data or {})))
                conn.commit()
                return True
            except sqlite3.Error:
                return False

    def get_user_interactions(self, user_id: str, interaction_type: Optional[str] = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get user interactions.

        Args:
            user_id (str): User ID
            interaction_type (str, optional): Filter by interaction type
            limit (int): Maximum number of interactions to return

        Returns:
            list: List of interactions
        """
        with self._get_connection() as conn:
            if interaction_type:
                cursor = conn.execute("""
                    SELECT * FROM user_interactions
                    WHERE user_id = ? AND interaction_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (user_id, interaction_type, limit))
            else:
                cursor = conn.execute("""
                    SELECT * FROM user_interactions
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (user_id, limit))

            interactions = []
            for row in cursor.fetchall():
                interaction = dict(row)
                interaction['interaction_data'] = json.loads(interaction['interaction_data'] or '{}')
                interactions.append(interaction)

            return interactions

    def update_topic_interest(self, user_id: str, topic_name: str, interest_score: float) -> bool:
        """
        Update user's interest score for a topic.

        Args:
            user_id (str): User ID
            topic_name (str): Topic name
            interest_score (float): Interest score (0.0 to 1.0)

        Returns:
            bool: True if successful, False otherwise
        """
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO user_topics
                    (user_id, topic_name, interest_score, last_updated)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (user_id, topic_name, max(0.0, min(1.0, interest_score))))
                conn.commit()
                return True
            except sqlite3.Error:
                return False

    def get_user_topics(self, user_id: str, min_score: float = 0.1) -> List[Dict[str, Any]]:
        """
        Get user's preferred topics with interest scores.

        Args:
            user_id (str): User ID
            min_score (float): Minimum interest score to include

        Returns:
            list: List of topics with interest scores
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM user_topics
                WHERE user_id = ? AND interest_score >= ?
                ORDER BY interest_score DESC
            """, (user_id, min_score))

            return [dict(row) for row in cursor.fetchall()]

    def track_search(self, user_id: str, search_query: str, search_type: str,
                    results_count: int) -> bool:
        """
        Track user search query.

        Args:
            user_id (str): User ID
            search_query (str): Search query
            search_type (str): Type of search ('keyword', 'semantic', 'topic')
            results_count (int): Number of results returned

        Returns:
            bool: True if successful, False otherwise
        """
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO search_history
                    (user_id, search_query, search_type, results_count)
                    VALUES (?, ?, ?, ?)
                """, (user_id, search_query, search_type, results_count))
                conn.commit()
                return True
            except sqlite3.Error:
                return False

    def get_search_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get user's search history.

        Args:
            user_id (str): User ID
            limit (int): Maximum number of searches to return

        Returns:
            list: List of search history
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM search_history
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, limit))

            return [dict(row) for row in cursor.fetchall()]

    def record_recommendation_feedback(self, user_id: str, story_id: str,
                                      recommendation_type: str, feedback_score: int,
                                      feedback_type: str) -> bool:
        """
        Record user feedback on recommendations.

        Args:
            user_id (str): User ID
            story_id (str): Story ID
            recommendation_type (str): Type of recommendation
            feedback_score (int): Feedback score (-1 to 5)
            feedback_type (str): Type of feedback ('rating', 'click', 'ignore')

        Returns:
            bool: True if successful, False otherwise
        """
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO recommendation_feedback
                    (user_id, story_id, recommendation_type, feedback_score, feedback_type)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, story_id, recommendation_type, feedback_score, feedback_type))
                conn.commit()
                return True
            except sqlite3.Error:
                return False

    def get_recommendation_feedback(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get user's recommendation feedback history.

        Args:
            user_id (str): User ID
            limit (int): Maximum number of feedback entries to return

        Returns:
            list: List of recommendation feedback
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM recommendation_feedback
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, limit))

            return [dict(row) for row in cursor.fetchall()]

    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive user analytics.

        Args:
            user_id (str): User ID

        Returns:
            dict: User analytics data
        """
        with self._get_connection() as conn:
            # Interaction counts by type
            cursor = conn.execute("""
                SELECT interaction_type, COUNT(*) as count
                FROM user_interactions
                WHERE user_id = ?
                GROUP BY interaction_type
            """, (user_id,))
            interaction_counts = {row['interaction_type']: row['count'] for row in cursor.fetchall()}

            # Top topics by interest
            cursor = conn.execute("""
                SELECT topic_name, interest_score
                FROM user_topics
                WHERE user_id = ?
                ORDER BY interest_score DESC
                LIMIT 10
            """, (user_id,))
            top_topics = [dict(row) for row in cursor.fetchall()]

            # Recent search frequency
            cursor = conn.execute("""
                SELECT COUNT(*) as total_searches,
                       AVG(results_count) as avg_results
                FROM search_history
                WHERE user_id = ? AND timestamp > datetime('now', '-7 days')
            """, (user_id,))
            search_stats = dict(cursor.fetchone() or {'total_searches': 0, 'avg_results': 0})

            # Recommendation feedback summary
            cursor = conn.execute("""
                SELECT AVG(feedback_score) as avg_rating,
                       COUNT(*) as total_feedback
                FROM recommendation_feedback
                WHERE user_id = ? AND feedback_score > 0
            """, (user_id,))
            feedback_stats = dict(cursor.fetchone() or {'avg_rating': 0, 'total_feedback': 0})

            return {
                'interaction_counts': interaction_counts,
                'top_topics': top_topics,
                'search_stats': search_stats,
                'feedback_stats': feedback_stats,
                'last_updated': datetime.now().isoformat()
            }

    def delete_user(self, user_id: str) -> bool:
        """
        Soft delete user (mark as inactive).

        Args:
            user_id (str): User ID to delete

        Returns:
            bool: True if successful, False otherwise
        """
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    UPDATE users SET is_active = 0 WHERE user_id = ?
                """, (user_id,))
                conn.commit()
                return conn.total_changes > 0
            except sqlite3.Error:
                return False

    def _generate_user_id(self, username: str) -> str:
        """
        Generate a unique user ID.

        Args:
            username (str): Username to include in ID

        Returns:
            str: Generated user ID
        """
        timestamp = str(int(datetime.now().timestamp()))
        unique_string = f"{username}{timestamp}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:16]

    def get_all_active_users(self) -> List[Dict[str, Any]]:
        """
        Get all active users.

        Returns:
            list: List of active user profiles
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT user_id, username, email, created_at, last_active
                FROM users
                WHERE is_active = 1
                ORDER BY last_active DESC
            """)

            users = []
            for row in cursor.fetchall():
                user = dict(row)
                users.append(user)

            return users