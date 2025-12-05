"""
UI Components Module for Phase 7.2
Streamlit UI components for user management and personalization features
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json

from .database import UserDatabase
from .user_profile import UserProfile
from .recommendations import PersonalizedRecommendations


class UIComponents:
    """
    Streamlit UI components for user management and personalization.
    Provides ready-to-use components for dashboard integration.
    """

    def __init__(self, user_id: str):
        """
        Initialize UI components for a specific user.

        Args:
            user_id (str): Unique user identifier
        """
        self.user_id = user_id
        self.db = UserDatabase()
        self.user_profile = UserProfile(user_id, self.db)
        self.recommendations = PersonalizedRecommendations(user_id, self.db)

    def render_user_profile_section(self) -> Dict[str, Any]:
        """
        Render comprehensive user profile section.

        Returns:
            dict: User interaction data
        """
        st.markdown("### 👤 Your Profile")

        # Get user profile
        profile = self.user_profile.get_profile()
        summary = self.user_profile.get_personalized_summary()

        # Welcome message
        st.info(summary['welcome_message'])

        # Profile stats in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Stories Read",
                value=summary['stats_highlights']['stories_read']
            )

        with col2:
            st.metric(
                label="Bookmarks",
                value=summary['stats_highlights']['bookmarks_saved']
            )

        with col3:
            st.metric(
                label="Topics",
                value=summary['stats_highlights']['topics_following']
            )

        with col4:
            st.metric(
                label="Searches",
                value=summary['stats_highlights']['searches_performed']
            )

        # Personalized insights
        if summary['recommendations']:
            st.markdown("#### 💡 Personalized Insights")
            for insight in summary['recommendations']:
                st.markdown(f"• {insight}")

        # Achievements
        if summary['achievements']:
            st.markdown("#### 🏆 Achievements")
            achievement_cols = st.columns(min(4, len(summary['achievements'])))
            for i, achievement in enumerate(summary['achievements']):
                with achievement_cols[i % len(achievement_cols)]:
                    st.markdown(f"{achievement['icon']} {achievement['name']}")

        # Improvement suggestions
        if summary['suggestions']:
            st.markdown("#### 📈 Suggestions")
            for suggestion in summary['suggestions']:
                st.markdown(f"• {suggestion}")

        return {'viewed_profile': True}

    def render_preferences_editor(self) -> Dict[str, Any]:
        """
        Render user preferences editor.

        Returns:
            dict: Updated preferences
        """
        st.markdown("### ⚙️ Preferences")

        # Get current preferences
        preferences = self.user_profile.get_enhanced_preferences()

        updated_prefs = {}

        # Theme selection
        with st.expander("🎨 Appearance", expanded=True):
            theme = st.selectbox(
                "Theme",
                options=['light', 'dark', 'auto'],
                index=['light', 'dark', 'auto'].index(preferences.get('theme', 'light')),
                help="Choose your preferred color theme"
            )
            updated_prefs['theme'] = theme

            stories_per_page = st.slider(
                "Stories per page",
                min_value=10,
                max_value=100,
                value=preferences.get('stories_per_page', 30),
                step=5,
                help="Number of stories to display per page"
            )
            updated_prefs['stories_per_page'] = stories_per_page

        # Notification settings
        with st.expander("🔔 Notifications"):
            notifications = st.checkbox(
                "Enable notifications",
                value=preferences.get('notifications', True),
                help="Receive notifications for new stories and updates"
            )
            updated_prefs['notifications'] = notifications

            email_alerts = st.checkbox(
                "Email alerts",
                value=preferences.get('email_alerts', False),
                help="Receive daily email summaries"
            )
            updated_prefs['email_alerts'] = email_alerts

        # Reading preferences
        with st.expander("📖 Reading Experience"):
            auto_refresh = st.checkbox(
                "Auto-refresh content",
                value=preferences.get('auto_refresh', False),
                help="Automatically refresh content every 30 minutes"
            )
            updated_prefs['auto_refresh'] = auto_refresh

            reading_mode = st.selectbox(
                "Reading mode",
                options=['standard', 'comfort', 'focused'],
                index=['standard', 'comfort', 'focused'].index(
                    preferences.get('reading_mode', 'standard')
                ),
                help="Optimize display for your reading preference"
            )
            updated_prefs['reading_mode'] = reading_mode

        # Content filtering
        with st.expander("🔍 Content Filter"):
            content_filter = st.select_slider(
                "Content filter level",
                options=['minimal', 'moderate', 'strict'],
                value=preferences.get('content_filter', 'moderate'),
                help="Filter content based on your preferences"
            )
            updated_prefs['content_filter'] = content_filter

        # Save button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("💾 Save Preferences", type="primary", use_container_width=True):
                success = self.user_profile.update_preferences(updated_prefs)
                if success:
                    st.success("Preferences saved successfully!")
                    st.rerun()
                else:
                    st.error("Failed to save preferences")

        return updated_prefs

    def render_topic_management(self) -> Dict[str, Any]:
        """
        Render topic interest management interface.

        Returns:
            dict: Topic interaction data
        """
        st.markdown("### 🏷️ Topic Interests")

        # Get current topics
        topics = self.user_profile.get_interest_topics()

        if topics:
            # Display current topics
            st.markdown("#### Your Interested Topics")

            # Create DataFrame for visualization
            topic_df = pd.DataFrame(topics)

            # Topic interest chart
            fig = px.bar(
                topic_df.head(10),
                x='topic_name',
                y='interest_score',
                title="Your Topic Interest Levels",
                labels={'interest_score': 'Interest Level', 'topic_name': 'Topic'},
                color='interest_score',
                color_continuous_scale='viridis'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Topic management
            st.markdown("#### Manage Topics")
            selected_topic = st.selectbox(
                "Select a topic to update",
                options=[topic['topic_name'] for topic in topics],
                format_func=lambda x: f"{x} (Score: {next(t['interest_score'] for t in topics if t['topic_name'] == x):.2f})"
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button("📈 Increase Interest", type="primary"):
                    self.user_profile.update_topic_interest(selected_topic, 0.1)
                    st.success(f"Increased interest in {selected_topic}")
                    st.rerun()

            with col2:
                if st.button("📉 Decrease Interest"):
                    self.user_profile.update_topic_interest(selected_topic, -0.1)
                    st.success(f"Decreased interest in {selected_topic}")
                    st.rerun()

        # Add new topic
        st.markdown("#### Add New Topic")
        new_topic = st.text_input("Enter a topic name", key="new_topic_input")

        if st.button("➕ Add Topic") and new_topic:
            self.user_profile.update_topic_interest(new_topic, 0.5)
            st.success(f"Added {new_topic} to your interests")
            st.rerun()

        # Recommended topics
        st.markdown("#### 🎯 Recommended Topics")
        recommended_topics = self._get_recommended_topics()

        for topic in recommended_topics:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"• {topic}")
            with col2:
                if st.button("Add", key=f"add_{topic}"):
                    self.user_profile.update_topic_interest(topic, 0.6)
                    st.rerun()

        return {'topics_managed': len(topics)}

    def render_recommendations_section(self) -> Dict[str, Any]:
        """
        Render personalized recommendations section.

        Returns:
            dict: Recommendation interaction data
        """
        st.markdown("### 🎯 Personalized Recommendations")

        # Recommendation type selector
        rec_type = st.selectbox(
            "Recommendation type",
            options=['mixed', 'topics', 'collaborative', 'trending', 'diverse'],
            format_func=lambda x: {
                'mixed': '🔀 Mixed Recommendations',
                'topics': '🏷️ Based on Your Topics',
                'collaborative': '👥 Similar Users',
                'trending': '🔥 Trending Now',
                'diverse': '🌟 Explore New'
            }[x]
        )

        # Get recommendations
        count = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)
        recommendations = self.recommendations.get_recommendations(count, rec_type)

        if recommendations:
            # Display recommendations
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"{i}. {rec.title}", expanded=i <= 3):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.write(f"**Score:** {rec.score:.2f}/1.0")
                        st.write(f"**Reason:** {rec.reason}")

                        # Feedback buttons
                        fb_col1, fb_col2, fb_col3, fb_col4 = st.columns(4)

                        with fb_col1:
                            if st.button("👍 Like", key=f"like_{rec.story_id}"):
                                self.recommendations.update_recommendation_feedback(
                                    rec.story_id, 'like', 4
                                )
                                st.success("Thanks for your feedback!")

                        with fb_col2:
                            if st.button("🔖 Bookmark", key=f"bookmark_{rec.story_id}"):
                                self.recommendations.update_recommendation_feedback(
                                    rec.story_id, 'bookmark', 5
                                )
                                st.success("Bookmarked!")

                        with fb_col3:
                            if st.button("↗️ Share", key=f"share_{rec.story_id}"):
                                self.recommendations.update_recommendation_feedback(
                                    rec.story_id, 'share', 4
                                )
                                st.success("Shared!")

                        with fb_col4:
                            if st.button("❌ Not Interested", key=f"dismiss_{rec.story_id}"):
                                self.recommendations.update_recommendation_feedback(
                                    rec.story_id, 'dismiss', -1
                                )
                                st.success("We'll show less of this")

                    with col2:
                        # Explanation
                        if st.button("🤔 Why this?", key=f"explain_{rec.story_id}"):
                            explanation = self.recommendations.get_recommendation_explanation(
                                rec.story_id
                            )
                            st.json(explanation)

            # Recommendation stats
            stats = self.recommendations.get_recommendation_stats()
            st.markdown("#### 📊 Recommendation Performance")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Total Recommendations",
                    stats['total_recommendations']
                )

            with col2:
                st.metric(
                    "Average Rating",
                    f"{stats['average_rating']:.1f}/5.0"
                )

            with col3:
                st.metric(
                    "Click Rate",
                    f"{stats['click_through_rate']:.1%}"
                )

        else:
            st.info("No recommendations available. Start reading some stories to get personalized suggestions!")

        return {'recommendations_viewed': len(recommendations) if recommendations else 0}

    def render_analytics_dashboard(self) -> Dict[str, Any]:
        """
        Render user analytics dashboard.

        Returns:
            dict: Analytics view data
        """
        st.markdown("### 📊 Your Analytics")

        # Get analytics data
        analytics = self.user_profile.get_analytics_summary()
        reading_patterns = self.user_profile.get_reading_patterns()
        reading_history = self.user_profile.get_reading_history(days=30)

        # Time period selector
        period = st.selectbox(
            "Time period",
            options=['7 days', '30 days', '90 days'],
            index=1
        )
        days = int(period.split()[0])

        # Reading activity chart
        st.markdown("#### 📈 Reading Activity")

        if reading_history:
            # Create activity DataFrame
            activity_df = pd.DataFrame(reading_history)
            activity_df['date'] = pd.to_datetime(activity_df['timestamp']).dt.date

            daily_counts = activity_df.groupby('date').size().reset_index(name='count')

            fig = px.line(
                daily_counts,
                x='date',
                y='count',
                title=f"Reading Activity - Last {days} Days",
                labels={'count': 'Stories Read', 'date': 'Date'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Reading patterns
        st.markdown("#### ⏰ Reading Patterns")

        pattern_cols = st.columns(3)

        with pattern_cols[0]:
            st.metric(
                "Daily Average",
                f"{reading_patterns.get('daily_average', 0):.1f} stories"
            )

        with pattern_cols[1]:
            current_streak = reading_patterns.get('reading_streaks', {}).get('current', 0)
            st.metric(
                "Current Streak",
                f"{current_streak} days"
            )

        with pattern_cols[2]:
            longest_streak = reading_patterns.get('reading_streaks', {}).get('longest', 0)
            st.metric(
                "Longest Streak",
                f"{longest_streak} days"
            )

        # Most active times
        most_active_hours = reading_patterns.get('most_active_hours', {})
        if most_active_hours:
            st.markdown("#### 🕐 Most Active Hours")

            hour_df = pd.DataFrame(
                list(most_active_hours.items()),
                columns=['hour', 'count']
            )

            fig = px.bar(
                hour_df,
                x='hour',
                y='count',
                title="Activity by Hour of Day",
                labels={'count': 'Stories Read', 'hour': 'Hour'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Topic distribution
        st.markdown("#### 🏷️ Topic Distribution")

        topics = self.user_profile.get_interest_topics()
        if topics:
            topic_df = pd.DataFrame(topics)

            # Create pie chart
            fig = px.pie(
                topic_df.head(8),
                values='interest_score',
                names='topic_name',
                title="Your Topic Interests"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Interaction breakdown
        interaction_counts = analytics.get('interaction_counts', {})
        if interaction_counts:
            st.markdown("#### 📊 Interaction Breakdown")

            interaction_df = pd.DataFrame(
                list(interaction_counts.items()),
                columns=['type', 'count']
            )

            fig = px.bar(
                interaction_df,
                x='type',
                y='count',
                title="Your Interactions",
                labels={'count': 'Number of Interactions', 'type': 'Type'}
            )
            st.plotly_chart(fig, use_container_width=True)

        return {'analytics_viewed': True}

    def render_activity_feed(self, limit: int = 20) -> Dict[str, Any]:
        """
        Render user's recent activity feed.

        Args:
            limit (int): Number of activities to show

        Returns:
            dict: Activity view data
        """
        st.markdown("### 🕐 Recent Activity")

        # Get recent activity
        activities = self.user_profile.get_recent_activity(limit)

        if activities:
            for activity in activities:
                # Format timestamp
                timestamp = datetime.fromisoformat(activity['timestamp'])
                time_ago = self._format_time_ago(timestamp)

                # Activity icon and color
                icon, color = self._get_activity_icon(activity)

                # Display activity
                st.markdown(
                    f"""
                    <div style='padding: 10px; margin: 5px 0; border-left: 3px solid {color}; background-color: #f0f2f6;'>
                        <span style='color: {color}; font-size: 1.2em;'>{icon}</span>
                        <strong>{activity['type'].title()} {activity['subtype'].title()}</strong>
                        <br>
                        <small>{activity['target']}</small>
                        <br>
                        <small style='color: #666;'>{time_ago}</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No recent activity to show")

        return {'activities_viewed': len(activities)}

    def render_quick_actions(self) -> Dict[str, Any]:
        """
        Render quick action buttons for common tasks.

        Returns:
            dict: Action click data
        """
        st.markdown("### ⚡ Quick Actions")

        action_data = {}

        # Action buttons in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📚 View Reading History", use_container_width=True):
                action_data['viewed_history'] = True
                st.session_state.show_reading_history = True

        with col2:
            if st.button("🏷️ Manage Topics", use_container_width=True):
                action_data['manage_topics'] = True
                st.session_state.show_topic_management = True

        with col3:
            if st.button("📊 View Analytics", use_container_width=True):
                action_data['view_analytics'] = True
                st.session_state.show_analytics = True

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🎯 Get Recommendations", use_container_width=True):
                action_data['get_recommendations'] = True
                st.session_state.show_recommendations = True

        with col2:
            if st.button("⚙️ Edit Preferences", use_container_width=True):
                action_data['edit_preferences'] = True
                st.session_state.edit_preferences = True

        return action_data

    def render_mini_profile_card(self) -> None:
        """Render a compact profile card for sidebar display."""
        profile = self.user_profile.get_profile()
        summary = self.user_profile.get_personalized_summary()

        st.markdown("---")
        st.markdown("### 👤 Profile")

        # Basic info
        st.write(f"**{profile.get('username', 'User')}**")

        # Quick stats
        stats_cols = st.columns(2)

        with stats_cols[0]:
            st.metric("Read", summary['stats_highlights']['stories_read'])

        with stats_cols[1]:
            st.metric("Topics", summary['stats_highlights']['topics_following'])

        # Engagement level
        engagement = summary.get('computed', {}).get('engagement_level', 0) * 100
        st.progress(engagement / 100)
        st.caption(f"Engagement: {engagement:.0f}%")

    # Private helper methods

    def _get_recommended_topics(self) -> List[str]:
        """Get recommended topics for the user."""
        # Placeholder implementation
        return ['Artificial Intelligence', 'Machine Learning', 'Web Development', 'Cloud Computing', 'Data Science']

    def _format_time_ago(self, timestamp: datetime) -> str:
        """Format timestamp as 'time ago' string."""
        now = datetime.now()
        diff = now - timestamp

        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "Just now"

    def _get_activity_icon(self, activity: Dict[str, Any]) -> Tuple[str, str]:
        """Get icon and color for activity type."""
        activity_type = activity['type']
        subtype = activity['subtype']

        icons = {
            ('interaction', 'view'): ('👁️', '#4CAF50'),
            ('interaction', 'like'): ('👍', '#2196F3'),
            ('interaction', 'bookmark'): ('🔖', '#FF9800'),
            ('interaction', 'share'): ('↗️', '#9C27B0'),
            ('search', 'keyword'): ('🔍', '#607D8B'),
            ('search', 'semantic'): ('🧠', '#3F51B5'),
            ('search', 'topic'): ('🏷️', '#009688')
        }

        return icons.get((activity_type, subtype), ('📝', '#9E9E9E'))

    def render_export_data_section(self) -> bool:
        """
        Render data export functionality.

        Returns:
            bool: Whether export was initiated
        """
        st.markdown("### 📤 Export Your Data")

        st.info("Export your profile data for backup or analysis")

        if st.button("📥 Download Profile Data", type="primary"):
            try:
                export_data = self.user_profile.export_profile_data()

                # Convert to JSON for download
                json_data = json.dumps(export_data, indent=2, default=str)

                # Provide download button
                st.download_button(
                    label="Download JSON file",
                    data=json_data,
                    file_name=f"profile_export_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )

                return True
            except Exception as e:
                st.error(f"Failed to export data: {str(e)}")
                return False

        return False

    def render_import_data_section(self) -> bool:
        """
        Render data import functionality.

        Returns:
            bool: Whether import was successful
        """
        st.markdown("### 📥 Import Profile Data")

        st.warning("⚠️ Importing data will overwrite your current preferences and topic interests")

        uploaded_file = st.file_uploader("Choose a JSON file", type=['json'])

        if uploaded_file is not None:
            try:
                # Read and parse the file
                content = uploaded_file.read().decode('utf-8')
                import_data = json.loads(content)

                # Show preview
                with st.expander("Preview Import Data"):
                    st.json(import_data)

                # Confirm import
                if st.button("🔄 Import Data", type="primary"):
                    success = self.user_profile.import_profile_data(import_data)

                    if success:
                        st.success("Profile data imported successfully!")
                        st.rerun()
                        return True
                    else:
                        st.error("Failed to import data")
                        return False

            except json.JSONDecodeError:
                st.error("Invalid JSON file")
                return False
            except Exception as e:
                st.error(f"Error importing data: {str(e)}")
                return False

        return False