"""
Tech-Pulse Dashboard: Interactive Streamlit application for tech news analysis.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import time

# Import our custom modules
from data_loader import fetch_hn_data, analyze_sentiment, get_topics
from dashboard_config import (
    PAGE_CONFIG, COLORS, SENTIMENT_COLORS, DEFAULT_SETTINGS,
    CHART_CONFIG, HELP_TEXT, ERROR_MESSAGES, SUCCESS_MESSAGES, LOADING_MESSAGES,
    REAL_TIME_SETTINGS
)

# Configure page
st.set_page_config(**PAGE_CONFIG)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = None
    if 'real_time_mode' not in st.session_state:
        st.session_state.real_time_mode = False
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = None
    if 'stories_count' not in st.session_state:
        st.session_state.stories_count = DEFAULT_SETTINGS['default_stories']
    # Remove old auto_refresh and refresh_countdown if they exist
    if 'auto_refresh' in st.session_state:
        del st.session_state.auto_refresh
    if 'refresh_countdown' in st.session_state:
        del st.session_state.refresh_countdown

def create_header():
    """Create dashboard header"""
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0; border-bottom: 2px solid #E8E8E8; margin-bottom: 2rem;'>
        <h1 style='color: #2C3E50; margin: 0; font-size: 2.5rem;'>
            ⚡ Tech-Pulse Dashboard
        </h1>
        <p style='color: #7F8C8D; margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
            Real-time analysis of trending tech stories from Hacker News
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create interactive sidebar"""
    with st.sidebar:
        # Header
        st.markdown("### 🎛️ Control Panel")

        # Stories count slider
        stories_count = st.slider(
            "Number of Stories",
            min_value=DEFAULT_SETTINGS['min_stories'],
            max_value=DEFAULT_SETTINGS['max_stories'],
            value=st.session_state.stories_count,
            step=5,
            help="Select the number of top stories to analyze"
        )

        # Update session state
        st.session_state.stories_count = stories_count

        st.markdown("---")

        # Refresh button
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            refresh_data()

        # Real-time mode toggle
        real_time_mode = st.toggle(
            "Enable Real-Time Mode",
            value=st.session_state.real_time_mode,
            help="Automatically refresh data every 60 seconds for live updates"
        )
        st.session_state.real_time_mode = real_time_mode

        # Visual indicator when real-time mode is active
        if st.session_state.real_time_mode:
            st.success("🟢 Real-time mode is active")

        # Last refresh info
        if st.session_state.last_refresh:
            time_str = st.session_state.last_refresh.strftime('%H:%M:%S')
            if st.session_state.real_time_mode:
                st.info(f"Last refreshed: {time_str}\nAuto-refreshing every 60 seconds")
            else:
                st.info(f"Last refreshed: {time_str}")

        st.markdown("---")

        # Filter options
        st.markdown("### 📊 Data Filters")

        # Sentiment filter (will be enabled after data is loaded)
        if st.session_state.data is not None:
            sentiment_filter = st.multiselect(
                "Filter by Sentiment",
                options=['All', 'Positive', 'Negative', 'Neutral'],
                default=['All'],
                help="Filter stories by sentiment"
            )
        else:
            st.info("Load data first to enable filters")

        st.markdown("---")

        # Export options
        st.markdown("### 💾 Export Options")
        export_format = st.selectbox(
            "Export Format",
            options=["CSV", "JSON", "None"],
            help="Choose format to export current data"
        )

        if export_format != "None" and st.session_state.data is not None:
            if st.button("Export Data", use_container_width=True):
                export_data(export_format)

        st.markdown("---")

        # Help section
        with st.expander("ℹ️ Help & Info"):
            st.markdown(f"""
            **Sentiment Analysis:**
            {HELP_TEXT['sentiment']}

            **Topic Modeling:**
            {HELP_TEXT['topics']}

            **Metrics:**
            {HELP_TEXT['metrics']}
            """)

def create_metrics_row(df):
    """Create metrics display row"""
    if df is None or df.empty:
        return

    # Calculate metrics
    avg_sentiment = df['sentiment_score'].mean()
    total_comments = df['descendants'].sum()

    # Get top topic
    if 'topic_keyword' in df.columns:
        top_topic = df[df['topic_keyword'] != 'Outlier/No Topic']['topic_keyword'].mode().iloc[0] if len(df[df['topic_keyword'] != 'Outlier/No Topic']) > 0 else "No topics"
    else:
        top_topic = "No topics"

    # Determine sentiment color
    if avg_sentiment > 0.05:
        sentiment_color = "normal"
        sentiment_icon = "😊"
    elif avg_sentiment < -0.05:
        sentiment_color = "inverse"
        sentiment_icon = "😟"
    else:
        sentiment_color = "off"
        sentiment_icon = "😐"

    # Display metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="🎭 Overall Vibe",
            value=f"{avg_sentiment:.3f}",
            delta=f"{sentiment_icon} {'Positive' if avg_sentiment > 0.05 else 'Negative' if avg_sentiment < -0.05 else 'Neutral'}",
            delta_color=sentiment_color
        )

    with col2:
        st.metric(
            label="💬 Total Comments",
            value=f"{total_comments:,}",
            delta="Engagement"
        )

    with col3:
        st.metric(
            label="🔥 Top Trend",
            value=top_topic.replace("_", " ").title(),
            delta="Trending Now"
        )

def create_charts_row(df):
    """Create charts display row"""
    if df is None or df.empty:
        return

    col1, col2 = st.columns(2)

    with col1:
        # Sentiment vs Score scatter plot
        fig_sentiment = px.scatter(
            df.head(50),  # Limit to top 50 for performance
            x='time',
            y='score',
            color='sentiment_label',
            color_discrete_map=SENTIMENT_COLORS,
            hover_data=['title'],
            title="📈 Story Impact Over Time",
            labels={
                'time': 'Publication Time',
                'score': 'Score',
                'sentiment_label': 'Sentiment'
            }
        )

        fig_sentiment.update_layout(
            height=CHART_CONFIG['height'],
            template=CHART_CONFIG['theme'],
            showlegend=True
        )

        st.plotly_chart(fig_sentiment, use_container_width=True)

    with col2:
        # Topic distribution bar chart
        if 'topic_keyword' in df.columns:
            # Exclude outliers and get top topics
            topic_data = df[df['topic_keyword'] != 'Outlier/No Topic']['topic_keyword'].value_counts().head(7)

            fig_topics = px.bar(
                x=topic_data.values,
                y=topic_data.index,
                orientation='h',
                title="📊 Trending Topics",
                labels={
                    'x': 'Number of Stories',
                    'y': 'Topic',
                    'topic_keyword': 'Topic Keywords'
                }
            )

            fig_topics.update_layout(
                height=CHART_CONFIG['height'],
                template=CHART_CONFIG['theme'],
                yaxis={'categoryorder': 'total ascending'}
            )

            # Custom colors
            fig_topics.update_traces(
                marker_color=[COLORS['primary']] * len(topic_data),
                marker_line_color=COLORS['text'],
                marker_line_width=1
            )

            st.plotly_chart(fig_topics, use_container_width=True)
        else:
            st.info("Topic data not available. Run topic analysis first.")

def create_data_table(df):
    """Create expandable data table"""
    if df is None or df.empty:
        return

    with st.expander("📋 View Raw Data", expanded=False):
        # Select display columns
        display_columns = ['title', 'score', 'sentiment_label', 'topic_keyword', 'time']
        available_columns = [col for col in display_columns if col in df.columns]

        if available_columns:
            # Format the data for display
            display_df = df[available_columns].copy()

            # Format time for better readability
            if 'time' in display_df.columns:
                display_df['time'] = display_df['time'].dt.strftime('%Y-%m-%d %H:%M')

            # Create clickable titles
            if 'url' in df.columns:
                display_df['title'] = df.apply(
                    lambda row: f"[{row['title']}]({row['url']})" if pd.notna(row['url']) else row['title'],
                    axis=1
                )

            st.dataframe(display_df, use_container_width=True)
        else:
            st.dataframe(df)

def refresh_data():
    """Refresh data from Hacker News"""
    try:
        # Show loading spinner
        with st.spinner(LOADING_MESSAGES['fetching']):
            # Fetch data
            df = fetch_hn_data(limit=st.session_state.stories_count)

            if df.empty:
                st.error(ERROR_MESSAGES['no_data'])
                return

        with st.spinner(LOADING_MESSAGES['analyzing']):
            # Analyze sentiment
            df = analyze_sentiment(df)

            # Extract topics
            df = get_topics(df)

        # Store in session state
        st.session_state.data = df
        st.session_state.last_refresh = datetime.now()
        st.session_state.last_update_time = datetime.now()

        # Show success message
        st.success(SUCCESS_MESSAGES['data_loaded'])

    except Exception as e:
        st.error(f"Error refreshing data: {str(e)}")

def export_data(format_type):
    """Export data in selected format"""
    if st.session_state.data is None:
        st.warning("No data to export")
        return

    try:
        df = st.session_state.data

        if format_type == "CSV":
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"tech_pulse_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        elif format_type == "JSON":
            json_data = df.to_json(orient='records', date_format='iso')
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"tech_pulse_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        st.success(SUCCESS_MESSAGES['export_successful'])

    except Exception as e:
        st.error(f"Error exporting data: {str(e)}")

def check_auto_refresh():
    """Check if real-time auto-refresh should trigger"""
    if not st.session_state.real_time_mode:
        return

    if st.session_state.last_update_time:
        time_since_refresh = (datetime.now() - st.session_state.last_update_time).seconds
        # Refresh every 60 seconds in real-time mode
        if time_since_refresh >= 60:
            refresh_data()

def display_timestamp():
    """Display current timestamp in top-right corner"""
    current_time = datetime.now().strftime('%H:%M:%S')
    st.caption(f"Last Update: {current_time}")

def create_realtime_display():
    """Create the main content area for real-time display"""
    # Initialize placeholder for dynamic content updates
    placeholder = st.empty()

    return placeholder

def create_content_in_placeholder(placeholder):
    """Create and render all content within the placeholder container"""
    with placeholder.container():
        # Display timestamp in top-right
        col1, col2, col3 = st.columns([1, 2, 1])
        with col3:
            display_timestamp()

        # Main content area
        if st.session_state.data is not None:
            # Create metrics row
            create_metrics_row(st.session_state.data)

            st.markdown("---")

            # Create charts row
            create_charts_row(st.session_state.data)

            st.markdown("---")

            # Create data table
            create_data_table(st.session_state.data)
        else:
            # Initial load prompt
            st.markdown("""
            <div style='text-align: center; padding: 3rem;'>
                <h2 style='color: #7F8C8D;'>Welcome to Tech-Pulse Dashboard! 🚀</h2>
                <p style='color: #BDC3C7; font-size: 1.1rem; margin: 1rem 0;'>
                    Loading real-time data from Hacker News...
                </p>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main dashboard function"""
    # Initialize session state
    initialize_session_state()

    # Create header
    create_header()

    # Create sidebar
    create_sidebar()

    # Check if real-time mode is enabled
    if st.session_state.real_time_mode:
        # Create placeholder for real-time content
        placeholder = create_realtime_display()

        # Real-time loop
        while True:
            try:
                # Check if we need to refresh data
                if st.session_state.last_update_time is None:
                    # First time load
                    refresh_data()
                else:
                    # Check if 60 seconds have passed
                    time_since_refresh = (datetime.now() - st.session_state.last_update_time).seconds
                    if time_since_refresh >= REAL_TIME_SETTINGS['refresh_interval']:
                        refresh_data()

                # Update content display
                create_content_in_placeholder(placeholder)

                # Sleep for 60 seconds before next update
                time.sleep(REAL_TIME_SETTINGS['refresh_interval'])

            except Exception as e:
                # Handle exceptions gracefully without breaking the loop
                st.error(f"Real-time update error: {str(e)}")
                # Continue the loop after a short delay
                time.sleep(REAL_TIME_SETTINGS['retry_delay'])
    else:
        # Manual mode - display timestamp and content once
        # Display timestamp in top-right
        col1, col2, col3 = st.columns([1, 2, 1])
        with col3:
            display_timestamp()

        # Check auto-refresh
        check_auto_refresh()

        # Main content area
        if st.session_state.data is not None:
            # Create metrics row
            create_metrics_row(st.session_state.data)

            st.markdown("---")

            # Create charts row
            create_charts_row(st.session_state.data)

            st.markdown("---")

            # Create data table
            create_data_table(st.session_state.data)

        else:
            # Initial load prompt
            st.markdown("""
            <div style='text-align: center; padding: 3rem;'>
                <h2 style='color: #7F8C8D;'>Welcome to Tech-Pulse Dashboard! 🚀</h2>
                <p style='color: #BDC3C7; font-size: 1.1rem; margin: 1rem 0;'>
                    Click the "🔄 Refresh Data" button in the sidebar to start analyzing trending tech stories from Hacker News.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Auto-refresh on first load
            if st.session_state.last_refresh is None:
                refresh_data()

if __name__ == "__main__":
    main()