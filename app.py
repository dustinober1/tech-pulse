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
from data_loader import (
    fetch_hn_data, analyze_sentiment, get_topics, setup_vector_db, semantic_search,
    fetch_multi_source_data, analyze_multi_source_sentiment, extract_multi_source_topics,
    get_multi_source_trends,
    # Phase 8: PDF Generation Functions
    generate_executive_briefing, get_pdf_generation_status, validate_pdf_generation_requirements,
    estimate_pdf_generation_time
)
from dashboard_config import (
    PAGE_CONFIG, COLORS, SENTIMENT_COLORS, DEFAULT_SETTINGS,
    CHART_CONFIG, HELP_TEXT, ERROR_MESSAGES, SUCCESS_MESSAGES, LOADING_MESSAGES,
    REAL_TIME_SETTINGS, SEMANTIC_SEARCH_SETTINGS, SEMANTIC_SEARCH_MESSAGES
)
from src.phase7.predictive_analytics.dashboard import PredictiveDashboard
from src.phase7.user_management.database import UserDatabase
from src.phase7.user_management.user_profile import UserProfile
from src.phase7.user_management.recommendations import PersonalizedRecommendations
from src.phase7.user_management.ui_components import UIComponents

# Configure page
st.set_page_config(**PAGE_CONFIG)

# Load custom CSS for accessibility
def load_css():
    """Load custom CSS for better accessibility"""
    with open("assets/styles.css", "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Add error handling for st_autorefresh import
try:
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH_AVAILABLE = True
except ImportError:
    AUTO_REFRESH_AVAILABLE = False
    st.warning("Auto-refresh not available. Install streamlit-autorefresh: pip install streamlit-autorefresh")

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
    # Vector DB session state
    if 'vector_collection' not in st.session_state:
        st.session_state.vector_collection = None
    if 'vector_db_initialized' not in st.session_state:
        st.session_state.vector_db_initialized = False
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    # Predictive Analytics session state
    if 'predictive_dashboard' not in st.session_state:
        st.session_state.predictive_dashboard = PredictiveDashboard()
    # User Management session state
    if 'current_user_id' not in st.session_state:
        # Create a default user for demo purposes
        db = UserDatabase()
        st.session_state.current_user_id = db.create_user(
            username=f"demo_user_{datetime.now().strftime('%Y%m%d')}",
            email="demo@techpulse.com",
            preferences={"theme": "light", "notifications": True}
        )
    if 'user_ui_components' not in st.session_state:
        st.session_state.user_ui_components = UIComponents(st.session_state.current_user_id)
    # Remove old auto_refresh and refresh_countdown if they exist
    if 'auto_refresh' in st.session_state:
        del st.session_state.auto_refresh
    if 'refresh_countdown' in st.session_state:
        del st.session_state.refresh_countdown
    # Onboarding tracking
    if 'onboarding_completed' not in st.session_state:
        st.session_state.onboarding_completed = False

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

def show_onboarding_if_needed():
    """Show onboarding dialog for first-time users"""
    if not st.session_state.onboarding_completed:
        with st.expander("👋 Welcome to Tech-Pulse Dashboard!", expanded=True):
            st.markdown("""
            ### Getting Started

            Tech-Pulse Dashboard provides real-time analysis of tech news from multiple sources. Here's what you need to know:

            #### Key Features:
            - **📊 Real-time Analysis**: Fetch and analyze trending stories from Hacker News, Reddit, RSS feeds, and Twitter
            - **🎭 Sentiment Analysis**: Understand the emotional tone of news stories (Positive 😊, Negative 😟, or Neutral 😐)
            - **🔍 Semantic Search**: Find stories by meaning, not just keywords
            - **📈 Trend Prediction**: AI-powered predictions for trending topics
            - **📄 Executive Briefings**: Generate professional PDF reports

            #### How to Use:
            1. **Start**: Click "Load Data" to fetch the latest stories
            2. **Customize**: Use the Control Panel on the left to adjust settings
            3. **Explore**: Browse through tabs for different analyses
            4. **Search**: Use semantic search to find specific topics
            5. **Export**: Download data or generate PDF reports

            #### Accessibility:
            - Use Tab key to navigate between elements
            - Press Enter to select buttons and options
            - High contrast colors are enabled for better visibility
            - All charts include alt text for screen readers

            Need help? Check the Help tab in the sidebar!
            """)

            if st.button("✅ Got it!", key="onboarding_complete"):
                st.session_state.onboarding_completed = True
                st.rerun()

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

        # Multi-Source Options (Phase 7.3)
        st.markdown("### 🔌 Multi-Source Settings")

        # Initialize multi-source settings in session state if not exists
        if 'multi_source_enabled' not in st.session_state:
            st.session_state.multi_source_enabled = False
        if 'include_reddit' not in st.session_state:
            st.session_state.include_reddit = True
        if 'include_rss' not in st.session_state:
            st.session_state.include_rss = True
        if 'include_twitter' not in st.session_state:
            st.session_state.include_twitter = True

        # Multi-source toggle
        multi_source_enabled = st.checkbox(
            "🌐 Enable Multi-Source",
            value=st.session_state.multi_source_enabled,
            help="Fetch data from multiple sources (Reddit, RSS feeds, Twitter) in addition to Hacker News"
        )
        st.session_state.multi_source_enabled = multi_source_enabled

        # Source selection (only show if multi-source is enabled)
        if multi_source_enabled:
            st.markdown("**Select Sources:**")
            include_reddit = st.checkbox("Reddit", value=st.session_state.include_reddit, help="Include posts from Reddit")
            include_rss = st.checkbox("RSS Feeds", value=st.session_state.include_rss, help="Include tech news from RSS feeds")
            include_twitter = st.checkbox("Twitter", value=st.session_state.include_twitter, help="Include tweets from tech influencers")

            # Update session state
            st.session_state.include_reddit = include_reddit
            st.session_state.include_rss = include_rss
            st.session_state.include_twitter = include_twitter
        else:
            # Update session state to false when multi-source is disabled
            st.session_state.include_reddit = False
            st.session_state.include_rss = False
            st.session_state.include_twitter = False

        st.markdown("---")

        # Refresh button
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            refresh_data()

        # Real-time mode toggle
        real_time_mode = st.checkbox(
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

        # Executive Briefing Section (Phase 8)
        st.markdown("### 📄 Executive Briefing")

        # Check PDF generation status
        pdf_status = get_pdf_generation_status()

        # Show status indicator
        if pdf_status['status'] == 'ready':
            st.success("✅ PDF generation ready")
        else:
            st.warning(f"⚠️ PDF generation incomplete ({pdf_status['missing_count']} missing components)")
            with st.expander("📋 Missing Components"):
                for missing in pdf_status['validation']['missing_modules']:
                    st.error(f"• {missing}")
                for rec in pdf_status['validation']['recommendations']:
                    st.info(f"💡 {rec}")

        # Add OpenAI API key input (optional)
        openai_key = st.text_input(
            "OpenAI API Key (Optional)",
            type="password",
            help="Enter your OpenAI API key to enable AI-powered summaries. Leave blank for rule-based summaries."
        )

        # Add briefing options
        col1, col2 = st.columns(2)
        with col1:
            pdf_stories = st.number_input(
                "Stories",
                min_value=10,
                max_value=100,
                value=30,
                step=5,
                help="Number of stories to include in the briefing"
            )

        with col2:
            include_charts = st.checkbox(
                "Include Charts",
                value=True,
                help="Include visual charts in the PDF"
            )

        # Show time estimation
        time_est = estimate_pdf_generation_time(pdf_stories, include_charts)
        st.info(f"⏱️ Estimated generation time: {time_est['estimated_seconds']}s")

        # Generate and download button
        if st.button("📥 Generate Briefing", help="Generate and download executive briefing PDF"):
            try:
                with st.spinner("🔄 Generating executive briefing..."):
                    # Show progress
                    progress_bar = st.progress(0)
                    progress_bar.progress(25)

                    # Generate PDF
                    pdf_bytes = generate_executive_briefing(
                        stories_count=int(pdf_stories),
                        include_charts=include_charts,
                        openai_api_key=openai_key if openai_key else None
                    )

                    progress_bar.progress(75)

                    # Create download button
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.sidebar.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"tech_pulse_briefing_{timestamp}.pdf",
                        mime="application/pdf",
                        type="primary"
                    )

                    progress_bar.progress(100)

                st.success("✅ Executive briefing generated successfully!")

                # Show file size
                file_size_mb = len(pdf_bytes) / (1024 * 1024)
                st.info(f"📊 File size: {file_size_mb:.2f} MB")

            except ImportError as e:
                st.error(f"❌ PDF generation modules not available: {str(e)}")
                st.info("💡 Run: pip install -r requirements.txt to install required dependencies")
            except RuntimeError as e:
                st.error(f"❌ Failed to generate briefing: {str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

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

            **Semantic Search:**
            {HELP_TEXT['semantic_search']}
            """)

        st.markdown("---")

        # User Profile Section
        st.session_state.user_ui_components.render_mini_profile_card()

        # Quick Actions
        quick_actions = st.session_state.user_ui_components.render_quick_actions()

def create_metrics_row(df):
    """Create metrics display row"""
    if df is None or df.empty:
        return

    # Check if this is multi-source data
    is_multi_source = 'source' in df.columns and len(df['source'].unique()) > 1

    # Calculate metrics
    if is_multi_source:
        # Multi-source metrics
        avg_sentiment = df['sentiment'].mean() if 'sentiment' in df.columns else 0
        total_comments = df['num_comments'].sum() if 'num_comments' in df.columns else 0
        total_stories = len(df)
    else:
        # Hacker News metrics (original)
        avg_sentiment = df['sentiment_score'].mean()
        total_comments = df['descendants'].sum()
        total_stories = len(df)

    # Get top topic
    if 'topic_keyword' in df.columns:
        # For multi-source, handle topics differently
        if is_multi_source:
            # Remove source prefixes from topics for cleaner display
            df_clean_topics = df.copy()
            if df_clean_topics['topic_keyword'].dtype == 'object':
                df_clean_topics['topic_clean'] = df_clean_topics['topic_keyword'].str.replace(r'^\[[^\]]+\]\s*', '', regex=True)
                valid_topics = df_clean_topics[df_clean_topics['topic_clean'] != 'Outlier/No Topic']
                if len(valid_topics) > 0:
                    top_topic = valid_topics['topic_clean'].mode().iloc[0]
                else:
                    top_topic = "No topics"
            else:
                top_topic = "No topics"
        else:
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
            delta=f"From {total_stories} stories" if is_multi_source else "Engagement"
        )

    with col3:
        st.metric(
            label="🔥 Top Trend",
            value=top_topic.replace("_", " ").title(),
            delta="Trending Now"
        )

    # Show source breakdown if multi-source
    if is_multi_source:
        st.markdown("---")
        st.markdown("### 📊 Source Breakdown")

        # Create source metrics
        source_cols = st.columns(len(df['source'].unique()))

        for i, (source, source_df) in enumerate(df.groupby('source')):
            with source_cols[i]:
                # Source-specific metrics
                source_count = len(source_df)
                source_comments = source_df['num_comments'].sum() if 'num_comments' in source_df.columns else 0
                source_sentiment = source_df['sentiment'].mean() if 'sentiment' in source_df.columns else 0

                # Determine source icon
                source_icons = {
                    'hackernews': '🍊',
                    'reddit': '🤖',
                    'rss': '📰',
                    'twitter': '🐦'
                }
                icon = source_icons.get(source, '📊')

                st.metric(
                    label=f"{icon} {source.title()}",
                    value=f"{source_count}",
                    delta=f"{source_comments} comments"
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

def create_semantic_search_section(df):
    """Create semantic search interface section"""
    if df is None or df.empty:
        return

    st.markdown("---")

    # Check if vector DB is initialized
    if not st.session_state.vector_db_initialized or st.session_state.vector_collection is None:
        # Show initialization button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("🔍 Semantic search is available. Initialize the search engine to find stories by meaning.")
        with col2:
            if st.button("🚀 Initialize Search", type="primary", use_container_width=True):
                initialize_vector_db()
        return

    # Vector DB is initialized, show search interface
    st.markdown("### 🔍 Semantic Search")
    st.markdown("Find stories by meaning, not just keywords. Enter a query to discover semantically similar content.")

    # Search input
    col1, col2 = st.columns([4, 1])
    with col1:
        search_query = st.text_input(
            "Search for stories:",
            placeholder="e.g., 'artificial intelligence', 'startup funding', 'machine learning'",
            help=SEMANTIC_SEARCH_MESSAGES['help'],
            key="semantic_search_query"
        )

    with col2:
        st.write("")  # Add spacing
        st.write("")  # Add spacing
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)

    # Perform search when button is clicked or Enter is pressed
    if search_button or (search_query and search_query != st.session_state.get('last_search_query', '')):
        if search_query and len(search_query.strip()) >= SEMANTIC_SEARCH_SETTINGS['min_query_length']:
            perform_semantic_search(search_query.strip())
            st.session_state.last_search_query = search_query
        elif search_query:
            st.warning(f"Query must be at least {SEMANTIC_SEARCH_SETTINGS['min_query_length']} characters long.")

    # Display search results if they exist
    if st.session_state.search_results:
        display_search_results(st.session_state.search_results)

def display_search_results(results):
    """Display search results in expandable sections"""
    if not results:
        st.info(SEMANTIC_SEARCH_MESSAGES['no_results'])
        return

    st.markdown(f"#### Found {len(results)} relevant stories")

    for i, result in enumerate(results, 1):
        with st.expander(f"{i}. {result['title'][:80]}{'...' if len(result['title']) > 80 else ''}", expanded=i <= 3):
            # Main content columns
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                # Title and URL
                title = result['title']
                url = result['metadata'].get('url', '')

                if url:
                    st.markdown(f"**[{title}]({url})**")
                else:
                    st.markdown(f"**{title}**")

                # Metadata info
                metadata = result['metadata']
                st.caption(f"Score: {metadata.get('score', 0)} | "
                          f"Comments: {metadata.get('descendants', 0)} | "
                          f"Sentiment: {metadata.get('sentiment_label', 'N/A')} | "
                          f"Topic: {metadata.get('topic_keyword', 'N/A').replace('_', ' ').title()}")

            with col2:
                # Similarity score
                similarity_pct = result['similarity_score'] * 100
                st.metric("Similarity", f"{similarity_pct:.1f}%")

            with col3:
                # Ranking
                st.metric("Rank", f"#{result['rank']}")

            # Additional details
            with st.expander("📊 Search Details", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Similarity Score:** {result['similarity_score']:.3f}")
                    st.write(f"**Distance:** {result['distance']:.3f}")
                    if 'time' in metadata:
                        time_str = pd.to_datetime(metadata['time']).strftime('%Y-%m-%d %H:%M')
                        st.write(f"**Posted:** {time_str}")

                with col2:
                    st.write(f"**Explanation:** {result['explanation']}")
                    st.write(f"**Collection ID:** {result['metadata'].get('index', 'N/A')}")

def initialize_vector_db():
    """Initialize vector database for semantic search"""
    if st.session_state.data is None or st.session_state.data.empty:
        st.error("No data available for search initialization. Please refresh data first.")
        return

    try:
        with st.spinner(SEMANTIC_SEARCH_MESSAGES['initializing']):
            # Set up vector database
            collection = setup_vector_db(st.session_state.data)

            if collection is not None:
                st.session_state.vector_collection = collection
                st.session_state.vector_db_initialized = True
                st.success(SUCCESS_MESSAGES['semantic_search_initialized'])
                st.rerun()
            else:
                st.error(ERROR_MESSAGES['semantic_search_initialization_error'])

    except Exception as e:
        st.error(f"Error initializing semantic search: {str(e)}")

def perform_semantic_search(query):
    """Perform semantic search and store results"""
    try:
        with st.spinner(SEMANTIC_SEARCH_MESSAGES['searching']):
            # Perform search using the vector collection
            results = semantic_search(
                collection=st.session_state.vector_collection,
                query=query,
                max_results=SEMANTIC_SEARCH_SETTINGS['max_results'],
                similarity_threshold=SEMANTIC_SEARCH_SETTINGS['similarity_threshold']
            )

            # Store results in session state
            st.session_state.search_results = results

            if results:
                st.success(f"Found {len(results)} relevant stories for '{query}'")
            else:
                st.info(f"No results found for '{query}'. Try different keywords.")

    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        st.session_state.search_results = None

def create_data_table(df):
    """Create expandable data table"""
    if df is None or df.empty:
        return

    with st.expander("📋 View Raw Data", expanded=False):
        # Check if multi-source data
        is_multi_source = 'source' in df.columns

        # Select display columns based on data type
        if is_multi_source:
            display_columns = ['title', 'source', 'score', 'sentiment_label', 'topic_keyword', 'published']
            # Map column names for consistency
            df_display = df.copy()
            if 'published' in df_display.columns:
                df_display['time'] = df_display['published']
            if 'num_comments' in df_display.columns:
                df_display['comments'] = df_display['num_comments']
        else:
            display_columns = ['title', 'score', 'sentiment_label', 'topic_keyword', 'time']
            df_display = df.copy()

        available_columns = [col for col in display_columns if col in df_display.columns]

        if available_columns:
            # Format the data for display
            display_df = df_display[available_columns].copy()

            # Format time for better readability
            time_col = 'published' if 'published' in display_df.columns else 'time'
            if time_col in display_df.columns:
                display_df[time_col] = display_df[time_col].dt.strftime('%Y-%m-%d %H:%M')

            # Add source badge if multi-source
            if is_multi_source and 'source' in display_df.columns:
                source_icons = {
                    'hackernews': '🍊',
                    'reddit': '🤖',
                    'rss': '📰',
                    'twitter': '🐦'
                }
                display_df['source'] = display_df['source'].apply(
                    lambda x: f"{source_icons.get(x, '📊')} {x.title()}"
                )

            # Create clickable titles
            if 'url' in df_display.columns:
                display_df['title'] = df_display.apply(
                    lambda row: f"[{row['title']}]({row['url']})" if pd.notna(row['url']) else row['title'],
                    axis=1
                )

            # Add comments if available
            if 'num_comments' in df_display.columns:
                display_df['comments'] = df_display['num_comments']
                if 'comments' not in available_columns and len(display_columns) < 7:
                    display_df = display_df[['title', 'source', 'score', 'comments', 'sentiment_label', 'topic_keyword', time_col]] if is_multi_source else display_df

            st.dataframe(display_df, use_container_width=True)
        else:
            st.dataframe(df_display)

def refresh_data():
    """Refresh data from Hacker News and optional multi-sources"""
    try:
        # Show loading spinner
        with st.spinner(LOADING_MESSAGES['fetching']):
            # Check if multi-source is enabled
            if st.session_state.multi_source_enabled:
                # Fetch from multiple sources
                import asyncio

                # Create a simple event loop for async operation
                try:
                    # Try to get existing loop
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is running, create new one in thread
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run,
                                fetch_multi_source_data(
                                    hn_limit=st.session_state.stories_count,
                                    include_reddit=st.session_state.include_reddit,
                                    include_rss=st.session_state.include_rss,
                                    include_twitter=st.session_state.include_twitter,
                                    use_cache=True,
                                    cache_duration_hours=1
                                )
                            )
                            df = future.result()
                    else:
                        # Use existing loop
                        df = asyncio.run(
                            fetch_multi_source_data(
                                hn_limit=st.session_state.stories_count,
                                include_reddit=st.session_state.include_reddit,
                                include_rss=st.session_state.include_rss,
                                include_twitter=st.session_state.include_twitter,
                                use_cache=True,
                                cache_duration_hours=1
                            )
                        )
                except Exception as async_error:
                    print(f"Async error, falling back to single source: {async_error}")
                    # Fallback to just Hacker News
                    df = fetch_hn_data(limit=st.session_state.stories_count)
            else:
                # Fetch only Hacker News data
                df = fetch_hn_data(limit=st.session_state.stories_count)

            if df.empty:
                st.error(ERROR_MESSAGES['no_data'])
                return

        # Skip analysis if multi-source data is already analyzed
        if not (st.session_state.multi_source_enabled and 'sentiment' in df.columns):
            with st.spinner(LOADING_MESSAGES['analyzing']):
                if st.session_state.multi_source_enabled and 'source' in df.columns:
                    # Multi-source data already analyzed
                    pass
                else:
                    # Analyze sentiment for Hacker News data
                    df = analyze_sentiment(df)

                    # Extract topics
                    df = get_topics(df)

        # Store in session state
        st.session_state.data = df
        st.session_state.last_refresh = datetime.now()
        st.session_state.last_update_time = datetime.now()

        # Clear vector DB state to force re-initialization if needed
        # Only clear if not in real-time mode to preserve performance
        if not st.session_state.real_time_mode:
            st.session_state.vector_db_initialized = False
            st.session_state.vector_collection = None
            st.session_state.search_results = None

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

def create_executive_briefing_section(df):
    """Create executive briefing preview section"""
    st.markdown("---")

    # Executive Briefing Preview
    with st.container():
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📊 Executive Briefing")
            st.write("Generate a professional PDF report containing:")

            briefing_features = [
                "🔍 AI-powered trend analysis",
                "📈 Sentiment distribution charts",
                "🏆 Top stories by engagement",
                "🎯 Topic coverage analysis",
                "⚡ Key metrics and insights",
                "📅 Timestamped analysis"
            ]

            for feature in briefing_features:
                st.markdown(f"• {feature}")

        with col2:
            # Check PDF generation status
            pdf_status = get_pdf_generation_status()

            if pdf_status['status'] == 'ready':
                st.success("✅ Ready to generate")
            else:
                st.warning(f"⚠️ Setup incomplete")

            # Quick generate button
            if st.button("⚡ Quick Generate", key="quick_pdf", help="Generate PDF with default settings"):
                try:
                    with st.spinner("Preparing your briefing..."):
                        pdf_bytes = generate_executive_briefing(
                            stories_count=30,
                            include_charts=True,
                            openai_api_key=None  # Use rule-based by default
                        )

                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        st.download_button(
                            label="📥 Download Now",
                            data=pdf_bytes,
                            file_name=f"tech_pulse_briefing_{timestamp}.pdf",
                            mime="application/pdf",
                            key="quick_download"
                        )

                    st.success("✅ Briefing ready for download!")
                    # Show file size
                    file_size_mb = len(pdf_bytes) / (1024 * 1024)
                    st.info(f"📊 File size: {file_size_mb:.2f} MB")

                except ImportError as e:
                    st.error("❌ PDF generation not available")
                    st.info("💡 Install dependencies: pip install -r requirements.txt")
                except Exception as e:
                    st.error(f"❌ Generation failed: {str(e)}")

            # Show features based on availability
            if pdf_status['status'] == 'incomplete':
                with st.expander("📋 Setup Requirements"):
                    for missing in pdf_status['validation']['missing_modules']:
                        st.error(f"• {missing}")
                    for rec in pdf_status['validation']['recommendations']:
                        st.info(f"💡 {rec}")

            # Show available features
            if pdf_status['status'] == 'ready':
                st.info("📋 Available features:")
                if pdf_status['can_generate_basic']:
                    st.success("✅ Basic PDF generation")
                if pdf_status['can_generate_charts']:
                    st.success("✅ Chart inclusion")
                if pdf_status['has_ai_summaries']:
                    st.success("✅ AI summaries (with API key)")

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

            # Executive Briefing Preview Section (Phase 8)
            create_executive_briefing_section(st.session_state.data)

            # Create semantic search section after metrics
            create_semantic_search_section(st.session_state.data)

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
    # Load custom CSS for accessibility
    load_css()

    # Initialize session state
    initialize_session_state()

    # Show onboarding for first-time users
    show_onboarding_if_needed()

    # Create header
    create_header()

    # Create sidebar
    create_sidebar()

    # Create tabs for different features
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 News Feed",
        "🔍 Smart Search",
        "📈 Trend Forecast",
        "🚨 Breaking News",
        "🤖 AI Insights",
        "👤 My Dashboard"
    ])

    with tab1:
        # Original news analysis tab
        render_news_analysis_tab()

    with tab2:
        # Semantic search tab
        render_semantic_search_tab()

    with tab3:
        # Predictive analytics tab
        render_prediction_tab()

    with tab4:
        # Anomaly detection tab
        render_anomaly_tab()

    with tab5:
        # Model training tab
        render_model_training_tab()

    with tab6:
        # User Profile tab
        render_user_profile_tab()

def render_news_analysis_tab():
    """Render the news analysis tab"""
    # Check if real-time mode is enabled
    if st.session_state.real_time_mode and AUTO_REFRESH_AVAILABLE:
        # Use st_autorefresh for non-blocking auto-refresh
        # Auto-refresh every 60 seconds when real-time mode is on
        count = st_autorefresh(
            interval=REAL_TIME_SETTINGS['refresh_interval'] * 1000,  # Convert to milliseconds
            limit=None,  # No limit on refreshes
            key="realtime_refresh"
        )

        # Check if we need to refresh data
        if st.session_state.last_update_time is None or count > 0:
            try:
                with st.spinner("Updating data..."):
                    refresh_data()
                    st.success(f"Data refreshed at {datetime.now().strftime('%H:%M:%S')}")
            except Exception as e:
                st.error(f"Real-time update error: {str(e)}")
                # Fall back to manual mode on error
                st.session_state.real_time_mode = False
    else:
        # Manual mode - display timestamp and content once
        # Display timestamp in top-right
        col1, col2, col3 = st.columns([1, 2, 1])
        with col3:
            display_timestamp()

        # Check auto-refresh
        check_auto_refresh()

        # Always show manual refresh button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            if st.button("🔄 Refresh Now", key="manual_refresh"):
                with st.spinner("Refreshing data..."):
                    refresh_data()
                    st.success("Data refreshed successfully!")
                    st.rerun()

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

def render_semantic_search_tab():
    """Render the semantic search tab"""
    st.header("🔍 Semantic Search")

    if st.session_state.data is not None and not st.session_state.data.empty:
        create_semantic_search_section(st.session_state.data)
    else:
        st.info("Please load data from the News Analysis tab first to enable semantic search.")

def render_prediction_tab():
    """Render the trend prediction tab"""
    st.header("📈 Technology Trend Predictions")
    st.session_state.predictive_dashboard.render_prediction_tab()

def render_anomaly_tab():
    """Render the anomaly detection tab"""
    st.header("🔍 Anomaly Detection")
    st.session_state.predictive_dashboard.render_anomaly_tab()

def render_model_training_tab():
    """Render the model training tab"""
    st.header("🤖 Model Training Management")
    st.session_state.predictive_dashboard.render_model_training_tab()

def render_user_profile_tab():
    """Render the user profile tab with personalization features"""
    st.header("👤 Your Profile & Personalization")

    # Create sub-tabs for user profile features
    profile_tab1, profile_tab2, profile_tab3, profile_tab4, profile_tab5 = st.tabs([
        "👤 Profile Overview",
        "⚙️ Preferences",
        "🎯 Recommendations",
        "📊 Analytics",
        "🏷️ Topics"
    ])

    with profile_tab1:
        # Profile overview section
        profile_data = st.session_state.user_ui_components.render_user_profile_section()

        # Activity feed
        st.markdown("---")
        st.session_state.user_ui_components.render_activity_feed(limit=10)

    with profile_tab2:
        # Preferences editor
        st.session_state.user_ui_components.render_preferences_editor()

        st.markdown("---")

        # Data export/import
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.user_ui_components.render_export_data_section()
        with col2:
            st.session_state.user_ui_components.render_import_data_section()

    with profile_tab3:
        # Recommendations section
        st.session_state.user_ui_components.render_recommendations_section()

    with profile_tab4:
        # Analytics dashboard
        st.session_state.user_ui_components.render_analytics_dashboard()

    with profile_tab5:
        # Topic management
        st.session_state.user_ui_components.render_topic_management()


def semantic_search(query: str, n_results: int = 10) -> List[Dict]:
    """
    Perform semantic search on the loaded data.

    Args:
        query: Search query
        n_results: Number of results to return

    Returns:
        List of matching documents
    """
    try:
        # Check if we have loaded data
        if not st.session_state.get('data_loaded', False):
            st.warning("No data loaded for search. Please refresh data first.")
            return []

        # Check if vector search is initialized
        if 'vector_manager' not in st.session_state:
            st.error("Vector search not initialized")
            return []

        vector_manager = st.session_state.vector_manager

        # Check if collection has documents
        try:
            count = vector_manager.collection.count()
            if count == 0:
                st.warning("No documents in search index. Loading data...")
                # Load current data into vector search
                if 'stories' in st.session_state and st.session_state.stories:
                    success = vector_manager.add_documents(st.session_state.stories)
                    if not success:
                        st.error("Failed to load documents for search")
                        return []
                else:
                    st.warning("No stories available for search")
                    return []
        except Exception as e:
            st.error(f"Error checking search index: {str(e)}")
            return []

        # Perform search
        results = vector_manager.search(
            query=query,
            n_results=min(n_results, count),  # Don't request more than available
            where=None  # No filters for general search
        )

        # Format results for display
        formatted_results = []
        if results:
            for doc in results:
                formatted_doc = {
                    'id': doc.get('id', ''),
                    'text': doc.get('text', ''),
                    'distance': doc.get('distance', 0),
                    'metadata': doc.get('metadata', {})
                }
                formatted_results.append(formatted_doc)

        return formatted_results

    except Exception as e:
        st.error(f"Semantic search error: {str(e)}")
        return []


if __name__ == "__main__":
    main()