"""
Dashboard configuration and constants for Tech-Pulse Streamlit application.
"""

# Page configuration
PAGE_CONFIG = {
    "page_title": "Tech-Pulse Dashboard",
    "page_icon": "⚡",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Color scheme
COLORS = {
    "primary": "#FF6B6B",
    "secondary": "#4ECDC4",
    "accent": "#45B7D1",
    "positive": "#2ECC71",
    "negative": "#E74C3C",
    "neutral": "#95A5A6",
    "background": "#F8F9FA",
    "surface": "#FFFFFF",
    "text": "#2C3E50"
}

# Sentiment colors
SENTIMENT_COLORS = {
    "Positive": COLORS["positive"],
    "Negative": COLORS["negative"],
    "Neutral": COLORS["neutral"]
}

# Dashboard settings
DEFAULT_SETTINGS = {
    "default_stories": 30,
    "min_stories": 10,
    "max_stories": 100,
    "refresh_interval": 300,  # 5 minutes
    "auto_refresh": False
}

# Real-time settings
REAL_TIME_SETTINGS = {
    "refresh_interval": 60,  # 60 seconds for real-time mode
    "enabled": True,
    "auto_start": False,
    "max_attempts": 3,
    "timeout": 30,
    "retry_delay": 5,
    "visual_indicator": {
        "enabled": True,
        "color": "#45B7D1",  # Accent color for real-time status
        "blink_interval": 1000,  # milliseconds
        "show_tooltip": True
    },
    "toggle_settings": {
        "enable_notifications": True,
        "enable_sound_alerts": False,
        "enable_progress_bar": True,
        "show_last_update": True,
        "enable_error_recovery": True
    }
}

# Chart settings
CHART_CONFIG = {
    "height": 400,
    "theme": "plotly_white",
    "color_sequence": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]
}

# Export options
EXPORT_FORMATS = ["CSV", "JSON", "Excel"]

# Help text
HELP_TEXT = {
    "sentiment": "Sentiment analysis uses VADER to classify story titles as Positive, Negative, or Neutral based on emotional tone.",
    "topics": "Topic modeling uses BERTopic to group similar stories and extract key themes from Hacker News titles.",
    "metrics": "Metrics provide insights into the current tech news landscape, including overall sentiment and trending topics.",
    "refresh": "Click 'Refresh Data' to fetch the latest stories from Hacker News and update all analyses.",
    "real_time_mode": "Real-time mode automatically refreshes data every 60 seconds to provide the latest Hacker News stories and insights.",
    "real_time_enable": "Enable real-time mode to activate automatic 60-second refresh intervals and live data updates.",
    "refresh_interval": "The 60-second refresh interval means the dashboard will automatically fetch and analyze new Hacker News stories every minute to keep your insights current.",
    "troubleshooting": "If real-time updates aren't working: check your internet connection, ensure the Hacker News API is accessible, and verify that your browser supports automatic refresh."
}

# Error messages
ERROR_MESSAGES = {
    "no_data": "No data available. Please refresh to fetch latest stories.",
    "api_error": "Unable to fetch data from Hacker News. Please try again later.",
    "analysis_error": "Error analyzing data. Please refresh and try again.",
    "connection_error": "Connection error. Please check your internet connection.",
    "real_time_failure": "Real-time mode encountered an error. Switching to manual refresh mode.",
    "rate_limit_error": "API rate limit exceeded. Please wait before refreshing or reduce refresh frequency.",
    "connection_during_real_time": "Connection lost during real-time updates. Reconnecting automatically...",
    "real_time_timeout": "Real-time update timed out. Please check your connection and try again.",
    "initialization_error": "Failed to initialize real-time mode. Please refresh the page and try again.",
    "configuration_error": "Real-time configuration error. Using default settings."
}

# Success messages
SUCCESS_MESSAGES = {
    "data_loaded": "Data successfully loaded and analyzed!",
    "refresh_complete": "Data refresh complete!",
    "export_successful": "Data exported successfully!",
    "real_time_activated": "Real-time mode activated! Data will refresh automatically every 60 seconds.",
    "mode_switched": "Mode switched successfully. Updates will occur automatically now.",
    "update_completed": "Real-time update completed successfully!",
    "reconnection_successful": "Successfully reconnected to real-time updates.",
    "configuration_updated": "Real-time configuration updated successfully."
}

# Loading messages
LOADING_MESSAGES = {
    "fetching": "Fetching latest stories from Hacker News...",
    "analyzing": "Analyzing sentiment and topics...",
    "loading": "Loading dashboard...",
    "refreshing": "Refreshing data..."
}

# Semantic search settings
SEMANTIC_SEARCH_SETTINGS = {
    "model_name": "all-MiniLM-L6-v2",
    "max_results": 10,
    "similarity_threshold": 0.7,
    "batch_size": 50,
    "enable_cache": True,
    "cache_duration": 3600,
    "min_query_length": 3,
    "max_query_length": 100
}

# Semantic search messages
SEMANTIC_SEARCH_MESSAGES = {
    "initializing": "Initializing semantic search engine...",
    "searching": "Performing semantic search...",
    "no_results": "No relevant results found for your query.",
    "error": "Error occurred during semantic search. Please try again.",
    "help": "Semantic search finds stories with similar meanings to your query. "
           "Enter at least 3 characters to search. "
           "Results are ranked by semantic similarity."
}

# Error messages
ERROR_MESSAGES = {
    "no_data": "No data available. Please refresh to fetch latest stories.",
    "api_error": "Unable to fetch data from Hacker News. Please try again later.",
    "analysis_error": "Error analyzing data. Please refresh and try again.",
    "connection_error": "Connection error. Please check your internet connection.",
    "real_time_failure": "Real-time mode encountered an error. Switching to manual refresh mode.",
    "rate_limit_error": "API rate limit exceeded. Please wait before refreshing or reduce refresh frequency.",
    "connection_during_real_time": "Connection lost during real-time updates. Reconnecting automatically...",
    "real_time_timeout": "Real-time update timed out. Please check your connection and try again.",
    "initialization_error": "Failed to initialize real-time mode. Please refresh the page and try again.",
    "configuration_error": "Real-time configuration error. Using default settings.",
    "semantic_search_error": "Error during semantic search. Please try again.",
    "semantic_search_initialization_error": "Failed to initialize semantic search. Some features may be unavailable."
}

# Success messages
SUCCESS_MESSAGES = {
    "data_loaded": "Data successfully loaded and analyzed!",
    "refresh_complete": "Data refresh complete!",
    "export_successful": "Data exported successfully!",
    "real_time_activated": "Real-time mode activated! Data will refresh automatically every 60 seconds.",
    "mode_switched": "Mode switched successfully. Updates will occur automatically now.",
    "update_completed": "Real-time update completed successfully!",
    "reconnection_successful": "Successfully reconnected to real-time updates.",
    "configuration_updated": "Real-time configuration updated successfully.",
    "semantic_search_initialized": "Semantic search initialized successfully!",
    "semantic_search_performed": "Semantic search completed successfully!"
}