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
    "sentiment": "Sentiment analysis evaluates the emotional tone of news headlines. Positive 😊 means optimistic/good news, Negative 😟 means concerning/bad news, and Neutral 😐 means factual/balanced reporting.",
    "topics": "Topic groups similar news stories together based on their content (e.g., 'AI & Machine Learning' or 'Cybersecurity'). This helps you understand what themes are trending in tech.",
    "metrics": "Key insights about current tech news: Overall Vibe shows if news is generally positive or negative, Total Comments indicates community engagement, and Top Trend highlights the most discussed topic.",
    "refresh": "Click 'Refresh Data' to get the latest stories. This fetches new articles and updates all analyses with current information.",
    "real_time_mode": "Automatically updates data every minute to keep you informed of breaking tech news without manual refresh.",
    "real_time_enable": "Turn on automatic updates. The dashboard will refresh itself every 60 seconds to show the latest stories.",
    "refresh_interval": "Updates happen every minute to balance freshness with performance. You'll always see recent stories without overwhelming the system.",
    "troubleshooting": "If updates aren't working: 1) Check your internet connection 2) Try manually clicking Refresh Data 3) Wait a moment and retry 4) Contact support if issues persist.",
    "semantic_search": "Smart search that understands meaning. Search for 'AI regulation' and it finds stories about 'AI laws', 'tech policy', etc., not just exact word matches.",
    "vader": "VADER is a sentiment analysis tool specialized for social media and news text. It's great at understanding emotions in headlines.",
    "bertopic": "An AI tool that reads all stories and automatically groups them by topic. It identifies trends like 'Cloud Computing' or 'Mobile Tech'.",
    "embeddings": "Mathematical representations of text that capture meaning. Similar stories have similar embeddings, enabling smart search.",
    "anomaly_detection": "Finds unusual patterns in tech news that might indicate breaking stories or unexpected developments.",
    "trend_prediction": "Uses historical data and current patterns to predict which topics will trend in the near future."
}

# Error messages
ERROR_MESSAGES = {
    "no_data": "No stories available. Click 'Refresh Data' to fetch the latest tech news.",
    "api_error": "Cannot reach the news servers right now. This is usually temporary. Try again in a few minutes.",
    "analysis_error": "Unable to analyze the stories. Please refresh to try again with fresh data.",
    "connection_error": "Internet connection issue. Please check your WiFi/network and try again.",
    "real_time_failure": "Auto-update failed. Switched to manual mode - click 'Refresh Data' for updates.",
    "rate_limit_error": "Too many requests! Please wait 1 minute before refreshing again.",
    "connection_during_real_time": "Connection interrupted. Trying to reconnect... (Will switch to manual if this continues)",
    "real_time_timeout": "Update took too long. Your connection might be slow. Try manual refresh.",
    "initialization_error": "Cannot start auto-update. Please refresh the page and try again.",
    "configuration_error": "Settings issue. Using safe default values instead.",
    "pdf_error": "Cannot generate PDF report. Check if you have data loaded and try again.",
    "search_error": "Search failed. Make sure you've loaded data first, then try searching again.",
    "model_loading": "AI models are loading... This may take a moment on first use.",
    "cache_error": "Cache issue detected. Refreshing with fresh data..."
    "semantic_search_error": "Error during semantic search. Please try again.",
    "semantic_search_initialization_error": "Failed to initialize semantic search. Some features may be unavailable."
}

SUCCESS_MESSAGES = {
    "data_loaded": "Data successfully loaded and analyzed!",
    "refresh_complete": "Data refresh complete!",
    "export_successful": "Data exported successfully!",
    "real_time_activated": "Real-time mode activated! Data will refresh automatically every 60 seconds.",
    "mode_switched": "Mode switched successfully. Updates will occur automatically now.",
    "update_completed": "Real-time update completed successfully!",
    "reconnection_successful": "Successfully reconnected to real-time updates.",
    "configuration_updated": "Real-time configuration updated successfully."
    "semantic_search_initialized": "Semantic search initialized successfully!",
    "semantic_search_performed": "Semantic search completed successfully!"
}

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

