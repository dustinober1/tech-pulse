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
    "refresh": "Click 'Refresh Data' to fetch the latest stories from Hacker News and update all analyses."
}

# Error messages
ERROR_MESSAGES = {
    "no_data": "No data available. Please refresh to fetch latest stories.",
    "api_error": "Unable to fetch data from Hacker News. Please try again later.",
    "analysis_error": "Error analyzing data. Please refresh and try again.",
    "connection_error": "Connection error. Please check your internet connection."
}

# Success messages
SUCCESS_MESSAGES = {
    "data_loaded": "Data successfully loaded and analyzed!",
    "refresh_complete": "Data refresh complete!",
    "export_successful": "Data exported successfully!"
}

# Loading messages
LOADING_MESSAGES = {
    "fetching": "Fetching latest stories from Hacker News...",
    "analyzing": "Analyzing sentiment and topics...",
    "loading": "Loading dashboard...",
    "refreshing": "Refreshing data..."
}