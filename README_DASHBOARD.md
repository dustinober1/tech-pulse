# Tech-Pulse Dashboard Guide

## 📋 Overview

The Tech-Pulse Dashboard is an interactive Streamlit application that provides real-time analysis of trending tech stories from Hacker News. It combines data fetching, sentiment analysis, and topic modeling into a user-friendly web interface.

## 🚀 Quick Start

### Prerequisites

- Python 3.13 or higher
- Virtual environment activated
- Phase 1 and Phase 2 dependencies installed

### Installation

1. **Install dashboard dependencies:**
   ```bash
   pip install -r requirements-dashboard.txt
   ```

2. **Run the dashboard:**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser:**
   Navigate to `http://localhost:8501`

## 🎛️ Dashboard Features

### Main Components

#### 1. Control Panel (Sidebar)
- **Stories Slider**: Select number of stories to analyze (10-100)
- **Refresh Button**: Manually fetch fresh data from Hacker News
- **Auto-Refresh**: Enable automatic data refresh every 5 minutes
- **Data Filters**: Filter stories by sentiment after data is loaded
- **Export Options**: Download data in CSV or JSON format
- **Help Section**: Expandable help information

#### 2. Metrics Row
- **Overall Vibe**: Average sentiment score with emoji indicator
- **Total Comments**: Sum of all story comments (engagement metric)
- **Top Trend**: Most frequently occurring topic keyword

#### 3. Visualizations
- **Story Impact Over Time**: Scatter plot showing sentiment-colored stories over time
- **Trending Topics**: Bar chart of top 7 topics by story count

#### 4. Data Table
- Expandable raw data view with all story details
- Clickable titles that link to original articles

## 📊 Understanding the Dashboard

### Sentiment Analysis
- **Positive** 😊 (Score > 0.05): Generally optimistic or favorable tone
- **Negative** 😟 (Score < -0.05): Generally pessimistic or critical tone
- **Neutral** 😐 (-0.05 ≤ Score ≤ 0.05): Objective or balanced tone

### Topic Modeling
- Stories are automatically grouped by similar themes
- Keywords are extracted and combined (e.g., "AI_Technology")
- Outliers don't fit any specific topic

### Metrics Explained
- **Overall Vibe**: Compound sentiment score across all stories
- **Total Comments**: Measure of community engagement
- **Top Trend**: Most popular topic keyword

## 🔧 Advanced Features

### Auto-Refresh
- When enabled, automatically refreshes data every 5 minutes
- Maintains current view settings
- Updates all metrics and visualizations

### Export Functionality
- **CSV**: Tabular data format for spreadsheet applications
- **JSON**: Structured data format for programming use
- Files are timestamped for easy organization

### Interactive Charts
- **Hover**: Hover over data points to see detailed information
- **Zoom**: Use mouse scroll to zoom into time ranges
- **Filter**: Click legend items to show/hide categories

## 🎨 Customization

### Color Scheme
The dashboard uses a carefully selected color palette:
- **Primary**: Coral red (#FF6B6B)
- **Secondary**: Turquoise (#4ECDC4)
- **Accent**: Sky blue (#45B7D1)
- **Positive**: Green (#2ECC71)
- **Negative**: Red (#E74C3C)
- **Neutral**: Gray (#95A5A6)

### Chart Configuration
- All charts use consistent styling
- Responsive design adapts to screen size
- High contrast for accessibility

## 🐛 Troubleshooting

### Common Issues

#### Dashboard Won't Load
```bash
# Check dependencies
pip install streamlit plotly

# Clear cache
streamlit cache clear

# Restart with specific port
streamlit run app.py --server.port 8502
```

#### Data Loading Errors
- **No Internet Connection**: Check your network connectivity
- **API Rate Limit**: Wait a few minutes before refreshing
- **Memory Issues**: Reduce the number of stories loaded

#### Display Issues
- **Blank Charts**: Try refreshing the page
- **Weird Formatting**: Clear browser cache and reload
- **Slow Loading**: Reduce story count or disable auto-refresh

### Error Messages

#### "No data available"
- Click "Refresh Data" to fetch current stories
- Check internet connection
- Try again in a few minutes

#### "Unable to fetch data from Hacker News"
- Hacker News API might be temporarily down
- Check your internet connection
- Try again later

#### "Error analyzing data"
- Temporary processing error
- Refresh data to retry
- Check system resources

## 📱 Mobile Support

The dashboard is fully responsive and works on:
- **Desktop**: Full functionality with all features
- **Tablet**: Optimized layout with touch support
- **Mobile**: Compact layout with essential features

### Mobile Navigation
- Swipe to access sidebar
- Tap charts to expand
- Long press for additional options

## 🔒 Privacy and Security

### Data Handling
- No personal data is stored
- All processing happens locally
- No data is sent to external servers

### Browser Security
- HTTPS support when available
- Secure connections to Hacker News API
- No third-party tracking scripts

## 🚀 Performance Tips

### For Better Performance
1. **Use appropriate story count**: 30-50 stories is optimal
2. **Disable auto-refresh** if not needed
3. **Close unused browser tabs**
4. **Use modern browsers** (Chrome, Firefox, Safari)

### System Requirements
- **Minimum**: 4GB RAM, dual-core processor
- **Recommended**: 8GB RAM, quad-core processor
- **Network**: Stable internet connection for data fetching

## 📚 API Reference

### Key Functions

#### `fetch_hn_data(limit=30)`
Fetches top stories from Hacker News API.

#### `analyze_sentiment(df)`
Analyzes sentiment using VADER sentiment analyzer.

#### `get_topics(df)`
Extracts topics using BERTopic clustering.

### Configuration

#### `dashboard_config.py`
Contains all dashboard settings:
- Color schemes
- Default values
- Error messages
- Help text

## 🔄 Updates and Maintenance

### Version History
- **v1.0**: Initial dashboard implementation
- **v1.1**: Added export functionality
- **v1.2**: Improved mobile support
- **v1.3**: Enhanced error handling

### Regular Updates
- Dashboard code updates automatically with git pulls
- Dependencies updated via pip
- Configuration changes require restart

## 🤝 Contributing

### Development Setup
1. Clone repository
2. Create virtual environment
3. Install all dependencies
4. Run development server:
   ```bash
   streamlit run app.py --server.developmentMode true
   ```

### Adding Features
1. Create feature branch
2. Add tests for new functionality
3. Update documentation
4. Submit pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add comprehensive docstrings
- Maintain test coverage >90%

## 📞 Support

### Getting Help
- **Documentation**: Check this README and inline help
- **Issues**: Report bugs on GitHub Issues
- **Community**: Join discussions in GitHub Discussions

### Contact Information
- **Maintainer**: Tech-Pulse Team
- **Repository**: https://github.com/dustinober1/tech-pulse
- **Documentation**: https://github.com/dustinober1/tech-pulse/wiki

## 🎓 Learning Resources

### Streamlit Documentation
- [Official Docs](https://docs.streamlit.io/)
- [Gallery Examples](https://streamlit.io/gallery)
- [Community Forums](https://discuss.streamlit.io/)

### Data Visualization
- [Plotly Documentation](https://plotly.com/python/)
- [Data Visualization Best Practices](https://www.data-to-viz.com/)

### Machine Learning
- [BERTopic Documentation](https://maartengr.github.io/BERTopic/)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)

---

**Tech-Pulse Dashboard** - Your window into the world of tech trends! 🚀