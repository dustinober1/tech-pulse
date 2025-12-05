# Tech-Pulse User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Dashboard Overview](#dashboard-overview)
4. [Features and Functionality](#features-and-functionality)
5. [Understanding the Analysis](#understanding-the-analysis)
6. [Advanced Features](#advanced-features)
7. [Tips and Best Practices](#tips-and-best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Glossary](#glossary)

## Introduction

Welcome to Tech-Pulse, your real-time dashboard for analyzing trending tech stories from Hacker News and beyond. This powerful tool transforms raw news data into actionable insights through sentiment analysis, topic modeling, and predictive analytics.

### What You Can Do With Tech-Pulse:
- Monitor trending tech stories in real-time
- Understand sentiment patterns in tech news
- Discover emerging topics and trends
- Search for stories by meaning, not just keywords
- Generate executive briefings and reports
- Track predictions for future trending stories

### Accessibility Commitment
Tech-Pulse is designed to be accessible to all users:
- **Full Keyboard Navigation**: Use Tab, Enter, and Arrow keys
- **High Contrast Colors**: Improved visibility for all users
- **Screen Reader Support**: Compatible with assistive technologies
- **Clear Error Messages**: Helpful guidance when issues occur
- **Plain Language**: Technical terms explained simply

## Getting Started

### Accessing the Dashboard

The easiest way to use Tech-Pulse is through our live deployment:
- **URL**: [https://tech-pulse.streamlit.app](https://tech-pulse.streamlit.app)
- No installation required
- Works on any modern web browser
- Mobile-responsive design

### First-Time User Experience

1. **Initial Load**: The dashboard automatically loads with the top 30 trending stories from Hacker News
2. **Welcome Message**: First-time users see a brief introduction to the features
3. **Auto-Analysis**: Stories are automatically analyzed for sentiment and topics

## Dashboard Overview

The dashboard is organized into several key sections:

### 1. Header Section
- **Tech-Pulse Dashboard**: Main title and navigation
- **Last Update Time**: Shows when data was last refreshed
- **Real-time Indicator**: Green dot when auto-refresh is active

### 2. Control Panel (Sidebar)
The left sidebar contains all controls for customizing your experience:

#### Data Controls
- **Number of Stories**: Slider to select 10-100 stories (default: 30)
- **Refresh Data**: Manual refresh button
- **Enable Real-Time Mode**: Toggle for automatic 60-second updates

#### Multi-Source Settings
- **Enable Multi-Source**: Fetch from Reddit, RSS feeds, and Twitter
- **Source Selection**: Choose which additional sources to include

#### Data Filters
- **Sentiment Filter**: Filter stories by Positive, Negative, or Neutral sentiment
- **Export Options**: Download data in CSV or JSON format

#### Executive Briefing
- **Generate Briefing**: Create PDF reports with analysis
- **OpenAI API Key**: Optional key for AI-powered summaries

### 3. Main Tabs
The dashboard content is organized into tabs:

#### 📊 Main Dashboard
- Metrics row showing key statistics
- Interactive charts and visualizations
- Expandable data table with story details

#### 🔍 Semantic Search
- AI-powered search for similar stories
- Search by meaning and context
- Results ranked by relevance

#### 📈 Predictive Analytics
- Trend predictions
- Story scoring model
- Future topic forecasting

#### 👤 User Profile
- Personalized recommendations
- Reading history
- Preference settings

## Features and Functionality

### Real-Time Updates

**How it works:**
- When enabled, the dashboard automatically refreshes every 60 seconds
- Data updates seamlessly without page reload
- Visual indicator shows when updates are active

**To enable:**
1. Check "Enable Real-Time Mode" in the sidebar
2. Watch for the green success message
3. Data will refresh automatically

### Sentiment Analysis

Understanding the emotional tone of tech news:

- **Positive** 🟢: Optimistic, encouraging, or exciting news
- **Negative** 🔴: Critical, warning, or concerning news
- **Neutral** ⚪: Factual, informational, or balanced content

**Visual Indicators:**
- Sentiment timeline chart showing trends over time
- Color-coded story titles in the data table
- Distribution pie chart for overall sentiment

### Topic Modeling

Automatic identification of key topics:

- Stories are grouped by similar themes
- Each topic gets a descriptive keyword label
- Topics are ranked by frequency and importance

**Common Topics:**
- AI/Machine Learning
- Security & Privacy
- Cloud Computing
- Startups & Funding
- Programming & Development

### Semantic Search

Find stories using natural language queries:

**Example Searches:**
- "artificial intelligence breakthroughs"
- "cybersecurity threats and solutions"
- "remote work productivity tools"
- "blockchain applications in finance"

**How to Use:**
1. Navigate to the Semantic Search tab
2. Click "Initialize Search Database" (first time only)
3. Enter your search query
4. Review results with similarity scores

### Multi-Source Aggregation

Expand your view beyond Hacker News:

**Available Sources:**
- **Hacker News**: Default source for tech discussions
- **Reddit**: Posts from technology subreddits
- **RSS Feeds**: Curated tech news feeds
- **Twitter**: Posts from tech influencers

**Benefits:**
- Comprehensive view of tech news landscape
- Diverse perspectives on trending topics
- Broader coverage of smaller stories

### Executive Briefing Generation

Create professional PDF reports:

**Report Contents:**
- Executive summary of key findings
- Top stories with sentiment analysis
- Topic distribution insights
- Predictive trend analysis
- Interactive charts and visualizations

**Options:**
- Number of stories to include (10-100)
- Include/exclude visual charts
- AI-powered summaries (with OpenAI API key)

## Understanding the Analysis

### Metrics Explained

**1. Average Sentiment Score**
- Range: -1.0 (very negative) to +1.0 (very positive)
- Example: +0.3 indicates slightly positive overall sentiment

**2. Top Topic**
- Most frequently discussed topic
- Example: "ai_machine_learning" appears in 25% of stories

**3. Engagement Score**
- Calculated from comments and upvotes
- Higher scores indicate community interest

### Reading the Charts

**Sentiment Timeline:**
- X-axis: Time stories were posted
- Y-axis: Sentiment score
- Colors: Green (positive), Gray (neutral), Red (negative)

**Topic Distribution:**
- Pie chart showing topic percentages
- Bar chart showing topic frequency
- Interactive hover for detailed numbers

**Story Impact Scatter Plot:**
- X-axis: Number of comments
- Y-axis: Story score (upvotes)
- Size: Sentiment intensity
- Color: Topic category

## Advanced Features

### Predictive Analytics (Phase 7)

**Trend Prediction:**
- Machine learning model predicts future trending topics
- Confidence scores for each prediction
- Historical accuracy tracking

**Story Scoring:**
- Predicts which stories will gain traction
- Factors in sentiment, topic, and engagement patterns
- Updates based on real-time data

### User Personalization

**Create Your Profile:**
- Track your reading preferences
- Get personalized story recommendations
- Save interesting searches and topics

**Preference Settings:**
- Favorite topics to highlight
- Sentiment filters to apply
- Notification preferences

### Phase 9: Market Intelligence

**Competitor Tracking:**
- Monitor mentions of companies/products
- Track sentiment trends over time
- Alert on significant changes

**Market Analysis:**
- Identify emerging market segments
- Track technology adoption trends
- Analyze competitive landscape

## Tips and Best Practices

### For Better Search Results

1. **Use Natural Language**: Search like you speak
   - Good: "cloud computing security best practices"
   - Bad: "cloud security"

2. **Be Specific**: Include context for better results
   - Good: "machine learning for healthcare diagnostics"
   - Bad: "ML healthcare"

3. **Iterate: Refine searches based on initial results**

### For Real-Time Monitoring

1. **Use Stable Internet**: Real-time updates require consistent connection
2. **Monitor Resources**: Extended sessions may use significant memory
3. **Take Breaks**: Disable auto-refresh when away from desk

### For Effective Briefing Reports

1. **Choose Right Scope**:
   - 10-20 stories for quick updates
   - 30-50 stories for comprehensive reports
   - 50+ for deep analysis

2. **Include Charts**: Visuals help communicate trends effectively

3. **Time Your Downloads**: Generate reports during off-peak hours for faster processing

## Troubleshooting

### Common Issues and Solutions

**1. Dashboard Not Loading**
- Check your internet connection
- Try refreshing the page
- Clear browser cache and cookies
- Ensure JavaScript is enabled

**2. Real-Time Updates Not Working**
- Verify "Enable Real-Time Mode" is checked
- Check for error messages in the sidebar
- Disable and re-enable real-time mode
- Try manual refresh

**3. Semantic Search Errors**
- Click "Initialize Search Database" first
- Wait for initialization to complete
- Check query length (3-100 characters)
- Lower similarity threshold if no results

**4. PDF Generation Fails**
- Verify all required modules are installed
- Check OpenAI API key if using AI summaries
- Reduce number of stories if generation is slow
- Try again with different settings

**5. Data Seems Outdated**
- Click "Refresh Data" button
- Check when stories were last updated
- Verify your time zone settings
- Consider enabling real-time mode

### Performance Tips

1. **Optimize Story Count**:
   - 10-20 stories for quick analysis
   - 30-50 for balanced performance
   - 100+ for comprehensive research

2. **Manage Browser Tabs**:
   - Close unused tabs to free memory
   - Avoid multiple dashboard instances

3. **Export Data:**
   - Use CSV for spreadsheet analysis
   - Use JSON for programmatic access

## Glossary

### Technical Terms

**API (Application Programming Interface)**
- A way for different software applications to communicate
- Tech-Pulse uses APIs to fetch data from Hacker News and other sources

**BERTopic**
- An AI algorithm for identifying topics in text
- Groups similar stories based on their content

**ChromaDB**
- A vector database for storing and searching text embeddings
- Powers Tech-Pulse's semantic search capabilities

**Embeddings**
- Numerical representations of text that capture meaning
- Allow computers to understand text similarity

**Sentiment Analysis**
- Process of determining emotional tone in text
- Uses VADER algorithm optimized for headlines

**Semantic Search**
- Search based on meaning, not just keywords
- Finds stories that are conceptually similar

**Vector Database**
- Database optimized for storing high-dimensional vectors
- Enables fast similarity searches

**VADER (Valence Aware Dictionary and sEntiment Reasoner)**
- Sentiment analysis tool specifically tuned for social media
- Provides sentiment scores from -1 to +1

### Dashboard Terms

**Collection**
- Stored set of story vectors for semantic search
- Must be initialized before first search

**Real-Time Mode**
- Automatic data refresh every 60 seconds
- Keeps dashboard current with latest stories

**Similarity Score**
- Measure of how closely a story matches your search query
- Ranges from 0 (no match) to 1 (perfect match)

**Topic ID**
- Numerical identifier for automatically detected topics
- -1 indicates story doesn't fit any identified topic

### Acronyms

**AI**: Artificial Intelligence
**CSV**: Comma-Separated Values (file format)
**JSON**: JavaScript Object Notation (data format)
**ML**: Machine Learning
**NLP**: Natural Language Processing
**PDF**: Portable Document Format

## Need Help?

For additional support:
1. Check the FAQ section in our documentation
2. Review our tutorial videos (coming soon)
3. Contact support through our GitHub repository
4. Join our community forum for user discussions

---

**Happy analyzing with Tech-Pulse!** 🚀

Remember: The dashboard updates continuously with the latest tech trends. Check back often to stay informed about the rapidly changing technology landscape.