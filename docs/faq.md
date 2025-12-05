# Tech-Pulse Frequently Asked Questions (FAQ)

## Table of Contents
1. [Getting Started](#getting-started)
2. [Dashboard Features](#dashboard-features)
3. [Data and Analysis](#data-and-analysis)
4. [Technical Issues](#technical-issues)
5. [Semantic Search](#semantic-search)
6. [Real-Time Mode](#real-time-mode)
7. [Exports and Reports](#exports-and-reports)
8. [Privacy and Security](#privacy-and-security)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)

## Getting Started

### Q: Do I need to install anything to use Tech-Pulse?
A: **No!** The easiest way is to use our live dashboard at [https://tech-pulse.streamlit.app](https://tech-pulse.streamlit.app). Installation is only required if you want to run it locally.

### Q: What browsers are supported?
A: Tech-Pulse works on all modern browsers:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

### Q: Is there a mobile app?
A: Tech-Pulse is a web application with a mobile-responsive design. Simply visit the URL on your mobile browser - no app download needed.

### Q: How much does it cost?
A: Tech-Pulse is completely free to use. Some advanced features (like AI-powered summaries) may require an OpenAI API key, but all core features are free.

## Dashboard Features

### Q: What do the sentiment colors mean?
A:
- 🟢 **Green**: Positive sentiment (optimistic news, success stories)
- 🔴 **Red**: Negative sentiment (warnings, issues, failures)
- ⚪ **Gray**: Neutral sentiment (announcements, factual news)

### Q: How are topics determined?
A: Topics are automatically identified using BERTopic, an AI algorithm that groups similar stories based on their content. Each topic gets a descriptive label (e.g., "ai_machine_learning").

### Q: What does "Number of Stories" do?
A: This controls how many top stories from Hacker News are analyzed. More stories provide broader coverage but take longer to process.

### Q: Can I customize which sources are used?
A: Yes! Enable "Multi-Source Mode" in the sidebar to include Reddit, RSS feeds, and Twitter in addition to Hacker News.

### Q: What's the difference between tabs?
A:
- **📊 Main Dashboard**: Overview with charts and metrics
- **🔍 Semantic Search**: AI-powered search for similar stories
- **📈 Predictive Analytics**: Trend predictions and forecasts
- **👤 User Profile**: Personalized features and preferences

## Data and Analysis

### Q: How often is the data updated?
A:
- **Manual Mode**: Updates when you click "Refresh Data"
- **Real-Time Mode**: Automatically updates every 60 seconds
- **Live Dashboard**: Continuously updated with latest stories

### Q: Where does the data come from?
A: Primary source is Hacker News API. With multi-source enabled, additional data comes from:
- Reddit (technology subreddits)
- RSS feeds (major tech news sites)
- Twitter (tech influencers)

### Q: How is sentiment calculated?
A: We use VADER (Valence Aware Dictionary and sEntiment Reasoner), specifically tuned for social media and headlines. It provides scores from -1.0 (very negative) to +1.0 (very positive).

### Q: What does the engagement score mean?
A: Engagement score combines story upvotes and comment count to indicate community interest. Higher scores suggest more important or controversial stories.

### Q: Are older stories included?
A: The dashboard shows currently trending stories from Hacker News, typically from the last 24-48 hours.

## Technical Issues

### Q: The dashboard is loading slowly. What can I do?
A: Try these solutions:
1. Check your internet connection speed
2. Reduce the number of stories (use 20-30 instead of 100)
3. Close other browser tabs
4. Try refreshing the page

### Q: I'm seeing an error message. What should I do?
A:
1. Note the exact error message
2. Try refreshing the page
3. Check if real-time mode is causing issues (disable it)
4. Clear your browser cache and cookies

### Q: The charts aren't displaying correctly. Why?
A: This could be due to:
- Ad blockers interfering with scripts
- Browser compatibility issues
- JavaScript being disabled
- Outdated browser version

### Q: Can I use Tech-Pulse offline?
A: No, an internet connection is required to fetch the latest stories and perform analysis. Some cached data may display briefly offline.

## Semantic Search

### Q: What is semantic search?
A: Semantic search finds stories based on meaning and context, not just keywords. For example, searching "machine learning" will also find stories about "AI" and "neural networks."

### Q: Why do I need to "Initialize Search Database"?
A: First-time setup requires processing stories through a machine learning model to create vector embeddings. This enables meaning-based search.

### Q: My search returned no results. Why?
A: Possible reasons:
- Query is too short (minimum 3 characters)
- Similarity threshold is too high
- Try different wording or more specific terms
- Lower the similarity threshold in settings

### Q: What does the similarity score mean?
A: It ranges from 0 (no similarity) to 1 (perfect match). Scores above 0.7 typically indicate good matches.

### Q: Can I save my searches?
A: Currently, searches aren't saved between sessions. Export the results if you want to keep them.

## Real-Time Mode

### Q: What does real-time mode do?
A: It automatically refreshes the dashboard data every 60 seconds without manual intervention.

### Q: Does real-time mode use more data?
A: Yes, it makes periodic API calls to fetch new data. Usage is minimal but may matter on limited data plans.

### Q: Can I change the refresh interval?
A: Currently fixed at 60 seconds to respect API rate limits and ensure optimal performance.

### Q: Real-time updates stopped working. What should I do?
A:
1. Uncheck and re-check the real-time mode toggle
2. Check your internet connection
3. Try a manual refresh
4. Restart your browser

## Exports and Reports

### Q: What formats can I export data in?
A:
- **CSV**: For spreadsheet software (Excel, Google Sheets)
- **JSON**: For developers and data analysis
- **PDF**: Executive briefing reports

### Q: What's included in the PDF report?
A: The executive briefing includes:
- Summary of key findings
- Top stories with sentiment analysis
- Topic distribution insights
- Predictive trends
- Charts and visualizations (optional)

### Q: Why is PDF generation taking so long?
A: Generation time depends on:
- Number of stories included
- Whether charts are included
- Server load
- Using AI summaries (requires API calls)

### Q: Do I need an OpenAI API key?
A: Only if you want AI-powered summaries in your PDF reports. All other features work without it.

## Privacy and Security

### Q: Is my data private?
A: Yes. Tech-Pulse only analyzes public tech stories. No personal data is collected or stored.

### Q: Are my searches saved?
A: No. Searches are processed in real-time and not stored or tracked.

### Q: Is my OpenAI API key safe?
A: Yes, it's handled securely and only used for generating summaries when you explicitly request them.

### Q: Can I see what data is being collected?
A: Tech-Pulse only fetches publicly available story titles, URLs, scores, and comment counts from APIs.

## Advanced Features

### Q: What is predictive analytics?
A: It uses machine learning to:
- Predict which stories will trend
- Forecast emerging topics
- Estimate story performance
- Identify patterns in tech news

### Q: How accurate are the predictions?
A: Accuracy varies but typically 70-80% for short-term predictions. Confidence scores are provided with each prediction.

### Q: What is multi-source aggregation?
A: It combines stories from multiple sources (Hacker News, Reddit, RSS, Twitter) for a comprehensive view of tech news.

### Q: Can I add my own sources?
A: Currently not supported, but this feature is being considered for future releases.

## Troubleshooting

### Common Issues and Solutions

**Issue: Dashboard shows "No Data Available"**
- Click "Refresh Data" button
- Check your internet connection
- Try reducing the number of stories
- Disable multi-source mode if enabled

**Issue: Semantic Search not working**
1. Make sure you've clicked "Initialize Search Database"
2. Wait for initialization to complete (may take 30-60 seconds)
3. Check your query length (3-100 characters)
4. Try a different search term

**Issue: Real-time mode not updating**
- Check that "Enable Real-Time Mode" is checked
- Look for green success message
- Verify your connection is stable
- Try disabling and re-enabling

**Issue: Export failing**
- Ensure you have data loaded first
- Check you've selected an export format
- Try smaller data sets if large exports fail
- Clear browser storage if needed

**Issue: Mobile display problems**
- Use landscape orientation for better viewing
- Ensure you're using a modern browser
- Try refreshing the page
- Report specific issues if they persist

### Performance Tips

1. **For Faster Loading**:
   - Use 20-30 stories instead of 100
   - Disable multi-source mode
   - Close other browser tabs
   - Use a stable internet connection

2. **For Better Search Results**:
   - Use complete sentences
   - Include relevant context
   - Try multiple phrasings
   - Check similarity scores

3. **For Extended Sessions**:
   - Take breaks to prevent memory issues
   - Export important data
   - Monitor system resources
   - Restart browser if needed

## Need More Help?

### Contact Options
- **GitHub Issues**: Report bugs and feature requests
- **Community Forum**: Discuss with other users
- **Documentation**: Check user guide for detailed information
- **Email Support**: For critical issues (coming soon)

### Feature Requests
We love feedback! Suggest new features through:
- GitHub discussions
- In-app feedback form
- Community polls

### Bug Reports
When reporting bugs, please include:
- Browser and version
- Steps to reproduce
- Expected vs actual behavior
- Any error messages
- Screenshots if applicable

---

**Still have questions?**

Don't hesitate to reach out. We're constantly improving Tech-Pulse based on user feedback, and your questions help us make the product better for everyone.