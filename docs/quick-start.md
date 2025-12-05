# Tech-Pulse Quick Start Guide

## Get Up and Running in 5 Minutes

### 🚀 Option 1: Use the Live Dashboard (Recommended)

No installation required! Start analyzing tech news immediately:

1. **Open Your Browser**
   - Go to [https://tech-pulse.streamlit.app](https://tech-pulse.streamlit.app)
   - Works on Chrome, Firefox, Safari, or Edge

2. **Explore the Dashboard**
   - Stories load automatically
   - View sentiment analysis and topics
   - Check out the interactive charts

3. **Try These First Actions**:
   - Click different tabs to see features
   - Hover over charts for details
   - Click story titles to read articles
   - Use the semantic search to find related stories

### 📦 Option 2: Local Installation

Want to run Tech-Pulse on your own machine?

#### Prerequisites
- Python 3.13 or newer
- 4GB+ RAM
- 1GB+ disk space

#### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/dustinober1/tech-pulse.git
cd tech-pulse

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

5. **Access Locally**
   - Open your browser to http://localhost:8501
   - The dashboard will load with sample data

## Your First Actions

### 1. Understanding the Dashboard Layout

```
┌─────────────────────────────────────────────────────────┐
│                     Header Section                      │
│               Tech-Pulse Dashboard                     │
├─────────────────────────────────────────────────────────┤
│                │                                        │
│   Control Panel │         Main Content Area             │
│    (Sidebar)    │                                        │
│                │                                        │
│ • Story Count   │    ┌─────────────┬─────────────────┐  │
│ • Refresh Btn   │    │   Charts    │   Data Table    │  │
│ • Real-Time     │    │             │                 │  │
│ • Filters       │    └─────────────┴─────────────────┘  │
│                │                                        │
└─────────────────────────────────────────────────────────┘
```

### 2. Basic Navigation

| What You Want to Do | How to Do It |
|---------------------|--------------|
| **See more stories** | Use the "Number of Stories" slider in the sidebar |
| **Update data** | Click "🔄 Refresh Data" button |
| **Enable auto-update** | Check "Enable Real-Time Mode" |
| **Find specific topics** | Go to "🔍 Semantic Search" tab |
| **Export data** | Choose format in "💾 Export Options" |

### 3. Understanding the Data

#### Sentiment Colors
- 🟢 **Green** = Positive stories (good news, launches, success)
- ⚪ **Gray** = Neutral stories (announcements, updates)
- 🔴 **Red** = Negative stories (issues, failures, warnings)

#### Key Metrics
- **Average Sentiment**: Overall mood (-1.0 to +1.0)
- **Top Topic**: Most discussed theme
- **Engagement**: Community interest level

## Quick Workflows

### Workflow 1: Daily Tech Briefing
1. Open dashboard at start of day
2. Check sentiment trends (are things positive or negative?)
3. Review top topics for major themes
4. Click stories with high engagement
5. Export CSV for personal notes

### Workflow 2: Research Specific Topic
1. Go to "🔍 Semantic Search" tab
2. Initialize search (first time only)
3. Search for your topic (e.g., "artificial intelligence")
4. Review related stories by similarity score
5. Export relevant findings

### Workflow 3: Trend Monitoring
1. Enable "Real-Time Mode" in sidebar
2. Watch the sentiment timeline for patterns
3. Note emerging topics in the distribution chart
4. Set refresh intervals based on your needs

## Pro Tips for Beginners

### For Better Analysis
- **Start Small**: Use 20-30 stories initially
- **Check Sources**: Click story URLs to read full articles
- **Compare Sentiments**: Note how similar topics have different sentiment
- **Track Topics**: Watch how topics change over time

### For Better Performance
- **Refresh Manually**: Only enable real-time when needed
- **Limit Stories**: More stories = slower processing
- **One Tab**: Keep only one dashboard tab open
- **Stable Internet**: Required for real-time features

### For Better Search Results
- **Use Full Sentences**: "machine learning in healthcare"
- **Include Context**: Add timeframes or specific technologies
- **Try Variations**: Different words for the same concept
- **Check Similarity**: Look at scores for result quality

## Common Questions

**Q: Why does it take time to load?**
A: Initial setup includes downloading ML models (~100MB) and analyzing stories.

**Q: Can I use this on mobile?**
A: Yes! The dashboard is mobile-responsive. Use landscape mode for best experience.

**Q: What do the topic names mean?**
A: Topics are auto-generated (e.g., "ai_machine_learning" groups AI-related stories).

**Q: How often should I refresh?**
A: For active monitoring: every 30-60 minutes. For daily use: morning and evening.

**Q: Is my data saved?**
A: No data is stored. Each session starts fresh. Export data if you want to save it.

## Getting Help

### In-App Help
- Click "ℹ️ Help & Info" in the sidebar
- Hover over elements for tooltips
- Check status messages for guidance

### Common Solutions
- **Slow Loading**: Check internet speed, try fewer stories
- **Search Errors**: Initialize database first
- **Real-Time Issues**: Disable/enable the toggle
- **No Data**: Click "Refresh Data" button

## Next Steps

Ready to dive deeper?

1. **Read the Full User Guide**: [docs/user-guide.md](user-guide.md)
2. **Check the FAQ**: [docs/faq.md](faq.md)
3. **Explore Advanced Features**:
   - Multi-source aggregation
   - Predictive analytics
   - Executive briefing reports
4. **Join Our Community**: Share feedback and feature requests

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Refresh Data | Ctrl/Cmd + R |
| Search | Ctrl/Cmd + K |
| Export | Ctrl/Cmd + E |

---

**Welcome to Tech-Pulse!** 🎉

You're now ready to explore the latest in tech news with powerful AI-driven insights. The dashboard will help you spot trends, understand sentiment, and stay ahead of what's happening in technology.

Remember: The tech world moves fast - check back often for the latest updates!