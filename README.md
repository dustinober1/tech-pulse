# Tech-Pulse 🚀

A Python-based tech news analysis dashboard that fetches, processes, and analyzes trending stories from Hacker News.

## 📋 Project Overview

Tech-Pulse transforms raw Hacker News data into actionable insights through sentiment analysis and topic modeling. The project follows a structured phase-based approach to build a comprehensive tech news intelligence system.

## 🎯 Features

### Phase 1: Data Pipeline ✅
- Fetch top stories from Hacker News API
- Extract story metadata (title, score, comments, timestamp, URL)
- Return structured data as pandas DataFrame
- Robust error handling and data validation

### Phase 2: Analysis Engine ✅
- **Sentiment Analysis**: VADER-based sentiment analysis of story titles
- **Topic Modeling**: BERTopic-based topic extraction using BERT embeddings
- **Enhanced Processing**: Adds sentiment and topic columns to DataFrame
- **Intelligent Insights**: Automatic distribution summaries and keyword extraction

## 🛠️ Technical Stack

- **Python 3.13+** - Core programming language
- **pandas** - Data manipulation and analysis
- **requests** - HTTP client for API calls
- **NLTK** - Natural language processing (VADER sentiment)
- **BERTopic** - Topic modeling with transformers
- **sentence-transformers** - Text embeddings
- **unittest** - Comprehensive testing framework

## 📁 Project Structure

```
tech-pulse/
├── data_loader.py          # Main data fetching and analysis module
├── README.md              # Project documentation
├── CLAUDE.md              # AI assistant guidelines and rules
├── plans/                 # Phase-based development plans
│   ├── Phase1.md          # Data pipeline implementation plan
│   ├── Phase2.md          # Analysis engine plan
│   ├── Phase3.md          # Future development plan
│   └── Phase4.md          # Future development plan
├── test/                  # Test suite
│   ├── __init__.py        # Package initialization
│   ├── test_data_loader.py # Comprehensive unit tests
│   ├── run_tests.py       # Test runner with detailed reporting
│   └── README.md          # Test documentation
└── test_results/          # Automatic test result storage
    ├── test_results_*.txt # Detailed test reports
    ├── test_summary_*.json # JSON summaries
    └── latest_test_results.txt # Latest results symlink
```

## 🚀 Quick Start

### Prerequisites

- Python 3.13 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dustinober1/tech-pulse.git
   cd tech-pulse
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Or install manually:
   ```bash
   pip install requests pandas nltk bertopic hdbscan
   ```

### Usage

#### Option 1: Interactive Dashboard (Recommended)
```bash
# Install dashboard dependencies
pip install -r requirements-dashboard.txt

# Run the dashboard
streamlit run app.py
```
Access at http://localhost:8501

The dashboard provides:
- Interactive controls for story count and filtering
- Real-time sentiment and topic analysis
- Interactive charts and visualizations
- Data export functionality
- Mobile-responsive design

#### Option 2: Command Line Analysis
```bash
# Run the main analysis
python data_loader.py
```

This will:
- Fetch 20 top stories from Hacker News
- Analyze sentiment of each story title
- Extract topics from the stories
- Display enhanced results with analysis

#### Option 3: Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up

# Or build manually
docker build -t tech-pulse .
docker run -p 8501:8501 tech-pulse
```

2. **Example output:**
   ```
   1. Fetching data...
   Successfully fetched 20 stories.

   2. Analyzing Sentiment...
   Sentiment Analysis Results:
     Neutral: 14 stories (70.0%)
     Positive: 4 stories (20.0%)
     Negative: 2 stories (10.0%)

   3. Extracting Topics...
   Topic Modeling Results:
     Total Topics Found: 3
     Topic 0 (ai_machine_learning): 7 stories (35.0%)
     Topic 1 (security_vulnerability): 6 stories (30.0%)
     Topic 2 (cloud_computing_aws): 4 stories (20.0%)

   Top 5 Rows with Analysis:
                                              title  ... score
   0                          Ghostty is now non-profit  ...   731
   1                       Everyone in Seattle hates AI  ...   565
   2  Reverse engineering a $1B Legal AI tool expose...  ...   474
   ```

## 🧪 Testing

### Run All Tests
```bash
python test/run_tests.py
```

### Run Tests Directly
```bash
python -m unittest test.test_data_loader -v
```

### Test Coverage
- **50 total unit tests**
- **100% pass rate**
- **All functions tested** (as required by CLAUDE.md)
- **Phase 1**: 22 tests (data fetching and processing)
- **Phase 2**: 11 tests (sentiment and topic analysis)
- **Phase 3**: 17 tests (dashboard functionality and configuration)

### Dashboard Testing
```bash
# Run dashboard configuration tests
python -m unittest test.test_dashboard_config -v

# Run dashboard functionality tests
python -m unittest test.test_dashboard_basic -v

# Run all tests
python -m unittest discover test/ -v
```

Test results are automatically saved to `test_results/` with timestamped reports.

## 📊 API Reference

### Core Functions

#### `fetch_hn_data(limit=30)`
Fetch top stories from Hacker News API.

**Parameters:**
- `limit` (int): Number of stories to fetch (default: 30)

**Returns:** pandas.DataFrame with columns:
- `title` (str): Story title
- `score` (int): Number of upvotes
- `descendants` (int): Comment count
- `time` (datetime): Story creation time
- `url` (str): Link to the article

#### `analyze_sentiment(df)`
Analyze sentiment of story titles using VADER.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame with 'title' column

**Returns:** Enhanced DataFrame with additional columns:
- `sentiment_score` (float): VADER compound sentiment score (-1 to 1)
- `sentiment_label` (str): 'Positive', 'Negative', or 'Neutral'

#### `get_topics(df, embedding_model='all-MiniLM-L6-v2')`
Extract topics from story titles using BERTopic.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame with 'title' column
- `embedding_model` (str): Sentence transformer model name

**Returns:** Enhanced DataFrame with additional columns:
- `topic_id` (int): Topic assignment (-1 for outliers)
- `topic_keyword` (str): Topic keywords joined with underscores

## 🧠 Analysis Capabilities

### Sentiment Analysis
- **VADER (Valence Aware Dictionary and sEntiment Reasoner)**
- Optimized for social media and headlines
- Compound scores from -1 (most negative) to 1 (most positive)
- Automatic labeling: Positive (>0.05), Negative (<-0.05), Neutral (otherwise)

### Topic Modeling
- **BERTopic** with transformer embeddings
- Uses `all-MiniLM-L6-v2` for efficient text embeddings
- Automatic topic extraction with keyword representation
- Configurable minimum topic size and embedding models
- Outlier detection for stories not fitting any topic

## 📈 Example Use Cases

1. **Tech Trend Monitoring**: Track emerging technologies and topics
2. **Sentiment Analysis**: Understand community sentiment toward tech news
3. **Content Strategy**: Identify popular topics for content creation
4. **Competitive Intelligence**: Monitor competitor mentions and sentiment
5. **Market Research**: Analyze trends in tech industry discussions

## 🔧 Configuration

### Customizing Analysis

```python
# Fetch more stories
df = fetch_hn_data(limit=50)

# Use different embedding model for topic modeling
df = get_topics(df, embedding_model='all-mpnet-base-v2')

# Custom sentiment thresholds
df['custom_label'] = df['sentiment_score'].apply(
    lambda x: 'Very Positive' if x > 0.5 else 'Positive' if x > 0.05 else 'Neutral'
    if x > -0.05 else 'Negative' if x > -0.5 else 'Very Negative'
)
```

## 🗺️ Development Roadmap

### ✅ Phase 1: Data Pipeline
- Hacker News API integration
- Data extraction and processing
- Basic error handling

### ✅ Phase 2: Analysis Engine
- Sentiment analysis with VADER
- Topic modeling with BERTopic
- Enhanced data processing

### ✅ Phase 3: Dashboard Visualization (Completed)
- **Interactive Streamlit dashboard** with real-time data visualization
- **Multi-agent implementation** executed with 7 specialized development teams
- **Advanced visualizations** using Plotly for interactive charts and metrics
- **Responsive design** optimized for mobile and desktop viewing
- **Real-time features** including auto-refresh and live data updates
- **Export functionality** for CSV and JSON data download
- **Comprehensive testing** with 17 passing unit tests covering all components
- **Docker deployment** ready with containerization support

**Dashboard Features:**
- **Control Panel**: Interactive sidebar with data filtering and refresh controls
- **Metrics Row**: Real-time KPIs showing sentiment, engagement, and trending topics
- **Visualizations**: Sentiment timeline, topic distribution, and story impact charts
- **Data Table**: Expandable raw data view with clickable story links
- **Export Options**: Download analyzed data in multiple formats

**Files Created:**
- `app.py` - Main Streamlit dashboard application
- `dashboard_config.py` - Configuration and styling constants
- `requirements-dashboard.txt` - Dashboard-specific dependencies
- `README_DASHBOARD.md` - Comprehensive dashboard documentation
- `Dockerfile` & `docker-compose.yml` - Containerization and deployment
- `test/test_dashboard_*.py` - Comprehensive test suite (17 tests)

### 🔄 Phase 4: Advanced Features (Planned)
- Real-time data streaming
- Historical trend analysis
- Advanced analytics and insights

## 🧰 Contributing

This project follows the **Test Driven Development** approach. All contributions must:

1. **Include comprehensive unit tests** for all new functions
2. **Maintain 100% test coverage** for contributed code
3. **Follow the existing code style** and structure
4. **Update documentation** for any new features
5. **Pass all existing tests** before submission

### Development Workflow
1. Create feature branch
2. Write failing tests first (TDD)
3. Implement functionality to pass tests
4. Run complete test suite
5. Update documentation
6. Submit pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Resources

- [Hacker News API Documentation](https://github.com/HackerNews/API)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [BERTopic Documentation](https://maartengr.github.io/BERTopic/)
- [Sentence Transformers](https://www.sbert.net/)

## 📞 Contact

For questions, issues, or contributions, please use the GitHub Issues page or contact the project maintainers.

---

**Tech-Pulse** - Transforming Tech News into Actionable Intelligence 🚀