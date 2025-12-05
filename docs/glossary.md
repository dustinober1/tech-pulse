# Tech-Pulse Glossary of Terms

Understanding the terminology used in Tech-Pulse helps you make the most of its powerful features. This glossary explains technical terms in simple, accessible language.

## Table of Contents
1. [AI and Machine Learning Terms](#ai-and-machine-learning-terms)
2. [Data Analysis Terms](#data-analysis-terms)
3. [Dashboard Features](#dashboard-features)
4. [Technical Architecture](#technical-architecture)
5. [News and Media Terms](#news-and-media-terms)
6. [Acronyms and Abbreviations](#acronyms-and-abbreviations)

---

## AI and Machine Learning Terms

### AI (Artificial Intelligence)
**Definition**: Computer systems that can perform tasks typically requiring human intelligence.
**In Tech-Pulse**: Powers features like semantic search and sentiment analysis.
**Example**: "AI helps identify which stories are similar beyond just matching keywords."

### BERTopic
**Definition**: An AI algorithm that automatically discovers topics in text documents.
**In Tech-Pulse**: Groups similar tech stories into meaningful categories.
**Simple Terms**: Think of it as automatically sorting news into "AI stories," "security stories," etc.
**Example**: "BERTopic identified 'cloud_computing_aws' as a trending topic."

### Embeddings (Text Embeddings)
**Definition**: Numerical representations of text that capture meaning and context.
**In Tech-Pulse**: Converts story titles into numbers that can be compared for similarity.
**Simple Terms**: Like turning "The cat sat" into [0.2, -0.5, 0.8, ...] where similar sentences have similar numbers.
**Example**: "Embeddings help find stories about 'AI' even if they don't use that exact word."

### Machine Learning (ML)
**Definition**: Computer programs that learn patterns from data without being explicitly programmed.
**In Tech-Pulse**: Learns from past stories to predict future trends.
**Simple Terms**: Teaching computers to recognize patterns by showing them many examples.
**Example**: "ML predicts that cybersecurity stories will trend more next week."

### Natural Language Processing (NLP)
**Definition**: Technology that helps computers understand and process human language.
**In Tech-Pulse**: Enables analysis of story text for sentiment and meaning.
**Example**: "NLP determines that this story has positive sentiment."

### Sentiment Analysis
**Definition**: Determining the emotional tone or attitude expressed in text.
**In Tech-Pulse**: Classifies stories as positive, negative, or neutral.
**Simple Terms**: Like a mood detector for news stories.
**Example**: "Sentiment analysis shows the tech community is excited about the new product launch."

### VADER (Valence Aware Dictionary and sEntiment Reasoner)
**Definition**: A specialized tool for analyzing sentiment in social media and headlines.
**In Tech-Pulse**: Provides more accurate sentiment analysis for tech news titles.
**Simple Terms**: A sentiment analyzer specifically tuned for short texts like headlines.
**Example**: "VADER scored the story +0.7, indicating strong positive sentiment."

### Vector Database
**Definition**: A database optimized for storing and searching high-dimensional vector data.
**In Tech-Pulse**: Stores text embeddings for fast semantic search.
**Simple Terms**: A special database that can quickly find similar texts by comparing their numerical representations.
**Example**: "The vector database found 5 stories similar to your search query in 0.1 seconds."

---

## Data Analysis Terms

### Aggregation
**Definition**: Combining data from multiple sources into a unified view.
**In Tech-Pulse**: Merges stories from Hacker News, Reddit, RSS, and Twitter.
**Example**: "Multi-source aggregation provides a comprehensive view of today's tech news."

### Confidence Score
**Definition**: A numerical measure of how certain the system is about a prediction or analysis.
**In Tech-Pulse**: Indicates reliability of trend predictions and search results.
**Simple Terms**: How sure the AI is about its answer (0% = guessing, 100% = certain).
**Example**: "The prediction has a 85% confidence score."

### Correlation
**Definition**: A statistical relationship between two variables.
**In Tech-Pulse**: Identifies patterns like "stories about AI often have positive sentiment."
**Example**: "There's a strong correlation between startup funding news and positive sentiment."

### Data Visualization
**Definition**: Presenting data in graphical format for easier understanding.
**In Tech-Pulse**: Charts, graphs, and visual displays of analysis results.
**Example**: "The sentiment timeline chart shows mood changes over the day."

### Engagement Score
**Definition**: A metric indicating how much the community interacts with a story.
**In Tech-Pulse**: Calculated from upvotes, comments, and shares.
**Simple Terms**: How popular or interesting a story is to the community.
**Example**: "This story has a high engagement score with 500 comments and 2000 upvotes."

### Metadata
**Definition**: Data about data - additional information that describes content.
**In Tech-Pulse**: Story details like score, comment count, timestamp, URL.
**Example**: "The metadata shows this story was posted 2 hours ago and has 150 comments."

### Outlier Detection
**Definition**: Identifying data points that differ significantly from other observations.
**In Tech-Pulse**: Stories that don't fit into any identified topic category.
**Simple Terms**: Finding the unusual stories that don't match common patterns.
**Example**: "This story is marked as an outlier because it doesn't fit any tech topic."

### Time Series Analysis
**Definition**: Analyzing data points collected over time to identify trends.
**In Tech-Pulse**: Tracking sentiment and topics throughout the day.
**Example**: "Time series analysis shows sentiment becomes more positive after major product announcements."

---

## Dashboard Features

### Control Panel
**Definition**: The sidebar containing all dashboard controls and settings.
**In Tech-Pulse**: Where you adjust story count, enable real-time mode, set filters.
**Simple Terms**: The control center for customizing your dashboard view.
**Example**: "Use the control panel to enable real-time updates."

### Executive Briefing
**Definition**: A professional PDF report summarizing key insights and trends.
**In Tech-Pulse**: Generated reports with analysis, charts, and summaries.
**Simple Terms**: A professional summary you can share with your team.
**Example**: "Generate an executive briefing for your morning meeting."

### Multi-Source Mode
**Definition**: Feature that fetches news from multiple platforms simultaneously.
**In Tech-Pulse**: Combines Hacker News, Reddit, RSS feeds, and Twitter.
**Example**: "Enable multi-source mode for comprehensive news coverage."

### Predictive Analytics
**Definition**: Using historical data to forecast future trends and events.
**In Tech-Pulse**: AI predictions for trending topics and popular stories.
**Simple Terms**: Looking at past patterns to guess what will happen next.
**Example**: "Predictive analytics suggests blockchain stories will trend tomorrow."

### Real-Time Mode
**Definition**: Automatic data refresh feature that updates dashboard every 60 seconds.
**In Tech-Pulse**: Keeps your dashboard current with latest stories.
**Example**: "Turn on real-time mode to monitor breaking tech news."

### Semantic Search
**Definition**: Search based on meaning and context rather than exact keywords.
**In Tech-Pulse**: Find stories by describing what you're looking for naturally.
**Simple Terms**: Search like you talk, not like a computer.
**Example**: "Semantic search found 'machine learning' stories even when you searched for 'AI'."

### Topic Modeling
**Definition**: Automatically discovering abstract topics in a collection of documents.
**In Tech-Pulse**: Groups stories by underlying themes and topics.
**Simple Terms**: Finding the main subjects being discussed across many stories.
**Example**: "Topic modeling identified 8 major themes in today's tech news."

---

## Technical Architecture

### API (Application Programming Interface)
**Definition**: A set of rules for how software applications should communicate.
**In Tech-Pulse**: Interfaces used to fetch data from Hacker News and other sources.
**Simple Terms**: A standardized way for programs to talk to each other.
**Example**: "Tech-Pulse uses the Hacker News API to fetch story data."

### Cache / Caching
**Definition**: Storing frequently accessed data temporarily for faster access.
**In Tech-Pulse**: Saves analysis results to avoid reprocessing the same data.
**Simple Terms**: Like keeping often-used items on your desk instead of in the filing cabinet.
**Example**: "The cache speeds up repeated searches."

### ChromaDB
**Definition**: An open-source vector database optimized for AI applications.
**In Tech-Pulse**: Stores and searches text embeddings for semantic features.
**Simple Terms**: A smart database designed for AI-powered text search.
**Example**: "ChromaDB enables fast semantic search across thousands of stories."

### Containerization
**Definition**: Packaging applications with all dependencies into isolated environments.
**In Tech-Pulse**: Docker allows running Tech-Pulse anywhere without setup issues.
**Simple Terms**: Putting an app and everything it needs in a box that works anywhere.
**Example**: "Run Tech-Pulse in a Docker container for easy deployment."

### Session State
**Definition**: Maintaining user data and preferences during a dashboard session.
**In Tech-Pulse**: Remembers your settings, filters, and search results.
**Example**: "Session state keeps your real-time mode setting active."

### Streamlit
**Definition**: A Python framework for creating data applications quickly.
**In Tech-Pulse**: The platform used to build the interactive dashboard.
**Simple Terms**: A tool that turns Python code into web apps.
**Example**: "Tech-Pulse is built with Streamlit for easy deployment."

### Vectorization
**Definition**: Converting text or other data into numerical vectors for ML processing.
**In Tech-Pulse**: Converting story titles into embeddings for similarity search.
**Example**: "Vectorization enables finding similar stories through mathematical comparison."

---

## News and Media Terms

### Engagement Metrics
**Definition**: Measurements of how audiences interact with content.
**In Tech-Pulse**: Upvotes, comments, shares, and discussion activity.
**Simple Terms**: How much people care about and interact with a story.
**Example**: "High engagement metrics indicate this story resonates with the community."

### News Aggregation
**Definition**: Collecting news from various sources into one location.
**In Tech-Pulse**: Combining multiple tech news sources for comprehensive coverage.
**Example**: "News aggregation provides a complete picture of today's tech landscape."

### RSS (Really Simple Syndication)
**Definition**: A web feed format for publishing frequently updated content.
**In Tech-Pulse**: One of the data sources for tech news stories.
**Simple Terms**: A subscription feed for website updates.
**Example**: "RSS feeds provide automated updates from major tech news sites."

### Sentiment Polarity
**Definition**: The positive or negative direction of sentiment expressed in text.
**In Tech-Pulse**: Ranges from -1.0 (very negative) to +1.0 (very positive).
**Example**: "The story has positive polarity with a score of +0.6."

### Social Proof
**Definition**: People's tendency to adopt actions or beliefs because many others are doing so.
**In Tech-Pulse**: High-scoring stories influence what other stories gain attention.
**Example**: "Social proof drives more engagement to already popular stories."

### Trend Detection
**Definition**: Identifying topics or stories gaining popularity over time.
**In Tech-Pulse**: AI algorithms detect emerging trends before they peak.
**Example**: "Trend detection identified growing interest in quantum computing."

### Virality Prediction
**Definition**: Estimating which content will spread rapidly across the internet.
**In Tech-Pulse**: Predicts which stories will gain significant traction.
**Example**: "Virality prediction suggests this startup announcement will trend."

---

## Acronyms and Abbreviations

### Common Tech Acronyms

| Acronym | Full Term | Context in Tech-Pulse |
|---------|-----------|----------------------|
| **AI** | Artificial Intelligence | Powers smart features like search and analysis |
| **API** | Application Programming Interface | Used to fetch data from sources |
| **CSV** | Comma-Separated Values | Export format for spreadsheets |
| **JSON** | JavaScript Object Notation | Export format for developers |
| **ML** | Machine Learning | Predicts trends and patterns |
| **NLP** | Natural Language Processing | Enables text understanding |
| **PDF** | Portable Document Format | Format for executive briefings |

### Technical Terms

| Acronym | Full Term | Simple Explanation |
|---------|-----------|-------------------|
| **BERT** | Bidirectional Encoder Representations from Transformers | AI model for understanding text context |
| **CPU** | Central Processing Unit | Computer's main processor |
| **GPU** | Graphics Processing Unit | Specialized processor for AI tasks |
| **RAM** | Random Access Memory | Computer's short-term memory |
| **SSD** | Solid State Drive | Fast storage for applications |

### Data Science Terms

| Acronym | Full Term | Meaning in Tech-Pulse |
|---------|-----------|----------------------|
| **EDA** | Exploratory Data Analysis | Initial examination of news data |
| **KPI** | Key Performance Indicator | Main metrics like sentiment, engagement |
| **MAE** | Mean Absolute Error | Average prediction error in trends |
| **NLP** | Natural Language Processing | Text analysis and understanding |
| **TF-IDF** | Term Frequency-Inverse Document Frequency | Method to identify important words |

---

## How Terms Relate

### The Analysis Pipeline

```
Raw Stories → NLP → Embeddings → Vector DB → Analysis → Visualization
     ↓           ↓         ↓          ↓         ↓          ↓
   API Data  Sentiment  Text as   ChromaDB   Topics    Charts
             Analysis   Numbers    Search    & Trends
```

### Key Relationships

- **NLP** enables **Sentiment Analysis** and **Topic Modeling**
- **Embeddings** are stored in **Vector Database** for **Semantic Search**
- **ML** models use **Aggregated** data for **Predictive Analytics**
- **APIs** provide raw data that becomes **Metadata** in stories

---

## Why These Terms Matter

Understanding these terms helps you:

1. **Use Features Effectively**: Know what each button and setting does
2. **Interpret Results Correctly**: Understand what the data means
3. **Troubleshoot Issues**: Identify what might be going wrong
4. **Communicate Insights**: Explain findings to others accurately
5. **Make Informed Decisions**: Choose the right features for your needs

---

## Need More Explanations?

If you encounter terms not covered here:

1. **Check the User Guide**: [docs/user-guide.md](user-guide.md)
2. **Review the FAQ**: [docs/faq.md](faq.md)
3. **Ask in Community**: Join our user discussions
4. **Contact Support**: Get help with specific questions

---

**Remember**: Tech-Pulse is designed to be user-friendly. You don't need to understand all these terms to benefit from the dashboard, but knowing them helps you use it more effectively! 🚀

*Last updated: December 2025*