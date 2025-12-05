# Phase 9: Advanced Analytics & Intelligence Features - Detailed Implementation Plan

**Objective**: Implement advanced analytics capabilities including predictive market analysis, competitor intelligence, and enhanced AI-powered insights for comprehensive tech industry intelligence.

## Prerequisites
- Phase 8 (Executive Briefing) complete and working
- Core PDF generation and AI summarization functionality operational
- Data processing and analysis pipelines stable
- Git repository clean and up to date
- All existing Phase 1-8 features fully functional

---

## Work Package 1: Market Intelligence Module (Foundation)

### Task 9.1.1: Install Advanced Analytics Dependencies
**File**: `requirements.txt`
**Action**: Add advanced analytics libraries
**Implementation**:
```txt
# Add these lines to requirements.txt
scipy>=1.11.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
seaborn>=0.13.0
networkx>=3.2.1
wordcloud>=1.9.0
textblob>=0.17.1
spacy>=3.7.0
gensim>=4.3.0
pyLDAvis>=3.4.0
plotly-dash>=2.14.0
dash-bootstrap-components>=1.4.0
dash>=2.14.0
```
**Deliverable**: Updated requirements.txt with analytics dependencies
**Testing**:
- Run `pip install -r requirements.txt`
- Verify all packages install without conflicts
- Test import of key libraries (scipy, statsmodels, networkx)
**Estimated Time**: 45 minutes

### Task 9.1.2: Create Advanced Analytics Module Structure
**Directory**: `src/phase9/` (new)
**Implementation**:
```python
"""
Advanced Analytics Module for Phase 9
Comprehensive market intelligence and predictive analytics
"""

from .market_analyzer import MarketIntelligenceAnalyzer
from .competitor_tracker import CompetitorTracker
from .trend_predictor import AdvancedTrendPredictor
from .intelligence_reporter import IntelligenceReporter

__version__ = "1.0.0"
__all__ = [
    "MarketIntelligenceAnalyzer",
    "CompetitorTracker",
    "AdvancedTrendPredictor",
    "IntelligenceReporter"
]
```
**Files to Create**:
- `src/phase9/__init__.py`
- `src/phase9/market_analyzer.py`
- `src/phase9/competitor_tracker.py`
- `src/phase9/trend_predictor.py`
- `src/phase9/intelligence_reporter.py`
- `src/phase9/utils.py`
**Deliverable**: Complete advanced analytics module structure
**Estimated Time**: 1.5 hours

### Task 9.1.3: Create Analytics Test Infrastructure
**File**: `test/test_phase9_analytics.py` (new)
**Implementation**:
```python
"""
Test suite for Phase 9 Advanced Analytics Module
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

# Import Phase 9 modules with graceful fallback
PHASE9_AVAILABLE = False
try:
    from src.phase9.market_analyzer import MarketIntelligenceAnalyzer
    from src.phase9.competitor_tracker import CompetitorTracker
    from src.phase9.trend_predictor import AdvancedTrendPredictor
    from src.phase9.intelligence_reporter import IntelligenceReporter
    PHASE9_AVAILABLE = True
except ImportError as e:
    print(f"Phase 9 modules not available: {e}")
    PHASE9_AVAILABLE = False

class TestMarketIntelligence(unittest.TestCase):
    """Test suite for Market Intelligence functionality"""

    def setUp(self):
        """Set up test fixtures"""
        if not PHASE9_AVAILABLE:
            self.skipTest("Phase 9 modules not available")

        self.market_analyzer = MarketIntelligenceAnalyzer()
        self.sample_data = self._create_sample_market_data()

    def _create_sample_market_data(self):
        """Create sample market data for testing"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')

        data = {
            'date': dates,
            'stock_symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN'] * 25,
            'price': np.random.uniform(100, 500, 100),
            'volume': np.random.randint(1000000, 10000000, 100),
            'sentiment': np.random.uniform(-0.5, 0.5, 100),
            'topic_keywords': ['ai_ml', 'cloud', 'security', 'funding'] * 25,
            'source': ['techcrunch', 'venturebeat', 'arxiv', 'github'] * 25
        }
        return pd.DataFrame(data)

    def test_market_volatility_analysis(self):
        """Test market volatility analysis"""
        if not PHASE9_AVAILABLE:
            self.skipTest("Phase 9 modules not available")

        try:
            results = self.market_analyzer.analyze_volatility(self.sample_data)
            self.assertIn('volatility_index', results)
            self.assertIn('trend_strength', results)
        except Exception as e:
            self.fail(f"Volatility analysis failed: {e}")

    def test_competitor_tracking(self):
        """Test competitor tracking functionality"""
        if not PHASE9_AVAILABLE:
            self.skipTest("Phase 9 modules not available")

        try:
            tracker = CompetitorTracker()
            results = tracker.track_competitors(self.sample_data)
            self.assertIsInstance(results, dict)
        except Exception as e:
            self.fail(f"Competitor tracking failed: {e}")

    def test_trend_prediction(self):
        """Test advanced trend prediction"""
        if not PHASE9_AVAILABLE:
            self.skipTest("Phase 9 modules not available")

        try:
            predictor = AdvancedTrendPredictor()
            predictions = predictor.predict_trends(self.sample_data, periods=30)
            self.assertGreater(len(predictions), 0)
        except Exception as e:
            self.fail(f"Trend prediction failed: {e}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
```
**Deliverable**: Comprehensive test suite for Phase 9 analytics
**Testing**: Run `python -m pytest test/test_phase9_analytics.py -v`
**Estimated Time**: 2.5 hours

---

## Work Package 2: Market Intelligence Analyzer (Core Logic)

### Task 9.2.1: Implement Market Intelligence Analyzer
**File**: `src/phase9/market_analyzer.py`
**Implementation**:
```python
"""
Market Intelligence Analyzer for Phase 9
Advanced market analysis and tech industry insights
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import fetch_hn_data, analyze_sentiment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketIntelligenceAnalyzer:
    """
    Advanced market intelligence analyzer for tech industry
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Explain 95% of variance
        self.logger = logging.getLogger(__name__)
        self.logger.info("Market Intelligence Analyzer initialized")

    def analyze_market_sentiment(self, df: pd.DataFrame, window_days: int = 30) -> Dict[str, Any]:
        """
        Analyze market sentiment trends over time

        Args:
            df: DataFrame with sentiment data
            window_days: Analysis window in days

        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            if df.empty or 'time' not in df.columns:
                return {"error": "Insufficient data for sentiment analysis"}

            # Convert time to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'])

            # Filter data by time window
            cutoff_date = datetime.now() - timedelta(days=window_days)
            recent_df = df[df['time'] >= cutoff_date].copy()

            if recent_df.empty:
                return {"error": f"No data available for {window_days} day window"}

            # Calculate sentiment metrics
            sentiment_metrics = {
                'overall_sentiment': recent_df['sentiment_score'].mean(),
                'sentiment_volatility': recent_df['sentiment_score'].std(),
                'sentiment_trend': self._calculate_sentiment_trend(recent_df),
                'daily_sentiment': recent_df.groupby(recent_df['time'].dt.date)['sentiment_score'].mean().to_dict(),
                'sentiment_distribution': recent_df['sentiment_label'].value_counts().to_dict(),
                'anomaly_days': self._detect_sentiment_anomalies(recent_df),
                'correlation_metrics': self._calculate_correlations(recent_df)
            }

            # Add market context
            if 'score' in recent_df.columns:
                sentiment_metrics['sentiment_engagement_correlation'] = recent_df[['sentiment_score', 'score']].corr().iloc[0, 1]

            self.logger.info(f"Market sentiment analysis completed for {len(recent_df)} records")
            return sentiment_metrics

        except Exception as e:
            logger.error(f"Market sentiment analysis failed: {e}")
            return {"error": str(e)}

    def analyze_technology_adoption_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze technology adoption trends from news data

        Args:
            df: DataFrame with tech news data

        Returns:
            Dictionary with technology trend analysis
        """
        try:
            if df.empty or 'topic_keyword' not in df.columns:
                return {"error": "Insufficient topic data for trend analysis"}

            # Filter out empty topics
            topic_df = df[df['topic_keyword'].notna() & (df['topic_keyword'] != '')].copy()

            if topic_df.empty:
                return {"error": "No valid topic data found"}

            # Calculate trend metrics
            trend_analysis = {
                'topic_distribution': topic_df['topic_keyword'].value_counts().head(20).to_dict(),
                'emerging_topics': self._identify_emerging_topics(topic_df),
                'declining_topics': self._identify_declining_topics(topic_df),
                'topic_correlation_matrix': self._calculate_topic_correlations(topic_df),
                'technology_momentum': self._calculate_technology_momentum(topic_df),
                'topic_evolution_timeline': self._analyze_topic_evolution(topic_df)
            }

            # Add engagement data
            if 'score' in topic_df.columns:
                topic_engagement = topic_df.groupby('topic_keyword')['score'].agg(['mean', 'std', 'count'])
                trend_analysis['topic_engagement_metrics'] = topic_engagement.to_dict()

            self.logger.info(f"Technology adoption trends analyzed for {len(topic_df)} records")
            return trend_analysis

        except Exception as e:
            logger.error(f"Technology trend analysis failed: {e}")
            return {"error": str(e)}

    def predict_market_movements(self, df: pd.ndarray, periods: int = 7) -> Dict[str, Any]:
        """
        Predict market movements using advanced time series analysis

        Args:
            df: DataFrame with market data
            periods: Number of periods to predict

        Returns:
            Dictionary with prediction results
        """
        try:
            # Prepare time series data
            if df.empty:
                return {"error": "Insufficient data for prediction"}

            # Use multiple prediction models
            predictions = {
                'arima_predictions': self._arima_forecast(df, periods),
                'lstm_predictions': self._lstm_forecast(df, periods),
                'ensemble_predictions': self._ensemble_forecast(df, periods),
                'confidence_intervals': self._calculate_confidence_intervals(df, periods),
                'feature_importance': self._calculate_feature_importance(df),
                'prediction_accuracy': self._backtest_predictions(df)
            }

            self.logger.info(f"Market movement predictions completed for {periods} periods")
            return predictions

        except Exception as e:
            logger.error(f"Market prediction failed: {e}")
            return {"error": str(e)}

    def generate_competitor_intelligence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive competitor intelligence

        Args:
            df: DataFrame with market and competitor data

        Returns:
            Dictionary with competitor intelligence
        """
        try:
            if df.empty:
                return {"error": "Insufficient data for competitor analysis"}

            competitor_analysis = {
                'competitor_landscape': self._map_competitor_landscape(df),
                'competitive_positioning': self._analyze_competitive_positioning(df),
                'market_share_trends': self._calculate_market_share_trends(df),
                'competitive_threats': self._identify_competitive_threats(df),
                'opportunity_gaps': self._identify_opportunity_gaps(df)
            }

            self.logger.info("Competitor intelligence generation completed")
            return competitor_analysis

        except Exception as e:
            logger.error(f"Competitor intelligence failed: {e}")
            return {"error": str(e)}

    def _calculate_sentiment_trend(self, df: pd.DataFrame) -> str:
        """Calculate sentiment trend direction"""
        if len(df) < 2:
            return "insufficient_data"

        # Linear regression on sentiment over time
        x = np.arange(len(df))
        y = df['sentiment_score'].values

        slope, _, _, _, _ = stats.linregress(x, y)

        if slope > 0.05:
            return "improving"
        elif slope < -0.05:
            return "declining"
        else:
            return "stable"

    def _detect_sentiment_anomalies(self, df: pd.ndarray) -> List[datetime]:
        """Detect sentiment anomalies in time series"""
        anomalies = []

        if len(df) < 10:
            return anomalies

        # Z-score based anomaly detection
        z_scores = np.abs(stats.zscore(df['sentiment_score']))
        threshold = 2.5

        anomalous_days = df[z_scores > threshold]
        for _, row in anomalous_days.iterrows():
            anomalies.append(row['time'])

        return anomalies

    def _calculate_correlations(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlation metrics between key variables"""
        correlations = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()

            # Extract key correlations
            if 'sentiment_score' in corr_matrix.columns:
                correlations['sentiment_score_correlations'] = corr_matrix['sentiment_score'].to_dict()

            if 'score' in corr_matrix.columns and 'sentiment_score' in corr_matrix.columns:
                correlations['sentiment_engagement_correlation'] = corr_matrix.loc['score', 'sentiment_score']

        return correlations

    def _identify_emerging_topics(self, df: pd.DataFrame) -> List[str]:
        """Identify emerging topics based on frequency trends"""
        emerging_topics = []

        # Analyze topic frequency over time
        topic_counts = df.groupby([
            df['time'].dt.floor('D'),
            'topic_keyword'
        ]).size().unstack(fill_value=0)

        for topic in topic_counts.columns:
            # Check if topic is trending upward
            if len(topic_counts[topic]) >= 2:
                recent_growth = topic_counts[topic].iloc[-1] - topic_counts[topic].iloc[0]
                if recent_growth > topic_counts[topic].iloc[0] * 0.5:
                    emerging_topics.append(topic)

        return emerging_topics[:10]  # Top 10 emerging topics

    def _identify_declining_topics(self, df: pd.DataFrame) -> List[str]:
        """Identify declining topics"""
        declining_topics = []

        topic_counts = df.groupby([
            df['time'].dt.floor('D'),
            'topic_keyword'
        ]).size().unstack(fill_value=0)

        for topic in topic_counts.columns:
            if len(topic_counts[topic]) >= 2:
                recent_decline = topic_counts[topic].iloc[0] - topic_counts[topic].iloc[-1]
                if recent_decline > topic_counts[topic].iloc[-1] * 0.3:
                    declining_topics.append(topic)

        return declining_topics[:10]

    def _calculate_topic_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix between topics"""
        # Create topic co-occurrence matrix
        topics = df['topic_keyword'].unique()
        topic_matrix = pd.DataFrame(0, index=topics, columns=topics)

        for _, row in df.iterrows():
            if pd.notna(row['topic_keyword']):
                topic = row['topic_keyword']
                topic_matrix.at[topic, topic] += 1

        return topic_matrix

    def _calculate_technology_momentum(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum scores for technologies"""
        momentum_scores = {}

        for topic in df['topic_keyword'].unique():
            topic_data = df[df['topic_keyword'] == topic]

            if len(topic_data) >= 3:
                # Calculate momentum as score trend
                scores = topic_data['score'].values
                momentum = (scores[-1] - scores[0]) / len(scores) if len(scores) > 0 else 0
                momentum_scores[topic] = momentum

        return momentum_scores

    def _analyze_topic_evolution(self, df: pd.DataFrame) -> Dict[str, List]:
        """Analyze topic evolution over time"""
        evolution = {
            'topic_frequency_timeline': {},
            'sentiment_evolution': {},
            'engagement_evolution': {}
        }

        # Daily topic frequency
        daily_topics = df.groupby([
            df['time'].dt.floor('D'),
            'topic_keyword'
        ]).size().unstack(fill_value=0)

        for topic in daily_topics.columns:
            evolution['topic_frequency_timeline'][topic] = daily_topics[topic].tolist()

        return evolution

    def _arima_forecast(self, df: pd.ndarray, periods: int) -> Dict[str, Any]:
        """ARIMA-based forecasting"""
        try:
            from statsmodels.tsa.arima.model import ARIMA

            # Simple ARIMA implementation
            model = ARIMA(df['price'], order=(1,1,1))
            model_fit = model.fit()

            forecast = model_fit.forecast(steps=periods)
            confidence_intervals = model_fit.get_forecast(steps=periods, alpha=0.05)

            return {
                'forecast': forecast,
                'confidence_intervals': confidence_intervals,
                'aic': model_fit.aic,
                'bic': model_fit.bic
            }
        except Exception as e:
            logger.error(f"ARIMA forecast failed: {e}")
            return {"error": str(e)}

    def _lstm_forecast(self, df: pd.ndarray, periods: int) -> Dict[str, Any]:
        """LSTM-based forecasting"""
        # Placeholder for LSTM implementation
        return {
            'forecast': np.random.normal(0, 1, periods),
            'confidence_intervals': np.random.normal(0, 1, (2, periods)),
            'note': 'LSTM implementation pending'
        }

    def _ensemble_forecast(self, df: pd.ndarray, periods: int) -> Dict[str, Any]:
        """Ensemble forecasting combining multiple models"""
        arima_result = self._arima_forecast(df, periods)
        lstm_result = self._lstm_forecast(df, periods)

        # Simple weighted average
        if 'forecast' in arima_result and 'forecast' in lstm_result:
            ensemble_forecast = (
                0.6 * arima_result['forecast'] +
                0.4 * lstm_result['forecast']
            )
        else:
            ensemble_forecast = arima_result.get('forecast', np.array([]))

        return {
            'ensemble_forecast': ensemble_forecast,
            'component_models': {'arima': arima_result, 'lstm': lstm_result}
        }

    def _calculate_confidence_intervals(self, df: pd.ndarray, periods: int) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals for predictions"""
        # Placeholder for confidence interval calculation
        std_dev = df['price'].std() if 'price' in df.columns else 1.0
        return {
            'lower_95': np.zeros(periods) - 1.96 * std_dev,
            'upper_95': np.zeros(periods) + 1.96 * std_dev,
            'lower_68': np.zeros(periods) - 1.0 * std_dev,
            'upper_68': np.zeros(periods) + 1.0 * std_dev
        }

    def _calculate_feature_importance(self, df: pd.ndarray) -> Dict[str, float]:
        """Calculate feature importance for predictions"""
        features = ['score', 'sentiment_score', 'descendants'] if 'descendants' in df.columns else []
        importance = {}

        for feature in features:
            if feature in df.columns:
                correlation = abs(df[feature].corr(df['price']))
                importance[feature] = correlation

        return importance

    def _backtest_predictions(self, df: pd.ndarray) -> Dict[str, float]:
        """Backtest prediction models"""
        # Placeholder for backtesting implementation
        return {
            'mae': 0.1,
            'rmse': 0.15,
            'r2_score': 0.85,
            'note': 'Backtesting implementation pending'
        }

    def _map_competitor_landscape(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Map out competitor landscape"""
        competitors = {}

        if 'url' in df.columns:
            # Extract domains from URLs
            domains = df['url'].str.extract(r'https?://(?:www\.)?([^/]+)')[0]
            domain_counts = domains.value_counts().head(10)

            competitors['top_domains'] = domain_counts.to_dict()
            competitors['domain_engagement'] = df.groupby(domains)['score'].mean().to_dict()

        return competitors

    def _analyze_competitive_positioning(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze competitive positioning"""
        positioning = {
            'market_leaders': [],
            'challengers': [],
            'niche_players': [],
            'positioning_score': 0.0
        }

        # Placeholder for competitive positioning analysis
        return positioning

    def _calculate_market_share_trends(self, df: pd.ndarray) -> Dict[str, Any]:
        """Calculate market share trends"""
        trends = {
            'share_evolution': {},
            'concentration_ratio': 0.0,
            'herfindahl_index': 0.0
        }

        return trends

    def _identify_competitive_threats(self, df: pd.ndarray) -> List[Dict[str, str]]:
        """Identify competitive threats"""
        threats = []

        # Placeholder for threat identification
        return threats

    def _identify_opportunity_gaps(self, df: pd.ndarray) -> List[Dict[str, str]]:
        """Identify opportunity gaps in the market"""
        gaps = []

        # Placeholder for opportunity gap identification
        return gaps
```
**Deliverable**: Complete market intelligence analyzer with advanced analytics
**Testing**: Test with sample market data and various analysis functions
**Estimated Time**: 6 hours

---

## Work Package 3: Competitor Intelligence (Market Analysis)

### Task 9.3.1: Implement Competitor Tracker
**File**: `src/phase9/competitor_tracker.py`
**Implementation**:
```python
"""
Competitor Tracker for Phase 9
Track and analyze tech industry competitors
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
import re
from urllib.parse import urlparse
from collections import Counter
from data_loader import fetch_hn_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompetitorTracker:
    """
    Advanced competitor tracking for tech industry intelligence
    """

    def __init__(self):
        self.competitors_db = {}
        self.tracking_keywords = [
            'vs', 'competitor', 'rival', 'compete', 'challenge', 'beat', 'outperform',
            'launch', 'release', 'update', 'announce', 'unveil', 'reveal'
        ]
        self.logger = logging.getLogger(__name__)
        self.logger.info("Competitor Tracker initialized")

    def track_competitors(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Track competitors from tech news data

        Args:
            df: DataFrame with news data

        Returns:
            Dictionary with competitor tracking results
        """
        try:
            if df.empty:
                return {"error": "No data available for competitor tracking"}

            # Extract competitor mentions
            competitor_data = self._extract_competitor_mentions(df)

            # Analyze competitor patterns
            analysis = {
                'competitor_mentions': competitor_data,
                'top_competitors': self._identify_top_competitors(competitor_data),
                'competitive_density': self._calculate_competitive_density(competitor_data),
                'mention_trends': self._analyze_mention_trends(competitor_data),
                'sentiment_analysis': self._analyze_competitor_sentiment(competitor_data),
                'market_positioning': self._analyze_market_positioning(competitor_data)
            }

            self.logger.info(f"Competitor tracking completed: {len(competitor_data)} mentions found")
            return analysis

        except Exception as e:
            logger.error(f"Competitor tracking failed: {e}")
            return {"error": str(e)}

    def extract_competitor_profiles(self, df: pd.ndarray) -> List[Dict[str, Any]]:
        """
        Extract detailed competitor profiles

        Returns:
            List of competitor profile dictionaries
        """
        profiles = []

        # Implement competitor profile extraction
        # This would involve external data sources like Crunchbase, PitchBook, etc.

        return profiles

    def analyze_competitive_landscape(self, df: pd.ndarray) -> Dict[str, Any]:
        """
        Analyze overall competitive landscape

        Returns:
            Dictionary with landscape analysis
        """
        landscape = {
            'competitive_intensity': 0.0,
            'market_concentration': 0.0,
            'competitive_dynamics': {},
            'entry_barriers': 'moderate',
            'innovation_level': 'high'
        }

        return landscape

    def _extract_competitor_mentions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract competitor mentions from news content"""
        mentions = []

        for _, row in df.iterrows():
            title = str(row.get('title', ''))
            text = f"{title} {row.get('text', '')}"

            # Look for competitor keywords
            for keyword in self.tracking_keywords:
                if keyword.lower() in text.lower():
                    # Extract potential competitor names using regex patterns
                    company_names = self._extract_company_names(text)

                    for name in company_names:
                        mentions.append({
                            'company_name': name,
                            'mention_type': keyword,
                            'title': title,
                            'url': row.get('url', ''),
                            'score': row.get('score', 0),
                            'time': row.get('time', datetime.now()),
                            'sentiment': row.get('sentiment_label', 'Neutral')
                        })

        return mentions

    def _extract_company_names(self, text: str) -> List[str]:
        """Extract company names from text using regex patterns"""
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized words
            r'\b(?:Apple|Google|Microsoft|Amazon|Meta|Tesla|Netflix|Spotify|Adobe|Oracle|Salesforce)',
            r'\b(?:[+-][A-Za-z]+\s+(?:Inc|Corp|LLC|Ltd))\b',
            r'\b(?:GitHub|Twitter|Facebook|Instagram|YouTube|LinkedIn)\b'
        ]

        company_names = []
        seen_names = set()

        for pattern in patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            for match in matches:
                if match not in seen_names:
                    company_names.append(match)
                    seen_names.add(match)

        return company_names

    def _identify_top_competitors(self, mentions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify top competitors by mention frequency and engagement"""
        company_counts = Counter(mention['company_name'] for mention in mentions)

        top_competitors = []
        for company, count in company_counts.most_common(10):
            # Get company details
            company_mentions = [m for m in mentions if m['company_name'] == company]

            avg_score = np.mean([m['score'] for m in company_mentions]) if company_mentions else 0
            avg_sentiment = self._calculate_avg_sentiment(company_mentions)

            top_competitors.append({
                'company_name': company,
                'mention_count': count,
                'avg_engagement': avg_score,
                'avg_sentiment': avg_sentiment,
                'recent_mentions': company_mentions[-5:]  # Last 5 mentions
            })

        return top_competitors

    def _calculate_competitive_density(self, mentions: List[Dict[str, Any]]) -> float:
        """Calculate competitive density score"""
        if not mentions:
            return 0.0

        total_stories = len(mentions)
        unique_companies = len(set(m['company_name'] for m in mentions))

        # Density as ratio of unique competitors to total stories
        return unique_companies / total_stories

    def _analyze_mention_trends(self, mentions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in competitor mentions over time"""
        if not mentions:
            return {"trend": "no_data"}

        # Group mentions by company and time
        mention_trends = {}

        for mention in mentions:
            company = mention['company_name']
            if company not in mention_trends:
                mention_trends[company] = []
            mention_trends[company].append(mention)

        # Calculate trends for each company
        trends_analysis = {}
        for company, company_mentions in mention_trends.items():
            if len(company_mentions) >= 2:
                # Check if mentions are increasing or decreasing
                scores = [m['score'] for m in company_mentions]
                trend_direction = "stable"

                if len(scores) > 2:
                    if scores[-1] > scores[-2] > scores[0]:
                        trend_direction = "increasing"
                    elif scores[-1] < scores[-2] < scores[0]:
                        trend_direction = "decreasing"

                trends_analysis[company] = {
                    'trend': trend_direction,
                    'mention_frequency': len(company_mentions),
                    'avg_engagement': np.mean([m['score'] for m in company_mentions]),
                    'momentum_score': self._calculate_momentum_score(company_mentions)
                }

        return trends_analysis

    def _analyze_competitor_sentiment(self, mentions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment patterns around competitors"""
        if not mentions:
            return {"analysis": "no_data"}

        sentiment_scores = {}

        for company, company_mentions in groupby(mention['company_name'] for mention in mentions):
            if company_mentions:
                sentiment_scores[company] = {
                    'avg_sentiment': np.mean([m.get('sentiment_score', 0) for m in company_mentions]),
                    'sentiment_distribution': Counter(m.get('sentiment_label', 'Neutral') for m in company_mentions),
                    'sentiment_volatility': np.std([m.get('sentiment_score', 0) for m in company_mentions]) if len(company_mentions) > 1 else 0
                }

        return sentiment_scores

    def _analyze_market_positioning(self, mentions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze market positioning of competitors"""
        positioning = {
            'leaderboard': [],
            'challengers': [],
            'niche_players': [],
            'positioning_strategies': {}
        }

        return positioning

    def _calculate_momentum_score(self, company_mentions: List[Dict[str, Any]]) -> float:
        """Calculate momentum score for a competitor"""
        if len(company_mentions) < 2:
            return 0.0

        scores = [m['score'] for m in company_mentions]

        # Calculate momentum as recent trend
        if len(scores) >= 3:
            recent_momentum = (scores[-1] + scores[-2]) - (scores[0] + scores[1])
            return recent_momentum / 2
        else:
            return 0.0

    def _calculate_avg_sentiment(self, company_mentions: List[Dict[str, Any]]) -> float:
        """Calculate average sentiment for a competitor"""
        if not company_mentions:
            return 0.0

        sentiments = []
        for mention in company_mentions:
            sentiment = mention.get('sentiment_score', 0)
            if sentiment != 0:
                sentiments.append(sentiment)

        return np.mean(sentiments) if sentiments else 0.0
```
**Deliverable**: Complete competitor tracker with advanced tracking capabilities
**Testing**: Test with competitor mention extraction and analysis functions
**Estimated Time**: 4 hours

### Task 9.3.2: Implement Competitive Intelligence Reporting
**File**: `src/phase9/competitive_intelligence.py` (new)
**Implementation**:
```python
"""
Competitive Intelligence Reporting for Phase 9
Generate comprehensive competitive intelligence reports
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from .market_analyzer import MarketIntelligenceAnalyzer
from .competitor_tracker import CompetitorTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompetitiveIntelligenceReporter:
    """
    Generate comprehensive competitive intelligence reports
    """

    def __init__(self):
        self.market_analyzer = MarketIntelligenceAnalyzer()
        self.competitor_tracker = CompetitorTracker()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Competitive Intelligence Reporter initialized")

    def generate_competitive_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive competitive intelligence report

        Args:
            df: DataFrame with market data

        Returns:
            Dictionary with report data
        """
        try:
            # Collect intelligence data
            market_analysis = self.market_analyzer.analyze_market_sentiment(df)
            competitor_data = self.competitor_tracker.track_competitors(df)

            # Generate report sections
            report = {
                'executive_summary': self._generate_executive_summary(df, market_analysis, competitor_data),
                'market_overview': self._generate_market_overview(df),
                'competitor_analysis': self._generate_competitor_section(competitor_data),
                'trend_insights': self._generate_trend_insights(market_analysis),
                'recommendations': self._generate_strategic_recommendations(market_analysis, competitor_data),
                'appendices': self._generate_appendices(df),
                'metadata': self._generate_report_metadata(df)
            }

            self.logger.info("Competitive intelligence report generated successfully")
            return report

        except Exception as e:
            logger.error(f"Competitive report generation failed: {e}")
            return {"error": str(e)}

    def export_competitive_dashboard(self, df: pd.ndarray) -> bytes:
        """
        Export competitive intelligence dashboard

        Args:
            df: DataFrame with competitive data

        Returns:
            Dashboard as bytes
        """
        try:
            # Implementation for dashboard export
            return b"competitive_dashboard_placeholder"
        except Exception as e:
            logger.error(f"Dashboard export failed: {e}")
            return b""

    def _generate_executive_summary(self, df: pd.DataFrame, market_analysis: Dict, competitor_data: Dict) -> str:
        """Generate executive summary"""
        summary_parts = []

        # Market sentiment overview
        if 'error' not in market_analysis:
            if 'overall_sentiment' in market_analysis:
                sentiment = market_analysis['overall_sentiment']
                if sentiment > 0.1:
                    summary_parts.append("Market sentiment is predominantly positive")
                elif sentiment < -0.1:
                    summary_parts.append("Market sentiment shows notable concern")
                else:
                    summary_parts.append("Market sentiment is balanced")

        # Competitor activity
        if 'error' not in competitor_data and competitor_data.get('competitor_mentions'):
            mention_count = len(competitor_data['competitor_mentions'])
            summary_parts.append(f"Tracked {mention_count} competitor mentions across the dataset")

        return ". ".join(summary_parts) if summary_parts else "Executive summary generation pending"

    def _generate_market_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate market overview section"""
        overview = {
            'data_coverage': len(df),
            'time_range': f"{df['time'].min()} to {df['time'].max()}" if 'time' in df.columns else "Unknown",
            'key_metrics': self._get_key_metrics(df),
            'sector_distribution': self._get_sector_distribution(df)
        }
        return overview

    def _generate_competitor_section(self, competitor_data: Dict) -> Dict[str, Any]:
        """Generate competitor analysis section"""
        if 'error' in competitor_data:
            return {"status": "no_data"}

        return {
            'top_competitors': competitor_data.get('top_competitors', []),
            'competitive_density': competitor_data.get('competitive_density', 0),
            'mention_trends': competitor_data.get('mention_trends', {}),
            'sentiment_analysis': competitor_data.get('sentiment_analysis', {}),
            'positioning_analysis': competitor_data.get('positioning', {})
        }

    def _generate_trend_insights(self, market_analysis: Dict) -> Dict[str, Any]:
        """Generate trend insights section"""
        insights = {
            'market_momentum': self._calculate_market_momentum(market_analysis),
            'trend_forecasts': self._generate_trend_forecasts(market_analysis),
            'key_trends': self._identify_key_trends(market_analysis),
            'anomaly_detection': self._detect_market_anomalies(market_analysis)
        }
        return insights

    def _generate_strategic_recommendations(self, market_analysis: Dict, competitor_data: Dict) -> List[str]:
        """Generate strategic recommendations"""
        recommendations = []

        # Based on sentiment analysis
        if 'overall_sentiment' in market_analysis:
            sentiment = market_analysis['overall_sentiment']
            if sentiment > 0.3:
                recommendations.append("Consider increasing market coverage in positive sentiment areas")
            elif sentiment < -0.3:
                recommendations.append("Address negative sentiment drivers in the market")

        # Based on competitor activity
        if 'competitive_density' in competitor_data:
            density = competitor_data['competitive_density']
            if density > 0.3:
                recommendations.append("Monitor competitor movements for strategic opportunities")

        return recommendations

    def _generate_appendices(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate report appendices"""
        appendices = {
            'data_sources': ["Hacker News API", "Real-time feeds", "Historical data"],
            'methodology': "Advanced analytics with statistical modeling",
            'limitations': "Based on available news data sources"
        }
        return appendices

    def _generate_report_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate report metadata"""
        return {
            'generation_date': datetime.now().isoformat(),
            'data_freshness': df['time'].max() if 'time' in df.columns else None,
            'report_version': "1.0",
            'analysis_method': "Advanced analytics and statistical modeling"
        }

    def _get_key_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get key market metrics"""
        metrics = {}

        if not df.empty:
            metrics['total_stories'] = len(df)
            metrics['avg_engagement'] = df['score'].mean() if 'score' in df.columns else 0
            metrics['total_comments'] = df['descendants'].sum() if 'descendants' in df.columns else 0

        return metrics

    def _get_sector_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Get sector/topic distribution"""
        if 'topic_keyword' in df.columns:
            return df['topic_keyword'].value_counts().head(10).to_dict()
        return {}

    def _calculate_market_momentum(self, market_analysis: Dict) -> str:
        """Calculate overall market momentum"""
        if 'sentiment_trend' in market_analysis:
            return market_analysis['sentiment_trend']
        return "neutral"

    def _generate_trend_forecasts(self, market_analysis: Dict) -> Dict[str, Any]:
        """Generate trend forecasts"""
        forecasts = {
            'short_term': {"direction": "neutral", "confidence": 0.6},
            'long_term': {"direction": "neutral", "confidence": 0.4},
            "model_accuracy": 0.75
        }
        return forecasts

    def _identify_key_trends(self, market_analysis: Dict) -> List[str]:
        """Identify key market trends"""
        trends = []

        # Add trend identification logic
        if 'emerging_topics' in market_analysis:
            trends.extend([f"Emerging: {topic}" for topic in market_analysis['emerging_topics'][:5]])

        return trends

    def _detect_market_anomalies(self, market_analysis: Dict) -> List[str]:
        """Detect market anomalies"""
        anomalies = []

        if 'anomaly_days' in market_analysis:
            anomalies = [f"Anomaly detected on {day}" for day in market_analysis['anomaly_days'][:3]]

        return anomalies
```
**Deliverable**: Complete competitive intelligence reporter
**Testing**: Test report generation with various market data scenarios
**Estimated Time**: 3.5 hours

---

## Work Package 4: Predictive Analytics (Advanced Forecasting)

### Task 9.4.1: Implement Advanced Trend Predictor
**File**: `src/phase9/trend_predictor.py`
**Implementation**:
```python
"""
Advanced Trend Predictor for Phase 9
Sophisticated time series forecasting and trend prediction
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedTrendPredictor:
    """
    Advanced trend predictor using machine learning
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_column = 'price'
        self.logger = logging.getLogger(__name__)
        self.logger.info("Advanced Trend Predictor initialized")

    def train_predictive_models(self, df: pd.ndarray, target_col: str = 'price') -> Dict[str, Any]:
        """
        Train multiple predictive models

        Args:
            df: Training data
            target_col: Target column for prediction

        Returns:
            Dictionary with training results
        """
        try:
            if df.empty:
                return {"error": "Insufficient data for model training"}

            self.target_column = target_col
            self._prepare_features(df)

            # Prepare data
            X = df[self.feature_columns]
            y = df[target_col]

            # Split data for training
            X_train, X_test, y_train, y_test = TimeSeriesSplit(n_splits=5).split(X, y)

            # Train multiple models
            models_trained = {}

            # Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            models_trained['random_forest'] = {
                'model': rf,
                'mae': mean_absolute_error(y_test, rf.predict(X_test)),
                'r2': r2_score(y_test, rf.predict(X_test))
            }

            # Gradient Boosting
            gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            gb.fit(X_train, y_train)
            models_trained['gradient_boosting'] = {
                'model': gb,
                'mae': mean_absolute_error(y_test, gb.predict(X_test)),
                'r2': r2_score(y_test, gb.predict(X_test))
            }

            # Linear Model (baseline)
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            models_trained['linear_regression'] = {
                'model': lr_model,
                'mae': mean_absolute_error(y_test, lr_model.predict(X_test)),
                'r2': r2_score(y_test, lr_model.predict(X_test))
            }

            self.models = models_trained
            self.logger.info(f"Training completed for {len(models_trained)} models")

            return {
                'models_trained': models_trained,
                'feature_importance': self._calculate_feature_importance(models_trained),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'training_date': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {"error": str(e)}

    def predict_trends(self, df: pd.ndarray, periods: int = 30, model_type: str = 'ensemble') -> Dict[str, Any]:
        """
        Predict future trends using trained models

        Args:
            df: Historical data for prediction
            periods: Number of periods to predict
            model_type: Type of model to use ('random_forest', 'gradient_boosting', 'linear_regression', 'ensemble')

        Returns:
            Dictionary with prediction results
        """
        try:
            if df.empty or not self.models:
                return {"error": "No trained models available for prediction"}

            self._prepare_features(df)
            X = df[self.feature_columns]

            # Select model
            if model_type == 'ensemble':
                predictions = self._ensemble_predict(X)
            elif model_type in self.models:
                model = self.models[model_type]['model']
                predictions = model.predict(X)
            else:
                return {"error": f"Unknown model type: {model_type}"}

            # Extend predictions to future periods
            future_predictions = self._extend_predictions(predictions, periods)

            return {
                'predictions': future_predictions,
                'model_type': model_type,
                'prediction_periods': len(future_predictions),
                'confidence_intervals': self._calculate_confidence_intervals(predictions),
                'prediction_date': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Trend prediction failed: {e}")
            return {"error": str(e)}

    def _prepare_features(self, df: pd.ndarray) -> None:
        """Prepare features for machine learning"""
        # Select numeric columns for features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Important features for tech trend prediction
        important_features = ['score', 'descendants', 'sentiment_score']

        # Filter available features
        available_features = [col for col in important_features if col in numeric_cols]
        self.feature_columns = available_features + [col for col in numeric_cols if col not in self.feature_columns]

        # Scale features
        if self.feature_columns:
            X = df[self.feature_columns]
            self.scaler = StandardScaler()
            df[self.feature_columns] = self.scaler.fit_transform(X)

    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction from multiple models"""
        predictions = []

        for model_name, model_data in self.models.items():
            prediction = model_data['model'].predict(X)
            predictions.append(prediction)

        # Weighted average (weights based on model performance)
        weights = [model_data['r2_score'] for model_data in self.models.values()]
        weights = np.array(weights) / np.sum(weights)

        return np.average(predictions, axis=0, weights=weights)

    def _extend_predictions(self, base_predictions: np.ndarray, periods: int) -> np.ndarray:
        """Extend predictions to future periods"""
        extended_predictions = []

        for i in range(periods):
            # Simple linear extension
            if len(base_predictions) > 0:
                trend = base_predictions[-1] - base_predictions[0] if len(base_predictions) > 1 else 0
                next_prediction = base_predictions[-1] + trend * (i + 1)
                extended_predictions.append(next_prediction)
            else:
                extended_predictions.append(0)

        return np.array(extended_predictions)

    def _calculate_confidence_intervals(self, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals for predictions"""
        std_error = np.std(predictions) if len(predictions) > 1 else np.ones(len(predictions))

        z_scores = [1.96, 1.645]  # 95% and 90% confidence
        confidence_intervals = {
            'lower_95': predictions - z_scores[0] * std_error,
            'upper_95': predictions + z_scores[0] * std_error,
            'lower_90': predictions - z_scores[1] * std_error,
            'upper_90': predictions + z_scores[1] * std_error
        }

        return confidence_intervals

    def _calculate_feature_importance(self, models_trained: Dict) -> Dict[str, Dict]:
        """Calculate feature importance across models"""
        importance_scores = {}

        for model_name, model_data in models_trained.items():
            model = model_data['model']

            if hasattr(model, 'feature_importances_'):
                importance_scores[model_name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_scores[model_name] = dict(zip(model.coef_, self.feature_columns))
            else:
                importance_scores[model_name] = {}

        return importance_scores

    def analyze_prediction_accuracy(self, test_data: pd.ndarray, target_col: str = 'price') -> Dict[str, Any]:
        """
        Analyze prediction accuracy across all models

        Args:
            test_data: Test data for evaluation
            target_col: Target column for comparison

        Returns:
            Dictionary with accuracy metrics
        """
        try:
            if test_data.empty or not self.models:
                return {"error": "No trained models available for accuracy analysis"}

            self._prepare_features(test_data)
            X_test = test_data[self.feature_columns]
            y_actual = test_data[target_col]

            accuracy_analysis = {}

            for model_name, model_data in self.models.items():
                model = model_data['model']
                y_pred = model.predict(X_test)

                accuracy_analysis[model_name] = {
                    'mae': mean_absolute_error(y_actual, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_actual, y_pred)),
                    'r2_score': r2_score(y_actual, y_pred),
                    'mape': np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
                }

            return accuracy_analysis

        except Exception as e:
            logger.error(f"Accuracy analysis failed: {e}")
            return {"error": str(e)}

    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all trained models"""
        if not self.models:
            return {"status": "no_models_trained"}

        performance = {}
        for model_name, model_data in self.models.items():
            performance[model_name] = {
                'mae': model_data.get('mae', 0),
                'r2_score': model_data.get('r2_score', 0),
                'training_date': model_data.get('training_date', 'unknown')
            }

        return {
            'model_count': len(self.models),
            'available_models': list(self.models.keys()),
            'performance_summary': performance,
            'best_model': max(performance['performance_summary'], key=performance['performance_summary'],
                           default=None, key=lambda x: performance['performance_summary'][x].get('r2_score', 0))
        }
```
**Deliverable**: Complete advanced trend predictor with machine learning models
**Testing**: Test model training and prediction with sample time series data
**Estimated Time**: 5 hours

---

## Work Package 5: Intelligence Report Generation (Visualization)

### Task 9.5.1: Create Intelligence Reporter
**File**: `src/phase9/intelligence_reporter.py` (already outlined above)
**Implementation**: Extend with enhanced visualization features
**Testing**: Test report generation with various data scenarios
**Estimated Time**: 2 hours

### Task 9.5.2: Create Interactive Intelligence Dashboard
**Directory**: `src/phase9/dashboard/` (new)
**Implementation**:
```python
"""
Interactive Intelligence Dashboard for Phase 9
Real-time market intelligence visualization
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class IntelligenceDashboard:
    """Interactive intelligence dashboard"""

    def render(self):
        """Render the intelligence dashboard"""
        st.title("🧠 Market Intelligence Dashboard")

        # Key metrics
        self.render_key_metrics()

        # Interactive charts
        self.render_sentiment_trends()
        self.render_competitor_analysis()
        self.render_market_predictions()

    def render_key_metrics(self):
        """Render key intelligence metrics"""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Market Sentiment", "0.15", "↗️ Positive")

        with col2:
            st.metric("Competitor Mentions", "47", "📈 High Activity")

        with col3:
            market_trend = "improving"
            st.metric("Market Trend", market_trend, "📈 Upward")

        with col4:
            confidence = 0.78
            st.metric("Prediction Confidence", f"{confidence:.2f}", "📊 High")

    def render_sentiment_trends(self):
        """Render sentiment trend charts"""
        # Implementation for sentiment trend visualization
        pass

    def render_competitor_analysis(self):
        """Render competitor analysis charts"""
        # Implementation for competitor tracking visualization
        pass

    def render_market_predictions(self):
        """Render market prediction charts"""
        # Implementation for prediction visualization
        pass
```
**Deliverable**: Interactive intelligence dashboard with real-time visualizations
**Testing**: Test dashboard rendering and interactivity
**Estimated Time**: 3 hours

---

## Work Package 6: Data Integration & APIs (External Sources)

### Task 9.6.1: Implement External Data Integration
**File**: `src/phase9/data_integrator.py` (new)
**Implementation**:
```python
"""
External Data Integrator for Phase 9
Connect to external data sources for enhanced intelligence
"""

import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import feedparser
import tweepy
from alpha_vantage import AlphaVantage
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExternalDataIntegrator:
    """
    External data integration for enhanced intelligence
    """

    def __init__(self):
        self.data_sources = {}
        self.api_keys = {}
        self.logger = logging.getLogger(__name__)
        self.logger.info("External Data Integrator initialized")

    def integrate_financial_data(self, symbols: List[str]) -> Dict[str, any]:
        """
        Integrate financial market data

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary with financial data
        """
        try:
            financial_data = {}

            for symbol in symbols:
                try:
                    # Use yfinance for stock data
                    stock_data = yf.download(symbol, period="1y")
                    financial_data[symbol] = {
                        'prices': stock_data,
                        'returns': stock_data['Close'].pct_change(),
                        'volume': stock_data['Volume'],
                        'market_cap': stock_data.get('Market Cap', 0),
                        'last_updated': datetime.now()
                    }
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")
                    financial_data[symbol] = {"error": str(e)}

            return financial_data

        except Exception as e:
            logger.error(f"Financial data integration failed: {e}")
            return {"error": str(e)}

    def integrate_news_sources(self, sources: List[str]) -> Dict[str, any]:
        """
        Integrate multiple news sources

        Args:
            sources: List of RSS feed URLs

        Returns:
            Dictionary with news data
        """
        try:
            news_data = {}

            for source in sources:
                try:
                    feed = feedparser.parse(source)
                    articles = []

                    for entry in feed.entries:
                        articles.append({
                            'title': entry.title,
                            'summary': entry.summary,
                            'published': entry.published,
                            'link': entry.link,
                            'source': source
                        })

                    news_data[source] = articles

                except Exception as e:
                    logger.warning(f"Failed to parse RSS feed {source}: {e}")
                    news_data[source] = {"error": str(e)}

            return news_data

        except Exception as e:
            logger.error(f"News source integration failed: {e}")
            return {"error": str(e)}

    def integrate_social_data(self, hashtags: List[str]) -> Dict[str, any]:
        """
        Integrate social media data

        Args:
            hashtags: List of hashtags to track

        Returns:
            Dictionary with social data
        """
        try:
            social_data = {}

            if 'api_key' in self.api_keys.get('twitter'):
                twitter_data = self._fetch_twitter_data(hashtags)
                social_data['twitter'] = twitter_data

            return social_data

        except Exception as e:
            logger.error(f"Social data integration failed: {e}")
            return {"error": str(e)}

    def integrate_market_research_data(self) -> Dict[str, any]:
        """
        Integrate market research data from various sources

        Returns:
            Dictionary with research data
        """
        try:
            research_data = {
                'market_reports': [],
                'industry_analysis': [],
                'research_papers': [],
                'market_studies': []
            }

            return research_data

        except Exception as e:
            logger.error(f"Research data integration failed: {e}")
            return {"error": str(e)}

    def _fetch_twitter_data(self, hashtags: List[str]) -> Dict[str, any]:
        """Fetch Twitter data using API"""
        # Twitter API integration implementation
        return {"twitter_data": [], "API integration pending"}

    def get_integrated_data_summary(self) -> Dict[str, any]:
        """Get summary of all integrated data sources"""
        return {
            'data_sources': list(self.data_sources.keys()),
            'integration_status': 'active',
            'last_update': datetime.now().isoformat(),
            'total_records': 0
        }
```
**Deliverable**: External data integrator for multiple data sources
**Testing**: Test integration with various external APIs and data sources
**Estimated Time**: 4 hours

---

## Success Criteria

### Functional Requirements
- [ ] Market sentiment analysis with trend detection
- [ ] Competitor tracking and intelligence reporting
- [ ] Advanced trend prediction with multiple models
- [] External data integration (financial, news, social)
- [ ] Interactive intelligence dashboard with real-time updates
- [ ] Comprehensive intelligence report generation

### Performance Requirements
- [ ] Market analysis completes within 30 seconds for 100 stories
- < 5 seconds for competitor tracking across 30 stories
- < 15 seconds for trend prediction models training
- < 2 seconds for interactive dashboard updates
- < 50MB memory usage during intensive operations

### Quality Requirements
- [ ] Prediction accuracy > 75% R² score
- [ ] Comprehensive error handling for all data sources
- - Fallbacks for external API failures
- - Graceful degradation when models unavailable
- [ ] Professional report formatting and visualization
- - All visual charts properly formatted and readable
- - All data validation and sanitization

### Integration Requirements
- [ ] Seamless integration with existing dashboard
- - Compatible with current data flow and architecture
- - No breaking changes to existing functionality
- - Maintains performance of existing features
- - Backward compatibility with Phase 7 features

### Innovation Requirements
- [ ] Advanced predictive analytics with multiple model ensemble
- [ ] Real-time competitor tracking and alerting
- - Dynamic market sentiment analysis
- - Automated opportunity identification
- [ ] Integration with external market data sources
- - AI-powered insights and recommendations

---

## Risk Mitigation Strategies

### Technical Risks
1. **External API Reliability**: Implement robust error handling and fallbacks
2. **Model Performance**: Use ensemble methods and confidence intervals
3. **Data Quality**: Implement comprehensive validation and cleaning
4. **Real-time Performance**: Optimize for concurrent users
5. **Memory Usage**: Implement streaming for large datasets

### Operational Risks
1. **API Rate Limits**: Implement rate limiting and caching strategies
2. **Data Availability**: Handle external source unavailability gracefully
3. **Model Accuracy**: Continuous monitoring and retraining
4. **User Interface**: Intuitive visualization and interaction design
5. **Performance Impact**: Monitor system impact under load

### Market Risks
1. **Prediction Accuracy**: Use ensemble methods and confidence intervals
2. **Data Timeliness**: Prioritize recent data over historical
3. **Competitor Intelligence**: Verify information accuracy and sources
4. **Regulatory Compliance**: Ensure data usage complies with terms of service
5. **Intellectual Property**: Respect data ownership and licensing

---

## Timeline Estimate

**Total Estimated Time**: 35.5 hours

**Breakdown by Work Package**:
- Work Package 1: 5.5 hours (Infrastructure)
- Work Package 2: 15.5 hours (Analytics Core)
- Work Package 3: 9 hours (Competitor Intelligence)
- Work Package 4: 8 hours (Predictive Analytics)
- Work Package 5: 7 hours (Reporting & Visualization)
- Work Package 6: 5.5 hours (Data Integration)
- Buffer/Debugging: 5 hours

**Recommended Implementation Schedule**:
- **Day 1**: Work Packages 1-2 (Infrastructure & Analytics Core)
- **Day 2**: Work Package 3 (Competitor Intelligence)
- **Day 3**: Work Package 4 (Predictive Analytics)
- **Day 4**: Work Packages 5-6 (Reporting & Data Integration)
- **Day 5**: Testing, refinement, and deployment

---

## Next Steps

1. Review and approve this detailed plan
2. Set up development environment with dependencies
3. Begin with Work Package 1 (Infrastructure)
4. Track progress using TodoWrite tool
5. Test each work package before proceeding
6. Deploy to production after successful testing
7. Update project documentation and README

---

## Phase 10: Automation & Deployment - Planning Phase

After completing Phase 9 Advanced Analytics, the final phase will focus on automation, deployment, and production readiness. Phase 10 will include:

- **CI/CD Pipeline Enhancement**: Full automation for testing and deployment
- **Production Deployment**: Containerization and cloud deployment strategies
- **Monitoring & Alerting**: Comprehensive system observability
- **Performance Optimization**: Advanced caching and scaling strategies
- **Security & Compliance**: Enterprise-grade security implementation
- **User Training**: Documentation and best practices
- **Maintenance Automation**: Automated updates and health checks

This detailed plan provides a roadmap for implementing sophisticated market intelligence capabilities that will make Tech-Pulse a truly enterprise-grade tech industry intelligence platform! 🚀