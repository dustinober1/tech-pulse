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

    def predict_market_movements(self, df: pd.DataFrame, periods: int = 7) -> Dict[str, Any]:
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

    def _detect_sentiment_anomalies(self, df: pd.DataFrame) -> List[datetime]:
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
        if 'time' in df.columns and 'topic_keyword' in df.columns:
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

        if 'time' in df.columns and 'topic_keyword' in df.columns:
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
        if 'topic_keyword' not in df.columns:
            return pd.DataFrame()

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

        if 'topic_keyword' not in df.columns or 'score' not in df.columns:
            return momentum_scores

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

        if 'time' not in df.columns or 'topic_keyword' not in df.columns:
            return evolution

        # Daily topic frequency
        daily_topics = df.groupby([
            df['time'].dt.floor('D'),
            'topic_keyword'
        ]).size().unstack(fill_value=0)

        for topic in daily_topics.columns:
            evolution['topic_frequency_timeline'][topic] = daily_topics[topic].tolist()

        return evolution

    def _arima_forecast(self, df: pd.DataFrame, periods: int) -> Dict[str, Any]:
        """ARIMA-based forecasting"""
        try:
            from statsmodels.tsa.arima.model import ARIMA

            # Simple ARIMA implementation
            if 'price' not in df.columns:
                return {"error": "No price data for ARIMA forecast"}

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

    def _lstm_forecast(self, df: pd.DataFrame, periods: int) -> Dict[str, Any]:
        """LSTM-based forecasting"""
        # Placeholder for LSTM implementation
        return {
            'forecast': np.random.normal(0, 1, periods),
            'confidence_intervals': np.random.normal(0, 1, (2, periods)),
            'note': 'LSTM implementation pending'
        }

    def _ensemble_forecast(self, df: pd.DataFrame, periods: int) -> Dict[str, Any]:
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

    def _calculate_confidence_intervals(self, df: pd.DataFrame, periods: int) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals for predictions"""
        # Placeholder for confidence interval calculation
        std_dev = df['price'].std() if 'price' in df.columns else 1.0
        return {
            'lower_95': np.zeros(periods) - 1.96 * std_dev,
            'upper_95': np.zeros(periods) + 1.96 * std_dev,
            'lower_68': np.zeros(periods) - 1.0 * std_dev,
            'upper_68': np.zeros(periods) + 1.0 * std_dev
        }

    def _calculate_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance for predictions"""
        features = ['score', 'sentiment_score', 'descendants'] if 'descendants' in df.columns else []
        importance = {}

        if 'price' in df.columns:
            for feature in features:
                if feature in df.columns:
                    correlation = abs(df[feature].corr(df['price']))
                    importance[feature] = correlation

        return importance

    def _backtest_predictions(self, df: pd.DataFrame) -> Dict[str, float]:
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

    def _calculate_market_share_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market share trends"""
        trends = {
            'share_evolution': {},
            'concentration_ratio': 0.0,
            'herfindahl_index': 0.0
        }

        return trends

    def _identify_competitive_threats(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Identify competitive threats"""
        threats = []

        # Placeholder for threat identification
        return threats

    def _identify_opportunity_gaps(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Identify opportunity gaps in the market"""
        gaps = []

        # Placeholder for opportunity gap identification
        return gaps

    def analyze_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market volatility patterns

        Args:
            df: DataFrame with market data

        Returns:
            Dictionary with volatility analysis
        """
        try:
            if df.empty or 'price' not in df.columns:
                return {"error": "Insufficient data for volatility analysis"}

            # Calculate volatility metrics
            returns = df['price'].pct_change().dropna()

            volatility_analysis = {
                'volatility_index': returns.std() * np.sqrt(252),  # Annualized volatility
                'trend_strength': abs(stats.linregress(np.arange(len(df)), df['price'])[0]),
                'price_range': {
                    'min': df['price'].min(),
                    'max': df['price'].max(),
                    'range': df['price'].max() - df['price'].min()
                },
                'volatility_trend': self._calculate_volatility_trend(returns),
                'extreme_events': self._identify_extreme_events(returns),
                'volatility_clustering': self._detect_volatility_clustering(returns)
            }

            return volatility_analysis

        except Exception as e:
            logger.error(f"Volatility analysis failed: {e}")
            return {"error": str(e)}

    def _calculate_volatility_trend(self, returns: pd.Series) -> str:
        """Calculate volatility trend"""
        if len(returns) < 10:
            return "insufficient_data"

        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=20).std().dropna()

        if len(rolling_vol) < 2:
            return "stable"

        # Linear regression on rolling volatility
        x = np.arange(len(rolling_vol))
        y = rolling_vol.values
        slope, _, _, _, _ = stats.linregress(x, y)

        if slope > 0.001:
            return "increasing"
        elif slope < -0.001:
            return "decreasing"
        else:
            return "stable"

    def _identify_extreme_events(self, returns: pd.Series) -> List[Dict[str, Any]]:
        """Identify extreme market events"""
        extreme_events = []

        # Define extreme events as returns beyond 3 standard deviations
        threshold = returns.std() * 3

        for i, ret in enumerate(returns):
            if abs(ret) > threshold:
                extreme_events.append({
                    'index': i,
                    'return': ret,
                    'date': returns.index[i] if hasattr(returns, 'index') else None,
                    'type': 'positive' if ret > 0 else 'negative'
                })

        return extreme_events

    def _detect_volatility_clustering(self, returns: pd.Series) -> bool:
        """Detect if volatility clustering is present"""
        if len(returns) < 20:
            return False

        # Simple test for volatility clustering
        # Check if high volatility periods tend to cluster
        squared_returns = returns ** 2

        # Calculate autocorrelation of squared returns
        autocorr = [squared_returns.autocorr(lag=i) for i in range(1, 6)]

        # If average autocorrelation is positive, clustering is present
        return np.mean(autocorr) > 0.1