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
from itertools import groupby
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

    def extract_competitor_profiles(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Extract detailed competitor profiles

        Returns:
            List of competitor profile dictionaries
        """
        profiles = []

        # Implement competitor profile extraction
        # This would involve external data sources like Crunchbase, PitchBook, etc.

        return profiles

    def analyze_competitive_landscape(self, df: pd.DataFrame) -> Dict[str, Any]:
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
        if not mentions:
            return []

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

        # Group by company
        for company, company_mentions in groupby(mention['company_name'] for mention in mentions):
            if company_mentions:
                company_mention_list = [m for m in mentions if m['company_name'] == company]
                sentiment_scores[company] = {
                    'avg_sentiment': np.mean([m.get('sentiment_score', 0) for m in company_mention_list]),
                    'sentiment_distribution': Counter(m.get('sentiment_label', 'Neutral') for m in company_mention_list),
                    'sentiment_volatility': np.std([m.get('sentiment_score', 0) for m in company_mention_list]) if len(company_mention_list) > 1 else 0
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

    def generate_competitive_intelligence_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive competitive intelligence report

        Args:
            df: DataFrame with market data

        Returns:
            Dictionary with competitive intelligence report
        """
        try:
            # Get competitor tracking data
            tracking_data = self.track_competitors(df)

            # Generate report sections
            report = {
                'executive_summary': self._generate_executive_summary(tracking_data),
                'competitor_landscape': tracking_data.get('top_competitors', []),
                'competitive_dynamics': self._analyze_competitive_dynamics(tracking_data),
                'threat_assessment': self._assess_competitive_threats(tracking_data),
                'opportunity_analysis': self._identify_competitive_opportunities(tracking_data),
                'recommendations': self._generate_competitive_recommendations(tracking_data),
                'market_insights': self._extract_market_insights(tracking_data)
            }

            return report

        except Exception as e:
            logger.error(f"Competitive intelligence report generation failed: {e}")
            return {"error": str(e)}

    def _generate_executive_summary(self, tracking_data: Dict[str, Any]) -> str:
        """Generate executive summary of competitive landscape"""
        summary_parts = []

        top_competitors = tracking_data.get('top_competitors', [])
        if top_competitors:
            summary_parts.append(f"Identified {len(top_competitors)} key competitors in the market")

        density = tracking_data.get('competitive_density', 0)
        if density > 0.3:
            summary_parts.append("High competitive density detected")
        elif density > 0.1:
            summary_parts.append("Moderate competitive density")
        else:
            summary_parts.append("Low competitive density")

        mention_trends = tracking_data.get('mention_trends', {})
        increasing_companies = [c for c, t in mention_trends.items() if t.get('trend') == 'increasing']
        if increasing_companies:
            summary_parts.append(f"{len(increasing_companies)} competitors showing increased activity")

        return ". ".join(summary_parts) if summary_parts else "Executive summary generation pending"

    def _analyze_competitive_dynamics(self, tracking_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze competitive dynamics"""
        dynamics = {
            'market_leader': None,
            'challenger_companies': [],
            'niche_players': [],
            'emerging_threats': [],
            'competitive_pressure': 'moderate'
        }

        top_competitors = tracking_data.get('top_competitors', [])
        if top_competitors:
            dynamics['market_leader'] = top_competitors[0]['company_name']
            dynamics['challenger_companies'] = [c['company_name'] for c in top_competitors[1:4]]

        return dynamics

    def _assess_competitive_threats(self, tracking_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Assess competitive threats"""
        threats = []

        mention_trends = tracking_data.get('mention_trends', {})
        for company, trend_data in mention_trends.items():
            if trend_data.get('trend') == 'increasing' and trend_data.get('avg_engagement', 0) > 50:
                threats.append({
                    'company': company,
                    'threat_level': 'high',
                    'reason': 'High mention frequency and engagement growth'
                })

        return threats

    def _identify_competitive_opportunities(self, tracking_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify competitive opportunities"""
        opportunities = []

        sentiment_analysis = tracking_data.get('sentiment_analysis', {})
        for company, sentiment_data in sentiment_analysis.items():
            avg_sentiment = sentiment_data.get('avg_sentiment', 0)
            if avg_sentiment < -0.2:  # Negative sentiment might indicate opportunity
                opportunities.append({
                    'opportunity_type': 'competitor_weakness',
                    'target_company': company,
                    'description': f'Negative sentiment ({avg_sentiment:.2f}) suggests potential weakness'
                })

        return opportunities

    def _generate_competitive_recommendations(self, tracking_data: Dict[str, Any]) -> List[str]:
        """Generate strategic recommendations based on competitive analysis"""
        recommendations = []

        competitive_density = tracking_data.get('competitive_density', 0)
        if competitive_density > 0.3:
            recommendations.append("Monitor competitor activities closely due to high market density")

        top_competitors = tracking_data.get('top_competitors', [])
        if top_competitors and len(top_competitors) > 5:
            recommendations.append("Consider differentiation strategy to stand out in crowded market")

        mention_trends = tracking_data.get('mention_trends', {})
        momentum_companies = [c for c, t in mention_trends.items() if t.get('momentum_score', 0) > 10]
        if momentum_companies:
            recommendations.append(f"Watch {', '.join(momentum_companies[:3])} for strategic moves")

        return recommendations

    def _extract_market_insights(self, tracking_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key market insights from competitive data"""
        insights = {
            'total_competitors_tracked': len(tracking_data.get('top_competitors', [])),
            'average_mention_frequency': 0,
            'most_active_competitor': None,
            'competitive_trends': []
        }

        top_competitors = tracking_data.get('top_competitors', [])
        if top_competitors:
            mention_counts = [c['mention_count'] for c in top_competitors]
            insights['average_mention_frequency'] = np.mean(mention_counts)
            insights['most_active_competitor'] = max(top_competitors, key=lambda x: x['mention_count'])['company_name']

        return insights

    def monitor_competitor_activity(self, competitor_name: str, days: int = 7) -> Dict[str, Any]:
        """
        Monitor specific competitor activity over recent period

        Args:
            competitor_name: Name of competitor to monitor
            days: Number of days to look back

        Returns:
            Dictionary with competitor activity data
        """
        try:
            # Get recent data
            recent_data = fetch_hn_data(limit=200)

            if recent_data.empty:
                return {"error": "No recent data available"}

            # Filter for competitor mentions
            competitor_mentions = []
            for _, row in recent_data.iterrows():
                text = f"{row.get('title', '')} {row.get('text', '')}"
                if competitor_name.lower() in text.lower():
                    competitor_mentions.append({
                        'title': row.get('title', ''),
                        'url': row.get('url', ''),
                        'score': row.get('score', 0),
                        'time': row.get('time', datetime.now()),
                        'sentiment': row.get('sentiment_label', 'Neutral')
                    })

            # Analyze recent activity
            activity_analysis = {
                'competitor_name': competitor_name,
                'period_days': days,
                'total_mentions': len(competitor_mentions),
                'average_engagement': np.mean([m['score'] for m in competitor_mentions]) if competitor_mentions else 0,
                'sentiment_breakdown': Counter(m['sentiment'] for m in competitor_mentions),
                'activity_trend': self._calculate_activity_trend(competitor_mentions),
                'key_mention_types': self._analyze_mention_types(competitor_mentions),
                'recent_highlights': competitor_mentions[:5]  # Top 5 recent mentions
            }

            return activity_analysis

        except Exception as e:
            logger.error(f"Competitor monitoring failed: {e}")
            return {"error": str(e)}

    def _calculate_activity_trend(self, mentions: List[Dict[str, Any]]) -> str:
        """Calculate activity trend for competitor"""
        if len(mentions) < 3:
            return "insufficient_data"

        # Sort by time
        mentions.sort(key=lambda x: x['time'])

        # Calculate trend in scores
        scores = [m['score'] for m in mentions[-10:]]  # Last 10 mentions
        if len(scores) >= 3:
            if scores[-1] > scores[-2] > scores[0]:
                return "increasing"
            elif scores[-1] < scores[-2] < scores[0]:
                return "decreasing"

        return "stable"

    def _analyze_mention_types(self, mentions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze types of mentions (launch, update, competition, etc.)"""
        mention_types = {
            'product_launch': 0,
            'feature_update': 0,
            'competitive': 0,
            'partnership': 0,
            'other': 0
        }

        for mention in mentions:
            title = mention.get('title', '').lower()
            text = f"{title} {mention.get('text', '')}".lower()

            if any(keyword in text for keyword in ['launch', 'release', 'unveiled', 'announced']):
                mention_types['product_launch'] += 1
            elif any(keyword in text for keyword in ['update', 'upgrade', 'improved', 'enhanced']):
                mention_types['feature_update'] += 1
            elif any(keyword in text for keyword in ['vs', 'compete', 'rival', 'challenge']):
                mention_types['competitive'] += 1
            elif any(keyword in text for keyword in ['partner', 'collaborate', 'joint']):
                mention_types['partnership'] += 1
            else:
                mention_types['other'] += 1

        return mention_types