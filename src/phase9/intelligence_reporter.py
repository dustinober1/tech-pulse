"""
Intelligence Reporter for Phase 9
Generate comprehensive market intelligence reports
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
import json

# Import Phase 9 modules
try:
    from .market_analyzer import MarketIntelligenceAnalyzer
    from .competitor_tracker import CompetitorTracker
    from .trend_predictor import AdvancedTrendPredictor
except ImportError:
    # Fallback imports for testing
    MarketIntelligenceAnalyzer = None
    CompetitorTracker = None
    AdvancedTrendPredictor = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntelligenceReporter:
    """
    Generate comprehensive market intelligence reports
    """

    def __init__(self):
        self.market_analyzer = MarketIntelligenceAnalyzer() if MarketIntelligenceAnalyzer else None
        self.competitor_tracker = CompetitorTracker() if CompetitorTracker else None
        self.trend_predictor = AdvancedTrendPredictor() if AdvancedTrendPredictor else None
        self.logger = logging.getLogger(__name__)
        self.logger.info("Intelligence Reporter initialized")

    def generate_comprehensive_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive market intelligence report

        Args:
            df: DataFrame with market data

        Returns:
            Dictionary with complete intelligence report
        """
        try:
            self.logger.info("Starting comprehensive intelligence report generation")

            # Initialize report structure
            report = {
                'report_metadata': self._generate_report_metadata(df),
                'executive_summary': {},
                'market_analysis': {},
                'competitive_intelligence': {},
                'trend_predictions': {},
                'risk_assessment': {},
                'opportunities': {},
                'strategic_recommendations': [],
                'appendices': {}
            }

            # Generate each section
            report['market_analysis'] = self._generate_market_analysis(df)
            report['competitive_intelligence'] = self._generate_competitive_intelligence(df)
            report['trend_predictions'] = self._generate_trend_predictions(df)
            report['executive_summary'] = self._generate_executive_summary(report)
            report['risk_assessment'] = self._generate_risk_assessment(report)
            report['opportunities'] = self._generate_opportunities(report)
            report['strategic_recommendations'] = self._generate_strategic_recommendations(report)
            report['appendices'] = self._generate_appendices(df, report)

            self.logger.info("Comprehensive intelligence report generated successfully")
            return report

        except Exception as e:
            logger.error(f"Comprehensive report generation failed: {e}")
            return {"error": str(e), "report_metadata": {"status": "failed", "error": str(e)}}

    def _generate_market_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate market analysis section"""
        market_analysis = {
            'sentiment_analysis': {},
            'technology_trends': {},
            'market_volatility': {},
            'key_metrics': {}
        }

        if self.market_analyzer:
            try:
                # Sentiment analysis
                sentiment_data = self.market_analyzer.analyze_market_sentiment(df)
                market_analysis['sentiment_analysis'] = sentiment_data

                # Technology trends
                trend_data = self.market_analyzer.analyze_technology_adoption_trends(df)
                market_analysis['technology_trends'] = trend_data

                # Market volatility
                volatility_data = self.market_analyzer.analyze_volatility(df)
                market_analysis['market_volatility'] = volatility_data

            except Exception as e:
                self.logger.warning(f"Market analysis partially failed: {e}")
                market_analysis['error'] = str(e)

        # Calculate key metrics
        market_analysis['key_metrics'] = self._calculate_key_market_metrics(df)

        return market_analysis

    def _generate_competitive_intelligence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate competitive intelligence section"""
        competitive_intelligence = {
            'competitor_landscape': {},
            'competitive_dynamics': {},
            'market_positioning': {},
            'threat_assessment': {}
        }

        if self.competitor_tracker:
            try:
                # Competitor tracking
                tracking_data = self.competitor_tracker.track_competitors(df)
                competitive_intelligence['competitor_landscape'] = tracking_data

                # Competitive intelligence report
                intelligence_data = self.competitor_tracker.generate_competitive_intelligence_report(df)
                competitive_intelligence['competitive_dynamics'] = intelligence_data

            except Exception as e:
                self.logger.warning(f"Competitive intelligence partially failed: {e}")
                competitive_intelligence['error'] = str(e)

        return competitive_intelligence

    def _generate_trend_predictions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate trend predictions section"""
        trend_predictions = {
            'market_forecast': {},
            'topic_trends': {},
            'prediction_accuracy': {},
            'confidence_levels': {}
        }

        if self.trend_predictor:
            try:
                # Train models if needed
                if not self.trend_predictor.models:
                    self.trend_predictor.train_predictive_models(df)

                # Market forecast
                forecast_data = self.trend_predictor.predict_trends(df, periods=30)
                trend_predictions['market_forecast'] = forecast_data

                # Topic trends
                topic_data = self.trend_predictor.predict_topic_trends(df, periods=30)
                trend_predictions['topic_trends'] = topic_data

                # Comprehensive trend report
                trend_report = self.trend_predictor.generate_trend_forecast_report(df, periods=30)
                trend_predictions.update(trend_report)

            except Exception as e:
                self.logger.warning(f"Trend predictions partially failed: {e}")
                trend_predictions['error'] = str(e)

        return trend_predictions

    def _generate_executive_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary"""
        executive_summary = {
            'key_findings': [],
            'critical_insights': [],
            'strategic_overview': '',
            'action_items': []
        }

        # Extract key findings from each section
        try:
            # Market findings
            market_analysis = report.get('market_analysis', {})
            if market_analysis.get('sentiment_analysis', {}).get('overall_sentiment'):
                sentiment = market_analysis['sentiment_analysis']['overall_sentiment']
                if sentiment > 0.1:
                    executive_summary['key_findings'].append("Market sentiment is predominantly positive")
                elif sentiment < -0.1:
                    executive_summary['key_findings'].append("Market sentiment shows notable concern")

            # Competitive findings
            competitive_intel = report.get('competitive_intelligence', {})
            if competitive_intel.get('competitor_landscape', {}).get('top_competitors'):
                top_competitors = competitive_intel['competitor_landscape']['top_competitors']
                executive_summary['key_findings'].append(f"Identified {len(top_competitors)} key competitors")

            # Trend findings
            trend_predictions = report.get('trend_predictions', {})
            if trend_predictions.get('topic_trends', {}).get('emerging_topics'):
                emerging = trend_predictions['topic_trends']['emerging_topics'][:3]
                executive_summary['key_findings'].append(f"Emerging trends: {', '.join(emerging)}")

            # Generate strategic overview
            executive_summary['strategic_overview'] = self._generate_strategic_overview(report)

            # Generate action items
            executive_summary['action_items'] = self._generate_action_items(report)

        except Exception as e:
            self.logger.warning(f"Executive summary generation partially failed: {e}")
            executive_summary['error'] = str(e)

        return executive_summary

    def _generate_risk_assessment(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk assessment section"""
        risk_assessment = {
            'market_risks': [],
            'competitive_risks': [],
            'technological_risks': [],
            'risk_mitigation': [],
            'risk_matrix': {}
        }

        try:
            # Market risks
            market_analysis = report.get('market_analysis', {})
            if market_analysis.get('market_volatility', {}).get('volatility_index', 0) > 0.3:
                risk_assessment['market_risks'].append("High market volatility detected")

            # Competitive risks
            competitive_intel = report.get('competitive_intelligence', {})
            if competitive_intel.get('competitor_landscape', {}).get('competitive_density', 0) > 0.4:
                risk_assessment['competitive_risks'].append("High competitive density increasing pressure")

            # Technological risks
            trend_predictions = report.get('trend_predictions', {})
            if trend_predictions.get('confidence_levels', {}).get('overall_confidence') == 'low':
                risk_assessment['technological_risks'].append("Low prediction confidence indicates high uncertainty")

            # Generate risk mitigation strategies
            risk_assessment['risk_mitigation'] = self._generate_risk_mitigation_strategies(risk_assessment)

        except Exception as e:
            self.logger.warning(f"Risk assessment generation partially failed: {e}")
            risk_assessment['error'] = str(e)

        return risk_assessment

    def _generate_opportunities(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate opportunities section"""
        opportunities = {
            'market_opportunities': [],
            'technology_opportunities': [],
            'competitive_opportunities': [],
            'strategic_opportunities': []
        }

        try:
            # Market opportunities
            market_analysis = report.get('market_analysis', {})
            if market_analysis.get('sentiment_analysis', {}).get('sentiment_trend') == 'improving':
                opportunities['market_opportunities'].append("Improving market sentiment indicates growth potential")

            # Technology opportunities
            trend_predictions = report.get('trend_predictions', {})
            if trend_predictions.get('topic_trends', {}).get('emerging_topics'):
                emerging = trend_predictions['topic_trends']['emerging_topics']
                for topic in emerging[:5]:
                    opportunities['technology_opportunities'].append(f"Growing interest in {topic}")

            # Competitive opportunities
            competitive_intel = report.get('competitive_intelligence', {})
            if competitive_intel.get('competitive_intelligence', {}).get('opportunity_analysis'):
                opp_analysis = competitive_intel['competitive_intelligence']['opportunity_analysis']
                opportunities['competitive_opportunities'].extend(opp_analysis)

            # Strategic opportunities
            opportunities['strategic_opportunities'] = self._identify_strategic_opportunities(report)

        except Exception as e:
            self.logger.warning(f"Opportunities generation partially failed: {e}")
            opportunities['error'] = str(e)

        return opportunities

    def _generate_strategic_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate strategic recommendations"""
        recommendations = []

        try:
            # Market-based recommendations
            market_analysis = report.get('market_analysis', {})
            sentiment = market_analysis.get('sentiment_analysis', {}).get('overall_sentiment', 0)
            if sentiment > 0.2:
                recommendations.append("Consider aggressive growth strategies in positive market conditions")
            elif sentiment < -0.2:
                recommendations.append("Implement defensive strategies and focus on core competencies")

            # Competitive-based recommendations
            competitive_intel = report.get('competitive_intelligence', {})
            density = competitive_intel.get('competitor_landscape', {}).get('competitive_density', 0)
            if density > 0.3:
                recommendations.append("Differentiate offerings to stand out in crowded market")

            # Trend-based recommendations
            trend_predictions = report.get('trend_predictions', {})
            emerging = trend_predictions.get('topic_trends', {}).get('emerging_topics', [])
            if emerging:
                recommendations.append(f"Invest in emerging technologies: {', '.join(emerging[:3])}")

            # Risk-based recommendations
            risk_assessment = report.get('risk_assessment', {})
            if risk_assessment.get('market_risks'):
                recommendations.append("Implement robust risk management strategies")

        except Exception as e:
            self.logger.warning(f"Strategic recommendations generation partially failed: {e}")
            recommendations.append("Error generating some recommendations")

        return recommendations

    def _generate_appendices(self, df: pd.DataFrame, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appendices section"""
        appendices = {
            'data_sources': ["Hacker News API", "Real-time feeds", "Historical data"],
            'methodology': "Advanced analytics with statistical modeling and machine learning",
            'limitations': "Based on available news data sources and time constraints",
            'technical_details': {},
            'data_quality_assessment': {}
        }

        # Add technical details
        appendices['technical_details'] = {
            'analysis_date': datetime.now().isoformat(),
            'data_points_analyzed': len(df),
            'time_period': f"{df['time'].min()} to {df['time'].max()}" if 'time' in df.columns else "Unknown",
            'models_used': list(self.trend_predictor.models.keys()) if self.trend_predictor else [],
            'confidence_level': 'Medium-High'
        }

        # Add data quality assessment
        appendices['data_quality_assessment'] = {
            'completeness': f"{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%",
            'freshness': 'Recent' if len(df) > 0 else 'No data',
            'coverage': 'Technology sector focus'
        }

        return appendices

    def _calculate_key_market_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate key market metrics"""
        metrics = {}

        if not df.empty:
            metrics['total_stories'] = len(df)
            metrics['avg_engagement'] = df['score'].mean() if 'score' in df.columns else 0
            metrics['total_comments'] = df['descendants'].sum() if 'descendants' in df.columns else 0
            metrics['unique_topics'] = df['topic_keyword'].nunique() if 'topic_keyword' in df.columns else 0
            metrics['sentiment_distribution'] = df['sentiment_label'].value_counts().to_dict() if 'sentiment_label' in df.columns else {}

            # Time-based metrics
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                metrics['date_range'] = f"{df['time'].min().date()} to {df['time'].max().date()}"
                metrics['stories_per_day'] = len(df) / ((df['time'].max() - df['time'].min()).days + 1)

        return metrics

    def _generate_strategic_overview(self, report: Dict[str, Any]) -> str:
        """Generate strategic overview"""
        overview_parts = []

        # Market overview
        market_analysis = report.get('market_analysis', {}).get('key_metrics', {})
        if market_analysis.get('total_stories'):
            overview_parts.append(f"Analysis of {market_analysis['total_stories']} market data points")

        # Competitive overview
        competitive_intel = report.get('competitive_intelligence', {}).get('competitor_landscape', {})
        if competitive_intel.get('top_competitors'):
            overview_parts.append(f"Competitive landscape includes {len(competitive_intel['top_competitors'])} key players")

        # Trend overview
        trend_predictions = report.get('trend_predictions', {})
        if trend_predictions.get('topic_trends', {}).get('emerging_topics'):
            overview_parts.append("Multiple emerging technology trends identified")

        return ". ".join(overview_parts) if overview_parts else "Strategic overview generation pending"

    def _generate_action_items(self, report: Dict[str, Any]) -> List[str]:
        """Generate actionable items"""
        action_items = []

        # Based on market analysis
        market_analysis = report.get('market_analysis', {})
        if market_analysis.get('technology_trends', {}).get('emerging_topics'):
            action_items.append("Monitor emerging technologies for investment opportunities")

        # Based on competitive intelligence
        competitive_intel = report.get('competitive_intelligence', {})
        if competitive_intel.get('competitive_dynamics', {}).get('threat_assessment'):
            action_items.append("Review competitive threat assessment and develop response strategies")

        # Based on trend predictions
        trend_predictions = report.get('trend_predictions', {})
        if trend_predictions.get('market_forecast', {}).get('predictions'):
            action_items.append("Align strategic planning with market forecast predictions")

        return action_items

    def _generate_risk_mitigation_strategies(self, risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate risk mitigation strategies"""
        mitigation_strategies = []

        if risk_assessment.get('market_risks'):
            mitigation_strategies.append("Implement diversified portfolio strategy to mitigate market volatility")

        if risk_assessment.get('competitive_risks'):
            mitigation_strategies.append("Focus on differentiation and innovation to maintain competitive advantage")

        if risk_assessment.get('technological_risks'):
            mitigation_strategies.append("Invest in R&D and talent development to address technological uncertainty")

        return mitigation_strategies

    def _identify_strategic_opportunities(self, report: Dict[str, Any]) -> List[str]:
        """Identify strategic opportunities"""
        strategic_opps = []

        # Look for market gaps
        market_analysis = report.get('market_analysis', {}).get('technology_trends', {})
        declining_topics = market_analysis.get('declining_topics', [])
        if declining_topics:
            strategic_opps.append(f"Consider acquisitions in declining areas: {', '.join(declining_topics[:2])}")

        # Look for partnership opportunities
        competitive_intel = report.get('competitive_intelligence', {})
        if competitive_intel.get('competitive_dynamics', {}).get('partnership_opportunities'):
            strategic_opps.append("Explore strategic partnerships to expand market reach")

        return strategic_opps

    def _generate_report_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate report metadata"""
        return {
            'report_type': 'Comprehensive Market Intelligence Report',
            'generation_date': datetime.now().isoformat(),
            'data_as_of': df['time'].max().isoformat() if 'time' in df.columns and not df.empty else None,
            'data_points': len(df),
            'version': '1.0',
            'generated_by': 'Tech-Pulse Intelligence System',
            'status': 'complete'
        }

    def export_report_to_dict(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Export report in dictionary format (for JSON serialization)"""
        try:
            # Convert any non-serializable objects to serializable formats
            serializable_report = self._make_serializable(report)
            return serializable_report
        except Exception as e:
            self.logger.error(f"Report export failed: {e}")
            return {"error": str(e)}

    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        else:
            return obj

    def generate_quick_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a quick summary for dashboard display"""
        try:
            summary = {
                'market_sentiment': 'neutral',
                'key_metrics': {},
                'top_topics': [],
                'competitive_alerts': []
            }

            # Quick sentiment analysis
            if 'sentiment_label' in df.columns:
                sentiment_dist = df['sentiment_label'].value_counts()
                if sentiment_dist.idxmax() == 'Positive':
                    summary['market_sentiment'] = 'positive'
                elif sentiment_dist.idxmax() == 'Negative':
                    summary['market_sentiment'] = 'negative'

            # Key metrics
            summary['key_metrics'] = {
                'total_stories': len(df),
                'avg_engagement': df['score'].mean() if 'score' in df.columns else 0,
                'topic_diversity': df['topic_keyword'].nunique() if 'topic_keyword' in df.columns else 0
            }

            # Top topics
            if 'topic_keyword' in df.columns:
                top_topics = df['topic_keyword'].value_counts().head(5).index.tolist()
                summary['top_topics'] = top_topics

            # Competitive alerts
            if self.competitor_tracker:
                competitor_data = self.competitor_tracker.track_competitors(df)
                if competitor_data.get('top_competitors'):
                    summary['competitive_alerts'] = [c['company_name'] for c in competitor_data['top_competitors'][:3]]

            return summary

        except Exception as e:
            self.logger.error(f"Quick summary generation failed: {e}")
            return {"error": str(e)}