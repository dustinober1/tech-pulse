"""
AI-powered summarization for executive briefings
"""

import pandas as pd
import os
from typing import List, Dict, Optional
import logging
from datetime import datetime
import re

# Try to import OpenAI, but provide graceful fallback
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI library not available. Will use rule-based summaries only.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AISummarizer:
    """
    Generates AI-powered summaries of tech trends
    Fallback to rule-based summaries if OpenAI unavailable
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = None

        if self.api_key and OPENAI_AVAILABLE:
            try:
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            logger.info("AI summarizer initialized in rule-based mode")

    def generate_executive_summary(self, df: pd.DataFrame, topics: Dict) -> str:
        """
        Generate an executive summary of current tech trends

        Args:
            df: DataFrame with story data
            topics: Dictionary with topic analysis

        Returns:
            String containing the executive summary
        """
        try:
            if self.client:
                return self._generate_ai_summary(df, topics)
            else:
                return self._generate_rule_based_summary(df, topics)
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return self._generate_fallback_summary()

    def _generate_ai_summary(self, df: pd.DataFrame, topics: Dict) -> str:
        """Generate summary using OpenAI GPT"""
        try:
            if not self.client:
                raise ValueError("OpenAI client not available")

            # Extract key insights
            top_stories = df.nlargest(5, 'score')['title'].tolist() if not df.empty else []
            sentiment_dist = df['sentiment_label'].value_counts().to_dict() if not df.empty else {}
            top_topic = max(topics.items(), key=lambda x: x[1]) if topics else ("General", 0)

            # Calculate engagement metrics
            avg_score = df['score'].mean() if not df.empty else 0
            total_comments = df['descendants'].sum() if not df.empty else 0

            # Time-based analysis
            if 'time' in df.columns and not df.empty:
                most_active_hour = df['time'].dt.hour.mode().iloc[0] if len(df) > 0 else 0
            else:
                most_active_hour = 0

            prompt = f"""
            Generate a concise executive summary (150-200 words) of today's tech trends based on:

            Top Stories: {', '.join(top_stories[:3])}
            Sentiment Distribution: {sentiment_dist}
            Dominant Topic: {top_topic[0]} ({top_topic[1]:.1f}% of stories)
            Average Engagement: {avg_score:.1f}
            Total Comments: {total_comments:,}
            Most Active Hour: {most_active_hour:00}:00

            Focus on:
            1. Major trends emerging in the tech landscape
            2. Sentiment patterns in the tech community
            3. Key technologies, companies, or themes mentioned
            4. Overall market mood and engagement levels
            5. Notable patterns or anomalies in today's data

            Style: Professional, executive-level briefing suitable for C-suite readers
            Avoid jargon where possible, focus on actionable insights
            """

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert tech analyst writing executive summaries for senior technology leaders."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=350,
                temperature=0.7
            )

            summary = response.choices[0].message.content.strip()
            logger.info(f"Generated AI summary: {len(summary)} characters")
            return summary

        except Exception as e:
            logger.error(f"AI summary generation failed: {e}")
            return self._generate_rule_based_summary(df, topics)

    def _generate_rule_based_summary(self, df: pd.DataFrame, topics: Dict) -> str:
        """Generate summary using rule-based approach"""
        try:
            # Extract insights
            total_stories = len(df)
            avg_score = df['score'].mean() if not df.empty else 0
            top_topic = max(topics.items(), key=lambda x: x[1]) if topics else ("Technology", 0)

            # Sentiment analysis
            if not df.empty and 'sentiment_label' in df.columns:
                positive_sentiment = df[df['sentiment_label'] == 'Positive'].shape[0]
                negative_sentiment = df[df['sentiment_label'] == 'Negative'].shape[0]
                neutral_sentiment = df[df['sentiment_label'] == 'Neutral'].shape[0]
                sentiment_ratio = positive_sentiment / total_stories if total_stories > 0 else 0
            else:
                positive_sentiment = negative_sentiment = neutral_sentiment = 0
                sentiment_ratio = 0

            # Engagement analysis
            total_comments = df['descendants'].sum() if not df.empty and 'descendants' in df.columns else 0

            # Rule-based summary templates
            summaries = []

            # Trend analysis
            if top_topic[1] > 30:
                topic_name = top_topic[0].replace('_', ' ').title()
                summaries.append(f"Today's tech news is dominated by {topic_name}, accounting for {top_topic[1]:.1f}% of coverage.")
            elif top_topic[1] > 20:
                topic_name = top_topic[0].replace('_', ' ').title()
                summaries.append(f"Significant focus on {topic_name} in today's tech landscape, representing {top_topic[1]:.1f}% of stories.")

            # Sentiment analysis
            if sentiment_ratio > 0.6:
                summaries.append("The tech sentiment is predominantly positive, indicating optimistic market conditions and favorable community response to developments.")
            elif negative_sentiment > positive_sentiment * 2:
                summaries.append("There's notable concern in the tech community today, with several critical issues and challenges being discussed.")
            elif neutral_sentiment > total_stories * 0.5:
                summaries.append("Tech coverage appears balanced and objective, reflecting diverse perspectives across different sectors and neutral reporting.")

            # Engagement analysis
            if avg_score > 300:
                summaries.append("Exceptional engagement levels suggest significant interest and importance in today's tech developments.")
            elif avg_score > 150:
                summaries.append("Strong engagement indicates substantial interest in ongoing tech conversations and announcements.")
            elif avg_score > 50:
                summaries.append("Moderate engagement suggests steady interest in tech industry developments.")

            # Story count insight
            if total_stories >= 30:
                summaries.append(f"With {total_stories} major stories tracked, there's substantial activity across the tech landscape with diverse topics gaining attention.")
            elif total_stories >= 15:
                summaries.append(f"Coverage of {total_stories} significant stories indicates healthy activity across multiple tech sectors.")
            elif total_stories > 0:
                summaries.append(f"Tracking {total_stories} key stories provides focused insights into the most important tech developments.")

            # Comment activity
            if total_comments > 1000:
                summaries.append(f"High community engagement with {total_comments:,} total comments indicates passionate discussion around tech topics.")
            elif total_comments > 500:
                summaries.append(f"Active community participation with {total_comments:,} comments shows strong interest in tech discussions.")

            # Time-based patterns
            if not df.empty and 'time' in df.columns:
                try:
                    most_active_hour = df['time'].dt.hour.mode().iloc[0] if len(df) > 0 else 0
                    if 9 <= most_active_hour <= 17:
                        summaries.append(f"Peak activity during business hours suggests professional audience engagement around {most_active_hour:00}:00.")
                    elif most_active_hour >= 18:
                        summaries.append(f"Evening peak activity around {most_active_hour:00}:00 indicates after-hours tech community engagement.")
                except:
                    pass  # Skip time analysis if there are issues

            # Combine and format summary
            if not summaries:
                summaries.append("Tech landscape analysis shows ongoing developments across various sectors with community engagement tracking key trends.")

            final_summary = " ".join(summaries)

            # Ensure appropriate length
            if len(final_summary) < 100:
                final_summary += " This analysis provides insights into current technology trends, community sentiment, and engagement patterns across the tech ecosystem."

            return final_summary

        except Exception as e:
            logger.error(f"Rule-based summary generation failed: {e}")
            return self._generate_fallback_summary()

    def _generate_fallback_summary(self) -> str:
        """Generate a basic fallback summary"""
        return ("Tech-Pulse analysis reveals current trends in the technology landscape, "
                "covering developments across software, hardware, and digital innovation sectors. "
                "Community engagement and sentiment tracking provide insights into market reception "
                "and industry priorities. This executive summary highlights key patterns and themes "
                "emerging from today's tech news coverage.")

    def analyze_sentiment_trends(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze sentiment trends for summary generation"""
        try:
            if df.empty or 'sentiment_label' not in df.columns:
                return {"trend": "insufficient_data", "insights": []}

            sentiment_counts = df['sentiment_label'].value_counts()
            total_stories = len(df)

            # Calculate percentages
            sentiment_pct = {
                sentiment: (count / total_stories) * 100
                for sentiment, count in sentiment_counts.items()
            }

            # Generate insights
            insights = []

            if sentiment_pct.get('Positive', 0) > 50:
                insights.append("Strong positive sentiment indicates favorable market conditions")
            elif sentiment_pct.get('Negative', 0) > 40:
                insights.append("Elevated negative sentiment suggests market concerns")

            if sentiment_pct.get('Neutral', 0) > 30:
                insights.append("Significant neutral coverage indicates objective reporting")

            return {
                "trend": "positive" if sentiment_pct.get('Positive', 0) > sentiment_pct.get('Negative', 0) else "negative",
                "percentages": sentiment_pct,
                "insights": insights,
                "total_analyzed": total_stories
            }

        except Exception as e:
            logger.error(f"Sentiment trend analysis failed: {e}")
            return {"trend": "error", "insights": [], "error": str(e)}

    def extract_key_topics(self, df: pd.DataFrame, topics: Dict) -> List[Dict[str, any]]:
        """Extract and analyze key topics for summary"""
        try:
            key_topics = []

            if topics:
                # Sort topics by percentage
                sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)

                for topic, percentage in sorted_topics[:5]:  # Top 5 topics
                    topic_name = topic.replace('_', ' ').title()

                    # Find stories related to this topic
                    topic_stories = []
                    if not df.empty and 'topic_keyword' in df.columns:
                        topic_stories = df[df['topic_keyword'] == topic]['title'].head(3).tolist()

                    key_topics.append({
                        "name": topic_name,
                        "percentage": percentage,
                        "stories": topic_stories,
                        "significance": "high" if percentage > 20 else "medium" if percentage > 10 else "low"
                    })

            return key_topics

        except Exception as e:
            logger.error(f"Key topics extraction failed: {e}")
            return []

    def get_engagement_insights(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze engagement patterns for insights"""
        try:
            if df.empty:
                return {"insights": [], "metrics": {}}

            insights = []
            metrics = {}

            # Score analysis
            if 'score' in df.columns:
                avg_score = df['score'].mean()
                max_score = df['score'].max()
                metrics['average_score'] = avg_score
                metrics['peak_score'] = max_score

                if avg_score > 200:
                    insights.append("High average engagement scores indicate strong community interest")
                elif avg_score < 50:
                    insights.append("Lower engagement may suggest niche topics or quiet news day")

            # Comment analysis
            if 'descendants' in df.columns:
                total_comments = df['descendants'].sum()
                avg_comments = df['descendants'].mean()
                metrics['total_comments'] = total_comments
                metrics['average_comments'] = avg_comments

                if total_comments > 1000:
                    insights.append("High comment activity indicates passionate community discussion")
                elif avg_comments > 50:
                    insights.append("Above-average comment engagement on stories")

            # Time-based analysis
            if 'time' in df.columns:
                try:
                    df['hour'] = pd.to_datetime(df['time']).dt.hour
                    peak_hour = df['hour'].mode().iloc[0] if len(df) > 0 else 0
                    metrics['peak_hour'] = peak_hour

                    if 9 <= peak_hour <= 17:
                        insights.append(f"Peak engagement during business hours ({peak_hour:00}:00)")
                    else:
                        insights.append(f"After-hours peak activity at {peak_hour:00}:00")
                except:
                    insights.append("Time-based analysis available with temporal data")

            return {
                "insights": insights,
                "metrics": metrics
            }

        except Exception as e:
            logger.error(f"Engagement insights analysis failed: {e}")
            return {"insights": [], "metrics": {}, "error": str(e)}


def create_sample_summary() -> str:
    """Create a sample summary for testing purposes"""
    summarizer = AISummarizer()  # Will use rule-based mode

    # Sample data
    sample_df = pd.DataFrame({
        'title': [
            'AI breakthrough in quantum computing',
            'New Python framework released',
            'Cybersecurity vulnerability discovered',
            'Tech startup raises $100M',
            'OpenAI announces new model'
        ],
        'score': [500, 300, 800, 200, 600],
        'sentiment_label': ['Positive', 'Neutral', 'Negative', 'Positive', 'Positive'],
        'descendants': [50, 30, 100, 20, 80],
        'time': pd.date_range('2024-01-01', periods=5, freq='h'),
        'topic_keyword': ['ai_ml', 'programming', 'security', 'funding', 'ai_ml']
    })

    sample_topics = {'ai_ml': 40, 'programming': 20, 'security': 20, 'funding': 20}

    return summarizer.generate_executive_summary(sample_df, sample_topics)


if __name__ == "__main__":
    # Test the summarizer
    summary = create_sample_summary()
    print("Generated Sample Summary:")
    print("=" * 50)
    print(summary)
    print("=" * 50)