"""
Main PDF report builder for Executive Briefings
"""

from .pdf_builder import ExecutiveBriefingPDF
from .ai_summarizer import AISummarizer
from .chart_exporter import ChartExporter
import pandas as pd
from datetime import datetime
import tempfile
import os
import logging
from typing import Optional, Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import data_loader, but provide graceful fallback
try:
    from data_loader import fetch_hn_data, analyze_sentiment, get_topics
    DATA_LOADER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Data loader not available: {e}")
    DATA_LOADER_AVAILABLE = False


class ExecutiveBriefingBuilder:
    """
    Complete PDF report builder for executive briefings
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        self.pdf_builder = ExecutiveBriefingPDF()
        self.ai_summarizer = AISummarizer(openai_api_key)
        self.chart_exporter = ChartExporter()
        logger.info("ExecutiveBriefingBuilder initialized")

    def generate_briefing(self,
                         stories_count: int = 30,
                         include_charts: bool = True,
                         include_ai_summary: bool = True,
                         custom_data: Optional[pd.DataFrame] = None) -> bytes:
        """
        Generate complete executive briefing PDF

        Args:
            stories_count: Number of stories to analyze
            include_charts: Whether to include charts
            include_ai_summary: Whether to include AI summary
            custom_data: Optional custom data to use instead of fetching

        Returns:
            PDF file as bytes
        """
        try:
            logger.info(f"Starting briefing generation: {stories_count} stories, charts={include_charts}, AI={include_ai_summary}")

            # Fetch and analyze data
            if custom_data is not None:
                df = custom_data
                logger.info("Using custom data provided")
            elif DATA_LOADER_AVAILABLE:
                df = fetch_hn_data(limit=stories_count)
                logger.info(f"Fetched {len(df)} stories from API")
            else:
                # Create sample data for testing
                df = self._create_sample_data(stories_count)
                logger.info("Using sample data for testing")

            if df.empty:
                raise ValueError("No data available for report generation")

            # Process data
            if DATA_LOADER_AVAILABLE:
                df = analyze_sentiment(df)
                df = get_topics(df)
            else:
                df = self._process_sample_data(df)

            # Generate content sections
            summary = self._generate_summary(df, include_ai_summary)
            metrics = self._calculate_metrics(df)
            top_stories = self._get_top_stories(df, 10)
            topics = self._extract_topics(df)

            # Generate charts
            charts = {}
            if include_charts:
                charts = self.chart_exporter.export_all_charts(df, topics)
                logger.info(f"Generated {len(charts)} charts")

            # Build PDF
            pdf_bytes = self.pdf_builder.build_report(
                title="Tech-Pulse Executive Briefing",
                date=datetime.now(),
                summary=summary,
                metrics=metrics,
                top_stories=top_stories,
                topics=topics,
                charts=charts
            )

            logger.info(f"PDF briefing generated successfully: {len(pdf_bytes)} bytes")
            return pdf_bytes

        except Exception as e:
            logger.error(f"Failed to generate briefing: {str(e)}")
            raise RuntimeError(f"Failed to generate briefing: {str(e)}")

    def _generate_summary(self, df: pd.DataFrame, use_ai: bool) -> str:
        """Generate executive summary"""
        topics = self._extract_topics(df)
        if use_ai:
            return self.ai_summarizer.generate_executive_summary(df, topics)
        else:
            return self.ai_summarizer._generate_rule_based_summary(df, topics)

    def _calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate key metrics for the briefing"""
        try:
            metrics = {
                'total_stories': len(df),
                'avg_sentiment': df['sentiment_score'].mean() if not df.empty and 'sentiment_score' in df.columns else 0,
                'sentiment_distribution': {},
                'avg_score': df['score'].mean() if not df.empty and 'score' in df.columns else 0,
                'total_comments': df['descendants'].sum() if not df.empty and 'descendants' in df.columns else 0,
                'most_active_hour': 0,
                'top_source': 'Various'
            }

            # Sentiment distribution
            if not df.empty and 'sentiment_label' in df.columns:
                metrics['sentiment_distribution'] = df['sentiment_label'].value_counts().to_dict()

            # Most active hour
            if not df.empty and 'time' in df.columns:
                try:
                    if not pd.api.types.is_datetime64_any_dtype(df['time']):
                        df['time'] = pd.to_datetime(df['time'])
                    metrics['most_active_hour'] = df['time'].dt.hour.mode().iloc[0] if len(df) > 0 else 0
                except:
                    pass  # Keep default 0

            # Top source
            if not df.empty and 'url' in df.columns:
                try:
                    domains = df['url'].str.extract(r'https?://(?:www\.)?([^/]+)')[0]
                    if not domains.empty:
                        metrics['top_source'] = domains.mode().iloc[0] if not pd.isna(domains.mode().iloc[0]) else 'Various'
                except:
                    pass  # Keep default 'Various'

            logger.info(f"Calculated metrics: {list(metrics.keys())}")
            return metrics

        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            return {
                'total_stories': len(df),
                'avg_sentiment': 0,
                'sentiment_distribution': {},
                'avg_score': 0,
                'total_comments': 0,
                'most_active_hour': 0,
                'top_source': 'Various'
            }

    def _get_top_stories(self, df: pd.DataFrame, limit: int = 10) -> List[Dict]:
        """Get top stories by score"""
        try:
            if df.empty or 'score' not in df.columns:
                return []

            # Sort by score and get top stories
            top_df = df.nlargest(min(limit, len(df)), 'score')

            top_stories = []
            for idx, (_, row) in enumerate(top_df.iterrows()):
                story = {
                    'rank': idx + 1,
                    'title': row.get('title', 'Unknown Title'),
                    'score': int(row.get('score', 0)),
                    'url': row.get('url', ''),
                    'sentiment': row.get('sentiment_label', 'Neutral')
                }
                top_stories.append(story)

            logger.info(f"Retrieved {len(top_stories)} top stories")
            return top_stories

        except Exception as e:
            logger.error(f"Failed to get top stories: {e}")
            return []

    def _extract_topics(self, df: pd.DataFrame) -> Dict:
        """Extract topic distribution"""
        try:
            if df.empty or 'topic_keyword' not in df.columns:
                return {}

            # Filter out empty topics and get distribution
            topic_mask = df['topic_keyword'].notna() & (df['topic_keyword'] != '')
            topic_counts = df[topic_mask]['topic_keyword'].value_counts()

            if topic_counts.empty:
                return {}

            total = topic_counts.sum()
            topics = {
                topic: round((count / total) * 100, 1)
                for topic, count in topic_counts.head(10).items()
            }

            logger.info(f"Extracted {len(topics)} topics")
            return topics

        except Exception as e:
            logger.error(f"Failed to extract topics: {e}")
            return {}

    def _create_sample_data(self, count: int) -> pd.DataFrame:
        """Create sample data for testing when real data is unavailable"""
        import numpy as np

        topics = ['ai_ml', 'programming', 'security', 'funding', 'cloud', 'mobile', 'web', 'data']
        sentiments = ['Positive', 'Negative', 'Neutral']
        sources = ['github.com', 'techcrunch.com', 'arxiv.org', 'ycombinator.com', 'medium.com']

        data = {
            'title': [
                f'Tech Story {i+1}: {np.random.choice(topics).replace("_", " ").title()} Development'
                for i in range(count)
            ],
            'url': [f'https://{np.random.choice(sources)}/story-{i+1}' for i in range(count)],
            'score': np.random.randint(10, 500, count),
            'descendants': np.random.randint(0, 200, count),
            'time': pd.date_range('2024-01-01', periods=count, freq='h'),
            'sentiment_score': np.random.uniform(-0.5, 0.5, count),
            'sentiment_label': np.random.choice(sentiments, count),
            'topic_keyword': np.random.choice(topics, count)
        }

        df = pd.DataFrame(data)
        logger.info(f"Created sample data with {count} stories")
        return df

    def _process_sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process sample data when data_loader is unavailable"""
        try:
            # Ensure required columns exist
            if 'sentiment_label' not in df.columns:
                df['sentiment_label'] = 'Neutral'

            if 'sentiment_score' not in df.columns:
                df['sentiment_score'] = 0.0

            if 'topic_keyword' not in df.columns:
                df['topic_keyword'] = 'technology'

            # Map sentiment scores to labels if needed
            if 'sentiment_score' in df.columns and 'sentiment_label' in df.columns:
                mask = df['sentiment_label'] == 'Neutral'
                if mask.any():
                    df.loc[mask, 'sentiment_label'] = df.loc[mask, 'sentiment_score'].apply(
                        lambda x: 'Positive' if x > 0.1 else 'Negative' if x < -0.1 else 'Neutral'
                    )

            logger.info("Processed sample data with required columns")
            return df

        except Exception as e:
            logger.error(f"Failed to process sample data: {e}")
            return df

    def generate_quick_summary(self, df: pd.DataFrame) -> str:
        """Generate a quick summary without full report"""
        try:
            if df.empty:
                return "No data available for summary generation."

            metrics = self._calculate_metrics(df)
            topics = self._extract_topics(df)

            summary_parts = []

            # Story count
            summary_parts.append(f"Analyzed {metrics['total_stories']} tech stories")

            # Top topic
            if topics:
                top_topic = max(topics.items(), key=lambda x: x[1])
                summary_parts.append(f"with {top_topic[0].replace('_', ' ').title()} dominating coverage at {top_topic[1]}%")

            # Sentiment
            if metrics['sentiment_distribution']:
                dominant_sentiment = max(metrics['sentiment_distribution'].items(), key=lambda x: x[1])
                summary_parts.append(f"Overall sentiment is {dominant_sentiment[0].lower()}")

            # Engagement
            if metrics['avg_score'] > 0:
                summary_parts.append(f"Average engagement score: {metrics['avg_score']:.1f}")

            return ". ".join(summary_parts) + "."

        except Exception as e:
            logger.error(f"Failed to generate quick summary: {e}")
            return "Unable to generate summary at this time."

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """Validate data quality before processing"""
        try:
            validation_results = {
                'is_valid': True,
                'issues': [],
                'warnings': [],
                'stats': {}
            }

            if df.empty:
                validation_results['is_valid'] = False
                validation_results['issues'].append("No data available")
                return validation_results

            # Basic stats
            validation_results['stats'] = {
                'total_rows': len(df),
                'columns': list(df.columns),
                'missing_values': df.isnull().sum().to_dict()
            }

            # Check required columns
            required_cols = ['title', 'score']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                validation_results['is_valid'] = False
                validation_results['issues'].append(f"Missing required columns: {missing_cols}")

            # Check for empty titles
            if 'title' in df.columns:
                empty_titles = df['title'].isnull().sum() + (df['title'] == '').sum()
                if empty_titles > 0:
                    validation_results['warnings'].append(f"{empty_titles} stories have empty titles")

            # Check score ranges
            if 'score' in df.columns:
                if df['score'].min() < 0:
                    validation_results['warnings'].append("Some scores are negative")

            # Check sentiment distribution
            if 'sentiment_label' in df.columns:
                unique_sentiments = df['sentiment_label'].nunique()
                validation_results['stats']['sentiment_types'] = unique_sentiments

            logger.info(f"Data validation completed: {'Valid' if validation_results['is_valid'] else 'Invalid'}")
            return validation_results

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return {
                'is_valid': False,
                'issues': [f"Validation error: {str(e)}"],
                'warnings': [],
                'stats': {}
            }


def create_sample_briefing(output_path: str = "sample_executive_briefing.pdf") -> str:
    """
    Create a sample executive briefing for testing

    Args:
        output_path: Path to save the sample PDF

    Returns:
        Path to the created PDF file
    """
    try:
        builder = ExecutiveBriefingBuilder()

        # Generate briefing with sample data
        pdf_bytes = builder.generate_briefing(
            stories_count=20,
            include_charts=True,
            include_ai_summary=False  # Use rule-based for testing
        )

        # Save to file
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)

        logger.info(f"Sample briefing created: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to create sample briefing: {e}")
        raise


if __name__ == "__main__":
    # Test the report builder
    try:
        output_path = create_sample_briefing()
        print(f"Sample briefing generated successfully!")
        print(f"Saved to: {output_path}")
        print(f"File size: {os.path.getsize(output_path)} bytes")
    except Exception as e:
        print(f"Failed to generate sample briefing: {e}")