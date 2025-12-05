"""
Chart export utilities for PDF generation
"""

import plotly.graph_objects as go
import plotly.io as pio
import io
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to set up Kaleido for static image generation
try:
    pio.kaleido.scope.mathjax = None
    KALEIDO_AVAILABLE = True
    logger.info("Kaleido engine available for chart export")
except Exception as e:
    KALEIDO_AVAILABLE = False
    logger.warning(f"Kaleido engine not available: {e}")


class ChartExporter:
    """
    Converts Plotly charts to images for PDF inclusion
    """

    def __init__(self, format='png', width=600, height=400, scale=2.0):
        self.format = format.lower()
        self.width = width
        self.height = height
        self.scale = scale
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ecc71',
            'danger': '#e74c3c',
            'warning': '#f39c12',
            'info': '#3498db',
            'light': '#ecf0f1',
            'dark': '#2c3e50'
        }

        logger.info(f"ChartExporter initialized: {format} {width}x{height}")

    def export_sentiment_chart(self, df: pd.DataFrame, title: str = "Sentiment Distribution") -> bytes:
        """Export sentiment distribution pie chart"""
        try:
            if df.empty or 'sentiment_label' not in df.columns:
                return self._create_empty_chart("No sentiment data available")

            sentiment_counts = df['sentiment_label'].value_counts()

            # Define color mapping
            colors = {
                'Positive': self.color_scheme['success'],
                'Negative': self.color_scheme['danger'],
                'Neutral': self.color_scheme['light']
            }

            # Map colors to actual sentiment values
            marker_colors = [colors.get(sentiment, self.color_scheme['info'])
                           for sentiment in sentiment_counts.index]

            fig = go.Figure(data=[
                go.Pie(
                    labels=sentiment_counts.index,
                    values=sentiment_counts.values,
                    hole=0.3,
                    marker_colors=marker_colors,
                    textinfo='label+percent',
                    textposition='auto',
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                )
            ])

            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': self.color_scheme['dark']}
                },
                font=dict(size=12, family='Arial'),
                width=self.width,
                height=self.height,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.01
                ),
                margin=dict(l=20, r=120, t=40, b=20),
                paper_bgcolor='white',
                plot_bgcolor='white'
            )

            return self._fig_to_bytes(fig)

        except Exception as e:
            logger.error(f"Failed to create sentiment chart: {e}")
            return self._create_empty_chart("Error generating sentiment chart")

    def export_topic_chart(self, topics: Dict[str, float], title: str = "Top Topics by Coverage") -> bytes:
        """Export topic distribution horizontal bar chart"""
        try:
            if not topics:
                return self._create_empty_chart("No topic data available")

            # Sort topics and take top 10
            sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10]

            if not sorted_topics:
                return self._create_empty_chart("No topic data available")

            # Format topic names
            formatted_topics = [topic.replace('_', ' ').title() for topic, _ in sorted_topics]
            topic_values = [value for _, value in sorted_topics]

            fig = go.Figure(data=[
                go.Bar(
                    x=topic_values,
                    y=formatted_topics,
                    orientation='h',
                    marker_color=self.color_scheme['primary'],
                    hovertemplate='<b>%{y}</b><br>Coverage: %{x:.1f}%<extra></extra>'
                )
            ])

            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': self.color_scheme['dark']}
                },
                xaxis_title="Coverage (%)",
                yaxis_title="Topics",
                font=dict(size=12, family='Arial'),
                width=self.width,
                height=self.height,
                margin=dict(l=150, r=40, t=40, b=60),
                paper_bgcolor='white',
                plot_bgcolor='white',
                xaxis=dict(
                    gridcolor=self.color_scheme['light'],
                    zeroline=True,
                    zerolinecolor=self.color_scheme['dark'],
                    zerolinewidth=1
                ),
                yaxis=dict(
                    gridcolor=self.color_scheme['light'],
                    automargin=True
                )
            )

            return self._fig_to_bytes(fig)

        except Exception as e:
            logger.error(f"Failed to create topic chart: {e}")
            return self._create_empty_chart("Error generating topic chart")

    def export_engagement_timeline(self, df: pd.DataFrame, title: str = "Engagement Timeline") -> bytes:
        """Export engagement timeline chart"""
        try:
            if df.empty or 'time' not in df.columns or 'score' not in df.columns:
                return self._create_empty_chart("No engagement data available")

            # Convert time to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'])

            # Group by hour and calculate average score
            hourly_data = df.groupby(df['time'].dt.floor('H')).agg({
                'score': 'mean',
                'descendants': 'sum' if 'descendants' in df.columns else 'count'
            }).reset_index()

            if hourly_data.empty:
                return self._create_empty_chart("No hourly engagement data")

            fig = go.Figure()

            # Add score line
            fig.add_trace(go.Scatter(
                x=hourly_data['time'],
                y=hourly_data['score'],
                mode='lines+markers',
                name='Average Score',
                line=dict(color=self.color_scheme['primary'], width=3),
                marker=dict(size=6),
                hovertemplate='Time: %{x}<br>Avg Score: %{y:.1f}<extra></extra>'
            ))

            # Add comments line if available
            if 'descendants' in hourly_data.columns:
                fig.add_trace(go.Scatter(
                    x=hourly_data['time'],
                    y=hourly_data['descendants'],
                    mode='lines+markers',
                    name='Total Comments',
                    yaxis='y2',
                    line=dict(color=self.color_scheme['secondary'], width=3),
                    marker=dict(size=6),
                    hovertemplate='Time: %{x}<br>Total Comments: %{y}<extra></extra>'
                ))

                # Create secondary y-axis
                fig.update_layout(
                    yaxis2=dict(
                        title="Total Comments",
                        overlaying='y',
                        side='right',
                        showgrid=False
                    )
                )

            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': self.color_scheme['dark']}
                },
                xaxis_title="Time",
                yaxis_title="Average Score",
                font=dict(size=12, family='Arial'),
                width=self.width,
                height=self.height,
                margin=dict(l=60, r=80, t=40, b=60),
                paper_bgcolor='white',
                plot_bgcolor='white',
                xaxis=dict(
                    gridcolor=self.color_scheme['light'],
                    showgrid=True
                ),
                yaxis=dict(
                    gridcolor=self.color_scheme['light'],
                    showgrid=True,
                    zeroline=True,
                    zerolinecolor=self.color_scheme['dark'],
                    zerolinewidth=1
                ),
                legend=dict(
                    x=0,
                    y=1,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor=self.color_scheme['light'],
                    borderwidth=1
                ),
                hovermode='x unified'
            )

            return self._fig_to_bytes(fig)

        except Exception as e:
            logger.error(f"Failed to create engagement timeline: {e}")
            return self._create_empty_chart("Error generating engagement timeline")

    def export_sentiment_timeline(self, df: pd.DataFrame, title: str = "Sentiment Timeline") -> bytes:
        """Export sentiment distribution over time"""
        try:
            if df.empty or 'time' not in df.columns or 'sentiment_label' not in df.columns:
                return self._create_empty_chart("No sentiment timeline data available")

            # Convert time to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'])

            # Group by hour and sentiment
            sentiment_timeline = df.groupby([
                df['time'].dt.floor('H'),
                'sentiment_label'
            ]).size().unstack(fill_value=0)

            if sentiment_timeline.empty:
                return self._create_empty_chart("No sentiment timeline data")

            fig = go.Figure()

            # Define colors
            colors = {
                'Positive': self.color_scheme['success'],
                'Negative': self.color_scheme['danger'],
                'Neutral': self.color_scheme['warning']
            }

            # Add traces for each sentiment
            for sentiment in sentiment_timeline.columns:
                fig.add_trace(go.Scatter(
                    x=sentiment_timeline.index,
                    y=sentiment_timeline[sentiment],
                    mode='lines+markers',
                    name=sentiment,
                    line=dict(color=colors.get(sentiment, self.color_scheme['info']), width=2),
                    marker=dict(size=5),
                    stackgroup='one' if len(sentiment_timeline.columns) > 1 else None,
                    hovertemplate='Time: %{x}<br>%{y} %{fullData.name} stories<extra></extra>'
                ))

            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': self.color_scheme['dark']}
                },
                xaxis_title="Time",
                yaxis_title="Number of Stories",
                font=dict(size=12, family='Arial'),
                width=self.width,
                height=self.height,
                margin=dict(l=60, r=40, t=40, b=60),
                paper_bgcolor='white',
                plot_bgcolor='white',
                xaxis=dict(
                    gridcolor=self.color_scheme['light'],
                    showgrid=True
                ),
                yaxis=dict(
                    gridcolor=self.color_scheme['light'],
                    showgrid=True,
                    zeroline=True,
                    zerolinecolor=self.color_scheme['dark'],
                    zerolinewidth=1
                ),
                legend=dict(
                    x=0,
                    y=1,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor=self.color_scheme['light'],
                    borderwidth=1
                ),
                hovermode='x unified'
            )

            return self._fig_to_bytes(fig)

        except Exception as e:
            logger.error(f"Failed to create sentiment timeline: {e}")
            return self._create_empty_chart("Error generating sentiment timeline")

    def export_source_distribution(self, df: pd.DataFrame, title: str = "Top Sources by Story Count") -> bytes:
        """Export source distribution chart"""
        try:
            if df.empty or 'url' not in df.columns:
                return self._create_empty_chart("No source data available")

            # Extract domains
            try:
                domains = df['url'].str.extract(r'https?://(?:www\.)?([^/]+)')[0]
                domain_counts = domains.value_counts().head(10)
            except:
                return self._create_empty_chart("Unable to extract source domains")

            if domain_counts.empty:
                return self._create_empty_chart("No source domains found")

            fig = go.Figure(data=[
                go.Bar(
                    x=domain_counts.values,
                    y=domain_counts.index,
                    orientation='h',
                    marker_color=self.color_scheme['info'],
                    hovertemplate='<b>%{y}</b><br>Stories: %{x}<extra></extra>'
                )
            ])

            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': self.color_scheme['dark']}
                },
                xaxis_title="Number of Stories",
                yaxis_title="Source Domains",
                font=dict(size=12, family='Arial'),
                width=self.width,
                height=self.height,
                margin=dict(l=120, r=40, t=40, b=60),
                paper_bgcolor='white',
                plot_bgcolor='white',
                xaxis=dict(
                    gridcolor=self.color_scheme['light'],
                    zeroline=True,
                    zerolinecolor=self.color_scheme['dark'],
                    zerolinewidth=1
                ),
                yaxis=dict(
                    gridcolor=self.color_scheme['light'],
                    automargin=True
                )
            )

            return self._fig_to_bytes(fig)

        except Exception as e:
            logger.error(f"Failed to create source distribution: {e}")
            return self._create_empty_chart("Error generating source distribution")

    def _fig_to_bytes(self, fig: go.Figure) -> bytes:
        """Convert Plotly figure to bytes"""
        try:
            if not KALEIDO_AVAILABLE:
                logger.warning("Kaleido not available, creating placeholder")
                return self._create_placeholder_image()

            # Configure for static image generation
            config = {
                'displayModeBar': False,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                'staticPlot': True
            }

            img_bytes = pio.to_image(
                fig,
                format=self.format,
                width=self.width,
                height=self.height,
                scale=self.scale,
                config=config
            )

            logger.debug(f"Chart exported successfully: {len(img_bytes)} bytes")
            return img_bytes

        except Exception as e:
            logger.error(f"Failed to convert figure to bytes: {e}")
            return self._create_placeholder_image()

    def _create_empty_chart(self, message: str) -> bytes:
        """Create a placeholder image for empty charts"""
        try:
            fig = go.Figure()

            # Add text annotation
            fig.add_annotation(
                text=message,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                xanchor='center',
                yanchor='middle',
                font=dict(
                    size=14,
                    color=self.color_scheme['dark']
                ),
                showarrow=False
            )

            fig.update_layout(
                title={
                    'text': "Data Unavailable",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': self.color_scheme['dark']}
                },
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                width=self.width,
                height=self.height,
                paper_bgcolor='white',
                plot_bgcolor='white',
                margin=dict(l=20, r=20, t=40, b=20)
            )

            return self._fig_to_bytes(fig)

        except Exception as e:
            logger.error(f"Failed to create empty chart: {e}")
            return self._create_placeholder_image()

    def _create_placeholder_image(self) -> bytes:
        """Create a simple placeholder image when chart generation fails"""
        try:
            # Create a simple SVG placeholder
            svg_content = f'''
            <svg width="{self.width}" height="{self.height}" xmlns="http://www.w3.org/2000/svg">
                <rect width="100%" height="100%" fill="white" stroke="#ddd" stroke-width="1"/>
                <text x="50%" y="50%" text-anchor="middle" dy=".3em"
                      font-family="Arial" font-size="14" fill="#666">
                    Chart Unavailable
                </text>
            </svg>
            '''
            return svg_content.encode('utf-8')
        except Exception as e:
            logger.error(f"Failed to create placeholder image: {e}")
            return b'<svg><text>Chart Error</text></svg>'

    def get_chart_info(self) -> Dict[str, any]:
        """Get information about the chart exporter configuration"""
        return {
            'format': self.format,
            'width': self.width,
            'height': self.height,
            'scale': self.scale,
            'kaleido_available': KALEIDO_AVAILABLE,
            'color_scheme': self.color_scheme
        }

    def export_all_charts(self, df: pd.DataFrame, topics: Dict) -> Dict[str, bytes]:
        """Export all available charts"""
        charts = {}

        try:
            # Sentiment distribution
            charts['sentiment'] = self.export_sentiment_chart(df)

            # Topic distribution
            charts['topics'] = self.export_topic_chart(topics)

            # Engagement timeline
            charts['engagement_timeline'] = self.export_engagement_timeline(df)

            # Sentiment timeline
            charts['sentiment_timeline'] = self.export_sentiment_timeline(df)

            # Source distribution
            charts['sources'] = self.export_source_distribution(df)

            logger.info(f"Exported {len(charts)} charts successfully")
            return charts

        except Exception as e:
            logger.error(f"Failed to export charts: {e}")
            # Return whatever charts we were able to create
            return charts


def create_sample_charts() -> Dict[str, bytes]:
    """Create sample charts for testing"""
    exporter = ChartExporter()

    # Sample data
    sample_df = pd.DataFrame({
        'title': [
            'AI breakthrough announced',
            'New security vulnerability',
            'Startup funding round',
            'Open source release',
            'Tech company acquisition'
        ],
        'score': [450, 380, 320, 290, 260],
        'sentiment_label': ['Positive', 'Negative', 'Positive', 'Positive', 'Neutral'],
        'descendants': [50, 100, 30, 25, 40],
        'time': pd.date_range('2024-01-01', periods=5, freq='h'),
        'url': [
            'https://techcrunch.com/ai-breakthrough',
            'https://security-blog.com/vulnerability',
            'https://news.ycombinator.com/startup',
            'https://github.com/new-release',
            'https://reuters.com/acquisition'
        ]
    })

    sample_topics = {
        'ai_ml': 30,
        'security': 25,
        'funding': 20,
        'programming': 15,
        'business': 10
    }

    return exporter.export_all_charts(sample_df, sample_topics)


if __name__ == "__main__":
    # Test the chart exporter
    charts = create_sample_charts()

    print("Generated Sample Charts:")
    print("=" * 40)

    exporter = ChartExporter()
    info = exporter.get_chart_info()
    print(f"Exporter Configuration: {info}")
    print(f"Charts Generated: {list(charts.keys())}")

    for chart_name, chart_bytes in charts.items():
        print(f"{chart_name}: {len(chart_bytes)} bytes")