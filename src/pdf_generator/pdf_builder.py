"""
PDF Builder Module for Tech-Pulse Executive Briefings
Generates professional PDF reports with tech trend analysis
"""

from fpdf import FPDF
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import io
import base64
import os
import logging
from typing import Optional, Dict, List, Any, Union
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutiveBriefingPDF:
    """
    Professional PDF generator for Tech-Pulse executive briefings
    """

    def __init__(self, page_size=A4, margin=0.75):
        self.page_size = page_size
        self.margin = margin * inch
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        logger.info("PDF Builder initialized with custom styles")

    def _setup_custom_styles(self):
        """Set up custom paragraph styles for the briefing"""
        try:
            # Title style
            self.title_style = ParagraphStyle(
                'CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.HexColor('#1f77b4'),
                alignment=1,  # Center
            )

            # Heading style
            self.heading_style = ParagraphStyle(
                'CustomHeading',
                parent=self.styles['Heading2'],
                fontSize=16,
                spaceAfter=12,
                spaceBefore=20,
                textColor=colors.HexColor('#2c3e50'),
            )

            # Subheading style
            self.subheading_style = ParagraphStyle(
                'CustomSubheading',
                parent=self.styles['Heading3'],
                fontSize=14,
                spaceAfter=8,
                spaceBefore=15,
                textColor=colors.HexColor('#34495e'),
            )

            # Body text style
            self.body_style = ParagraphStyle(
                'CustomBody',
                parent=self.styles['Normal'],
                fontSize=11,
                spaceAfter=6,
                leading=14,
            )

            # Metric style
            self.metric_style = ParagraphStyle(
                'CustomMetric',
                parent=self.styles['Normal'],
                fontSize=12,
                spaceAfter=4,
                leading=16,
                fontName='Helvetica-Bold',
            )

            # Footer style
            self.footer_style = ParagraphStyle(
                'CustomFooter',
                parent=self.styles['Normal'],
                fontSize=9,
                spaceAfter=2,
                leading=10,
                textColor=colors.HexColor('#7f8c8d'),
                alignment=1,  # Center
            )

            logger.info("Custom styles setup completed")
        except Exception as e:
            logger.error(f"Failed to setup custom styles: {e}")
            raise

    def build_report(self,
                     title: str,
                     date: datetime,
                     summary: str,
                     metrics: Dict[str, Any],
                     top_stories: List[Dict],
                     topics: Dict[str, float],
                     charts: Dict[str, bytes] = None) -> bytes:
        """
        Build complete executive briefing PDF

        Args:
            title: Report title
            date: Generation date
            summary: Executive summary text
            metrics: Dictionary of key metrics
            top_stories: List of top stories
            topics: Topic distribution dictionary
            charts: Dictionary of chart images (bytes)

        Returns:
            PDF file as bytes
        """
        try:
            # Create temporary buffer
            buffer = io.BytesIO()

            # Create PDF document
            doc = SimpleDocTemplate(
                buffer,
                pagesize=self.page_size,
                rightMargin=self.margin,
                leftMargin=self.margin,
                topMargin=self.margin,
                bottomMargin=self.margin
            )

            # Build content
            story = []

            # Title Page
            story.extend(self._build_title_page(title, date))
            story.append(PageBreak())

            # Executive Summary
            story.extend(self._build_executive_summary(summary))

            # Key Metrics
            story.extend(self._build_metrics_section(metrics))

            # Charts
            if charts:
                story.extend(self._build_charts_section(charts))

            # Top Stories
            story.extend(self._build_top_stories_section(top_stories))

            # Topics
            if topics:
                story.extend(self._build_topics_section(topics))

            # Footer
            story.extend(self._build_footer(date))

            # Build PDF
            doc.build(story)

            # Get bytes
            pdf_bytes = buffer.getvalue()
            buffer.close()

            logger.info(f"PDF report generated successfully: {len(pdf_bytes)} bytes")
            return pdf_bytes

        except Exception as e:
            logger.error(f"Failed to build PDF report: {e}")
            raise RuntimeError(f"PDF generation failed: {str(e)}")

    def _build_title_page(self, title: str, date: datetime) -> List:
        """Build title page content"""
        content = []

        # Main title
        content.append(Paragraph(title, self.title_style))
        content.append(Spacer(1, 50))

        # Subtitle
        content.append(Paragraph(
            f"Executive Briefing Report",
            ParagraphStyle(
                'Subtitle',
                parent=self.styles['Heading2'],
                fontSize=18,
                alignment=1,
                spaceAfter=20,
                textColor=colors.HexColor('#34495e')
            )
        ))

        # Date
        content.append(Paragraph(
            f"Generated on {date.strftime('%B %d, %Y at %I:%M %p')}",
            ParagraphStyle(
                'Date',
                parent=self.styles['Normal'],
                fontSize=14,
                alignment=1,
                spaceAfter=30,
                textColor=colors.HexColor('#7f8c8d')
            )
        ))

        # Description
        content.append(Paragraph(
            "A comprehensive analysis of current technology trends,<br/>"
            "sentiment patterns, and emerging topics from Tech-Pulse.",
            ParagraphStyle(
                'Description',
                parent=self.styles['Normal'],
                fontSize=12,
                alignment=1,
                spaceAfter=40,
                leading=16,
                textColor=colors.HexColor('#5d6d7e')
            )
        ))

        # Separator
        content.append(Spacer(1, 100))

        # Footer notice
        content.append(Paragraph(
            "Generated by Tech-Pulse - Tech Trend Analysis Dashboard",
            self.footer_style
        ))

        return content

    def _build_executive_summary(self, summary: str) -> List:
        """Build executive summary section"""
        content = []

        content.append(Paragraph("Executive Summary", self.heading_style))
        content.append(Spacer(1, 12))

        # Summary text
        summary_paragraph = Paragraph(summary, self.body_style)
        content.append(summary_paragraph)
        content.append(Spacer(1, 20))

        return content

    def _build_metrics_section(self, metrics: Dict[str, Any]) -> List:
        """Build key metrics section"""
        content = []

        content.append(Paragraph("Key Metrics", self.heading_style))
        content.append(Spacer(1, 12))

        # Create metrics table
        metric_data = []

        # Total stories
        metric_data.append(['<b>Total Stories Analyzed</b>', str(metrics.get('total_stories', 'N/A'))])

        # Average sentiment
        avg_sentiment = metrics.get('avg_sentiment', 0)
        sentiment_label = 'Positive' if avg_sentiment > 0.1 else 'Negative' if avg_sentiment < -0.1 else 'Neutral'
        metric_data.append(['<b>Average Sentiment</b>', f"{sentiment_label} ({avg_sentiment:.2f})"])

        # Sentiment distribution
        sent_dist = metrics.get('sentiment_distribution', {})
        if sent_dist:
            dist_text = ", ".join([f"{k}: {v}" for k, v in sent_dist.items()])
            metric_data.append(['<b>Sentiment Breakdown</b>', dist_text])

        # Average score
        metric_data.append(['<b>Average Engagement Score</b>', f"{metrics.get('avg_score', 0):.1f}"])

        # Total comments
        metric_data.append(['<b>Total Comments</b>', f"{metrics.get('total_comments', 0):,}"])

        # Most active hour
        metric_data.append(['<b>Most Active Hour</b>', f"{metrics.get('most_active_hour', 0):00}:00"])

        # Top source
        metric_data.append(['<b>Top Source Domain</b>', metrics.get('top_source', 'Various')])

        # Create table
        table = Table(metric_data, colWidths=[2.5*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
        ]))

        content.append(table)
        content.append(Spacer(1, 20))

        return content

    def _build_charts_section(self, charts: Dict[str, bytes]) -> List:
        """Build charts section"""
        content = []

        content.append(Paragraph("Visual Analytics", self.heading_style))
        content.append(Spacer(1, 12))

        # Add charts
        for chart_name, chart_bytes in charts.items():
            try:
                # Save chart to temporary file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp.write(chart_bytes)
                    tmp_path = tmp.name

                # Add to content
                img = Image(tmp_path, width=5*inch, height=3.3*inch)
                content.append(img)
                content.append(Spacer(1, 15))

                # Clean up temp file
                os.unlink(tmp_path)

            except Exception as e:
                logger.error(f"Failed to include chart {chart_name}: {e}")
                continue

        return content

    def _build_top_stories_section(self, top_stories: List[Dict]) -> List:
        """Build top stories section"""
        content = []

        content.append(Paragraph("Top Stories by Engagement", self.heading_style))
        content.append(Spacer(1, 12))

        for story in top_stories:
            # Title with rank
            title = f"<b>{story['rank']}. {story['title']}</b>"
            content.append(Paragraph(title, self.subheading_style))

            # Story details
            details = [
                f"Score: {story['score']}",
                f"Sentiment: {story['sentiment']}",
                f"URL: <a href='{story['url']}'>{story['url']}</a>"
            ]

            for detail in details:
                content.append(Paragraph(detail, self.body_style))

            content.append(Spacer(1, 10))

        return content

    def _build_topics_section(self, topics: Dict[str, float]) -> List:
        """Build topics section"""
        content = []

        content.append(Paragraph("Topic Analysis", self.heading_style))
        content.append(Spacer(1, 12))

        # Sort topics by percentage
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)

        for topic, percentage in sorted_topics:
            topic_name = topic.replace('_', ' ').title()
            content.append(Paragraph(
                f"<b>{topic_name}</b>: {percentage}%",
                self.metric_style
            ))

        content.append(Spacer(1, 20))

        return content

    def _build_footer(self, date: datetime) -> List:
        """Build footer section"""
        content = []

        content.append(Spacer(1, 30))
        content.append(Paragraph(
            "---",
            self.footer_style
        ))
        content.append(Paragraph(
            f"This report was generated on {date.strftime('%B %d, %Y at %I:%M %p')} "
            f"by Tech-Pulse, an AI-powered tech trend analysis dashboard.",
            self.footer_style
        ))
        content.append(Paragraph(
            "Data sources: Hacker News API and real-time tech news feeds.",
            self.footer_style
        ))

        return content


def create_sample_pdf(output_path: str = "sample_briefing.pdf") -> str:
    """
    Create a sample PDF for testing purposes

    Args:
        output_path: Path to save the sample PDF

    Returns:
        Path to the created PDF file
    """
    try:
        pdf_builder = ExecutiveBriefingPDF()

        # Sample data
        sample_data = {
            "title": "Tech-Pulse Executive Briefing",
            "date": datetime.now(),
            "summary": "This is a sample executive briefing demonstrating the PDF generation capabilities of Tech-Pulse. The system analyzes current tech trends, sentiment patterns, and emerging topics from real-time data sources.",
            "metrics": {
                "total_stories": 25,
                "avg_sentiment": 0.15,
                "sentiment_distribution": {"Positive": 12, "Neutral": 8, "Negative": 5},
                "avg_score": 145.6,
                "total_comments": 1250,
                "most_active_hour": 14,
                "top_source": "github.com"
            },
            "top_stories": [
                {
                    "rank": 1,
                    "title": "New AI Framework Breakthrough Announced",
                    "score": 450,
                    "sentiment": "Positive",
                    "url": "https://example.com/ai-framework"
                },
                {
                    "rank": 2,
                    "title": "Major Security Vulnerability Discovered",
                    "score": 380,
                    "sentiment": "Negative",
                    "url": "https://example.com/security-vuln"
                }
            ],
            "topics": {
                "ai_ml": 35.2,
                "security": 20.1,
                "programming": 18.7,
                "funding": 15.3,
                "cloud": 10.7
            }
        }

        # Generate PDF
        pdf_bytes = pdf_builder.build_report(
            title=sample_data["title"],
            date=sample_data["date"],
            summary=sample_data["summary"],
            metrics=sample_data["metrics"],
            top_stories=sample_data["top_stories"],
            topics=sample_data["topics"]
        )

        # Save to file
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)

        logger.info(f"Sample PDF created: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to create sample PDF: {e}")
        raise