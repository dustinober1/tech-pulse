"""
PDF Generation Package for Tech-Pulse Executive Briefings

This package provides comprehensive PDF generation capabilities for creating
professional executive briefings from Tech-Pulse dashboard data.
"""

from .pdf_builder import ExecutiveBriefingPDF
from .ai_summarizer import AISummarizer
from .chart_exporter import ChartExporter
from .report_builder import ExecutiveBriefingBuilder

__version__ = "1.0.0"
__all__ = [
    "ExecutiveBriefingPDF",
    "AISummarizer",
    "ChartExporter",
    "ExecutiveBriefingBuilder"
]