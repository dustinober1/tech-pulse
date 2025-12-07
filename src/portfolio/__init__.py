"""
Portfolio Enhancement Module for Tech-Pulse

This module provides advanced data science capabilities to showcase
ML/DS best practices, model governance, and production-ready systems.
"""

__version__ = "0.1.0"

from src.portfolio.utils.config import PortfolioConfig
from src.portfolio.utils.logging import setup_logging

__all__ = ["PortfolioConfig", "setup_logging"]
