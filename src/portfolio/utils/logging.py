"""
Logging utilities for portfolio enhancement.

Provides structured logging with context information for debugging
and monitoring portfolio enhancement features.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration for portfolio enhancement.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        log_format: Optional custom log format
        
    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(message)s'
        )
    
    # Create logger
    logger = logging.getLogger('portfolio')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class PortfolioLogger:
    """Context-aware logger for portfolio operations"""
    
    def __init__(self, name: str, log_level: str = "INFO"):
        """
        Initialize portfolio logger.
        
        Args:
            name: Logger name (typically module name)
            log_level: Logging level
        """
        self.logger = logging.getLogger(f'portfolio.{name}')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.context = {}
    
    def add_context(self, **kwargs) -> None:
        """Add context information to all log messages"""
        self.context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all context information"""
        self.context.clear()
    
    def _format_message(self, message: str) -> str:
        """Format message with context"""
        if self.context:
            context_str = " | ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{message} | {context_str}"
        return message
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with context"""
        self.logger.debug(self._format_message(message), extra=kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with context"""
        self.logger.info(self._format_message(message), extra=kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with context"""
        self.logger.warning(self._format_message(message), extra=kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message with context"""
        self.logger.error(self._format_message(message), exc_info=exc_info, extra=kwargs)
    
    def critical(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log critical message with context"""
        self.logger.critical(self._format_message(message), exc_info=exc_info, extra=kwargs)
    
    def log_experiment_start(self, experiment_name: str, params: dict) -> None:
        """Log experiment start with parameters"""
        self.add_context(experiment=experiment_name)
        self.info(f"Starting experiment: {experiment_name}", params=params)
    
    def log_experiment_end(self, experiment_name: str, metrics: dict) -> None:
        """Log experiment completion with metrics"""
        self.info(f"Completed experiment: {experiment_name}", metrics=metrics)
        self.clear_context()
    
    def log_model_training(self, model_name: str, status: str, **kwargs) -> None:
        """Log model training status"""
        self.info(f"Model training {status}: {model_name}", **kwargs)
    
    def log_data_quality_issue(self, issue_type: str, details: dict) -> None:
        """Log data quality issue"""
        self.warning(f"Data quality issue detected: {issue_type}", details=details)
    
    def log_error_with_context(self, error: Exception, operation: str) -> None:
        """Log error with full context"""
        self.error(
            f"Error during {operation}: {str(error)}",
            exc_info=True,
            operation=operation,
            error_type=type(error).__name__
        )
