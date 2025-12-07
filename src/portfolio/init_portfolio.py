"""
Initialize portfolio enhancement infrastructure.

This script sets up the necessary directories, databases, and
configuration for portfolio enhancement features.
"""

from pathlib import Path
from src.portfolio.utils.config import PortfolioConfig
from src.portfolio.utils.logging import setup_logging
from src.portfolio.experimentation.database import init_experiment_db


def initialize_portfolio(config_file: str = "config/portfolio_config.json") -> None:
    """
    Initialize portfolio enhancement infrastructure.
    
    Args:
        config_file: Path to configuration file
    """
    # Load configuration
    config = PortfolioConfig(config_file)
    
    # Set up logging
    logger = setup_logging(
        log_level=config.get('experiment_tracking', 'log_level', 'INFO'),
        log_file='logs/portfolio.log'
    )
    
    logger.info("Initializing portfolio enhancement infrastructure")
    
    # Create necessary directories
    directories = [
        config.get('experiment_tracking', 'artifact_dir'),
        config.get('model_registry', 'storage_path'),
        'data',
        'logs',
        'notebooks',
        'docs/portfolio',
        'docs/portfolio/model_cards',
        'docs/portfolio/experiment_reports'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Initialize experiment tracking database
    db_path = config.get('experiment_tracking', 'database_path')
    init_experiment_db(db_path)
    logger.info(f"Initialized experiment database: {db_path}")
    
    # Validate configuration
    try:
        config.validate()
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    
    logger.info("Portfolio enhancement infrastructure initialized successfully")


if __name__ == "__main__":
    initialize_portfolio()
