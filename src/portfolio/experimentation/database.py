"""
Database schema and initialization for experiment tracking.

This module defines the SQLite database schema for storing experiments,
parameters, metrics, and artifacts.
"""

import sqlite3
from pathlib import Path
from typing import Optional


def init_experiment_db(db_path: str) -> None:
    """
    Initialize experiment tracking database with schema.
    
    Args:
        db_path: Path to SQLite database file
    """
    # Create directory if it doesn't exist
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Experiments table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'running',
            notes TEXT,
            model_path TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Parameters table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parameters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT NOT NULL,
            param_name TEXT NOT NULL,
            param_value TEXT NOT NULL,
            param_type TEXT,
            FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
            UNIQUE(experiment_id, param_name)
        )
    """)
    
    # Metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            step INTEGER DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
        )
    """)
    
    # Artifacts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT NOT NULL,
            artifact_name TEXT NOT NULL,
            artifact_path TEXT NOT NULL,
            artifact_type TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
            UNIQUE(experiment_id, artifact_name)
        )
    """)
    
    # Tags table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT NOT NULL,
            tag_name TEXT NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
            UNIQUE(experiment_id, tag_name)
        )
    """)
    
    # Create indexes for better query performance
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_experiments_name 
        ON experiments(name)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_experiments_timestamp 
        ON experiments(timestamp)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_metrics_experiment 
        ON metrics(experiment_id, metric_name)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_parameters_experiment 
        ON parameters(experiment_id)
    """)
    
    conn.commit()
    conn.close()


def get_connection(db_path: str) -> sqlite3.Connection:
    """
    Get database connection with row factory.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        Database connection
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn
