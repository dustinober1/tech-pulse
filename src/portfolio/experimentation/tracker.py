"""
Experiment tracking system for ML experiments.

This module provides the ExperimentTracker class for logging, retrieving,
and comparing machine learning experiments with parameters, metrics, and artifacts.
"""

import uuid
import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd

from src.portfolio.experimentation.database import get_connection
from src.portfolio.utils.logging import PortfolioLogger


@dataclass
class Experiment:
    """Represents a single ML experiment"""
    experiment_id: str
    name: str
    timestamp: datetime
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: Dict[str, str]
    model_path: Optional[str] = None
    status: str = 'running'
    tags: List[str] = None
    notes: str = ''
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


class ExperimentTracker:
    """Tracks ML experiments with parameters, metrics, and artifacts"""
    
    def __init__(self, db_path: str = 'data/experiments.db'):
        """
        Initialize experiment tracker.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.logger = PortfolioLogger('experimentation')
    
    def log_experiment(
        self,
        name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Dict[str, str],
        model_path: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: str = ''
    ) -> str:
        """
        Log a new experiment with all its components.
        
        Args:
            name: Experiment name
            params: Dictionary of hyperparameters
            metrics: Dictionary of performance metrics
            artifacts: Dictionary of artifact names to file paths
            model_path: Optional path to saved model
            tags: Optional list of tags
            notes: Optional notes about the experiment
            
        Returns:
            Experiment ID
        """
        experiment_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        self.logger.log_experiment_start(name, params)
        
        try:
            conn = get_connection(self.db_path)
            cursor = conn.cursor()
            
            # Insert experiment
            cursor.execute("""
                INSERT INTO experiments 
                (experiment_id, name, timestamp, status, notes, model_path)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (experiment_id, name, timestamp.isoformat(), 'completed', notes, model_path))
            
            # Insert parameters
            for param_name, param_value in params.items():
                param_type = type(param_value).__name__
                cursor.execute("""
                    INSERT INTO parameters 
                    (experiment_id, param_name, param_value, param_type)
                    VALUES (?, ?, ?, ?)
                """, (experiment_id, param_name, json.dumps(param_value), param_type))
            
            # Insert metrics
            for metric_name, metric_value in metrics.items():
                cursor.execute("""
                    INSERT INTO metrics 
                    (experiment_id, metric_name, metric_value)
                    VALUES (?, ?, ?)
                """, (experiment_id, metric_name, float(metric_value)))
            
            # Insert artifacts
            for artifact_name, artifact_path in artifacts.items():
                cursor.execute("""
                    INSERT INTO artifacts 
                    (experiment_id, artifact_name, artifact_path)
                    VALUES (?, ?, ?)
                """, (experiment_id, artifact_name, artifact_path))
            
            # Insert tags
            if tags:
                for tag in tags:
                    cursor.execute("""
                        INSERT INTO tags (experiment_id, tag_name)
                        VALUES (?, ?)
                    """, (experiment_id, tag))
            
            conn.commit()
            conn.close()
            
            self.logger.log_experiment_end(name, metrics)
            return experiment_id
            
        except Exception as e:
            self.logger.log_error_with_context(e, 'log_experiment')
            raise
    
    def get_experiment(self, experiment_id: str) -> Experiment:
        """
        Retrieve an experiment by ID.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment object
            
        Raises:
            ValueError: If experiment not found
        """
        conn = get_connection(self.db_path)
        cursor = conn.cursor()
        
        # Get experiment
        cursor.execute("""
            SELECT experiment_id, name, timestamp, status, notes, model_path
            FROM experiments WHERE experiment_id = ?
        """, (experiment_id,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Get parameters
        cursor.execute("""
            SELECT param_name, param_value FROM parameters
            WHERE experiment_id = ?
        """, (experiment_id,))
        params = {row['param_name']: json.loads(row['param_value']) 
                  for row in cursor.fetchall()}
        
        # Get metrics
        cursor.execute("""
            SELECT metric_name, metric_value FROM metrics
            WHERE experiment_id = ?
        """, (experiment_id,))
        metrics = {row['metric_name']: row['metric_value'] 
                   for row in cursor.fetchall()}
        
        # Get artifacts
        cursor.execute("""
            SELECT artifact_name, artifact_path FROM artifacts
            WHERE experiment_id = ?
        """, (experiment_id,))
        artifacts = {row['artifact_name']: row['artifact_path'] 
                     for row in cursor.fetchall()}
        
        # Get tags
        cursor.execute("""
            SELECT tag_name FROM tags WHERE experiment_id = ?
        """, (experiment_id,))
        tags = [row['tag_name'] for row in cursor.fetchall()]
        
        conn.close()
        
        return Experiment(
            experiment_id=row['experiment_id'],
            name=row['name'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            parameters=params,
            metrics=metrics,
            artifacts=artifacts,
            model_path=row['model_path'],
            status=row['status'],
            tags=tags,
            notes=row['notes'] or ''
        )
    
    def list_experiments(
        self,
        name_filter: Optional[str] = None,
        tag_filter: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Experiment]:
        """
        List experiments with optional filtering.
        
        Args:
            name_filter: Optional name pattern to filter by
            tag_filter: Optional tag to filter by
            limit: Optional maximum number of experiments to return
            
        Returns:
            List of Experiment objects
        """
        conn = get_connection(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT experiment_id FROM experiments WHERE 1=1"
        params = []
        
        if name_filter:
            query += " AND name LIKE ?"
            params.append(f"%{name_filter}%")
        
        if tag_filter:
            query += """ AND experiment_id IN (
                SELECT experiment_id FROM tags WHERE tag_name = ?
            )"""
            params.append(tag_filter)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor.execute(query, params)
        experiment_ids = [row['experiment_id'] for row in cursor.fetchall()]
        conn.close()
        
        return [self.get_experiment(exp_id) for exp_id in experiment_ids]
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: Optional list of specific metrics to compare
            
        Returns:
            DataFrame with experiment comparison
        """
        experiments = [self.get_experiment(exp_id) for exp_id in experiment_ids]
        
        comparison_data = []
        for exp in experiments:
            row = {
                'experiment_id': exp.experiment_id,
                'name': exp.name,
                'timestamp': exp.timestamp,
                'status': exp.status
            }
            
            # Add parameters
            for param_name, param_value in exp.parameters.items():
                row[f'param_{param_name}'] = param_value
            
            # Add metrics (filter if specified)
            for metric_name, metric_value in exp.metrics.items():
                if metrics is None or metric_name in metrics:
                    row[f'metric_{metric_name}'] = metric_value
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_best_model(
        self,
        metric: str,
        mode: str = 'max',
        name_filter: Optional[str] = None
    ) -> Tuple[str, Experiment]:
        """
        Get the best model based on a metric.
        
        Args:
            metric: Metric name to optimize
            mode: 'max' or 'min' for optimization direction
            name_filter: Optional name pattern to filter experiments
            
        Returns:
            Tuple of (experiment_id, Experiment)
            
        Raises:
            ValueError: If no experiments found or metric not found
        """
        experiments = self.list_experiments(name_filter=name_filter)
        
        if not experiments:
            raise ValueError("No experiments found")
        
        # Filter experiments that have the metric
        valid_experiments = [
            exp for exp in experiments 
            if metric in exp.metrics
        ]
        
        if not valid_experiments:
            raise ValueError(f"No experiments found with metric '{metric}'")
        
        # Find best experiment
        if mode == 'max':
            best_exp = max(valid_experiments, key=lambda e: e.metrics[metric])
        else:
            best_exp = min(valid_experiments, key=lambda e: e.metrics[metric])
        
        return best_exp.experiment_id, best_exp
    
    def update_experiment_status(
        self,
        experiment_id: str,
        status: str,
        notes: Optional[str] = None
    ) -> None:
        """
        Update experiment status and notes.
        
        Args:
            experiment_id: Experiment ID
            status: New status ('running', 'completed', 'failed')
            notes: Optional notes to add
        """
        conn = get_connection(self.db_path)
        cursor = conn.cursor()
        
        update_parts = ["status = ?", "updated_at = ?"]
        params = [status, datetime.now().isoformat()]
        
        if notes is not None:
            update_parts.append("notes = ?")
            params.append(notes)
        
        params.append(experiment_id)
        
        cursor.execute(f"""
            UPDATE experiments 
            SET {', '.join(update_parts)}
            WHERE experiment_id = ?
        """, params)
        
        conn.commit()
        conn.close()
    
    def delete_experiment(self, experiment_id: str) -> None:
        """
        Delete an experiment and all associated data.
        
        Args:
            experiment_id: Experiment ID to delete
        """
        conn = get_connection(self.db_path)
        cursor = conn.cursor()
        
        # Delete in order due to foreign key constraints
        cursor.execute("DELETE FROM tags WHERE experiment_id = ?", (experiment_id,))
        cursor.execute("DELETE FROM artifacts WHERE experiment_id = ?", (experiment_id,))
        cursor.execute("DELETE FROM metrics WHERE experiment_id = ?", (experiment_id,))
        cursor.execute("DELETE FROM parameters WHERE experiment_id = ?", (experiment_id,))
        cursor.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Deleted experiment: {experiment_id}")
    
    def get_experiment_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of all experiments.
        
        Returns:
            DataFrame with experiment summary
        """
        conn = get_connection(self.db_path)
        
        query = """
            SELECT 
                e.name,
                COUNT(DISTINCT e.experiment_id) as num_experiments,
                AVG(m.metric_value) as avg_metric_value,
                MAX(m.metric_value) as max_metric_value,
                MIN(m.metric_value) as min_metric_value
            FROM experiments e
            LEFT JOIN metrics m ON e.experiment_id = m.experiment_id
            GROUP BY e.name
            ORDER BY num_experiments DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
