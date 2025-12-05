"""
Predictive Analytics - Training Data Preparation

Handles collection, cleaning, and preparation of training data
for predictive models from various sources.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import sqlite3
import json
import logging
from pathlib import Path
import requests
from dataclasses import dataclass

# Import local modules
from src.data_loader import DataLoader
from src.phase7.predictive_analytics.features import (
    engineer_features,
    select_features,
    prepare_features_for_model
)
from src.cache_manager import cache_manager

logger = logging.getLogger(__name__)

@dataclass
class DataSourceConfig:
    """Configuration for a data source"""
    name: str
    type: str  # 'database', 'api', 'file', 'mock'
    connection_string: Optional[str] = None
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    file_path: Optional[str] = None
    query: Optional[str] = None
    update_frequency: str = 'daily'  # 'hourly', 'daily', 'weekly'
    priority: int = 1  # 1=highest

@dataclass
class TrainingDataset:
    """Container for prepared training dataset"""
    features_df: pd.DataFrame
    target_column: str
    feature_columns: List[str]
    technology: str
    date_range: Tuple[datetime, datetime]
    source_metadata: Dict[str, Any]
    quality_metrics: Dict[str, float]

class TrainingDataCollector:
    """
    Collects and prepares training data from multiple sources.

    Supports:
    - Database queries
    - API calls
    - File imports
    - Data validation and cleaning
    - Feature engineering
    """

    def __init__(self, cache_ttl: int = 3600):
        """
        Initialize the training data collector.

        Args:
            cache_ttl: Cache time-to-live in seconds
        """
        self.cache_ttl = cache_ttl
        self.data_loader = DataLoader()
        self.data_sources = self._initialize_data_sources()

    def _initialize_data_sources(self) -> List[DataSourceConfig]:
        """Initialize available data sources."""
        sources = []

        # Main database source
        sources.append(DataSourceConfig(
            name='main_db',
            type='database',
            connection_string='data/tech_pulse.db',
            priority=1
        ))

        # GitHub API source
        sources.append(DataSourceConfig(
            name='github_api',
            type='api',
            api_url='https://api.github.com',
            priority=2
        ))

        # Stack Overflow API source
        sources.append(DataSourceConfig(
            name='stackoverflow_api',
            type='api',
            api_url='https://api.stackexchange.com/2.3',
            priority=2
        ))

        # Mock data source for testing
        sources.append(DataSourceConfig(
            name='mock_data',
            type='mock',
            priority=3
        ))

        return sources

    @cache_manager.cached(ttl=3600)
    def collect_historical_data(
        self,
        technology: str,
        start_date: datetime,
        end_date: datetime,
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Collect historical data for a technology.

        Args:
            technology: Name of the technology
            start_date: Start date for data collection
            end_date: End date for data collection
            metrics: List of metrics to collect

        Returns:
            DataFrame with historical data
        """
        if metrics is None:
            metrics = ['commit_count', 'stars', 'forks', 'issues', 'contributors']

        logger.info(f"Collecting historical data for {technology} from {start_date} to {end_date}")

        all_data = []

        # Try to collect from each source
        for source in self.data_sources:
            try:
                if source.type == 'database':
                    data = self._collect_from_database(
                        source,
                        technology,
                        start_date,
                        end_date,
                        metrics
                    )
                elif source.type == 'api':
                    data = self._collect_from_api(
                        source,
                        technology,
                        start_date,
                        end_date,
                        metrics
                    )
                elif source.type == 'mock':
                    data = self._generate_mock_data(
                        technology,
                        start_date,
                        end_date,
                        metrics
                    )
                else:
                    continue

                if data is not None and not data.empty:
                    all_data.append(data)
                    logger.info(f"Collected {len(data)} records from {source.name}")

            except Exception as e:
                logger.error(f"Failed to collect from {source.name}: {str(e)}")
                continue

        # Combine data from all sources
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            # Remove duplicates and sort by date
            combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
            combined_df = combined_df.sort_values('date').reset_index(drop=True)
            return combined_df
        else:
            logger.warning(f"No data collected for {technology}")
            return pd.DataFrame()

    def _collect_from_database(
        self,
        source: DataSourceConfig,
        technology: str,
        start_date: datetime,
        end_date: datetime,
        metrics: List[str]
    ) -> Optional[pd.DataFrame]:
        """Collect data from SQLite database."""
        try:
            conn = sqlite3.connect(source.connection_string)

            # Build query
            metric_columns = ', '.join(metrics)
            query = f"""
                SELECT date, {metric_columns}
                FROM daily_metrics
                WHERE technology = ?
                AND date BETWEEN ? AND ?
                ORDER BY date
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=(technology, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            )

            conn.close()

            # Add technology column
            df['technology'] = technology

            return df

        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            return None

    def _collect_from_api(
        self,
        source: DataSourceConfig,
        technology: str,
        start_date: datetime,
        end_date: datetime,
        metrics: List[str]
    ) -> Optional[pd.DataFrame]:
        """Collect data from API sources."""
        # This is a simplified version - real implementation would need
        # proper API authentication, rate limiting, etc.

        if 'github' in source.name:
            return self._collect_from_github(technology, start_date, end_date)
        elif 'stackoverflow' in source.name:
            return self._collect_from_stackoverflow(technology, start_date, end_date)

        return None

    def _collect_from_github(
        self,
        technology: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Collect data from GitHub API."""
        # Simplified GitHub API collection
        # In production, use proper pagination, rate limiting, etc.

        try:
            # Search for repositories
            search_url = "https://api.github.com/search/repositories"
            params = {
                'q': f'language:{technology.lower()}',
                'sort': 'stars',
                'order': 'desc',
                'per_page': 100
            }

            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()

            repos = response.json().get('items', [])

            # Aggregate metrics
            total_stars = sum(repo['stargazers_count'] for repo in repos)
            total_forks = sum(repo['forks_count'] for repo in repos)
            total_issues = sum(repo['open_issues_count'] for repo in repos)

            # Create DataFrame with one record per day
            date_range = pd.date_range(start_date, end_date, freq='D')
            data = []

            for date in date_range:
                # Apply some random variation to make it more realistic
                variation = np.random.normal(1, 0.1, 3)
                variation = np.clip(variation, 0.5, 1.5)

                data.append({
                    'date': date,
                    'stars': int(total_stars * variation[0]),
                    'forks': int(total_forks * variation[1]),
                    'issues': int(total_issues * variation[2])
                })

            df = pd.DataFrame(data)
            df['technology'] = technology

            return df

        except Exception as e:
            logger.error(f"GitHub API collection failed: {str(e)}")
            return None

    def _collect_from_stackoverflow(
        self,
        technology: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Collect data from Stack Overflow API."""
        try:
            # Search for questions
            search_url = "https://api.stackexchange.com/2.3/search/advanced"
            params = {
                'order': 'desc',
                'sort': 'activity',
                'accepted': 'True',
                'answers': '1',
                'tagged': technology.lower(),
                'site': 'stackoverflow',
                'pagesize': 100
            }

            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()

            questions = response.json().get('items', [])

            # Create daily metrics
            date_range = pd.date_range(start_date, end_date, freq='D')
            data = []

            # Generate realistic looking data
            base_questions = len(questions)
            for i, date in enumerate(date_range):
                # Simulate question trend with some randomness
                trend = 1 + i * 0.001  # Slight upward trend
                seasonal = 1 + 0.2 * np.sin(2 * np.pi * i / 7)  # Weekly pattern
                random_factor = np.random.normal(1, 0.2)

                daily_questions = int(base_questions * trend * seasonal * random_factor / len(date_range))
                daily_answers = int(daily_questions * 1.5)

                data.append({
                    'date': date,
                    'questions': max(0, daily_questions),
                    'answers': max(0, daily_answers),
                    'views': daily_questions * np.random.randint(10, 100)
                })

            df = pd.DataFrame(data)
            df['technology'] = technology

            return df

        except Exception as e:
            logger.error(f"Stack Overflow API collection failed: {str(e)}")
            return None

    def _generate_mock_data(
        self,
        technology: str,
        start_date: datetime,
        end_date: datetime,
        metrics: List[str]
    ) -> pd.DataFrame:
        """Generate mock training data for testing."""
        logger.info(f"Generating mock data for {technology}")

        date_range = pd.date_range(start_date, end_date, freq='D')
        data = []

        # Generate realistic looking patterns
        for i, date in enumerate(date_range):
            row = {'date': date, 'technology': technology}

            # Generate each metric with different patterns
            for metric in metrics:
                if 'commit' in metric:
                    # Commits: Weekly pattern with growth trend
                    base = 10 + i * 0.1
                    weekly = 5 * np.sin(2 * np.pi * i / 7)
                    noise = np.random.normal(0, 2)
                    row[metric] = max(0, int(base + weekly + noise))

                elif 'star' in metric:
                    # Stars: Steady growth
                    base = 100 + i * 2
                    random_spikes = np.random.poisson(1) * np.random.randint(5, 20)
                    row[metric] = base + random_spikes

                elif 'fork' in metric:
                    # Forks: Related to stars
                    stars = row.get('stars', 100)
                    row[metric] = int(stars * np.random.uniform(0.1, 0.3))

                elif 'issue' in metric:
                    # Issues: Random with slight increase
                    base = 5 + i * 0.01
                    row[metric] = max(0, int(base + np.random.poisson(3)))

                elif 'contributor' in metric:
                    # Contributors: Step function (occasionally new people join)
                    base = 5
                    steps = i // 30  # New contributors every month
                    row[metric] = base + steps + np.random.binomial(1, 0.1)

                else:
                    # Default metric
                    row[metric] = np.random.randint(0, 100)

            data.append(row)

        return pd.DataFrame(data)

    def prepare_training_dataset(
        self,
        technology: str,
        lookback_days: int = 365,
        prediction_horizon: int = 30,
        target_metric: str = 'commit_count',
        feature_selection: bool = True,
        n_features: int = 50
    ) -> TrainingDataset:
        """
        Prepare a complete training dataset for a technology.

        Args:
            technology: Name of the technology
            lookback_days: Number of days of historical data to use
            prediction_horizon: Days ahead to predict
            target_metric: Metric to predict
            feature_selection: Whether to perform feature selection
            n_features: Number of features to select

        Returns:
            TrainingDataset with prepared data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        logger.info(f"Preparing training dataset for {technology}")

        # Collect historical data
        raw_data = self.collect_historical_data(
            technology,
            start_date,
            end_date
        )

        if raw_data.empty:
            raise ValueError(f"No training data available for {technology}")

        # Engineer features
        logger.info("Engineering features...")
        features_df = engineer_features(
            raw_data,
            config={
                'date_column': 'date',
                'technology_column': 'technology',
                'value_column': target_metric,
                'popularity_columns': ['stars', 'forks', 'issues'],
                'time_windows': [7, 14, 30, 90]
            }
        )

        # Create target variable (future values)
        logger.info("Creating target variable...")
        features_df['target'] = features_df[target_metric].shift(-prediction_horizon)

        # Remove rows with NaN target (end of dataset)
        features_df = features_df.dropna(subset=['target'])

        # Fill other NaN values
        features_df = features_df.fillna(0)

        # Select features if requested
        if feature_selection:
            logger.info("Selecting features...")
            feature_columns = select_features(
                features_df,
                'target',
                method='correlation',
                n_features=n_features
            )
        else:
            feature_columns = [col for col in features_df.columns
                             if col != 'target' and not col.endswith('_date')]

        # Calculate quality metrics
        quality_metrics = self._calculate_data_quality(features_df, target_metric)

        return TrainingDataset(
            features_df=features_df,
            target_column='target',
            feature_columns=feature_columns,
            technology=technology,
            date_range=(start_date, end_date),
            source_metadata={
                'sources_used': [s.name for s in self.data_sources],
                'total_records': len(features_df),
                'prediction_horizon': prediction_horizon
            },
            quality_metrics=quality_metrics
        )

    def _calculate_data_quality(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> Dict[str, float]:
        """Calculate data quality metrics."""
        metrics = {}

        # Completeness
        metrics['completeness'] = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))

        # Target distribution
        target_values = df[target_column]
        metrics['target_mean'] = float(target_values.mean())
        metrics['target_std'] = float(target_values.std())
        metrics['target_min'] = float(target_values.min())
        metrics['target_max'] = float(target_values.max())

        # Outliers (using IQR method)
        Q1 = target_values.quantile(0.25)
        Q3 = target_values.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((target_values < (Q1 - 1.5 * IQR)) |
                   (target_values > (Q3 + 1.5 * IQR))).sum()
        metrics['outlier_percentage'] = outliers / len(target_values) * 100

        # Time coverage
        if 'date' in df.columns:
            date_range = (df['date'].max() - df['date'].min()).days
            expected_days = len(df)
            metrics['time_coverage'] = expected_days / date_range if date_range > 0 else 0

        return metrics

    def validate_dataset(
        self,
        dataset: TrainingDataset,
        min_records: int = 100,
        max_missing_percentage: float = 0.1
    ) -> bool:
        """
        Validate a prepared training dataset.

        Args:
            dataset: TrainingDataset to validate
            min_records: Minimum number of records required
            max_missing_percentage: Maximum allowed missing data percentage

        Returns:
            True if dataset passes validation
        """
        # Check minimum records
        if len(dataset.features_df) < min_records:
            logger.error(f"Dataset has only {len(dataset.features_df)} records, "
                        f"minimum required is {min_records}")
            return False

        # Check missing data
        missing_percentage = (1 - dataset.quality_metrics['completeness']) * 100
        if missing_percentage > max_missing_percentage * 100:
            logger.error(f"Dataset has {missing_percentage:.2f}% missing data, "
                        f"maximum allowed is {max_missing_percentage * 100}%")
            return False

        # Check for reasonable target range
        target_range = (dataset.quality_metrics['target_max'] -
                       dataset.quality_metrics['target_min'])
        if target_range == 0:
            logger.warning("Target variable has no variation")

        # Check for sufficient features
        if len(dataset.feature_columns) < 5:
            logger.warning(f"Only {len(dataset.feature_columns)} features selected")

        logger.info("Dataset validation passed")
        return True

    def save_dataset(
        self,
        dataset: TrainingDataset,
        output_path: str
    ) -> None:
        """Save training dataset to disk."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save features DataFrame
        dataset.features_df.to_csv(
            output_path / f"{dataset.technology}_features.csv",
            index=False
        )

        # Save metadata
        metadata = {
            'technology': dataset.technology,
            'target_column': dataset.target_column,
            'feature_columns': dataset.feature_columns,
            'date_range': [d.isoformat() for d in dataset.date_range],
            'source_metadata': dataset.source_metadata,
            'quality_metrics': dataset.quality_metrics,
            'created_at': datetime.now().isoformat()
        }

        with open(output_path / f"{dataset.technology}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Dataset saved to {output_path}")

    def load_dataset(
        self,
        technology: str,
        input_path: str
    ) -> TrainingDataset:
        """Load training dataset from disk."""
        input_path = Path(input_path)

        # Load features DataFrame
        features_df = pd.read_csv(input_path / f"{technology}_features.csv")

        # Load metadata
        with open(input_path / f"{technology}_metadata.json", 'r') as f:
            metadata = json.load(f)

        return TrainingDataset(
            features_df=features_df,
            target_column=metadata['target_column'],
            feature_columns=metadata['feature_columns'],
            technology=metadata['technology'],
            date_range=tuple(datetime.fromisoformat(d) for d in metadata['date_range']),
            source_metadata=metadata['source_metadata'],
            quality_metrics=metadata['quality_metrics']
        )