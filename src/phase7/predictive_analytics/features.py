"""
Predictive Analytics - Feature Engineering

Implements feature extraction and engineering for
predictive models including time, technology, growth,
and temporal features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import math
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)

def extract_time_features(
    df: pd.DataFrame,
    date_column: str = 'date'
) -> pd.DataFrame:
    """
    Extract time-based features from date column.

    Args:
        df: Input DataFrame with date column
        date_column: Name of the date column

    Returns:
        DataFrame with additional time features
    """
    # Ensure date column is datetime
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    # Extract basic time components
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['dayofweek'] = df[date_column].dt.dayofweek
    df['dayofyear'] = df[date_column].dt.dayofyear
    df['quarter'] = df[date_column].dt.quarter
    df['week'] = df[date_column].dt.isocalendar().week

    # Cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

    # Special dates
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = df[date_column].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df[date_column].dt.is_year_start.astype(int)
    df['is_year_end'] = df[date_column].dt.is_year_end.astype(int)

    # Holiday indicator (simplified - could be enhanced with actual holiday data)
    df['is_holiday_period'] = ((df['month'].isin([12, 1])) |  # December/January
                               ((df['month'] == 7) & (df['day'].between(1, 7))) |  # July 4th week
                               ((df['month'] == 11) & (df['day'].between(22, 28)))).astype(int)  # Thanksgiving

    return df

def extract_technology_features(
    df: pd.DataFrame,
    technology_column: str = 'technology',
    category_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Extract technology-specific features.

    Args:
        df: Input DataFrame with technology column
        technology_column: Name of the technology column
        category_mapping: Optional mapping of technologies to categories

    Returns:
        DataFrame with additional technology features
    """
    df = df.copy()

    # Technology name features
    df['tech_name_length'] = df[technology_column].str.len()
    df['tech_has_numbers'] = df[technology_column].str.contains(r'\d').astype(int)
    df['tech_has_dots'] = df[technology_column].str.contains(r'\.').astype(int)
    df['tech_word_count'] = df[technology_column].str.split().str.len()

    # Categorize technologies if mapping provided
    if category_mapping:
        df['tech_category'] = df[technology_column].map(category_mapping)
        df['tech_category'] = df['tech_category'].fillna('Other')

        # One-hot encode categories
        category_dummies = pd.get_dummies(df['tech_category'], prefix='tech_cat')
        df = pd.concat([df, category_dummies], axis=1)

    # Technology age indicator (simplified - could use actual first appearance dates)
    tech_age_mapping = {
        'Python': 34,
        'JavaScript': 28,
        'Java': 29,
        'C++': 38,
        'C#': 23,
        'PHP': 29,
        'TypeScript': 10,
        'Go': 15,
        'Rust': 13,
        'Swift': 11,
        'Kotlin': 10,
        'Dart': 14,
        'Ruby': 28,
        'Swift': 11,
        'Scala': 20,
        'Elixir': 13,
        'Clojure': 16
    }

    df['tech_age'] = df[technology_column].map(tech_age_mapping)
    df['tech_age'] = df['tech_age'].fillna(10)  # Default for unknown technologies

    # Technology generation indicator
    df['is_modern_tech'] = (df['tech_age'] <= 15).astype(int)
    df['is_legacy_tech'] = (df['tech_age'] >= 25).astype(int)

    return df

def extract_growth_features(
    df: pd.DataFrame,
    value_column: str,
    date_column: str = 'date',
    technology_column: str = 'technology',
    windows: List[int] = [7, 14, 30, 90]
) -> pd.DataFrame:
    """
    Extract growth-based features for time series data.

    Args:
        df: Input DataFrame with values and dates
        value_column: Name of the value column to analyze
        date_column: Name of the date column
        technology_column: Name of the technology column
        windows: List of window sizes for calculations

    Returns:
        DataFrame with additional growth features
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values([technology_column, date_column])

    # Group by technology for per-technology calculations
    for tech in df[technology_column].unique():
        tech_mask = df[technology_column] == tech
        tech_values = df.loc[tech_mask, value_column]

        # Growth rates for different windows
        for window in windows:
            if len(tech_values) >= window:
                # Calculate growth rate
                growth_rate = tech_values.pct_change(window)
                df.loc[tech_mask, f'growth_rate_{window}d'] = growth_rate

                # Calculate absolute growth
                abs_growth = tech_values.diff(window)
                df.loc[tech_mask, f'abs_growth_{window}d'] = abs_growth

                # Calculate volatility (standard deviation of growth)
                if len(tech_values) >= window * 2:
                    volatility = growth_rate.rolling(window=window).std()
                    df.loc[tech_mask, f'volatility_{window}d'] = volatility

        # Recent trend (last 7 days vs previous 7 days)
        if len(tech_values) >= 14:
            recent_avg = tech_values.tail(7).mean()
            previous_avg = tech_values.tail(14).head(7).mean()
            trend_change = (recent_avg - previous_avg) / (previous_avg + 1e-6)
            df.loc[tech_mask, 'recent_trend_change'] = trend_change

        # Moving averages
        for ma_window in [7, 14, 30]:
            if len(tech_values) >= ma_window:
                ma = tech_values.rolling(window=ma_window).mean()
                df.loc[tech_mask, f'ma_{ma_window}d'] = ma

                # Distance from moving average
                df.loc[tech_mask, f'dist_from_ma_{ma_window}d'] = (
                    (tech_values - ma) / (ma + 1e-6)
                )

        # Exponential moving average
        if len(tech_values) >= 10:
            ema = tech_values.ewm(span=10).mean()
            df.loc[tech_mask, 'ema_10'] = ema
            df.loc[tech_mask, 'dist_from_ema'] = (
                (tech_values - ema) / (ema + 1e-6)
            )

    return df

def extract_popularity_features(
    df: pd.DataFrame,
    popularity_columns: List[str],
    technology_column: str = 'technology',
    normalize: bool = True
) -> pd.DataFrame:
    """
    Extract popularity-based features.

    Args:
        df: Input DataFrame with popularity metrics
        popularity_columns: List of popularity column names
        technology_column: Name of the technology column
        normalize: Whether to normalize popularity features

    Returns:
        DataFrame with additional popularity features
    """
    df = df.copy()

    for col in popularity_columns:
        if col in df.columns:
            # Normalize popularity if requested
            if normalize:
                scaler = MinMaxScaler()
                df[f'{col}_normalized'] = scaler.fit_transform(
                    df[[col]].fillna(0)
                ).flatten()
            else:
                df[f'{col}_normalized'] = df[col].fillna(0)

            # Log transformation for skewed distributions
            df[f'{col}_log'] = np.log1p(df[col].fillna(0))

            # Popularity rank within technology
            df[f'{col}_rank'] = df.groupby(technology_column)[col].rank(
                pct=True,
                method='min'
            )

            # Popularity percentile
            df[f'{col}_percentile'] = df[col].rank(pct=True) * 100

            # High popularity indicator (top 25%)
            df[f'{col}_is_high'] = (df[f'{col}_rank'] > 0.75).astype(int)

            # Low popularity indicator (bottom 25%)
            df[f'{col}_is_low'] = (df[f'{col}_rank'] < 0.25).astype(int)

    # Combine popularity metrics if multiple provided
    if len(popularity_columns) > 1:
        # Average popularity
        normalized_cols = [f'{col}_normalized' for col in popularity_columns
                          if f'{col}_normalized' in df.columns]
        if normalized_cols:
            df['avg_popularity'] = df[normalized_cols].mean(axis=1)

            # Popularity variance (consistency)
            df['popularity_variance'] = df[normalized_cols].var(axis=1)

            # Max popularity
            df['max_popularity'] = df[normalized_cols].max(axis=1)

    return df

def extract_temporal_features(
    df: pd.DataFrame,
    value_column: str,
    date_column: str = 'date',
    technology_column: str = 'technology'
) -> pd.DataFrame:
    """
    Extract advanced temporal features.

    Args:
        df: Input DataFrame with time series data
        value_column: Name of the value column
        date_column: Name of the date column
        technology_column: Name of the technology column

    Returns:
        DataFrame with additional temporal features
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values([technology_column, date_column])

    # Group by technology
    for tech in df[technology_column].unique():
        tech_mask = df[technology_column] == tech
        tech_data = df[tech_mask].copy()
        tech_values = tech_data[value_column]

        if len(tech_values) >= 10:
            # Time since first appearance
            first_date = tech_data[date_column].min()
            df.loc[tech_mask, 'days_since_first'] = (
                (tech_data[date_column] - first_date).dt.days
            )

            # Acceleration (change in growth rate)
            if len(tech_values) >= 20:
                growth_rates = tech_values.pct_change()
                acceleration = growth_rates.diff()
                df.loc[tech_mask, 'acceleration'] = acceleration

            # Momentum (weighted recent growth)
            weights = np.exp(-np.arange(len(tech_values)) / 10)  # Exponential decay
            weights = weights[::-1]  # Recent values have higher weight
            momentum = np.convolve(tech_values, weights, mode='valid')
            if len(momentum) > 0:
                df.loc[tech_mask.iloc[-len(momentum):].index, 'momentum'] = momentum

            # Seasonal decomposition components (simplified)
            if len(tech_values) >= 90:  # Need at least 3 months
                # Simple seasonal pattern detection
                monthly_avg = tech_data.groupby(
                    tech_data[date_column].dt.month
                )[value_column].mean()

                if len(monthly_avg) == 12:
                    # Calculate seasonal strength
                    seasonal_strength = np.std(monthly_avg) / (np.mean(monthly_avg) + 1e-6)
                    df.loc[tech_mask, 'seasonal_strength'] = seasonal_strength

            # Trend strength (R² of linear fit)
            if len(tech_values) >= 30:
                x = np.arange(len(tech_values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x, tech_values
                )
                df.loc[tech_mask, 'trend_slope'] = slope
                df.loc[tech_mask, 'trend_r2'] = r_value ** 2
                df.loc[tech_mask, 'trend_p_value'] = p_value

    return df

def engineer_features(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Apply full feature engineering pipeline.

    Args:
        df: Input DataFrame
        config: Configuration dictionary for feature engineering

    Returns:
        DataFrame with all engineered features
    """
    if config is None:
        config = {
            'date_column': 'date',
            'technology_column': 'technology',
            'value_column': 'value',
            'popularity_columns': ['stars', 'forks', 'issues'],
            'time_windows': [7, 14, 30, 90]
        }

    result_df = df.copy()

    # Extract features in order
    logger.info("Extracting time features...")
    result_df = extract_time_features(result_df, config['date_column'])

    logger.info("Extracting technology features...")
    result_df = extract_technology_features(
        result_df,
        config['technology_column'],
        config.get('category_mapping')
    )

    logger.info("Extracting growth features...")
    result_df = extract_growth_features(
        result_df,
        config['value_column'],
        config['date_column'],
        config['technology_column'],
        config['time_windows']
    )

    if 'popularity_columns' in config:
        logger.info("Extracting popularity features...")
        result_df = extract_popularity_features(
            result_df,
            config['popularity_columns'],
            config['technology_column']
        )

    logger.info("Extracting temporal features...")
    result_df = extract_temporal_features(
        result_df,
        config['value_column'],
        config['date_column'],
        config['technology_column']
    )

    # Clean up any infinite values
    numeric_columns = result_df.select_dtypes(include=[np.number]).columns
    result_df[numeric_columns] = result_df[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    return result_df

def select_features(
    df: pd.DataFrame,
    target_column: str,
    method: str = 'correlation',
    n_features: int = 50
) -> List[str]:
    """
    Select most relevant features for prediction.

    Args:
        df: DataFrame with features and target
        target_column: Name of the target column
        method: Selection method ('correlation', 'variance', 'mutual_info')
        n_features: Number of features to select

    Returns:
        List of selected feature names
    """
    # Exclude target and date columns
    feature_columns = [col for col in df.columns
                      if col != target_column and not col.endswith('_date')]

    if method == 'correlation':
        # Select features with highest correlation to target
        correlations = df[feature_columns].corrwith(df[target_column]).abs()
        correlations = correlations.fillna(0)
        selected = correlations.nlargest(n_features).index.tolist()

    elif method == 'variance':
        # Select features with highest variance
        variances = df[feature_columns].var()
        variances = variances.fillna(0)
        selected = variances.nlargest(n_features).index.tolist()

    elif method == 'mutual_info':
        # Use mutual information (requires sklearn)
        try:
            from sklearn.feature_selection import mutual_info_regression

            X = df[feature_columns].fillna(0)
            y = df[target_column].fillna(0)

            mi_scores = mutual_info_regression(X, y)
            mi_scores = pd.Series(mi_scores, index=feature_columns)
            selected = mi_scores.nlargest(n_features).index.tolist()

        except ImportError:
            logger.warning("sklearn not available, falling back to correlation")
            return select_features(df, target_column, 'correlation', n_features)

    else:
        raise ValueError(f"Unknown feature selection method: {method}")

    logger.info(f"Selected {len(selected)} features using {method} method")
    return selected

def prepare_features_for_model(
    df: pd.DataFrame,
    selected_features: List[str],
    scale_features: bool = True
) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    """
    Prepare features for model training/prediction.

    Args:
        df: DataFrame with features
        selected_features: List of feature names to use
        scale_features: Whether to scale features

    Returns:
        Tuple of (features_array, scaler)
    """
    # Extract selected features
    features = df[selected_features].fillna(0)

    if scale_features:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        return features_scaled, scaler
    else:
        return features.values, None

def get_feature_importance_ranking(
    model: Any,
    feature_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Get and rank feature importance from trained model.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with feature importance rankings
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        raise ValueError("Model does not have feature importance attributes")

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })

    # Sort by importance
    importance_df = importance_df.sort_values(
        'importance',
        ascending=False
    ).reset_index(drop=True)

    return importance_df.head(top_n)