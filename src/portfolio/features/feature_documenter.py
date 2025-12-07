"""
Feature documentation system for machine learning features.

This module provides comprehensive documentation capabilities for features,
including their rationale, statistics, and expected impact on model performance.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import json
import warnings
from pathlib import Path
import hashlib

# Statistical imports
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder

# Suppress warnings
warnings.filterwarnings('ignore')


@dataclass
class FeatureStats:
    """Statistical summary of a feature."""
    name: str
    dtype: str
    count: int
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float

    # Numeric statistics
    mean: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    q25: Optional[float] = None
    q50: Optional[float] = None
    q75: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None

    # Categorical statistics
    most_frequent: Optional[str] = None
    most_frequent_count: Optional[int] = None
    most_frequent_percentage: Optional[float] = None
    categories: Optional[List[str]] = None
    n_categories: Optional[int] = None

    # Text statistics (if applicable)
    avg_length: Optional[float] = None
    max_length: Optional[int] = None
    min_length: Optional[int] = None

    # Temporal statistics (if applicable)
    time_span_days: Optional[float] = None
    time_start: Optional[str] = None
    time_end: Optional[str] = None


@dataclass
class FeatureAnalysis:
    """Analysis of feature's relationship with target and other features."""
    name: str

    # Relationship with target
    target_correlation: Optional[float] = None
    mutual_info: Optional[float] = None
    chi2_stat: Optional[float] = None
    chi2_p_value: Optional[float] = None
    f_stat: Optional[float] = None
    f_p_value: Optional[float] = None

    # Correlation with other features
    max_correlation: Optional[float] = None
    highly_correlated_features: List[str] = field(default_factory=list)
    vif_value: Optional[float] = None

    # Predictive power
    feature_importance: Optional[float] = None
    predictive_score: Optional[float] = None
    stability_score: Optional[float] = None

    # Data quality indicators
    outlier_percentage: Optional[float] = None
    outlier_count: Optional[int] = None
    completeness_score: Optional[float] = None
    consistency_score: Optional[float] = None


@dataclass
class FeatureDescription:
    """Complete feature documentation."""
    # Basic information
    name: str
    description: str
    feature_type: str  # 'numeric', 'categorical', 'text', 'datetime', etc.
    source: str  # Where the feature comes from
    creation_date: str

    # Business context
    business_meaning: str
    rationale: str  # Why this feature was created
    expected_impact: str  # Expected impact on model performance
    domain_knowledge: str  # Domain-specific information

    # Technical details
    transformation_history: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # Statistics and analysis
    statistics: Optional[FeatureStats] = None
    analysis: Optional[FeatureAnalysis] = None

    # Metadata
    tags: List[str] = field(default_factory=list)
    owner: str = ""
    status: str = "active"  # active, deprecated, experimental
    version: str = "1.0"

    # Validation
    validation_results: Dict[str, Any] = field(default_factory=dict)
    last_validated: Optional[str] = None


@dataclass
class DocumentationConfig:
    """Configuration for feature documentation."""
    # Analysis parameters
    correlation_threshold: float = 0.8
    outlier_method: str = 'iqr'  # 'iqr', 'zscore', 'isolation'
    outlier_threshold: float = 3.0

    # Statistical tests
    chi2_threshold: float = 0.05
    f_test_threshold: float = 0.05

    # File output
    output_dir: str = "docs/features"
    export_format: str = "markdown"  # 'markdown', 'html', 'json'
    include_plots: bool = True

    # Documentation content
    include_statistics: bool = True
    include_analysis: bool = True
    include_recommendations: bool = True

    # Visualization settings
    plot_samples: int = 1000
    plot_bins: int = 50


class FeatureDocumenter:
    """Comprehensive feature documentation system."""

    def __init__(self, config: Optional[DocumentationConfig] = None):
        """
        Initialize the feature documenter.

        Args:
            config: Configuration for documentation
        """
        self.config = config or DocumentationConfig()
        self.features_: Dict[str, FeatureDescription] = {}
        self.catalog_: pd.DataFrame = pd.DataFrame()
        self.templates_ = self._load_templates()

    def analyze_feature(
        self,
        feature_name: str,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        description: Optional[str] = None,
        **kwargs
    ) -> FeatureDescription:
        """
        Analyze a single feature and create documentation.

        Args:
            feature_name: Name of the feature
            X: Feature matrix (should contain the feature)
            y: Target variable (optional)
            description: Feature description (optional)
            **kwargs: Additional metadata

        Returns:
            FeatureDescription with complete analysis
        """
        if feature_name not in X.columns:
            raise ValueError(f"Feature '{feature_name}' not found in data")

        # Extract feature series
        feature_series = X[feature_name]

        # Create basic feature description
        feature_desc = FeatureDescription(
            name=feature_name,
            description=description or f"Feature: {feature_name}",
            feature_type=self._detect_feature_type(feature_series),
            source=kwargs.get('source', 'unknown'),
            creation_date=datetime.now().isoformat(),
            business_meaning=kwargs.get('business_meaning', ''),
            rationale=kwargs.get('rationale', ''),
            expected_impact=kwargs.get('expected_impact', ''),
            domain_knowledge=kwargs.get('domain_knowledge', ''),
            tags=kwargs.get('tags', []),
            owner=kwargs.get('owner', ''),
        )

        # Calculate statistics
        feature_desc.statistics = self._calculate_statistics(feature_series)

        # Analyze relationships if target provided
        if y is not None:
            feature_desc.analysis = self._analyze_relationships(
                feature_series, y, X.drop(columns=[feature_name])
            )

        # Store feature
        self.features_[feature_name] = feature_desc

        return feature_desc

    def analyze_dataset(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        feature_descriptions: Optional[Dict[str, str]] = None,
        **metadata
    ) -> Dict[str, FeatureDescription]:
        """
        Analyze all features in a dataset.

        Args:
            X: Feature matrix
            y: Target variable (optional)
            feature_descriptions: Dictionary of feature descriptions
            **metadata: Additional metadata for all features

        Returns:
            Dictionary of feature descriptions
        """
        if feature_descriptions is None:
            feature_descriptions = {}

        feature_docs = {}

        for feature_name in X.columns:
            try:
                # Get description if available
                desc = feature_descriptions.get(feature_name)

                # Analyze feature
                feature_doc = self.analyze_feature(
                    feature_name=feature_name,
                    X=X,
                    y=y,
                    description=desc,
                    **metadata
                )

                feature_docs[feature_name] = feature_doc

            except Exception as e:
                print(f"Warning: Could not analyze feature '{feature_name}': {e}")
                continue

        # Update catalog
        self._update_catalog()

        return feature_docs

    def _detect_feature_type(self, series: pd.Series) -> str:
        """Detect the type of a feature."""
        dtype = series.dtype

        # Check boolean first (before checking for few unique values)
        if pd.api.types.is_bool_dtype(dtype):
            return 'boolean'
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return 'datetime'
        elif pd.api.types.is_numeric_dtype(dtype):
            # Check if it's boolean masquerading as numeric
            if set(series.dropna().unique()) <= {0, 1}:
                return 'boolean'
            elif series.nunique() < 10 and series.nunique() / len(series) < 0.05:
                return 'ordinal'
            return 'numeric'
        elif dtype == 'object' or (hasattr(pd, 'StringDtype') and pd.api.types.is_string_dtype(dtype)):
            # Check if it's boolean masquerading as object
            unique_vals = set(series.dropna().astype(str).unique())
            if unique_vals <= {'0', '1', 'True', 'False', 'true', 'false'}:
                return 'boolean'
            elif hasattr(series, 'str') and series.str.len().mean() > 50:  # Heuristic for text
                return 'text'
            return 'categorical'
        else:
            return 'unknown'

    def _calculate_statistics(self, series: pd.Series) -> FeatureStats:
        """Calculate comprehensive statistics for a feature."""
        # Basic statistics
        count = len(series)
        null_count = series.isna().sum()
        unique_count = series.nunique()

        feature_stats = FeatureStats(
            name=series.name,
            dtype=str(series.dtype),
            count=count,
            null_count=null_count,
            null_percentage=100 * null_count / count,
            unique_count=unique_count,
            unique_percentage=100 * unique_count / count
        )

        # Numeric statistics
        if pd.api.types.is_numeric_dtype(series):
            clean_series = series.dropna()
            if len(clean_series) > 0:
                feature_stats.mean = clean_series.mean()
                feature_stats.std = clean_series.std()
                feature_stats.min_value = clean_series.min()
                feature_stats.max_value = clean_series.max()
                feature_stats.q25 = clean_series.quantile(0.25)
                feature_stats.q50 = clean_series.quantile(0.5)
                feature_stats.q75 = clean_series.quantile(0.75)

                # Skewness and kurtosis
                feature_stats.skewness = stats.skew(clean_series)
                feature_stats.kurtosis = stats.kurtosis(clean_series)

        # Categorical statistics
        elif series.dtype == 'object' or not pd.api.types.is_numeric_dtype(series):
            clean_series = series.dropna()
            if len(clean_series) > 0:
                value_counts = clean_series.value_counts()
                feature_stats.most_frequent = str(value_counts.index[0])
                feature_stats.most_frequent_count = value_counts.iloc[0]
                feature_stats.most_frequent_percentage = 100 * value_counts.iloc[0] / len(clean_series)
                feature_stats.categories = [str(x) for x in value_counts.index[:20]]  # Top 20
                feature_stats.n_categories = len(value_counts)

        # Text statistics
        if series.dtype == 'object' and not pd.api.types.is_numeric_dtype(series):
            clean_series = series.dropna().astype(str)
            if len(clean_series) > 0:
                lengths = clean_series.str.len()
                feature_stats.avg_length = lengths.mean()
                feature_stats.max_length = lengths.max()
                feature_stats.min_length = lengths.min()

        # Datetime statistics
        if pd.api.types.is_datetime64_any_dtype(series):
            clean_series = series.dropna()
            if len(clean_series) > 0:
                time_span = clean_series.max() - clean_series.min()
                feature_stats.time_span_days = time_span.total_seconds() / 86400
                feature_stats.time_start = str(clean_series.min())
                feature_stats.time_end = str(clean_series.max())

        return feature_stats

    def _analyze_relationships(
        self,
        feature: pd.Series,
        target: Union[pd.Series, np.ndarray],
        other_features: pd.DataFrame
    ) -> FeatureAnalysis:
        """Analyze feature's relationships with target and other features."""
        analysis = FeatureAnalysis(name=feature.name)

        # Clean data
        feature_clean = feature.dropna()
        target_clean = pd.Series(target).iloc[feature_clean.index]

        if len(feature_clean) == 0:
            return analysis

        # Target correlation (numeric target)
        if pd.api.types.is_numeric_dtype(target_clean):
            if pd.api.types.is_numeric_dtype(feature_clean):
                # Pearson correlation
                corr, p_value = stats.pearsonr(feature_clean, target_clean)
                analysis.target_correlation = corr

                # Mutual information
                try:
                    mi = mutual_info_score(feature_clean, target_clean)
                    analysis.mutual_info = mi
                except:
                    pass

        # F-test (numeric feature)
        if pd.api.types.is_numeric_dtype(feature_clean):
            try:
                f_stat, f_p = stats.f_oneway(
                    feature_clean[target_clean == 0] if (target_clean == 0).any() else feature_clean,
                    feature_clean[target_clean == 1] if (target_clean == 1).any() else feature_clean
                )
                analysis.f_stat = f_stat
                analysis.f_p_value = f_p
            except:
                pass

        # Chi-square test (categorical)
        if not pd.api.types.is_numeric_dtype(feature_clean):
            try:
                # Create contingency table
                contingency = pd.crosstab(feature_clean, target_clean)
                chi2, chi2_p, _, _ = stats.chi2_contingency(contingency)
                analysis.chi2_stat = chi2
                analysis.chi2_p_value = chi2_p
            except:
                pass

        # Correlation with other features
        if not other_features.empty and pd.api.types.is_numeric_dtype(feature_clean):
            correlations = []
            for col in other_features.columns:
                if pd.api.types.is_numeric_dtype(other_features[col]):
                    other_clean = other_features[col].dropna()
                    common_idx = feature_clean.index.intersection(other_clean.index)

                    if len(common_idx) > 1:
                        corr = np.corrcoef(
                            feature_clean.loc[common_idx],
                            other_clean.loc[common_idx]
                        )[0, 1]
                        if not np.isnan(corr):
                            correlations.append((col, abs(corr)))

            if correlations:
                correlations.sort(key=lambda x: x[1], reverse=True)
                analysis.max_correlation = correlations[0][1]

                # Find highly correlated features
                for col, corr in correlations:
                    if corr >= self.config.correlation_threshold:
                        analysis.highly_correlated_features.append(col)

        # Calculate outlier percentage
        analysis.outlier_count, analysis.outlier_percentage = self._detect_outliers(
            feature_clean, self.config.outlier_method, self.config.outlier_threshold
        )

        # Completeness score (inverse of null percentage)
        analysis.completeness_score = 1 - (feature.isna().sum() / len(feature))

        # Consistency score (based on value distribution stability)
        analysis.consistency_score = self._calculate_consistency(feature_clean)

        return analysis

    def _detect_outliers(
        self,
        series: pd.Series,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> Tuple[int, float]:
        """Detect outliers in a series."""
        if not pd.api.types.is_numeric_dtype(series):
            return 0, 0.0

        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((series < lower_bound) | (series > upper_bound)).sum()

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series))
            outliers = (z_scores > threshold).sum()

        else:
            outliers = 0

        return int(outliers), 100 * outliers / len(series)

    def _calculate_consistency(self, series: pd.Series) -> float:
        """Calculate consistency score based on value distribution."""
        if len(series) == 0:
            return 0.0

        # For numeric features: check coefficient of variation
        if pd.api.types.is_numeric_dtype(series):
            cv = series.std() / (series.mean() + 1e-10)
            # Lower CV is more consistent
            consistency = 1 / (1 + cv)
        else:
            # For categorical: check entropy
            value_counts = series.value_counts(normalize=True)
            entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
            max_entropy = np.log2(len(value_counts))
            consistency = 1 - (entropy / max_entropy)

        return min(max(consistency, 0.0), 1.0)

    def _update_catalog(self):
        """Update the feature catalog DataFrame."""
        if not self.features_:
            self.catalog_ = pd.DataFrame()
            return

        catalog_data = []
        for name, feature in self.features_.items():
            row = {
                'name': name,
                'type': feature.feature_type,
                'description': feature.description,
                'source': feature.source,
                'owner': feature.owner,
                'status': feature.status,
                'tags': ', '.join(feature.tags),
                'creation_date': feature.creation_date
            }

            # Add statistics
            if feature.statistics:
                row.update({
                    'null_percentage': feature.statistics.null_percentage,
                    'unique_count': feature.statistics.unique_count,
                    'mean': feature.statistics.mean,
                    'std': feature.statistics.std,
                    'min_value': feature.statistics.min_value,
                    'max_value': feature.statistics.max_value
                })

            # Add analysis
            if feature.analysis:
                row.update({
                    'target_correlation': feature.analysis.target_correlation,
                    'outlier_percentage': feature.analysis.outlier_percentage,
                    'completeness_score': feature.analysis.completeness_score,
                    'consistency_score': feature.analysis.consistency_score
                })

            catalog_data.append(row)

        self.catalog_ = pd.DataFrame(catalog_data)

    def generate_catalog(self) -> pd.DataFrame:
        """
        Generate a feature catalog.

        Returns:
            DataFrame with feature catalog
        """
        self._update_catalog()
        return self.catalog_.copy()

    def export_feature_documentation(
        self,
        feature_name: str,
        format: str = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Export documentation for a single feature.

        Args:
            feature_name: Name of the feature
            format: Export format ('markdown', 'html', 'json')
            output_path: Path to save documentation

        Returns:
            Documentation content as string
        """
        if feature_name not in self.features_:
            raise ValueError(f"Feature '{feature_name}' not documented")

        feature = self.features_[feature_name]
        format = format or self.config.export_format

        if format == 'markdown':
            content = self._generate_markdown(feature)
        elif format == 'html':
            content = self._generate_html(feature)
        elif format == 'json':
            content = self._generate_json(feature)
        else:
            raise ValueError(f"Unsupported format: {format}")

        if output_path:
            with open(output_path, 'w') as f:
                f.write(content)

        return content

    def export_all_documentation(
        self,
        format: str = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Export documentation for all features.

        Args:
            format: Export format
            output_dir: Directory to save documentation

        Returns:
            Dictionary mapping feature names to documentation content
        """
        format = format or self.config.export_format
        output_dir = output_dir or self.config.output_dir

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        all_docs = {}
        for feature_name in self.features_:
            output_path = Path(output_dir) / f"{feature_name}.{format}"
            content = self.export_feature_documentation(
                feature_name,
                format=format,
                output_path=str(output_path)
            )
            all_docs[feature_name] = content

        return all_docs

    def _generate_markdown(self, feature: FeatureDescription) -> str:
        """Generate markdown documentation for a feature."""
        md = f"# {feature.name}\n\n"

        # Basic information
        md += f"**Description**: {feature.description}\n\n"
        md += f"**Type**: {feature.feature_type}\n"
        md += f"**Source**: {feature.source}\n"
        md += f"**Owner**: {feature.owner}\n"
        md += f"**Status**: {feature.status}\n"
        md += f"**Created**: {feature.creation_date}\n\n"

        # Business context
        if feature.business_meaning:
            md += f"## Business Meaning\n\n{feature.business_meaning}\n\n"

        if feature.rationale:
            md += f"## Rationale\n\n{feature.rationale}\n\n"

        if feature.expected_impact:
            md += f"## Expected Impact\n\n{feature.expected_impact}\n\n"

        # Statistics
        if feature.statistics and self.config.include_statistics:
            md += "## Statistics\n\n"
            md += "| Metric | Value |\n"
            md += "|--------|-------|\n"

            md += f"| Count | {feature.statistics.count:,} |\n"
            md += f"| Null Count | {feature.statistics.null_count:,} ({feature.statistics.null_percentage:.1f}%) |\n"
            md += f"| Unique Values | {feature.statistics.unique_count:,} ({feature.statistics.unique_percentage:.1f}%) |\n"

            if feature.statistics.mean is not None:
                md += f"| Mean | {feature.statistics.mean:.4f} |\n"
                md += f"| Std Dev | {feature.statistics.std:.4f} |\n"
                md += f"| Min | {feature.statistics.min_value:.4f} |\n"
                md += f"| 25% | {feature.statistics.q25:.4f} |\n"
                md += f"| 50% | {feature.statistics.q50:.4f} |\n"
                md += f"| 75% | {feature.statistics.q75:.4f} |\n"
                md += f"| Max | {feature.statistics.max_value:.4f} |\n"
                md += f"| Skewness | {feature.statistics.skewness:.4f} |\n"
                md += f"| Kurtosis | {feature.statistics.kurtosis:.4f} |\n"

            md += "\n"

        # Analysis
        if feature.analysis and self.config.include_analysis:
            md += "## Analysis\n\n"

            if feature.analysis.target_correlation is not None:
                md += f"- **Target Correlation**: {feature.analysis.target_correlation:.4f}\n"

            if feature.analysis.outlier_percentage is not None:
                md += f"- **Outliers**: {feature.analysis.outlier_count} ({feature.analysis.outlier_percentage:.1f}%)\n"

            if feature.analysis.completeness_score is not None:
                md += f"- **Completeness**: {feature.analysis.completeness_score:.2%}\n"

            if feature.analysis.consistency_score is not None:
                md += f"- **Consistency**: {feature.analysis.consistency_score:.2%}\n"

            if feature.analysis.highly_correlated_features:
                md += f"- **Highly Correlated With**: {', '.join(feature.analysis.highly_correlated_features)}\n"

            md += "\n"

        # Notes and dependencies
        if feature.notes:
            md += "## Notes\n\n"
            for note in feature.notes:
                md += f"- {note}\n"
            md += "\n"

        if feature.dependencies:
            md += f"## Dependencies\n\n{', '.join(feature.dependencies)}\n\n"

        return md

    def _generate_html(self, feature: FeatureDescription) -> str:
        """Generate HTML documentation for a feature."""
        # Convert markdown to HTML and add styling
        md_content = self._generate_markdown(feature)

        # Simple HTML template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{feature.name} - Feature Documentation</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <pre><code>{md_content}</code></pre>
        </body>
        </html>
        """

        return html

    def _generate_json(self, feature: FeatureDescription) -> str:
        """Generate JSON documentation for a feature."""
        # Convert to dict and serialize
        feature_dict = {
            'name': feature.name,
            'description': feature.description,
            'feature_type': feature.feature_type,
            'source': feature.source,
            'creation_date': feature.creation_date,
            'business_meaning': feature.business_meaning,
            'rationale': feature.rationale,
            'expected_impact': feature.expected_impact,
            'domain_knowledge': feature.domain_knowledge,
            'transformation_history': feature.transformation_history,
            'dependencies': feature.dependencies,
            'notes': feature.notes,
            'tags': feature.tags,
            'owner': feature.owner,
            'status': feature.status,
            'version': feature.version,
            'validation_results': feature.validation_results,
            'last_validated': feature.last_validated
        }

        # Add statistics
        if feature.statistics:
            feature_dict['statistics'] = {
                'name': feature.statistics.name,
                'dtype': feature.statistics.dtype,
                'count': feature.statistics.count,
                'null_count': feature.statistics.null_count,
                'null_percentage': feature.statistics.null_percentage,
                'unique_count': feature.statistics.unique_count,
                'unique_percentage': feature.statistics.unique_percentage,
                'mean': feature.statistics.mean,
                'std': feature.statistics.std,
                'min_value': feature.statistics.min_value,
                'max_value': feature.statistics.max_value,
                'q25': feature.statistics.q25,
                'q50': feature.statistics.q50,
                'q75': feature.statistics.q75,
                'skewness': feature.statistics.skewness,
                'kurtosis': feature.statistics.kurtosis,
                'most_frequent': feature.statistics.most_frequent,
                'most_frequent_count': feature.statistics.most_frequent_count,
                'categories': feature.statistics.categories,
                'n_categories': feature.statistics.n_categories
            }

        # Add analysis
        if feature.analysis:
            feature_dict['analysis'] = {
                'name': feature.analysis.name,
                'target_correlation': feature.analysis.target_correlation,
                'mutual_info': feature.analysis.mutual_info,
                'chi2_stat': feature.analysis.chi2_stat,
                'chi2_p_value': feature.analysis.chi2_p_value,
                'f_stat': feature.analysis.f_stat,
                'f_p_value': feature.analysis.f_p_value,
                'max_correlation': feature.analysis.max_correlation,
                'highly_correlated_features': feature.analysis.highly_correlated_features,
                'vif_value': feature.analysis.vif_value,
                'outlier_percentage': feature.analysis.outlier_percentage,
                'outlier_count': feature.analysis.outlier_count,
                'completeness_score': feature.analysis.completeness_score,
                'consistency_score': feature.analysis.consistency_score
            }

        return json.dumps(feature_dict, indent=2)

    def _load_templates(self) -> Dict[str, str]:
        """Load documentation templates."""
        return {
            'numeric': """Numeric Feature Template:
- Check distribution normality
- Identify outliers using IQR method
- Consider transformations for skewness
- Validate correlation assumptions
""",
            'categorical': """Categorical Feature Template:
- Check cardinality
- Identify rare categories
- Consider encoding strategies
- Validate chi-square assumptions
""",
            'text': """Text Feature Template:
- Analyze text length distribution
- Check for encoding issues
- Consider preprocessing steps
- Validate text cleaning procedures
""",
            'datetime': """Datetime Feature Template:
- Check time range coverage
- Identify gaps in time series
- Consider seasonal patterns
- Validate timezone consistency
"""
        }

    def get_feature_summary(self, feature_name: str) -> Dict[str, Any]:
        """
        Get a summary of a feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Dictionary with feature summary
        """
        if feature_name not in self.features_:
            raise ValueError(f"Feature '{feature_name}' not documented")

        feature = self.features_[feature_name]
        summary = {
            'name': feature.name,
            'type': feature.feature_type,
            'description': feature.description,
            'status': feature.status
        }

        if feature.statistics:
            summary['statistics'] = {
                'null_percentage': feature.statistics.null_percentage,
                'unique_count': feature.statistics.unique_count,
                'mean': feature.statistics.mean,
                'std': feature.statistics.std,
                'min': feature.statistics.min_value,
                'max': feature.statistics.max_value
            }

        if feature.analysis:
            summary['analysis'] = {
                'target_correlation': feature.analysis.target_correlation,
                'outlier_percentage': feature.analysis.outlier_percentage,
                'completeness_score': feature.analysis.completeness_score,
                'highly_correlated_features': len(feature.analysis.highly_correlated_features)
            }

        return summary

    def search_features(
        self,
        query: Optional[str] = None,
        feature_type: Optional[str] = None,
        tag: Optional[str] = None,
        owner: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[str]:
        """
        Search for features based on criteria.

        Args:
            query: Text to search in descriptions
            feature_type: Filter by feature type
            tag: Filter by tag
            owner: Filter by owner
            status: Filter by status

        Returns:
            List of matching feature names
        """
        matching_features = []

        for name, feature in self.features_.items():
            match = True

            # Text search
            if query:
                query_lower = query.lower()
                if (query_lower not in feature.description.lower() and
                    query_lower not in feature.business_meaning.lower() and
                    query_lower not in feature.rationale.lower()):
                    match = False

            # Feature type filter
            if feature_type and feature.feature_type != feature_type:
                match = False

            # Tag filter
            if tag and tag not in feature.tags:
                match = False

            # Owner filter
            if owner and feature.owner != owner:
                match = False

            # Status filter
            if status and feature.status != status:
                match = False

            if match:
                matching_features.append(name)

        return matching_features

    def validate_documentation(self, feature_name: str) -> Dict[str, Any]:
        """
        Validate feature documentation completeness.

        Args:
            feature_name: Name of the feature

        Returns:
            Dictionary with validation results
        """
        if feature_name not in self.features_:
            raise ValueError(f"Feature '{feature_name}' not documented")

        feature = self.features_[feature_name]
        validation = {
            'complete': True,
            'warnings': [],
            'errors': [],
            'score': 100
        }

        # Check required fields
        required_fields = ['description', 'feature_type', 'source']
        for field in required_fields:
            if not getattr(feature, field):
                validation['errors'].append(f"Missing required field: {field}")
                validation['complete'] = False
                validation['score'] -= 20

        # Check business context
        if not feature.business_meaning:
            validation['warnings'].append("No business meaning provided")
            validation['score'] -= 10

        if not feature.rationale:
            validation['warnings'].append("No rationale provided")
            validation['score'] -= 10

        # Check statistics
        if not feature.statistics:
            validation['warnings'].append("No statistics calculated")
            validation['score'] -= 15

        # Check analysis
        if not feature.analysis:
            validation['warnings'].append("No analysis performed")
            validation['score'] -= 15

        # Check for issues in statistics
        if feature.statistics:
            if feature.statistics.null_percentage > 50:
                validation['warnings'].append(f"High null percentage: {feature.statistics.null_percentage:.1f}%")
                validation['score'] -= 5

            if feature.statistics.unique_count == 1:
                validation['warnings'].append("Feature has only one unique value")
                validation['score'] -= 10

        # Store validation results
        feature.validation_results = validation
        feature.last_validated = datetime.now().isoformat()

        return validation