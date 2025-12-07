"""
Model registry for tracking and versioning models.

This module provides a comprehensive model registry system for
tracking model versions, performance metrics, and lineage.
"""

import os
import re
import json
import yaml
import hashlib
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import joblib
import pickle
from pathlib import Path
import warnings

from sklearn.base import BaseEstimator
from src.portfolio.utils.logging import PortfolioLogger
from src.portfolio.documentation.model_card import ModelCard


@dataclass
class ModelMetadata:
    """Metadata for a registered model"""
    model_name: str
    model_version: str
    model_type: str
    created_at: datetime
    created_by: str
    framework: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    features: List[str]
    target: Optional[str] = None
    task_type: Optional[str] = None
    model_path: Optional[str] = None
    model_card_path: Optional[str] = None
    training_data_hash: Optional[str] = None
    model_hash: Optional[str] = None
    parent_model: Optional[str] = None
    tags: Optional[List[str]] = None
    description: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    file_size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert datetime to string
        if data['created_at']:
            data['created_at'] = data['created_at'].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        if data.get('created_at'):
            if isinstance(data['created_at'], str):
                data['created_at'] = datetime.fromisoformat(data['created_at'])
            elif isinstance(data['created_at'], datetime):
                pass  # Already datetime
        return cls(**data)


@dataclass
class ModelVersion:
    """Version information for a model"""
    version: str
    created_at: datetime
    changelog: Optional[str] = None
    improvements: Optional[List[str]] = None
    degradations: Optional[List[str]] = None
    bug_fixes: Optional[List[str]] = None
    is_production: bool = False
    is_staging: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        if data['created_at']:
            data['created_at'] = data['created_at'].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary"""
        if data.get('created_at'):
            if isinstance(data['created_at'], str):
                data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class ModelRegistry:
    """
    Model registry for tracking and versioning ML models.

    Provides comprehensive model tracking including versioning,
    lineage, performance metrics, and deployment status.
    """

    def __init__(
        self,
        registry_path: str = "data/model_registry",
        storage_backend: str = "local",
        storage_config: Optional[Dict[str, Any]] = None,
        enable_versioning: bool = True
    ):
        """
        Initialize model registry.

        Args:
            registry_path: Path to registry storage
            storage_backend: Storage backend ('local', 's3', 'gcs')
            storage_config: Configuration for storage backend
            enable_versioning: Whether to enable model versioning
        """
        self.registry_path = Path(registry_path)
        self.storage_backend = storage_backend
        self.storage_config = storage_config or {}
        self.enable_versioning = enable_versioning
        self.logger = PortfolioLogger('model_registry')

        # Create registry directories
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.registry_path / "models"
        self.metadata_dir = self.registry_path / "metadata"
        self.model_cards_dir = self.registry_path / "model_cards"

        for dir_path in [self.models_dir, self.metadata_dir, self.model_cards_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize registry database
        self.registry_file = self.registry_path / "registry.json"
        self._load_registry()

        self.logger.info(f"Model registry initialized at {registry_path}")

    def _load_registry(self) -> None:
        """Load registry database"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
                self.models = data.get('models', {})
                self.version_history = data.get('version_history', {})
        else:
            self.models = {}
            self.version_history = {}

    def _save_registry(self) -> None:
        """Save registry database"""
        data = {
            'models': self.models,
            'version_history': self.version_history,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _generate_hash(
        self,
        obj: Any,
        algorithm: str = 'sha256'
    ) -> str:
        """Generate hash for object"""
        if isinstance(obj, (str, bytes)):
            content = obj if isinstance(obj, bytes) else obj.encode()
        elif isinstance(obj, BaseEstimator):
            # Hash model parameters
            content = json.dumps(obj.get_params(), sort_keys=True).encode()
        elif isinstance(obj, (dict, list)):
            content = json.dumps(obj, sort_keys=True).encode()
        else:
            # For other objects, use pickle
            content = pickle.dumps(obj)

        return hashlib.sha256(content).hexdigest()

    def register_model(
        self,
        model: BaseEstimator,
        model_name: str,
        model_version: Optional[str] = None,
        model_card: Optional[ModelCard] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        created_by: Optional[str] = None,
        training_data: Optional[Any] = None,
        parent_model: Optional[str] = None,
        changelog: Optional[str] = None,
        is_production: bool = False,
        is_staging: bool = False,
        **kwargs
    ) -> str:
        """
        Register a model in the registry.

        Args:
            model: Fitted sklearn model
            model_name: Name of the model
            model_version: Version string (auto-generated if None)
            model_card: ModelCard object
            tags: List of tags
            description: Model description
            created_by: Who created the model
            training_data: Training data used
            parent_model: Parent model for lineage tracking
            changelog: Version changelog
            is_production: Whether this is production version
            is_staging: Whether this is staging version
            **kwargs: Additional metadata

        Returns:
            Model version string
        """
        # Generate version if not provided
        if model_version is None:
            model_version = self._generate_version(model_name)

        self.logger.info(f"Registering model {model_name} v{model_version}")

        # Create model metadata
        metadata = ModelMetadata(
            model_name=model_name,
            model_version=model_version,
            model_type=type(model).__name__,
            created_at=datetime.now(timezone.utc),
            created_by=created_by or os.getenv('USER', 'unknown'),
            framework='scikit-learn',
            algorithm=type(model).__name__,
            hyperparameters=model.get_params(),
            features=kwargs.get('features', []),
            target=kwargs.get('target'),
            task_type=kwargs.get('task_type'),
            tags=tags or [],
            description=description,
            parent_model=parent_model,
            model_hash=self._generate_hash(model)
        )

        # Add training data hash if provided
        if training_data is not None:
            metadata.training_data_hash = self._generate_hash(training_data)

        # Add performance metrics if provided
        if 'performance_metrics' in kwargs:
            metadata.performance_metrics = kwargs['performance_metrics']

        # Save model
        model_filename = f"{model_name}_{model_version}.joblib"
        model_path = self.models_dir / model_filename
        joblib.dump(model, model_path)
        metadata.model_path = str(model_path)
        metadata.file_size = model_path.stat().st_size

        # Save model card if provided
        if model_card:
            model_card_filename = f"{model_name}_{model_version}.md"
            model_card_path = self.model_cards_dir / model_card_filename
            model_card.save(str(model_card_path))
            metadata.model_card_path = str(model_card_path)

        # Store metadata
        if model_name not in self.models:
            self.models[model_name] = {
                'name': model_name,
                'created_at': metadata.created_at.isoformat(),
                'versions': {},
                'latest_version': model_version,
                'production_version': model_version if is_production else None,
                'staging_version': model_version if is_staging else None
            }
        else:
            # Update production/staging versions
            if is_production:
                self.models[model_name]['production_version'] = model_version
            if is_staging:
                self.models[model_name]['staging_version'] = model_version

            # Update latest version if newer
            if self._compare_versions(model_version, self.models[model_name]['latest_version']) > 0:
                self.models[model_name]['latest_version'] = model_version

        # Store version metadata
        self.models[model_name]['versions'][model_version] = metadata.to_dict()

        # Add to version history
        version_info = ModelVersion(
            version=model_version,
            created_at=metadata.created_at,
            changelog=changelog,
            is_production=is_production,
            is_staging=is_staging
        )

        if model_name not in self.version_history:
            self.version_history[model_name] = []
        self.version_history[model_name].append(version_info.to_dict())

        # Save registry
        self._save_registry()

        self.logger.info(f"Model {model_name} v{model_version} registered successfully")

        return model_version

    def _generate_version(self, model_name: str) -> str:
        """Generate next version number"""
        if model_name in self.models:
            latest_version = self.models[model_name]['latest_version']
            # Parse semantic version if possible
            try:
                parts = latest_version.split('.')
                if len(parts) == 3 and all(p.isdigit() for p in parts):
                    # Increment patch version
                    parts[2] = str(int(parts[2]) + 1)
                    return '.'.join(parts)
            except:
                pass
            # Fall back to timestamp-based version
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"v{timestamp}"
        else:
            return "1.0.0"

    def _compare_versions(self, version1: str, version2: str) -> int:
        """Compare two version strings.

        Returns:
            -1 if version1 < version2
             0 if version1 == version2
             1 if version1 > version2
        """
        def normalize(v):
            return [int(x) for x in re.findall(r'\d+', v)] or [v]

        import re
        v1_parts = normalize(version1)
        v2_parts = normalize(version2)

        # Compare part by part
        for p1, p2 in zip(v1_parts, v2_parts):
            if p1 < p2:
                return -1
            elif p1 > p2:
                return 1

        # If one has more parts
        if len(v1_parts) < len(v2_parts):
            return -1
        elif len(v1_parts) > len(v2_parts):
            return 1

        return 0

    def load_model(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        return_metadata: bool = False
    ) -> Union[BaseEstimator, Tuple[BaseEstimator, ModelMetadata]]:
        """
        Load a model from the registry.

        Args:
            model_name: Name of the model
            model_version: Version to load (latest if None)
            return_metadata: Whether to return metadata

        Returns:
            Loaded model and optionally metadata
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")

        # Get version
        if model_version is None:
            model_version = self.models[model_name]['latest_version']
        elif model_version not in self.models[model_name]['versions']:
            raise ValueError(f"Version {model_version} not found for model {model_name}")

        # Get metadata
        version_data = self.models[model_name]['versions'][model_version]
        metadata = ModelMetadata.from_dict(version_data)

        # Load model
        if not metadata.model_path or not os.path.exists(metadata.model_path):
            raise FileNotFoundError(f"Model file not found for {model_name} v{model_version}")

        model = joblib.load(metadata.model_path)

        self.logger.info(f"Loaded model {model_name} v{model_version}")

        if return_metadata:
            return model, metadata
        else:
            return model

    def list_models(self, include_metadata: bool = False) -> Union[List[str], Dict[str, Dict]]:
        """
        List all models in the registry.

        Args:
            include_metadata: Whether to include model metadata

        Returns:
            List of model names or dict with metadata
        """
        if include_metadata:
            return {
                name: {
                    'latest_version': info['latest_version'],
                    'production_version': info.get('production_version'),
                    'staging_version': info.get('staging_version'),
                    'versions': list(info['versions'].keys()),
                    'created_at': info['created_at']
                }
                for name, info in self.models.items()
            }
        else:
            return list(self.models.keys())

    def get_model_versions(self, model_name: str) -> List[str]:
        """
        Get all versions for a model.

        Args:
            model_name: Name of the model

        Returns:
            List of version strings
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")

        versions = list(self.models[model_name]['versions'].keys())
        versions.sort(key=lambda v: tuple(int(x) for x in re.findall(r'\d+', v)) or (0,))
        return versions

    def get_model_metadata(
        self,
        model_name: str,
        model_version: Optional[str] = None
    ) -> ModelMetadata:
        """
        Get metadata for a model version.

        Args:
            model_name: Name of the model
            model_version: Version (latest if None)

        Returns:
            ModelMetadata object
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")

        if model_version is None:
            model_version = self.models[model_name]['latest_version']
        elif model_version not in self.models[model_name]['versions']:
            raise ValueError(f"Version {model_version} not found for model {model_name}")

        version_data = self.models[model_name]['versions'][model_version]
        return ModelMetadata.from_dict(version_data)

    def compare_models(
        self,
        model1_name: str,
        model1_version: Optional[str] = None,
        model2_name: Optional[str] = None,
        model2_version: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare two models or two versions of the same model.

        Args:
            model1_name: Name of first model
            model1_version: Version of first model
            model2_name: Name of second model (same as first if None)
            model2_version: Version of second model
            metrics: List of metrics to compare

        Returns:
            DataFrame with comparison
        """
        if model2_name is None:
            model2_name = model1_name

        # Get metadata
        meta1 = self.get_model_metadata(model1_name, model1_version)
        meta2 = self.get_model_metadata(model2_name, model2_version)

        # Build comparison dataframe
        comparison_data = []

        # Basic info
        comparison_data.append({
            'Aspect': 'Model Name',
            meta1.model_version: meta1.model_name,
            meta2.model_version: meta2.model_name
        })
        comparison_data.append({
            'Aspect': 'Model Type',
            meta1.model_version: meta1.model_type,
            meta2.model_version: meta2.model_type
        })
        comparison_data.append({
            'Aspect': 'Created At',
            meta1.model_version: meta1.created_at.strftime('%Y-%m-%d %H:%M'),
            meta2.model_version: meta2.created_at.strftime('%Y-%m-%d %H:%M')
        })
        comparison_data.append({
            'Aspect': 'File Size (MB)',
            meta1.model_version: f"{meta1.file_size / 1024 / 1024:.2f}" if meta1.file_size else "N/A",
            meta2.model_version: f"{meta2.file_size / 1024 / 1024:.2f}" if meta2.file_size else "N/A"
        })

        # Performance metrics
        if meta1.performance_metrics and meta2.performance_metrics:
            for metric in set(meta1.performance_metrics.keys()) | set(meta2.performance_metrics.keys()):
                if metrics is None or metric in metrics:
                    comparison_data.append({
                        'Aspect': f'Metric: {metric}',
                        meta1.model_version: meta1.performance_metrics.get(metric, 'N/A'),
                        meta2.model_version: meta2.performance_metrics.get(metric, 'N/A')
                    })

        df = pd.DataFrame(comparison_data)
        return df.set_index('Aspect')

    def get_model_lineage(self, model_name: str) -> pd.DataFrame:
        """
        Get lineage information for a model.

        Args:
            model_name: Name of the model

        Returns:
            DataFrame with lineage information
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")

        lineage_data = []

        # Get version history
        if model_name in self.version_history:
            for version_info in self.version_history[model_name]:
                version = ModelVersion.from_dict(version_info)

                # Get metadata for this version
                metadata = self.get_model_metadata(model_name, version.version)

                lineage_data.append({
                    'Version': version.version,
                    'Created At': version.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                    'Parent': metadata.parent_model or 'None',
                    'Production': 'Yes' if version.is_production else 'No',
                    'Staging': 'Yes' if version.is_staging else 'No',
                    'Changes': version.changelog or 'N/A'
                })

        return pd.DataFrame(lineage_data)

    def promote_model(
        self,
        model_name: str,
        model_version: str,
        target_environment: str = 'production'
    ) -> None:
        """
        Promote a model to staging or production.

        Args:
            model_name: Name of the model
            model_version: Version to promote
            target_environment: 'staging' or 'production'
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")

        if model_version not in self.models[model_name]['versions']:
            raise ValueError(f"Version {model_version} not found for model {model_name}")

        # Update registry version tracking
        if target_environment == 'production':
            self.models[model_name]['production_version'] = model_version
            self.logger.info(f"Promoted {model_name} v{model_version} to production")
        elif target_environment == 'staging':
            self.models[model_name]['staging_version'] = model_version
            self.logger.info(f"Promoted {model_name} v{model_version} to staging")
        else:
            raise ValueError("Target environment must be 'staging' or 'production'")

        # Also update version history with promotion status
        if model_name in self.version_history:
            for i, version_info in enumerate(self.version_history[model_name]):
                if version_info['version'] == model_version:
                    # Update the version info with promotion status
                    if target_environment == 'production':
                        self.version_history[model_name][i]['is_production'] = True
                    elif target_environment == 'staging':
                        self.version_history[model_name][i]['is_staging'] = True
                    break

        self._save_registry()

    def delete_model(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        force: bool = False
    ) -> None:
        """
        Delete a model or specific version from registry.

        Args:
            model_name: Name of the model
            model_version: Version to delete (all if None)
            force: Force deletion even if in production
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")

        if model_version is None:
            # Delete entire model
            if not force:
                if self.models[model_name].get('production_version'):
                    raise ValueError("Cannot delete model with production version. Use force=True")

            # Delete all version files
            for version in list(self.models[model_name]['versions'].keys()):
                self._delete_version_files(model_name, version)

            # Remove from registry
            del self.models[model_name]
            if model_name in self.version_history:
                del self.version_history[model_name]

            self.logger.info(f"Deleted model {model_name} from registry")

        else:
            # Delete specific version
            if model_version not in self.models[model_name]['versions']:
                raise ValueError(f"Version {model_version} not found for model {model_name}")

            if not force:
                if self.models[model_name].get('production_version') == model_version:
                    raise ValueError("Cannot delete production version. Use force=True")
                if self.models[model_name].get('staging_version') == model_version:
                    raise ValueError("Cannot delete staging version. Use force=True")

            # Delete files
            self._delete_version_files(model_name, model_version)

            # Remove from versions
            del self.models[model_name]['versions'][model_version]

            # Update version history
            if model_name in self.version_history:
                self.version_history[model_name] = [
                    v for v in self.version_history[model_name]
                    if v['version'] != model_version
                ]

            # Update latest version
            remaining_versions = list(self.models[model_name]['versions'].keys())
            if remaining_versions:
                self.models[model_name]['latest_version'] = max(
                    remaining_versions,
                    key=lambda v: tuple(int(x) for x in re.findall(r'\d+', v)) or (0,)
                )
            else:
                # No versions left, remove model entirely
                del self.models[model_name]
                if model_name in self.version_history:
                    del self.version_history[model_name]

            self.logger.info(f"Deleted {model_name} v{model_version} from registry")

        self._save_registry()

    def _delete_version_files(self, model_name: str, model_version: str) -> None:
        """Delete files for a specific version"""
        metadata = self.get_model_metadata(model_name, model_version)

        # Delete model file
        if metadata.model_path and os.path.exists(metadata.model_path):
            os.remove(metadata.model_path)

        # Delete model card file
        if metadata.model_card_path and os.path.exists(metadata.model_card_path):
            os.remove(metadata.model_card_path)

    def export_registry(self, filepath: str) -> None:
        """Export registry to file"""
        export_data = {
            'models': self.models,
            'version_history': self.version_history,
            'exported_at': datetime.now(timezone.utc).isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        self.logger.info(f"Registry exported to {filepath}")

    def import_registry(self, filepath: str, merge: bool = True) -> None:
        """Import registry from file"""
        with open(filepath, 'r') as f:
            import_data = json.load(f)

        if not merge:
            # Replace entire registry
            self.models = import_data.get('models', {})
            self.version_history = import_data.get('version_history', {})
        else:
            # Merge with existing
            for model_name, model_data in import_data.get('models', {}).items():
                if model_name in self.models:
                    # Merge versions
                    self.models[model_name]['versions'].update(model_data.get('versions', {}))

                    # Update latest version if newer
                    if self._compare_versions(
                        model_data['latest_version'],
                        self.models[model_name]['latest_version']
                    ) > 0:
                        self.models[model_name]['latest_version'] = model_data['latest_version']
                else:
                    self.models[model_name] = model_data

            # Merge version history
            for model_name, history in import_data.get('version_history', {}).items():
                if model_name in self.version_history:
                    self.version_history[model_name].extend(history)
                else:
                    self.version_history[model_name] = history

        self._save_registry()
        self.logger.info(f"Registry imported from {filepath}")

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        total_models = len(self.models)
        total_versions = sum(len(info['versions']) for info in self.models.values())

        production_models = sum(
            1 for info in self.models.values()
            if info.get('production_version')
        )

        # Calculate total storage
        total_storage = 0
        for model_info in self.models.values():
            for version_data in model_info['versions'].values():
                if version_data.get('file_size'):
                    total_storage += version_data['file_size']

        return {
            'total_models': total_models,
            'total_versions': total_versions,
            'production_models': production_models,
            'total_storage_mb': total_storage / 1024 / 1024,
            'storage_backend': self.storage_backend,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }