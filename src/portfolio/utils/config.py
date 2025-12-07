"""
Configuration management for portfolio enhancement features.

This module handles loading configuration from environment variables,
config files, and provides default values for all portfolio components.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import json


class PortfolioConfig:
    """Centralized configuration management for portfolio enhancement"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration from file and environment variables.
        
        Args:
            config_file: Optional path to JSON config file
        """
        self.config_file = config_file
        self._config = self._load_defaults()
        
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
        
        self._load_from_env()
    
    def _load_defaults(self) -> Dict[str, Any]:
        """Load default configuration values"""
        return {
            'experiment_tracking': {
                'database_path': 'data/experiments.db',
                'artifact_dir': 'artifacts/',
                'auto_log': True,
                'log_level': 'INFO'
            },
            'model_registry': {
                'storage_path': 'models/',
                'versioning': 'semantic',
                'retention_policy': 'keep_best_n',
                'n_models_to_keep': 5
            },
            'feature_engineering': {
                'nlp_model': 'en_core_web_sm',
                'embedding_model': 'all-MiniLM-L6-v2',
                'max_features': 100,
                'feature_selection_method': 'mutual_info'
            },
            'interpretability': {
                'shap_background_samples': 100,
                'lime_num_features': 10,
                'pdp_grid_resolution': 50
            },
            'data_quality': {
                'missing_threshold': 0.3,
                'outlier_method': 'iqr',
                'outlier_threshold': 3.0,
                'min_samples': 100
            },
            'testing': {
                'coverage_threshold': 0.80,
                'property_test_iterations': 100,
                'random_seed': 42
            }
        }
    
    def _load_from_file(self, config_file: str) -> None:
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            file_config = json.load(f)
        self._merge_config(file_config)
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables"""
        # Experiment tracking
        if db_path := os.getenv('PORTFOLIO_EXPERIMENT_DB'):
            self._config['experiment_tracking']['database_path'] = db_path
        
        if artifact_dir := os.getenv('PORTFOLIO_ARTIFACT_DIR'):
            self._config['experiment_tracking']['artifact_dir'] = artifact_dir
        
        # Model registry
        if model_path := os.getenv('PORTFOLIO_MODEL_PATH'):
            self._config['model_registry']['storage_path'] = model_path
        
        # Testing
        if seed := os.getenv('PORTFOLIO_RANDOM_SEED'):
            self._config['testing']['random_seed'] = int(seed)
    
    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """Merge new configuration into existing config"""
        for key, value in new_config.items():
            if key in self._config and isinstance(value, dict):
                self._config[key].update(value)
            else:
                self._config[key] = value
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            section: Configuration section (e.g., 'experiment_tracking')
            key: Optional key within section
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        if section not in self._config:
            return default
        
        if key is None:
            return self._config[section]
        
        return self._config[section].get(key, default)
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            section: Configuration section
            key: Key within section
            value: Value to set
        """
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value
    
    def save(self, config_file: str) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_file: Path to save configuration
        """
        Path(config_file).parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    def validate(self) -> bool:
        """
        Validate configuration values.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate paths exist or can be created
        for path_key in ['database_path', 'artifact_dir']:
            path = self._config['experiment_tracking'].get(path_key)
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Validate numeric ranges
        if not 0 <= self._config['data_quality']['missing_threshold'] <= 1:
            raise ValueError("missing_threshold must be between 0 and 1")
        
        if self._config['data_quality']['outlier_threshold'] <= 0:
            raise ValueError("outlier_threshold must be positive")
        
        if not 0 <= self._config['testing']['coverage_threshold'] <= 1:
            raise ValueError("coverage_threshold must be between 0 and 1")
        
        return True
    
    @property
    def experiment_tracking(self) -> Dict[str, Any]:
        """Get experiment tracking configuration"""
        return self._config['experiment_tracking']
    
    @property
    def model_registry(self) -> Dict[str, Any]:
        """Get model registry configuration"""
        return self._config['model_registry']
    
    @property
    def feature_engineering(self) -> Dict[str, Any]:
        """Get feature engineering configuration"""
        return self._config['feature_engineering']
    
    @property
    def interpretability(self) -> Dict[str, Any]:
        """Get interpretability configuration"""
        return self._config['interpretability']
    
    @property
    def data_quality(self) -> Dict[str, Any]:
        """Get data quality configuration"""
        return self._config['data_quality']
    
    @property
    def testing(self) -> Dict[str, Any]:
        """Get testing configuration"""
        return self._config['testing']
