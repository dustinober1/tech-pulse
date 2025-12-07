"""
Unit tests for configuration management.

Tests environment variable loading, config file parsing, and default value handling.
"""

import os
import json
import tempfile
import pytest
from pathlib import Path
from src.portfolio.utils.config import PortfolioConfig


class TestPortfolioConfig:
    """Test suite for PortfolioConfig class"""
    
    def test_default_values_loaded(self):
        """Test that default configuration values are loaded correctly"""
        config = PortfolioConfig()
        
        # Test experiment tracking defaults
        assert config.get('experiment_tracking', 'database_path') == 'data/experiments.db'
        assert config.get('experiment_tracking', 'artifact_dir') == 'artifacts/'
        assert config.get('experiment_tracking', 'auto_log') is True
        assert config.get('experiment_tracking', 'log_level') == 'INFO'
        
        # Test model registry defaults
        assert config.get('model_registry', 'storage_path') == 'models/'
        assert config.get('model_registry', 'versioning') == 'semantic'
        assert config.get('model_registry', 'n_models_to_keep') == 5
        
        # Test testing defaults
        assert config.get('testing', 'coverage_threshold') == 0.80
        assert config.get('testing', 'property_test_iterations') == 100
        assert config.get('testing', 'random_seed') == 42
    
    def test_config_file_parsing(self):
        """Test that configuration is loaded from JSON file"""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                'experiment_tracking': {
                    'database_path': 'custom/path/experiments.db',
                    'log_level': 'DEBUG'
                },
                'testing': {
                    'random_seed': 123
                }
            }
            json.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            config = PortfolioConfig(temp_config_path)
            
            # Test that custom values override defaults
            assert config.get('experiment_tracking', 'database_path') == 'custom/path/experiments.db'
            assert config.get('experiment_tracking', 'log_level') == 'DEBUG'
            assert config.get('testing', 'random_seed') == 123
            
            # Test that non-overridden defaults remain
            assert config.get('experiment_tracking', 'auto_log') is True
            assert config.get('testing', 'coverage_threshold') == 0.80
        finally:
            os.unlink(temp_config_path)
    
    def test_environment_variable_loading(self):
        """Test that environment variables override config values"""
        # Set environment variables
        os.environ['PORTFOLIO_EXPERIMENT_DB'] = 'env/experiments.db'
        os.environ['PORTFOLIO_ARTIFACT_DIR'] = 'env/artifacts/'
        os.environ['PORTFOLIO_MODEL_PATH'] = 'env/models/'
        os.environ['PORTFOLIO_RANDOM_SEED'] = '999'
        
        try:
            config = PortfolioConfig()
            
            # Test that environment variables are loaded
            assert config.get('experiment_tracking', 'database_path') == 'env/experiments.db'
            assert config.get('experiment_tracking', 'artifact_dir') == 'env/artifacts/'
            assert config.get('model_registry', 'storage_path') == 'env/models/'
            assert config.get('testing', 'random_seed') == 999
        finally:
            # Clean up environment variables
            del os.environ['PORTFOLIO_EXPERIMENT_DB']
            del os.environ['PORTFOLIO_ARTIFACT_DIR']
            del os.environ['PORTFOLIO_MODEL_PATH']
            del os.environ['PORTFOLIO_RANDOM_SEED']
    
    def test_get_with_default(self):
        """Test get method with default values"""
        config = PortfolioConfig()
        
        # Test getting existing value
        assert config.get('testing', 'random_seed') == 42
        
        # Test getting non-existent key with default
        assert config.get('testing', 'nonexistent', 'default_value') == 'default_value'
        
        # Test getting non-existent section with default
        assert config.get('nonexistent_section', 'key', 'default') == 'default'
    
    def test_set_configuration_value(self):
        """Test setting configuration values"""
        config = PortfolioConfig()
        
        # Set new value
        config.set('testing', 'new_key', 'new_value')
        assert config.get('testing', 'new_key') == 'new_value'
        
        # Override existing value
        config.set('testing', 'random_seed', 100)
        assert config.get('testing', 'random_seed') == 100
        
        # Set value in new section
        config.set('new_section', 'key', 'value')
        assert config.get('new_section', 'key') == 'value'
    
    def test_save_configuration(self):
        """Test saving configuration to file"""
        config = PortfolioConfig()
        config.set('testing', 'random_seed', 777)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save(temp_path)
            
            # Load saved config and verify
            with open(temp_path, 'r') as f:
                saved_config = json.load(f)
            
            assert saved_config['testing']['random_seed'] == 777
            assert 'experiment_tracking' in saved_config
        finally:
            os.unlink(temp_path)
    
    def test_validation_success(self):
        """Test that valid configuration passes validation"""
        config = PortfolioConfig()
        assert config.validate() is True
    
    def test_validation_missing_threshold_range(self):
        """Test validation fails for invalid missing_threshold"""
        config = PortfolioConfig()
        
        # Test value > 1
        config.set('data_quality', 'missing_threshold', 1.5)
        with pytest.raises(ValueError, match="missing_threshold must be between 0 and 1"):
            config.validate()
        
        # Test value < 0
        config.set('data_quality', 'missing_threshold', -0.1)
        with pytest.raises(ValueError, match="missing_threshold must be between 0 and 1"):
            config.validate()
    
    def test_validation_outlier_threshold_positive(self):
        """Test validation fails for non-positive outlier_threshold"""
        config = PortfolioConfig()
        config.set('data_quality', 'outlier_threshold', -1.0)
        
        with pytest.raises(ValueError, match="outlier_threshold must be positive"):
            config.validate()
    
    def test_validation_coverage_threshold_range(self):
        """Test validation fails for invalid coverage_threshold"""
        config = PortfolioConfig()
        
        # Test value > 1
        config.set('testing', 'coverage_threshold', 1.2)
        with pytest.raises(ValueError, match="coverage_threshold must be between 0 and 1"):
            config.validate()
        
        # Test value < 0
        config.set('testing', 'coverage_threshold', -0.1)
        with pytest.raises(ValueError, match="coverage_threshold must be between 0 and 1"):
            config.validate()
    
    def test_property_accessors(self):
        """Test property accessors for configuration sections"""
        config = PortfolioConfig()
        
        # Test all property accessors return dictionaries
        assert isinstance(config.experiment_tracking, dict)
        assert isinstance(config.model_registry, dict)
        assert isinstance(config.feature_engineering, dict)
        assert isinstance(config.interpretability, dict)
        assert isinstance(config.data_quality, dict)
        assert isinstance(config.testing, dict)
        
        # Test property values match get method
        assert config.experiment_tracking == config.get('experiment_tracking')
        assert config.testing == config.get('testing')
    
    def test_config_file_not_found(self):
        """Test that non-existent config file doesn't cause error"""
        # Should not raise exception, just use defaults
        config = PortfolioConfig('nonexistent_config.json')
        assert config.get('testing', 'random_seed') == 42
    
    def test_merge_config_nested(self):
        """Test that nested config merging works correctly"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                'experiment_tracking': {
                    'database_path': 'new/path.db'
                    # Note: not including other keys
                }
            }
            json.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            config = PortfolioConfig(temp_config_path)
            
            # Test that new value is set
            assert config.get('experiment_tracking', 'database_path') == 'new/path.db'
            
            # Test that other keys in same section are preserved
            assert config.get('experiment_tracking', 'auto_log') is True
            assert config.get('experiment_tracking', 'log_level') == 'INFO'
        finally:
            os.unlink(temp_config_path)
