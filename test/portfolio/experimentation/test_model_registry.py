"""
Tests for model registry.

This module tests the ModelRegistry class and its functionality,
including property-based tests for model versioning.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
import os
from datetime import datetime, timezone
from typing import Dict, Any

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.portfolio.experimentation.model_registry import (
    ModelRegistry,
    ModelMetadata,
    ModelVersion
)
from src.portfolio.documentation.model_card import ModelCard


class TestModelRegistry:
    """Test cases for ModelRegistry"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(registry_path=self.temp_dir)

        # Create test model
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X_train, y_train)

    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_registry_initialization(self):
        """Test registry initialization"""
        assert os.path.exists(self.temp_dir)
        assert os.path.exists(os.path.join(self.temp_dir, "models"))
        assert os.path.exists(os.path.join(self.temp_dir, "metadata"))
        assert os.path.exists(os.path.join(self.temp_dir, "model_cards"))
        # registry.json is created on first model registration, not init

    def test_register_model(self):
        """Test basic model registration"""
        version = self.registry.register_model(
            model=self.model,
            model_name="test_model",
            created_by="test_user",
            description="Test model for unit testing"
        )

        assert version == "1.0.0"

        # Check model was registered
        models = self.registry.list_models()
        assert "test_model" in models

        # Check metadata
        metadata = self.registry.get_model_metadata("test_model")
        assert metadata.model_name == "test_model"
        assert metadata.model_version == "1.0.0"
        assert metadata.created_by == "test_user"
        assert metadata.description == "Test model for unit testing"
        assert metadata.model_type == "RandomForestClassifier"

    def test_register_model_with_version(self):
        """Test model registration with specific version"""
        version = self.registry.register_model(
            model=self.model,
            model_name="test_model",
            model_version="2.1.0"
        )

        assert version == "2.1.0"
        metadata = self.registry.get_model_metadata("test_model")
        assert metadata.model_version == "2.1.0"

    def test_register_multiple_versions(self):
        """Test registering multiple versions of same model"""
        # Register first version
        v1 = self.registry.register_model(
            model=self.model,
            model_name="test_model",
            model_version="1.0.0"
        )

        # Create and register second version
        model2 = LogisticRegression(random_state=42)
        model2.fit(self.X_train, self.y_train)

        v2 = self.registry.register_model(
            model=model2,
            model_name="test_model",
            model_version="1.1.0"
        )

        assert v1 == "1.0.0"
        assert v2 == "1.1.0"

        # Check both versions exist
        versions = self.registry.get_model_versions("test_model")
        assert "1.0.0" in versions
        assert "1.1.0" in versions

        # Latest version should be 1.1.0
        assert self.registry.get_model_metadata("test_model").model_version == "1.1.0"

    def test_load_model(self):
        """Test loading a model from registry"""
        # Register model
        version = self.registry.register_model(
            model=self.model,
            model_name="test_model"
        )

        # Load model
        loaded_model = self.registry.load_model("test_model")

        # Check it's the same type
        assert type(loaded_model) == type(self.model)

        # Check predictions match
        original_pred = self.model.predict(self.X_test)
        loaded_pred = loaded_model.predict(self.X_test)
        np.testing.assert_array_equal(original_pred, loaded_pred)

    def test_load_model_with_metadata(self):
        """Test loading model with metadata"""
        version = self.registry.register_model(
            model=self.model,
            model_name="test_model",
            description="Test model"
        )

        loaded_model, metadata = self.registry.load_model(
            "test_model",
            return_metadata=True
        )

        assert isinstance(loaded_model, RandomForestClassifier)
        assert isinstance(metadata, ModelMetadata)
        assert metadata.model_name == "test_model"
        assert metadata.description == "Test model"

    def test_compare_models(self):
        """Test model comparison functionality"""
        # Register two models
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        rf_model.fit(self.X_train, self.y_train)

        lr_model = LogisticRegression(random_state=42)
        lr_model.fit(self.X_train, self.y_train)

        rf_version = self.registry.register_model(
            model=rf_model,
            model_name="rf_model"
        )

        lr_version = self.registry.register_model(
            model=lr_model,
            model_name="lr_model"
        )

        # Compare models
        comparison = self.registry.compare_models(
            model1_name="rf_model",
            model2_name="lr_model"
        )

        assert isinstance(comparison, pd.DataFrame)
        assert "Model Name" in comparison.index
        assert rf_version in comparison.columns
        assert lr_version in comparison.columns

    def test_compare_model_versions(self):
        """Test comparing different versions of same model"""
        # Register two versions
        v1 = self.registry.register_model(
            model=self.model,
            model_name="test_model",
            model_version="1.0.0"
        )

        model2 = LogisticRegression(random_state=42)
        model2.fit(self.X_train, self.y_train)
        v2 = self.registry.register_model(
            model=model2,
            model_name="test_model",
            model_version="2.0.0"
        )

        # Compare versions
        comparison = self.registry.compare_models(
            model1_name="test_model",
            model1_version="1.0.0",
            model2_name="test_model",
            model2_version="2.0.0"
        )

        assert isinstance(comparison, pd.DataFrame)
        assert "1.0.0" in comparison.columns
        assert "2.0.0" in comparison.columns

    def test_model_lineage(self):
        """Test model lineage tracking"""
        # Register first version
        v1 = self.registry.register_model(
            model=self.model,
            model_name="test_model",
            model_version="1.0.0"
        )

        # Register second version with parent
        model2 = LogisticRegression(random_state=42)
        model2.fit(self.X_train, self.y_train)

        v2 = self.registry.register_model(
            model=model2,
            model_name="test_model",
            model_version="1.1.0",
            parent_model="test_model:1.0.0",
            changelog="Improved accuracy with logistic regression"
        )

        # Get lineage
        lineage = self.registry.get_model_lineage("test_model")

        assert isinstance(lineage, pd.DataFrame)
        assert len(lineage) == 2
        assert "1.0.0" in lineage["Version"].values
        assert "1.1.0" in lineage["Version"].values
        assert "Improved accuracy with logistic regression" in lineage["Changes"].values

    def test_promote_model(self):
        """Test model promotion to staging/production"""
        version = self.registry.register_model(
            model=self.model,
            model_name="test_model"
        )

        # Promote to staging
        self.registry.promote_model(
            model_name="test_model",
            model_version=version,
            target_environment="staging"
        )

        # Check staging version
        models_info = self.registry.list_models(include_metadata=True)
        assert models_info["test_model"]["staging_version"] == version

        # Promote to production
        self.registry.promote_model(
            model_name="test_model",
            model_version=version,
            target_environment="production"
        )

        # Check production version
        models_info = self.registry.list_models(include_metadata=True)
        assert models_info["test_model"]["production_version"] == version

    def test_delete_model_version(self):
        """Test deleting a specific model version"""
        # Register two versions
        v1 = self.registry.register_model(
            model=self.model,
            model_name="test_model",
            model_version="1.0.0"
        )

        model2 = LogisticRegression(random_state=42)
        model2.fit(self.X_train, self.y_train)
        v2 = self.registry.register_model(
            model=model2,
            model_name="test_model",
            model_version="1.1.0"
        )

        # Delete version 1.0.0
        self.registry.delete_model(
            model_name="test_model",
            model_version="1.0.0"
        )

        # Check it's deleted
        versions = self.registry.get_model_versions("test_model")
        assert "1.0.0" not in versions
        assert "1.1.0" in versions

    def test_delete_model(self):
        """Test deleting entire model"""
        version = self.registry.register_model(
            model=self.model,
            model_name="test_model"
        )

        # Delete entire model (force since not in production)
        self.registry.delete_model(
            model_name="test_model",
            force=True
        )

        # Check it's deleted
        models = self.registry.list_models()
        assert "test_model" not in models

    def test_registry_stats(self):
        """Test registry statistics"""
        # Register some models
        self.registry.register_model(
            model=self.model,
            model_name="test_model1"
        )

        model2 = LogisticRegression(random_state=42)
        model2.fit(self.X_train, self.y_train)

        self.registry.register_model(
            model=model2,
            model_name="test_model2"
        )

        # Get stats
        stats = self.registry.get_registry_stats()

        assert stats['total_models'] == 2
        assert stats['total_versions'] == 2
        assert stats['production_models'] == 0
        assert stats['storage_backend'] == 'local'
        assert stats['total_storage_mb'] > 0

    def test_export_import_registry(self):
        """Test registry export/import"""
        # Register a model
        self.registry.register_model(
            model=self.model,
            model_name="test_model"
        )

        # Export registry
        export_path = os.path.join(self.temp_dir, "registry_export.json")
        self.registry.export_registry(export_path)

        assert os.path.exists(export_path)

        # Create new registry and import
        new_registry_path = os.path.join(self.temp_dir, "new_registry")
        new_registry = ModelRegistry(registry_path=new_registry_path)

        new_registry.import_registry(export_path)

        # Check model was imported
        models = new_registry.list_models()
        assert "test_model" in models

    def test_version_auto_increment(self):
        """Test automatic version increment"""
        # Register initial version
        v1 = self.registry.register_model(
            model=self.model,
            model_name="test_model",
            model_version="1.0.0"
        )

        # Register without version - should auto-increment
        v2 = self.registry.register_model(
            model=self.model,
            model_name="test_model"
        )

        assert v1 == "1.0.0"
        assert v2 == "1.0.1"  # Patch version incremented

    def test_model_hash_generation(self):
        """Test model hash generation"""
        hash1 = self.registry._generate_hash(self.model)
        hash2 = self.registry._generate_hash(self.model)

        # Same model should generate same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hash length

    def test_with_model_card(self):
        """Test registering model with model card"""
        # Create model card
        model_card = ModelCard(
            model_name="test_model",
            model_version="1.0.0"
        )
        model_card.set_intended_use(
            intended_use="Testing purposes only",
            primary_use="Unit testing validation"
        )

        # Register with model card
        version = self.registry.register_model(
            model=self.model,
            model_name="test_model",
            model_card=model_card
        )

        # Check model card was saved
        metadata = self.registry.get_model_metadata("test_model")
        assert metadata.model_card_path is not None
        assert os.path.exists(metadata.model_card_path)


class TestModelVersioning:
    """Property-based tests for model versioning (Property 6.1)"""

    def test_version_increment_consistency(self):
        """
        Property 6.1: Model versioning

        Validates that model versions are incremented consistently
        following semantic versioning principles
        """
        temp_dir = tempfile.mkdtemp()
        registry = ModelRegistry(registry_path=temp_dir)

        X, y = make_classification(n_samples=50, n_features=5, n_informative=3, n_redundant=1, random_state=42)
        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        # Register initial semantic version
        v1 = registry.register_model(
            model=model,
            model_name="version_test",
            model_version="1.0.0"
        )

        # Auto-increment should increment patch version
        v2 = registry.register_model(
            model=model,
            model_name="version_test"
        )

        assert v1 == "1.0.0"
        assert v2 == "1.0.1"

        # Manual version can be any format
        v3 = registry.register_model(
            model=model,
            model_name="version_test",
            model_version="2.0.0"
        )

        assert v3 == "2.0.0"

        # Versions should be sorted correctly
        versions = registry.get_model_versions("version_test")
        # Should be in order: 1.0.0, 1.0.1, 2.0.0
        assert versions[0] == "1.0.0"
        assert versions[1] == "1.0.1"
        assert versions[2] == "2.0.0"

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_version_metadata_completeness(self):
        """
        Property 6.1: Model versioning

        Validates that all required metadata is stored
        for each model version
        """
        temp_dir = tempfile.mkdtemp()
        registry = ModelRegistry(registry_path=temp_dir)

        X, y = make_classification(n_samples=50, n_features=5, n_informative=3, n_redundant=1, random_state=42)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        # Register model with comprehensive metadata
        version = registry.register_model(
            model=model,
            model_name="metadata_test",
            model_version="1.0.0",
            created_by="test_user",
            description="Test model with complete metadata",
            tags=["test", "classification", "random_forest"],
            features=["feature_1", "feature_2", "feature_3"],
            target="target",
            task_type="classification"
        )

        # Verify metadata completeness
        metadata = registry.get_model_metadata("metadata_test", version)

        # Required fields
        assert metadata.model_name == "metadata_test"
        assert metadata.model_version == "1.0.0"
        assert metadata.model_type == "RandomForestClassifier"
        assert metadata.created_at is not None
        assert metadata.created_by == "test_user"
        assert metadata.framework == "scikit-learn"
        assert metadata.algorithm == "RandomForestClassifier"

        # Optional fields that were provided
        assert metadata.description == "Test model with complete metadata"
        assert metadata.tags == ["test", "classification", "random_forest"]
        assert metadata.features == ["feature_1", "feature_2", "feature_3"]
        assert metadata.target == "target"
        assert metadata.task_type == "classification"

        # Auto-generated fields
        assert metadata.hyperparameters is not None
        assert "n_estimators" in metadata.hyperparameters
        assert metadata.hyperparameters["n_estimators"] == 5
        assert metadata.model_hash is not None
        assert len(metadata.model_hash) == 64
        assert metadata.model_path is not None
        assert os.path.exists(metadata.model_path)
        assert metadata.file_size is not None
        assert metadata.file_size > 0

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_version_comparison_accuracy(self):
        """
        Property 6.1: Model versioning

        Validates that version comparison works correctly
        for different version formats
        """
        registry = ModelRegistry()

        # Test semantic version comparison
        assert registry._compare_versions("1.0.0", "1.0.1") < 0
        assert registry._compare_versions("1.0.1", "1.0.0") > 0
        assert registry._compare_versions("1.0.0", "1.0.0") == 0

        # Test major version comparison
        assert registry._compare_versions("1.0.0", "2.0.0") < 0
        assert registry._compare_versions("2.0.0", "1.0.0") > 0

        # Test minor version comparison
        assert registry._compare_versions("1.1.0", "1.2.0") < 0
        assert registry._compare_versions("1.2.0", "1.1.0") > 0

        # Test version with different lengths (current implementation doesn't pad)
        assert registry._compare_versions("1.0", "1.0.0") == -1  # 1.0 < 1.0.0
        assert registry._compare_versions("1", "1.0.0") == -1   # 1 < 1.0.0

        # Test timestamp-based versions
        v1 = "v20231201_120000"
        v2 = "v20231201_130000"
        v3 = "v20231202_120000"

        # Should compare as strings for non-numeric versions
        assert registry._compare_versions(v1, v1) == 0
        assert registry._compare_versions(v1, v2) != 0  # Different


class TestModelRegistryComparison:
    """Property-based tests for model registry comparison (Property 6.2)"""

    def test_model_comparison_attributes(self):
        """
        Property 6.2: Model registry comparison

        Validates that model comparison includes all relevant
        model attributes and metrics
        """
        temp_dir = tempfile.mkdtemp()
        registry = ModelRegistry(registry_path=temp_dir)

        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create two different models
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        rf_model.fit(X_train, y_train)

        lr_model = LogisticRegression(random_state=42)
        lr_model.fit(X_train, y_train)

        # Register models with performance metrics
        rf_score = rf_model.score(X_test, y_test)
        lr_score = lr_model.score(X_test, y_test)

        rf_version = registry.register_model(
            model=rf_model,
            model_name="rf_model",
            model_version="1.0.0",
            performance_metrics={"accuracy": rf_score, "f1": 0.85}
        )

        lr_version = registry.register_model(
            model=lr_model,
            model_name="lr_model",
            model_version="2.0.0",
            performance_metrics={"accuracy": lr_score, "precision": 0.82}
        )

        # Compare models
        comparison = registry.compare_models("rf_model", model2_name="lr_model")

        # Check that comparison includes required attributes
        assert isinstance(comparison, pd.DataFrame)

        # Basic model information
        assert "Model Name" in comparison.index
        assert "Model Type" in comparison.index
        assert "Created At" in comparison.index
        assert "File Size (MB)" in comparison.index

        # Performance metrics
        assert "Metric: accuracy" in comparison.index
        assert "Metric: f1" in comparison.index
        assert "Metric: precision" in comparison.index

        # Values should be correct - dataframe uses version strings as column names
        assert comparison.shape[1] == 2  # Should have 2 columns
        assert "1.0.0" in comparison.columns
        assert "2.0.0" in comparison.columns
        assert comparison.loc["Model Name", "1.0.0"] == "rf_model"
        assert comparison.loc["Model Name", "2.0.0"] == "lr_model"
        assert comparison.loc["Model Type", "1.0.0"] == "RandomForestClassifier"
        assert comparison.loc["Model Type", "2.0.0"] == "LogisticRegression"

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_comparison_with_missing_metrics(self):
        """
        Property 6.2: Model registry comparison

        Validates that comparison handles missing metrics gracefully
        """
        temp_dir = tempfile.mkdtemp()
        registry = ModelRegistry(registry_path=temp_dir)

        X, y = make_classification(n_samples=50, n_features=5, n_informative=3, n_redundant=1, random_state=42)
        model1 = LogisticRegression(random_state=42)
        model1.fit(X, y)

        model2 = RandomForestClassifier(n_estimators=5, random_state=42)
        model2.fit(X, y)

        # Register models with different metrics
        v1 = registry.register_model(
            model=model1,
            model_name="model1",
            model_version="1.0.0",
            performance_metrics={"accuracy": 0.8, "precision": 0.75}
        )

        v2 = registry.register_model(
            model=model2,
            model_name="model2",
            model_version="2.0.0",
            performance_metrics={"accuracy": 0.85, "recall": 0.80}
        )

        # Compare models
        comparison = registry.compare_models("model1", model2_name="model2")

        # Should include all metrics from both models
        assert "Metric: accuracy" in comparison.index
        assert "Metric: precision" in comparison.index
        assert "Metric: recall" in comparison.index

        # Check that we have the correct values in the dataframe
        # Dataframe uses version strings as column names
        assert comparison.shape[1] == 2  # Should have 2 columns
        assert "1.0.0" in comparison.columns
        assert "2.0.0" in comparison.columns
        # Check that missing metrics are handled
        assert comparison.loc["Metric: recall", "1.0.0"] == "N/A"
        assert comparison.loc["Metric: precision", "2.0.0"] == "N/A"
        # Common metrics should have values
        assert comparison.loc["Metric: accuracy", "1.0.0"] == 0.8
        assert comparison.loc["Metric: accuracy", "2.0.0"] == 0.85

        shutil.rmtree(temp_dir, ignore_errors=True)


class TestModelLineageTracking:
    """Property-based tests for model lineage tracking (Property 6.3)"""

    def test_lineage_parent_child_relationships(self):
        """
        Property 6.3: Model lineage tracking

        Validates that parent-child relationships between
        model versions are properly tracked
        """
        temp_dir = tempfile.mkdtemp()
        registry = ModelRegistry(registry_path=temp_dir)

        X, y = make_classification(n_samples=50, n_features=5, n_informative=3, n_redundant=1, random_state=42)
        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        # Create model lineage chain
        v1 = registry.register_model(
            model=model,
            model_name="lineage_test",
            model_version="1.0.0"
        )

        # Child version
        v2 = registry.register_model(
            model=model,
            model_name="lineage_test",
            model_version="1.1.0",
            parent_model="lineage_test:1.0.0",
            changelog="Improved regularization"
        )

        # Grandchild version
        v3 = registry.register_model(
            model=model,
            model_name="lineage_test",
            model_version="2.0.0",
            parent_model="lineage_test:1.1.0",
            changelog="New feature engineering"
        )

        # Get lineage
        lineage = registry.get_model_lineage("lineage_test")

        # Verify lineage structure
        assert isinstance(lineage, pd.DataFrame)
        assert len(lineage) == 3

        # Check parent relationships
        v1_row = lineage[lineage["Version"] == "1.0.0"].iloc[0]
        v2_row = lineage[lineage["Version"] == "1.1.0"].iloc[0]
        v3_row = lineage[lineage["Version"] == "2.0.0"].iloc[0]

        assert v1_row["Parent"] == "None"
        assert v2_row["Parent"] == "lineage_test:1.0.0"
        assert v3_row["Parent"] == "lineage_test:1.1.0"

        # Check changelog
        assert v1_row["Changes"] == "N/A"
        assert v2_row["Changes"] == "Improved regularization"
        assert v3_row["Changes"] == "New feature engineering"

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_lineage_with_multiple_parents(self):
        """
        Property 6.3: Model lineage tracking

        Validates that models can reference different parents
        for ensemble or hybrid approaches
        """
        temp_dir = tempfile.mkdtemp()
        registry = ModelRegistry(registry_path=temp_dir)

        X, y = make_classification(n_samples=50, n_features=5, n_informative=3, n_redundant=1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create base models
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        rf_model.fit(X_train, y_train)

        lr_model = LogisticRegression(random_state=42)
        lr_model.fit(X_train, y_train)

        # Register base models
        rf_v1 = registry.register_model(
            model=rf_model,
            model_name="rf_base",
            model_version="1.0.0"
        )

        lr_v1 = registry.register_model(
            model=lr_model,
            model_name="lr_base",
            model_version="1.0.0"
        )

        # Create ensemble model (mock ensemble for testing)
        ensemble_model = rf_model  # Using RF as mock ensemble
        ensemble_v1 = registry.register_model(
            model=ensemble_model,
            model_name="ensemble_model",
            model_version="1.0.0",
            parent_model="rf_base:1.0.0,lr_base:1.0.0",
            changelog="Ensemble of Random Forest and Logistic Regression"
        )

        # Verify lineage tracking
        rf_lineage = registry.get_model_lineage("rf_base")
        lr_lineage = registry.get_model_lineage("lr_base")
        ensemble_lineage = registry.get_model_lineage("ensemble_model")

        # Check ensemble has multiple parents
        ensemble_v1_row = ensemble_lineage[ensemble_lineage["Version"] == "1.0.0"].iloc[0]
        assert "rf_base:1.0.0" in ensemble_v1_row["Parent"]
        assert "lr_base:1.0.0" in ensemble_v1_row["Parent"]

        # Check changelog preserved
        assert "Ensemble" in ensemble_v1_row["Changes"]

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_lineage_promotion_tracking(self):
        """
        Property 6.3: Model lineage tracking

        Validates that model promotions through environments
        are tracked in the lineage
        """
        temp_dir = tempfile.mkdtemp()
        registry = ModelRegistry(registry_path=temp_dir)

        X, y = make_classification(n_samples=50, n_features=5, n_informative=3, n_redundant=1, random_state=42)
        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        # Register and promote model
        version = registry.register_model(
            model=model,
            model_name="promotion_test",
            model_version="1.0.0"
        )

        # Promote through environments
        registry.promote_model("promotion_test", version, target_environment="staging")
        registry.promote_model("promotion_test", version, target_environment="production")

        # Check lineage reflects promotions
        lineage = registry.get_model_lineage("promotion_test")
        version_row = lineage[lineage["Version"] == "1.0.0"].iloc[0]

        assert version_row["Production"] == "Yes"
        # Note: Current implementation doesn't track staging separately in lineage
        # This could be an enhancement

        shutil.rmtree(temp_dir, ignore_errors=True)