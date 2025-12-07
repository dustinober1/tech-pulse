"""
Tests for experiment tracking system.

Includes property-based tests and unit tests for experiment logging,
retrieval, and comparison.
"""

import os
import tempfile
import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime

from src.portfolio.experimentation import ExperimentTracker, Experiment
from src.portfolio.experimentation.database import init_experiment_db


# Strategies for property-based testing
@st.composite
def experiment_params(draw):
    """Generate random experiment parameters"""
    n_params = draw(st.integers(min_value=1, max_value=10))
    params = {}
    for _ in range(n_params):
        key = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll'), min_codepoint=65, max_codepoint=122)))
        value = draw(st.one_of(
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
            st.integers(min_value=-1000, max_value=1000),
            st.text(min_size=1, max_size=50)
        ))
        params[key] = value
    return params


@st.composite
def experiment_metrics(draw):
    """Generate random experiment metrics"""
    n_metrics = draw(st.integers(min_value=1, max_value=10))
    metrics = {}
    for _ in range(n_metrics):
        key = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll'), min_codepoint=65, max_codepoint=122)))
        value = draw(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))
        metrics[key] = value
    return metrics


class TestExperimentTrackerPropertyBased:
    """Property-based tests for ExperimentTracker"""
    
    @given(
        name=st.text(min_size=1, max_size=100),
        params=experiment_params(),
        metrics=experiment_metrics()
    )
    @settings(max_examples=100, deadline=None)
    def test_property_experiment_logging_completeness(self, name, params, metrics):
        """
        **Feature: portfolio-enhancement, Property 4: Experiment logging completeness**
        
        Property: For any model training run, the experiment tracker should log 
        all hyperparameters, metrics, and artifacts without omission.
        
        **Validates: Requirements 2.1**
        """
        # Create temporary tracker for this test
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            init_experiment_db(db_path)
            tracker = ExperimentTracker(db_path)
            
            artifacts = {'test_artifact': 'path/to/artifact.pkl'}
            
            # Log experiment
            exp_id = tracker.log_experiment(
                name=name,
                params=params,
                metrics=metrics,
                artifacts=artifacts
            )
            
            # Retrieve experiment
            retrieved = tracker.get_experiment(exp_id)
            
            # Verify all parameters are logged
            assert retrieved.parameters == params, "Parameters not logged completely"
            
            # Verify all metrics are logged
            assert retrieved.metrics == metrics, "Metrics not logged completely"
            
            # Verify all artifacts are logged
            assert retrieved.artifacts == artifacts, "Artifacts not logged completely"
            
            # Verify experiment name
            assert retrieved.name == name, "Experiment name not logged correctly"
        finally:
            # Cleanup
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    @given(
        name=st.text(min_size=1, max_size=100),
        params=experiment_params(),
        metrics=experiment_metrics()
    )
    @settings(max_examples=100, deadline=None)
    def test_property_experiment_reproducibility(self, name, params, metrics):
        """
        **Feature: portfolio-enhancement, Property 6: Reproducibility artifact storage**
        
        Property: For any completed experiment, the system should store training scripts 
        and data versions sufficient to reproduce the results.
        
        **Validates: Requirements 2.3**
        """
        # Create temporary tracker
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            init_experiment_db(db_path)
            tracker = ExperimentTracker(db_path)
            
            # Simulate reproducibility artifacts
            artifacts = {
                'training_script': 'scripts/train.py',
                'data_version': 'data/v1.0/dataset.csv',
                'requirements': 'requirements.txt',
                'config': 'config/experiment.json'
            }
            model_path = 'models/model_v1.pkl'
            
            # Log experiment with reproducibility artifacts
            exp_id = tracker.log_experiment(
                name=name,
                params=params,
                metrics=metrics,
                artifacts=artifacts,
                model_path=model_path
            )
            
            # Retrieve experiment
            retrieved = tracker.get_experiment(exp_id)
            
            # Verify all reproducibility artifacts are stored
            assert 'training_script' in retrieved.artifacts, "Training script not stored"
            assert 'data_version' in retrieved.artifacts, "Data version not stored"
            assert 'requirements' in retrieved.artifacts, "Requirements not stored"
            assert 'config' in retrieved.artifacts, "Config not stored"
            assert retrieved.model_path == model_path, "Model path not stored"
            
            # Verify parameters are stored (needed for reproducibility)
            assert retrieved.parameters == params, "Parameters not stored for reproducibility"
            
            # Verify experiment is marked as completed
            assert retrieved.status == 'completed', "Experiment not marked as completed"
        finally:
            # Cleanup
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestExperimentTracker:
    """Unit tests for ExperimentTracker"""
    
    @pytest.fixture
    def temp_tracker(self):
        """Create temporary tracker for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        init_experiment_db(db_path)
        tracker = ExperimentTracker(db_path)
        
        yield tracker
        
        # Cleanup
        os.unlink(db_path)
    
    def test_log_experiment_returns_id(self, temp_tracker):
        """Test that logging experiment returns valid ID"""
        exp_id = temp_tracker.log_experiment(
            name='test',
            params={'lr': 0.01},
            metrics={'acc': 0.95},
            artifacts={}
        )
        
        assert exp_id is not None
        assert isinstance(exp_id, str)
        assert len(exp_id) > 0
    
    def test_get_experiment_not_found(self, temp_tracker):
        """Test that getting non-existent experiment raises error"""
        with pytest.raises(ValueError, match="not found"):
            temp_tracker.get_experiment('nonexistent-id')
    
    def test_log_experiment_with_tags(self, temp_tracker):
        """Test logging experiment with tags"""
        tags = ['test', 'baseline', 'v1']
        
        exp_id = temp_tracker.log_experiment(
            name='tagged_experiment',
            params={'lr': 0.01},
            metrics={'acc': 0.95},
            artifacts={},
            tags=tags
        )
        
        retrieved = temp_tracker.get_experiment(exp_id)
        assert set(retrieved.tags) == set(tags)
    
    def test_log_experiment_with_notes(self, temp_tracker):
        """Test logging experiment with notes"""
        notes = "This is a test experiment with important notes"
        
        exp_id = temp_tracker.log_experiment(
            name='noted_experiment',
            params={'lr': 0.01},
            metrics={'acc': 0.95},
            artifacts={},
            notes=notes
        )
        
        retrieved = temp_tracker.get_experiment(exp_id)
        assert retrieved.notes == notes
    
    def test_log_experiment_with_model_path(self, temp_tracker):
        """Test logging experiment with model path"""
        model_path = 'models/test_model.pkl'
        
        exp_id = temp_tracker.log_experiment(
            name='model_experiment',
            params={'lr': 0.01},
            metrics={'acc': 0.95},
            artifacts={},
            model_path=model_path
        )
        
        retrieved = temp_tracker.get_experiment(exp_id)
        assert retrieved.model_path == model_path
    
    def test_list_experiments(self, temp_tracker):
        """Test listing experiments"""
        # Log multiple experiments
        exp_ids = []
        for i in range(3):
            exp_id = temp_tracker.log_experiment(
                name=f'experiment_{i}',
                params={'lr': 0.01 * (i + 1)},
                metrics={'acc': 0.9 + i * 0.01},
                artifacts={}
            )
            exp_ids.append(exp_id)
        
        # List all experiments
        experiments = temp_tracker.list_experiments()
        assert len(experiments) == 3
    
    def test_list_experiments_with_name_filter(self, temp_tracker):
        """Test listing experiments with name filter"""
        # Log experiments with different names
        temp_tracker.log_experiment('baseline_v1', {}, {}, {})
        temp_tracker.log_experiment('baseline_v2', {}, {}, {})
        temp_tracker.log_experiment('advanced_v1', {}, {}, {})
        
        # Filter by name
        baseline_exps = temp_tracker.list_experiments(name_filter='baseline')
        assert len(baseline_exps) == 2
        assert all('baseline' in exp.name for exp in baseline_exps)
    
    def test_list_experiments_with_tag_filter(self, temp_tracker):
        """Test listing experiments with tag filter"""
        # Log experiments with different tags
        temp_tracker.log_experiment('exp1', {}, {}, {}, tags=['production'])
        temp_tracker.log_experiment('exp2', {}, {}, {}, tags=['production', 'v2'])
        temp_tracker.log_experiment('exp3', {}, {}, {}, tags=['development'])
        
        # Filter by tag
        prod_exps = temp_tracker.list_experiments(tag_filter='production')
        assert len(prod_exps) == 2
    
    def test_list_experiments_with_limit(self, temp_tracker):
        """Test listing experiments with limit"""
        # Log multiple experiments
        for i in range(5):
            temp_tracker.log_experiment(f'exp_{i}', {}, {}, {})
        
        # Get limited results
        limited = temp_tracker.list_experiments(limit=3)
        assert len(limited) == 3
    
    def test_update_experiment_status(self, temp_tracker):
        """Test updating experiment status"""
        exp_id = temp_tracker.log_experiment('test', {}, {}, {})
        
        # Update status
        temp_tracker.update_experiment_status(exp_id, 'failed', 'Test failure')
        
        # Verify update
        retrieved = temp_tracker.get_experiment(exp_id)
        assert retrieved.status == 'failed'
        assert retrieved.notes == 'Test failure'
    
    def test_delete_experiment(self, temp_tracker):
        """Test deleting experiment"""
        exp_id = temp_tracker.log_experiment('test', {}, {}, {})
        
        # Delete experiment
        temp_tracker.delete_experiment(exp_id)
        
        # Verify deletion
        with pytest.raises(ValueError, match="not found"):
            temp_tracker.get_experiment(exp_id)
    
    def test_get_best_model_max(self, temp_tracker):
        """Test getting best model with max optimization"""
        # Log experiments with different accuracies
        exp_ids = []
        for i, acc in enumerate([0.85, 0.92, 0.88]):
            exp_id = temp_tracker.log_experiment(
                f'exp_{i}',
                {'lr': 0.01},
                {'accuracy': acc},
                {}
            )
            exp_ids.append(exp_id)
        
        # Get best model
        best_id, best_exp = temp_tracker.get_best_model('accuracy', mode='max')
        assert best_exp.metrics['accuracy'] == 0.92
    
    def test_get_best_model_min(self, temp_tracker):
        """Test getting best model with min optimization"""
        # Log experiments with different losses
        for i, loss in enumerate([0.15, 0.08, 0.12]):
            temp_tracker.log_experiment(
                f'exp_{i}',
                {'lr': 0.01},
                {'loss': loss},
                {}
            )
        
        # Get best model
        best_id, best_exp = temp_tracker.get_best_model('loss', mode='min')
        assert best_exp.metrics['loss'] == 0.08
    
    def test_get_best_model_with_filter(self, temp_tracker):
        """Test getting best model with name filter"""
        # Log experiments
        temp_tracker.log_experiment('baseline_v1', {}, {'acc': 0.85}, {})
        temp_tracker.log_experiment('baseline_v2', {}, {'acc': 0.90}, {})
        temp_tracker.log_experiment('advanced_v1', {}, {'acc': 0.95}, {})
        
        # Get best baseline model
        best_id, best_exp = temp_tracker.get_best_model(
            'acc', mode='max', name_filter='baseline'
        )
        assert best_exp.metrics['acc'] == 0.90
        assert 'baseline' in best_exp.name
    
    def test_get_best_model_no_experiments(self, temp_tracker):
        """Test getting best model when no experiments exist"""
        with pytest.raises(ValueError, match="No experiments found"):
            temp_tracker.get_best_model('acc')
    
    def test_get_best_model_metric_not_found(self, temp_tracker):
        """Test getting best model when metric doesn't exist"""
        temp_tracker.log_experiment('test', {}, {'acc': 0.9}, {})
        
        with pytest.raises(ValueError, match="No experiments found with metric"):
            temp_tracker.get_best_model('nonexistent_metric')
    
    def test_get_experiment_summary(self, temp_tracker):
        """Test getting experiment summary"""
        # Log multiple experiments
        for i in range(3):
            temp_tracker.log_experiment(
                'test_model',
                {'lr': 0.01},
                {'acc': 0.9 + i * 0.01},
                {}
            )
        
        # Get summary
        summary = temp_tracker.get_experiment_summary()
        assert len(summary) > 0
        assert 'name' in summary.columns
        assert 'num_experiments' in summary.columns
    
    def test_compare_experiments_basic(self, temp_tracker):
        """Test comparison with multiple experiments"""
        # Log experiments with different configurations
        exp_ids = []
        for i in range(3):
            exp_id = temp_tracker.log_experiment(
                f'experiment_{i}',
                {'learning_rate': 0.01 * (i + 1), 'n_estimators': 100 + i * 50},
                {'accuracy': 0.85 + i * 0.03, 'f1_score': 0.80 + i * 0.04},
                {}
            )
            exp_ids.append(exp_id)
        
        # Compare experiments
        comparison = temp_tracker.compare_experiments(exp_ids)
        
        # Verify comparison DataFrame structure
        assert len(comparison) == 3
        assert 'experiment_id' in comparison.columns
        assert 'name' in comparison.columns
        assert 'timestamp' in comparison.columns
        
        # Verify parameters are included
        assert 'param_learning_rate' in comparison.columns
        assert 'param_n_estimators' in comparison.columns
        
        # Verify metrics are included
        assert 'metric_accuracy' in comparison.columns
        assert 'metric_f1_score' in comparison.columns
        
        # Verify values are correct (with floating point tolerance)
        import numpy as np
        assert comparison['param_learning_rate'].tolist() == [0.01, 0.02, 0.03]
        np.testing.assert_array_almost_equal(
            comparison['metric_accuracy'].tolist(), 
            [0.85, 0.88, 0.91], 
            decimal=10
        )
    
    def test_compare_experiments_with_metric_filter(self, temp_tracker):
        """Test comparison with specific metrics"""
        # Log experiments
        exp_ids = []
        for i in range(2):
            exp_id = temp_tracker.log_experiment(
                f'exp_{i}',
                {'lr': 0.01},
                {'accuracy': 0.9, 'precision': 0.85, 'recall': 0.88, 'f1': 0.86},
                {}
            )
            exp_ids.append(exp_id)
        
        # Compare with specific metrics
        comparison = temp_tracker.compare_experiments(exp_ids, metrics=['accuracy', 'f1'])
        
        # Verify only specified metrics are included
        assert 'metric_accuracy' in comparison.columns
        assert 'metric_f1' in comparison.columns
        assert 'metric_precision' not in comparison.columns
        assert 'metric_recall' not in comparison.columns
    
    def test_compare_experiments_empty_list(self, temp_tracker):
        """Test comparison with empty experiment list"""
        comparison = temp_tracker.compare_experiments([])
        assert len(comparison) == 0
    
    def test_compare_experiments_visualization_data(self, temp_tracker):
        """Test that comparison provides data suitable for visualization"""
        # Log experiments with varying performance
        exp_ids = []
        accuracies = [0.75, 0.82, 0.88, 0.91, 0.85]
        
        for i, acc in enumerate(accuracies):
            exp_id = temp_tracker.log_experiment(
                f'model_v{i}',
                {'version': i},
                {'accuracy': acc, 'loss': 1.0 - acc},
                {}
            )
            exp_ids.append(exp_id)
        
        # Get comparison
        comparison = temp_tracker.compare_experiments(exp_ids)
        
        # Verify data is suitable for plotting
        assert len(comparison) == len(accuracies)
        assert comparison['metric_accuracy'].tolist() == accuracies
        
        # Verify we can identify best and worst
        best_idx = comparison['metric_accuracy'].idxmax()
        worst_idx = comparison['metric_accuracy'].idxmin()
        assert comparison.loc[best_idx, 'metric_accuracy'] == 0.91
        assert comparison.loc[worst_idx, 'metric_accuracy'] == 0.75
    
    def test_compare_experiments_with_missing_metrics(self, temp_tracker):
        """Test comparison when experiments have different metrics"""
        # Log experiments with different metric sets
        exp1_id = temp_tracker.log_experiment(
            'exp1',
            {},
            {'accuracy': 0.9, 'f1': 0.85},
            {}
        )
        
        exp2_id = temp_tracker.log_experiment(
            'exp2',
            {},
            {'accuracy': 0.92, 'precision': 0.88},
            {}
        )
        
        # Compare experiments
        comparison = temp_tracker.compare_experiments([exp1_id, exp2_id])
        
        # Verify all metrics are included
        assert 'metric_accuracy' in comparison.columns
        assert 'metric_f1' in comparison.columns
        assert 'metric_precision' in comparison.columns
        
        # Verify NaN handling for missing metrics
        import pandas as pd
        assert pd.isna(comparison.loc[comparison['experiment_id'] == exp1_id, 'metric_precision'].values[0])
        assert pd.isna(comparison.loc[comparison['experiment_id'] == exp2_id, 'metric_f1'].values[0])
