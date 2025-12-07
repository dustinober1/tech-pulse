"""
Tests for hyperparameter optimization framework.

This module tests the HyperparameterOptimizer class and its strategies,
including property-based tests for hyperparameter tuning documentation.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any
import tempfile
import json
import os

from src.portfolio.experimentation.optimizer import (
    HyperparameterOptimizer,
    SearchSpace,
    GridSearchStrategy,
    RandomSearchStrategy,
    BayesianOptimizationStrategy,
    OptimizationResult
)


class TestHyperparameterOptimizer:
    """Test cases for HyperparameterOptimizer"""

    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = HyperparameterOptimizer(random_state=42)

        # Simple objective function for testing
        def quadratic_objective(params: Dict[str, Any]) -> float:
            """Simple quadratic function: (x-2)^2 + (y-3)^2"""
            x = params.get('x', 0)
            y = params.get('y', 0)
            return (x - 2) ** 2 + (y - 3) ** 2

        self.objective = quadratic_objective

        # Test search space
        self.search_space = [
            SearchSpace(name='x', type='real', low=0, high=5),
            SearchSpace(name='y', type='real', low=0, high=5)
        ]

    def test_grid_search_optimization(self):
        """Test grid search optimization strategy"""
        result = self.optimizer.optimize(
            objective=self.objective,
            search_space=self.search_space,
            n_calls=25,
            strategy='grid'
        )

        assert isinstance(result, OptimizationResult)
        assert result.optimization_method == 'grid'
        assert result.total_evaluations <= 25
        assert len(result.all_results) == result.total_evaluations
        assert 'x' in result.best_params
        assert 'y' in result.best_params
        assert abs(result.best_params['x'] - 2) <= 0.5  # Should be close to optimum
        assert abs(result.best_params['y'] - 3) <= 0.5  # Should be close to optimum

    def test_random_search_optimization(self):
        """Test random search optimization strategy"""
        result = self.optimizer.optimize(
            objective=self.objective,
            search_space=self.search_space,
            n_calls=50,
            strategy='random'
        )

        assert isinstance(result, OptimizationResult)
        assert result.optimization_method == 'random'
        assert result.total_evaluations == 50
        assert len(result.all_results) == 50

    def test_bayesian_optimization(self):
        """Test Bayesian optimization strategy"""
        # Skip if scikit-optimize not available
        pytest.importorskip('skopt')

        result = self.optimizer.optimize(
            objective=self.objective,
            search_space=self.search_space,
            n_calls=20,
            strategy='bayesian_gp'
        )

        assert isinstance(result, OptimizationResult)
        assert result.optimization_method == 'bayesian_gp'
        assert result.total_evaluations == 20

    def test_categorical_parameters(self):
        """Test optimization with categorical parameters"""
        categorical_space = [
            SearchSpace(name='optimizer', type='categorical', values=['sgd', 'adam', 'rmsprop']),
            SearchSpace(name='learning_rate', type='real', low=0.001, high=0.1, log=True)
        ]

        def categorical_objective(params: Dict[str, Any]) -> float:
            # Mock objective that prefers 'adam' with specific learning rate
            lr = params['learning_rate']
            if params['optimizer'] == 'adam':
                return abs(lr - 0.01)  # Prefer lr=0.01 for adam
            else:
                return 1.0  # Higher score for other optimizers

        result = self.optimizer.optimize(
            objective=categorical_objective,
            search_space=categorical_space,
            n_calls=30,
            strategy='grid'
        )

        assert result.best_params['optimizer'] == 'adam'
        assert abs(result.best_params['learning_rate'] - 0.01) < 0.005

    def test_integer_parameters(self):
        """Test optimization with integer parameters"""
        integer_space = [
            SearchSpace(name='n_estimators', type='integer', low=10, high=100),
            SearchSpace(name='max_depth', type='integer', low=1, high=10, log=True)
        ]

        def integer_objective(params: Dict[str, Any]) -> float:
            # Mock objective that prefers specific values
            n_estimators = params['n_estimators']
            max_depth = params['max_depth']
            return abs(n_estimators - 50) + abs(max_depth - 5)

        result = self.optimizer.optimize(
            objective=integer_objective,
            search_space=integer_space,
            n_calls=30,
            strategy='grid'
        )

        assert isinstance(result.best_params['n_estimators'], int)
        assert isinstance(result.best_params['max_depth'], int)

    def test_maximization_objective(self):
        """Test optimization with maximization objective"""
        def linear_objective(params: Dict[str, Any]) -> float:
            """Linear function that increases with x"""
            x = params.get('x', 0)
            return x  # Maximized at highest x value

        space = [SearchSpace(name='x', type='real', low=0, high=5)]

        result = self.optimizer.optimize(
            objective=linear_objective,
            search_space=space,
            n_calls=20,
            strategy='grid',
            minimize=False
        )

        assert result.best_score > 0  # Should be positive
        assert result.best_params['x'] > 0  # Should be positive

    def test_create_search_space_from_dict(self):
        """Test creating search space from dictionary"""
        space_dict = {
            'learning_rate': {
                'type': 'real',
                'low': 1e-4,
                'high': 1e-1,
                'log': True
            },
            'n_estimators': {
                'type': 'integer',
                'low': 50,
                'high': 500
            },
            'max_depth': {
                'type': 'categorical',
                'values': [3, 5, 7, 9]
            }
        }

        search_space = self.optimizer.create_search_space(space_dict)

        assert len(search_space) == 3
        assert search_space[0].name == 'learning_rate'
        assert search_space[0].type == 'real'
        assert search_space[0].log == True
        assert search_space[1].name == 'n_estimators'
        assert search_space[1].type == 'integer'
        assert search_space[2].name == 'max_depth'
        assert search_space[2].type == 'categorical'

    def test_save_and_load_results(self):
        """Test saving and loading optimization results"""
        result = self.optimizer.optimize(
            objective=self.objective,
            search_space=self.search_space,
            n_calls=10,
            strategy='random'
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            # Save results
            self.optimizer.save_results(result, filepath)
            assert os.path.exists(filepath)

            # Load results
            loaded_result = self.optimizer.load_results(filepath)

            assert loaded_result.best_params == result.best_params
            assert loaded_result.best_score == result.best_score
            assert len(loaded_result.all_results) == len(result.all_results)
            assert loaded_result.optimization_method == result.optimization_method
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_optimization_report(self):
        """Test generation of optimization report"""
        result = self.optimizer.optimize(
            objective=self.objective,
            search_space=self.search_space,
            n_calls=10,
            strategy='random'
        )

        report = self.optimizer.get_optimization_report(result)

        assert "Hyperparameter Optimization Report" in report
        assert result.optimization_method in report
        assert str(result.total_evaluations) in report
        assert "Best Parameters:" in report
        assert "Search Space:" in report
        assert "Top 5 Results:" in report

    def test_reproducibility_with_random_state(self):
        """Test that results are reproducible with random state"""
        # Test that grid search is reproducible (deterministic)
        optimizer1 = HyperparameterOptimizer(random_state=42)
        optimizer2 = HyperparameterOptimizer(random_state=42)

        result1 = optimizer1.optimize(
            objective=self.objective,
            search_space=self.search_space,
            n_calls=20,
            strategy='grid'
        )

        result2 = optimizer2.optimize(
            objective=self.objective,
            search_space=self.search_space,
            n_calls=20,
            strategy='grid'
        )

        # Grid search results should be identical (deterministic)
        assert result1.best_params == result2.best_params
        assert result1.best_score == result2.best_score

    def test_invalid_strategy(self):
        """Test handling of invalid optimization strategy"""
        with pytest.raises(ValueError, match="Unknown strategy"):
            self.optimizer.optimize(
                objective=self.objective,
                search_space=self.search_space,
                n_calls=10,
                strategy='invalid_strategy'
            )


class TestHyperparameterOptimizationDocumentation:
    """Property-based tests for hyperparameter tuning documentation (Property 7)"""

    def test_search_space_documentation_completeness(self):
        """
        Property 7: Hyperparameter tuning documentation

        Validates that the search space and optimization strategy
        are properly documented with all required metadata
        """
        optimizer = HyperparameterOptimizer(random_state=42)

        # Create a complex search space
        space_dict = {
            'learning_rate': {
                'type': 'real',
                'low': 1e-4,
                'high': 1e-1,
                'log': True
            },
            'n_estimators': {
                'type': 'integer',
                'low': 10,
                'high': 1000
            },
            'max_depth': {
                'type': 'categorical',
                'values': [3, 5, 7, 9, None]
            },
            'subsample': {
                'type': 'real',
                'low': 0.5,
                'high': 1.0
            }
        }

        search_space = optimizer.create_search_space(space_dict)

        # Verify documentation completeness
        for space in search_space:
            # Each parameter should have a name
            assert hasattr(space, 'name')
            assert space.name is not None
            assert isinstance(space.name, str)

            # Each parameter should have a type
            assert hasattr(space, 'type')
            assert space.type in ['categorical', 'real', 'integer']

            # Type-specific documentation
            if space.type == 'categorical':
                assert hasattr(space, 'values')
                assert space.values is not None
                assert len(space.values) > 0
                # Document the value choices
                assert isinstance(space.values, list)
            else:
                # Real/integer should have bounds
                assert hasattr(space, 'low')
                assert hasattr(space, 'high')
                assert space.low is not None
                assert space.high is not None
                assert space.low < space.high

                # Document whether log scale is used
                assert hasattr(space, 'log')
                assert isinstance(space.log, bool)

    def test_optimization_strategy_documentation(self):
        """
        Property 7: Hyperparameter tuning documentation

        Validates that optimization strategies are documented
        with their characteristics and parameters
        """
        optimizer = HyperparameterOptimizer(random_state=42)

        # Test that each strategy has proper documentation
        for strategy_name, strategy in optimizer.strategies.items():
            # Strategy should have a name
            assert strategy_name is not None
            assert isinstance(strategy_name, str)

            # Strategy should be callable
            assert hasattr(strategy, 'optimize')
            assert callable(strategy.optimize)

            # Strategy should accept standard parameters
            # This is verified by successful optimization calls

    def test_optimization_results_documentation(self):
        """
        Property 7: Hyperparameter tuning documentation

        Validates that optimization results are documented
        with complete metadata
        """
        optimizer = HyperparameterOptimizer(random_state=42)

        def test_objective(params):
            # Handle both numeric and categorical parameters
            score = 0
            for p in params.values():
                if isinstance(p, (int, float)):
                    score += p ** 2
                else:
                    # Categorical parameter - just count it
                    score += 1
            return score

        search_space = [
            SearchSpace(name='x1', type='real', low=-1, high=1),
            SearchSpace(name='x2', type='integer', low=1, high=10),
            SearchSpace(name='x3', type='categorical', values=['a', 'b', 'c'])
        ]

        # Run optimization
        result = optimizer.optimize(
            objective=test_objective,
            search_space=search_space,
            n_calls=20,
            strategy='random',
            experiment_name='test_optimization'
        )

        # Verify result documentation completeness
        assert isinstance(result, OptimizationResult)

        # Best parameters should be documented
        assert hasattr(result, 'best_params')
        assert isinstance(result.best_params, dict)
        assert len(result.best_params) == 3  # One for each parameter

        # Best score should be documented
        assert hasattr(result, 'best_score')
        assert isinstance(result.best_score, (int, float))

        # All results should be documented
        assert hasattr(result, 'all_results')
        assert isinstance(result.all_results, list)
        assert len(result.all_results) == 20  # n_calls

        # Each result should have params and score
        for r in result.all_results:
            assert 'params' in r
            assert 'score' in r
            assert isinstance(r['params'], dict)
            assert isinstance(r['score'], (int, float))

        # Search space should be documented
        assert hasattr(result, 'search_space')
        assert len(result.search_space) == 3

        # Optimization method should be documented
        assert hasattr(result, 'optimization_method')
        assert result.optimization_method == 'random'

        # Total evaluations should be documented
        assert hasattr(result, 'total_evaluations')
        assert isinstance(result.total_evaluations, int)
        assert result.total_evaluations == 20

        # Optimization time should be documented
        assert hasattr(result, 'optimization_time')
        assert isinstance(result.optimization_time, (int, float))
        assert result.optimization_time > 0

    def test_experiment_tracking_integration(self):
        """
        Property 7: Hyperparameter tuning documentation

        Validates that hyperparameter tuning is properly
        tracked in the experiment tracking system
        """
        pytest.importorskip('src.portfolio.experimentation.tracker')

        from src.portfolio.experimentation.tracker import ExperimentTracker
        from src.portfolio.experimentation.database import init_experiment_db

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            init_experiment_db(db_path)
            tracker = ExperimentTracker(db_path)
            optimizer = HyperparameterOptimizer(experiment_tracker=tracker)

            def test_objective(params):
                return params.get('x', 0) ** 2

            search_space = [SearchSpace(name='x', type='real', low=-1, high=1)]

            # Run optimization with experiment tracking
            result = optimizer.optimize(
                objective=test_objective,
                search_space=search_space,
                n_calls=5,
                strategy='random',
                experiment_name='hyperparameter_test'
            )

            # Verify experiments were logged
            experiments = tracker.list_experiments(name_filter='hyperparameter_test')
            assert len(experiments) == 5  # One for each evaluation

            # Verify each experiment has required documentation
            for exp in experiments:
                assert 'hyperparameter_optimization' in exp.tags
                assert 'random' in exp.tags
                assert 'x' in exp.parameters
                assert 'score' in exp.metrics