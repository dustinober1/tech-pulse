"""
Hyperparameter optimization framework for ML models.

This module provides the HyperparameterOptimizer class with multiple
optimization strategies including grid search, random search, and Bayesian optimization.
"""

import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import itertools
import random
from datetime import datetime

try:
    from skopt import gp_minimize, forest_minimize, gbrt_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

from src.portfolio.experimentation.tracker import ExperimentTracker
from src.portfolio.utils.logging import PortfolioLogger


@dataclass
class SearchSpace:
    """Defines the search space for hyperparameters"""
    name: str
    type: str  # 'categorical', 'real', 'integer'
    values: Optional[List[Any]] = None  # For categorical
    low: Optional[float] = None  # For real/integer
    high: Optional[float] = None  # For real/integer
    log: bool = False  # Whether to use log scale for real/integer

    def to_skopt_space(self):
        """Convert to skopt space if available"""
        if not SKOPT_AVAILABLE:
            return None

        if self.type == 'categorical':
            return Categorical(self.values)
        elif self.type == 'real':
            return Real(self.low, self.high, prior='log-uniform' if self.log else 'uniform')
        elif self.type == 'integer':
            return Integer(self.low, self.high, prior='log-uniform' if self.log else 'uniform')
        else:
            raise ValueError(f"Unknown type: {self.type}")


@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization"""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    search_space: List[SearchSpace]
    optimization_method: str
    total_evaluations: int
    optimization_time: float


class OptimizationStrategy(ABC):
    """Abstract base class for optimization strategies"""

    @abstractmethod
    def optimize(
        self,
        objective: Callable[[Dict[str, Any]], float],
        search_space: List[SearchSpace],
        n_calls: int,
        **kwargs
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Optimize the objective function"""
        pass


class GridSearchStrategy(OptimizationStrategy):
    """Grid search optimization strategy"""

    def optimize(
        self,
        objective: Callable[[Dict[str, Any]], float],
        search_space: List[SearchSpace],
        n_calls: int,
        **kwargs
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Perform grid search optimization"""
        # Generate all combinations
        param_grids = []
        for space in search_space:
            if space.type == 'categorical':
                param_grids.append(space.values)
            elif space.type == 'real':
                # For real values, sample uniformly within the range
                n_values = min(int(np.sqrt(n_calls)), 10)
                if space.log:
                    values = np.logspace(np.log10(space.low), np.log10(space.high), n_values)
                else:
                    values = np.linspace(space.low, space.high, n_values)
                param_grids.append(values.tolist())
            elif space.type == 'integer':
                # For integers, sample within the range
                n_values = min(int(np.sqrt(n_calls)), int(space.high - space.low + 1))
                if space.log:
                    values = np.logspace(np.log10(space.low), np.log10(space.high), n_values)
                    values = np.unique(np.round(values)).astype(int)
                else:
                    values = np.linspace(space.low, space.high, n_values, dtype=int)
                param_grids.append(values.tolist())

        # Generate all combinations
        all_combinations = list(itertools.product(*param_grids))

        # Limit to n_calls if too many combinations
        if len(all_combinations) > n_calls:
            all_combinations = random.sample(all_combinations, n_calls)

        # Evaluate all combinations
        results = []
        for combination in all_combinations:
            params = {
                search_space[i].name: combination[i]
                for i in range(len(search_space))
            }
            score = objective(params)
            results.append({'params': params, 'score': score})

        # Sort by score (assuming lower is better)
        results.sort(key=lambda x: x['score'])

        return results[0]['params'], results


class RandomSearchStrategy(OptimizationStrategy):
    """Random search optimization strategy"""

    def optimize(
        self,
        objective: Callable[[Dict[str, Any]], float],
        search_space: List[SearchSpace],
        n_calls: int,
        **kwargs
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Perform random search optimization"""
        results = []

        for _ in range(n_calls):
            params = {}
            for space in search_space:
                if space.type == 'categorical':
                    params[space.name] = random.choice(space.values)
                elif space.type == 'real':
                    if space.log:
                        params[space.name] = 10 ** random.uniform(
                            np.log10(space.low), np.log10(space.high)
                        )
                    else:
                        params[space.name] = random.uniform(space.low, space.high)
                elif space.type == 'integer':
                    if space.log:
                        params[space.name] = int(round(10 ** random.uniform(
                            np.log10(space.low), np.log10(space.high)
                        )))
                    else:
                        params[space.name] = random.randint(space.low, space.high)

            score = objective(params)
            results.append({'params': params, 'score': score})

        # Sort by score (assuming lower is better)
        results.sort(key=lambda x: x['score'])

        return results[0]['params'], results


class BayesianOptimizationStrategy(OptimizationStrategy):
    """Bayesian optimization strategy using scikit-optimize"""

    def __init__(self, base_estimator='GP', acq_func='EI', random_state=None):
        """
        Initialize Bayesian optimization strategy.

        Args:
            base_estimator: 'GP', 'RF', or 'ET' for Gaussian Process,
                          Random Forest, or Extra Trees
            acq_func: Acquisition function ('EI', 'PI', 'LCB')
            random_state: Random seed for reproducibility
        """
        self.base_estimator = base_estimator
        self.acq_func = acq_func
        self.random_state = random_state

        if not SKOPT_AVAILABLE:
            raise ImportError(
                "scikit-optimize is required for Bayesian optimization. "
                "Install with: pip install scikit-optimize"
            )

    def optimize(
        self,
        objective: Callable[[Dict[str, Any]], float],
        search_space: List[SearchSpace],
        n_calls: int,
        n_initial_points: int = 10,
        **kwargs
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Perform Bayesian optimization"""
        # Convert search space to skopt format
        dimensions = [space.to_skopt_space() for space in search_space]

        # Create objective wrapper
        @use_named_args(dimensions)
        def objective_wrapped(**params):
            return objective(params)

        # Choose optimizer based on base estimator
        if self.base_estimator == 'GP':
            result = gp_minimize(
                objective_wrapped,
                dimensions,
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                acq_func=self.acq_func,
                random_state=self.random_state
            )
        elif self.base_estimator == 'RF':
            result = forest_minimize(
                objective_wrapped,
                dimensions,
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                acq_func=self.acq_func,
                random_state=self.random_state
            )
        elif self.base_estimator == 'ET':
            result = gbrt_minimize(
                objective_wrapped,
                dimensions,
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                acq_func=self.acq_func,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown base_estimator: {self.base_estimator}")

        # Convert results to standard format
        results = []
        for i, (params, score) in enumerate(zip(result.x_iters, result.func_vals)):
            param_dict = {
                search_space[j].name: params[j]
                for j in range(len(search_space))
            }
            results.append({'params': param_dict, 'score': score})

        best_params = {
            search_space[j].name: result.x[j]
            for j in range(len(search_space))
        }

        return best_params, results


class HyperparameterOptimizer:
    """Hyperparameter optimization framework with multiple strategies"""

    def __init__(
        self,
        experiment_tracker: Optional[ExperimentTracker] = None,
        default_strategy: str = 'random',
        random_state: Optional[int] = None
    ):
        """
        Initialize hyperparameter optimizer.

        Args:
            experiment_tracker: Optional experiment tracker for logging results
            default_strategy: Default optimization strategy
            random_state: Random seed for reproducibility
        """
        self.experiment_tracker = experiment_tracker
        self.default_strategy = default_strategy
        self.random_state = random_state
        self.logger = PortfolioLogger('hyperparameter_optimization')

        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

        # Initialize strategies
        self.strategies = {
            'grid': GridSearchStrategy(),
            'random': RandomSearchStrategy()
        }

        # Add Bayesian strategies if scikit-optimize is available
        if SKOPT_AVAILABLE:
            self.strategies.update({
                'bayesian_gp': BayesianOptimizationStrategy(
                    base_estimator='GP',
                    random_state=random_state
                ),
                'bayesian_rf': BayesianOptimizationStrategy(
                    base_estimator='RF',
                    random_state=random_state
                ),
                'bayesian_et': BayesianOptimizationStrategy(
                    base_estimator='ET',
                    random_state=random_state
                )
            })

    def optimize(
        self,
        objective: Callable[[Dict[str, Any]], float],
        search_space: List[SearchSpace],
        n_calls: int = 50,
        strategy: Optional[str] = None,
        experiment_name: Optional[str] = None,
        minimize: bool = True,
        **strategy_kwargs
    ) -> OptimizationResult:
        """
        Perform hyperparameter optimization.

        Args:
            objective: Function to minimize/maximize. Should take params dict and return score
            search_space: List of SearchSpace objects defining the parameter space
            n_calls: Number of function evaluations
            strategy: Optimization strategy to use
            experiment_name: Optional name for experiment tracking
            minimize: Whether to minimize (True) or maximize (False) the objective
            **strategy_kwargs: Additional arguments for the optimization strategy

        Returns:
            OptimizationResult with best parameters and all results
        """
        start_time = datetime.now()

        # Use default strategy if not specified
        if strategy is None:
            strategy = self.default_strategy

        # Validate strategy
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.strategies.keys())}")

        # Wrap objective to handle maximization and logging
        def objective_wrapped(params: Dict[str, Any]) -> float:
            score = objective(params)
            return -score if not minimize else score

        # Perform optimization
        self.logger.info(f"Starting hyperparameter optimization with strategy: {strategy}")
        self.logger.info(f"Search space: {[s.name for s in search_space]}")
        self.logger.info(f"Number of evaluations: {n_calls}")

        best_params, all_results = self.strategies[strategy].optimize(
            objective_wrapped,
            search_space,
            n_calls,
            **strategy_kwargs
        )

        # Convert scores back if maximizing
        if not minimize:
            best_score = -min(r['score'] for r in all_results)
            for result in all_results:
                result['score'] = -result['score']
        else:
            best_score = min(r['score'] for r in all_results)

        # Log to experiment tracker if provided
        if self.experiment_tracker and experiment_name:
            for i, result in enumerate(all_results):
                self.experiment_tracker.log_experiment(
                    name=f"{experiment_name}_{strategy}_{i}",
                    params=result['params'],
                    metrics={'score': result['score']},
                    artifacts={},
                    tags=[strategy, 'hyperparameter_optimization']
                )

        optimization_time = (datetime.now() - start_time).total_seconds()

        # Create result object
        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            search_space=search_space,
            optimization_method=strategy,
            total_evaluations=len(all_results),
            optimization_time=optimization_time
        )

        self.logger.info(f"Optimization completed. Best score: {best_score:.4f}")
        self.logger.info(f"Best parameters: {json.dumps(best_params, indent=2)}")

        return result

    def create_search_space(self, space_dict: Dict[str, Dict[str, Any]]) -> List[SearchSpace]:
        """
        Create search space from dictionary.

        Args:
            space_dict: Dictionary mapping parameter names to their specifications

        Returns:
            List of SearchSpace objects

        Example:
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
                    'values': [3, 5, 7, 9, None]
                }
            }
        """
        search_space = []
        for name, spec in space_dict.items():
            search_space.append(SearchSpace(name=name, **spec))
        return search_space

    def save_results(self, results: OptimizationResult, filepath: str) -> None:
        """
        Save optimization results to file.

        Args:
            results: OptimizationResult to save
            filepath: Path to save results
        """
        # Convert to serializable format
        serializable = asdict(results)

        # Convert search space objects
        serializable['search_space'] = [asdict(s) for s in results.search_space]

        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)

        self.logger.info(f"Results saved to: {filepath}")

    def load_results(self, filepath: str) -> OptimizationResult:
        """
        Load optimization results from file.

        Args:
            filepath: Path to load results from

        Returns:
            OptimizationResult
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Convert search space back to objects
        data['search_space'] = [SearchSpace(**s) for s in data['search_space']]

        return OptimizationResult(**data)

    def get_optimization_report(self, results: OptimizationResult) -> str:
        """
        Generate a text report of the optimization results.

        Args:
            results: OptimizationResult to report on

        Returns:
            String report
        """
        report = []
        report.append("Hyperparameter Optimization Report")
        report.append("=" * 50)
        report.append(f"Optimization Method: {results.optimization_method}")
        report.append(f"Total Evaluations: {results.total_evaluations}")
        report.append(f"Optimization Time: {results.optimization_time:.2f} seconds")
        report.append("")

        report.append("Best Parameters:")
        report.append("-" * 20)
        for param, value in results.best_params.items():
            report.append(f"  {param}: {value}")
        report.append(f"\nBest Score: {results.best_score:.6f}")
        report.append("")

        report.append("Search Space:")
        report.append("-" * 20)
        for space in results.search_space:
            if space.type == 'categorical':
                report.append(f"  {space.name}: {space.values}")
            else:
                log_str = " (log scale)" if space.log else ""
                report.append(f"  {space.name}: [{space.low}, {space.high}]{log_str}")
        report.append("")

        # Top 5 results
        sorted_results = sorted(results.all_results, key=lambda x: x['score'])
        report.append("Top 5 Results:")
        report.append("-" * 20)
        for i, result in enumerate(sorted_results[:5]):
            report.append(f"  {i+1}. Score: {result['score']:.6f}")
            report.append(f"     Params: {json.dumps(result['params'], separators=(',', ':'))}")

        return "\n".join(report)