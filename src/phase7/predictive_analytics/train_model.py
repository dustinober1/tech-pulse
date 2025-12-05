"""
Predictive Analytics - Model Training Pipeline

Handles training, evaluation, and persistence of
predictive models with automated hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import pickle
import json
import logging
from pathlib import Path
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb

# Import local modules
from src.phase7.predictive_analytics.training_data import (
    TrainingDataset,
    TrainingDataCollector
)
from src.phase7.predictive_analytics.predictor import PredictiveEngine
from src.phase7.predictive_analytics.features import prepare_features_for_model
from src.cache_manager import cache_manager

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Handles training and evaluation of predictive models.

    Features:
    - Multiple algorithm support
    - Hyperparameter optimization
    - Time series cross-validation
    - Model persistence
    - Performance tracking
    """

    def __init__(
        self,
        models_dir: str = "src/phase7/models",
        results_dir: str = "src/phase7/results",
        n_trials: int = 50,
        cv_folds: int = 5
    ):
        """
        Initialize the model trainer.

        Args:
            models_dir: Directory to save trained models
            results_dir: Directory to save training results
            n_trials: Number of hyperparameter optimization trials
            cv_folds: Number of cross-validation folds
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.n_trials = n_trials
        self.cv_folds = cv_folds

        # Initialize data collector
        self.data_collector = TrainingDataCollector()

        # Supported models
        self.model_types = {
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'xgboost': xgb.XGBRegressor,
            'lightgbm': lgb.LGBMRegressor,
            'linear': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'svr': SVR
        }

    def train_all_models(
        self,
        technologies: List[str],
        lookback_days: int = 365,
        prediction_horizon: int = 30,
        target_metric: str = 'commit_count',
        optimize_hyperparams: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train models for multiple technologies.

        Args:
            technologies: List of technology names
            lookback_days: Historical data window
            prediction_horizon: Prediction horizon in days
            target_metric: Metric to predict
            optimize_hyperparams: Whether to optimize hyperparameters

        Returns:
            Dictionary with training results for each technology
        """
        all_results = {}

        for technology in technologies:
            logger.info(f"Training models for {technology}")

            try:
                results = self.train_technology_models(
                    technology,
                    lookback_days,
                    prediction_horizon,
                    target_metric,
                    optimize_hyperparams
                )
                all_results[technology] = results

            except Exception as e:
                logger.error(f"Failed to train models for {technology}: {str(e)}")
                all_results[technology] = {'error': str(e)}

        # Save summary
        self._save_training_summary(all_results)

        return all_results

    def train_technology_models(
        self,
        technology: str,
        lookback_days: int = 365,
        prediction_horizon: int = 30,
        target_metric: str = 'commit_count',
        optimize_hyperparams: bool = True,
        model_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Train models for a single technology.

        Args:
            technology: Name of the technology
            lookback_days: Historical data window
            prediction_horizon: Prediction horizon in days
            target_metric: Metric to predict
            optimize_hyperparams: Whether to optimize hyperparameters
            model_types: List of model types to train (default: all)

        Returns:
            Dictionary with training results
        """
        if model_types is None:
            model_types = list(self.model_types.keys())

        # Prepare training data
        dataset = self.data_collector.prepare_training_dataset(
            technology,
            lookback_days,
            prediction_horizon,
            target_metric
        )

        # Validate dataset
        if not self.data_collector.validate_dataset(dataset):
            raise ValueError(f"Invalid training dataset for {technology}")

        logger.info(f"Training {len(model_types)} models for {technology}")

        # Train each model type
        results = {
            'technology': technology,
            'target_metric': target_metric,
            'training_date': datetime.now().isoformat(),
            'dataset_info': {
                'records': len(dataset.features_df),
                'features': len(dataset.feature_columns),
                'date_range': [d.isoformat() for d in dataset.date_range]
            },
            'models': {}
        }

        best_score = float('inf')
        best_model = None
        best_model_type = None

        for model_type in model_types:
            logger.info(f"Training {model_type} model")

            try:
                # Train model
                model_result = self._train_single_model(
                    dataset,
                    model_type,
                    optimize_hyperparams
                )

                results['models'][model_type] = model_result

                # Track best model
                if model_result['cv_score'] < best_score:
                    best_score = model_result['cv_score']
                    best_model = model_result['model']
                    best_model_type = model_type

                # Save model
                self._save_model(technology, model_type, model_result['model'])

            except Exception as e:
                logger.error(f"Failed to train {model_type}: {str(e)}")
                results['models'][model_type] = {'error': str(e)}

        # Save best model as default
        if best_model:
            self._save_model(technology, 'best', best_model)
            results['best_model'] = best_model_type
            results['best_score'] = best_score

        # Save results
        self._save_training_results(technology, results)

        return results

    def _train_single_model(
        self,
        dataset: TrainingDataset,
        model_type: str,
        optimize_hyperparams: bool
    ) -> Dict[str, Any]:
        """Train a single model type."""
        # Prepare features
        X, scaler = prepare_features_for_model(
            dataset.features_df,
            dataset.feature_columns,
            scale_features=True
        )
        y = dataset.features_df[dataset.target_column].values

        # Create time series split
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)

        # Initialize model
        model_class = self.model_types[model_type]

        # Hyperparameter optimization
        if optimize_hyperparams and model_type in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']:
            best_params = self._optimize_hyperparameters(
                model_class,
                X, y,
                tscv,
                model_type
            )
            model = model_class(**best_params, random_state=42)
        else:
            # Use default parameters
            default_params = self._get_default_params(model_type)
            model = model_class(**default_params)

        # Train model
        model.fit(X, y)

        # Evaluate with cross-validation
        cv_scores = cross_val_score(
            model, X, y,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        cv_score = -cv_scores.mean()

        # Make predictions on full dataset
        predictions = model.predict(X)

        # Calculate metrics
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)

        # Feature importance (if available)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(
                dataset.feature_columns,
                model.feature_importances_
            ))

        return {
            'model': model,
            'scaler': scaler,
            'cv_score': cv_score,
            'cv_std': cv_scores.std(),
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'feature_importance': feature_importance,
            'hyperparameters': model.get_params()
        }

    def _optimize_hyperparameters(
        self,
        model_class,
        X: np.ndarray,
        y: np.ndarray,
        cv,
        model_type: str
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        def objective(trial):
            # Define hyperparameter search space
            if model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
            elif model_type == 'gradient_boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
            elif model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0)
                }
            else:
                return float('inf')

            params['random_state'] = 42

            # Create and evaluate model
            model = model_class(**params)
            scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            return -scores.mean()

        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)

        return study.best_params

    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default hyperparameters for a model type."""
        defaults = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42,
                'verbose': -1
            },
            'linear': {},
            'ridge': {'alpha': 1.0},
            'lasso': {'alpha': 1.0},
            'svr': {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale'
            }
        }
        return defaults.get(model_type, {})

    def _save_model(
        self,
        technology: str,
        model_type: str,
        model: Any
    ) -> None:
        """Save trained model to disk."""
        model_path = self.models_dir / f"{technology}_{model_type}_model.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        logger.info(f"Model saved to {model_path}")

    def _save_training_results(
        self,
        technology: str,
        results: Dict[str, Any]
    ) -> None:
        """Save training results to disk."""
        results_path = self.results_dir / f"{technology}_training_results.json"

        # Remove model objects from results (can't serialize)
        results_copy = results.copy()
        if 'models' in results_copy:
            for model_name, model_data in results_copy['models'].items():
                if isinstance(model_data, dict) and 'model' in model_data:
                    model_data_copy = model_data.copy()
                    model_data_copy.pop('model', None)
                    results_copy['models'][model_name] = model_data_copy

        with open(results_path, 'w') as f:
            json.dump(results_copy, f, indent=2)

        logger.info(f"Training results saved to {results_path}")

    def _save_training_summary(
        self,
        all_results: Dict[str, Dict[str, Any]]
    ) -> None:
        """Save summary of all training results."""
        summary = {
            'training_date': datetime.now().isoformat(),
            'technologies': {}
        }

        for tech, results in all_results.items():
            if 'error' not in results:
                summary['technologies'][tech] = {
                    'best_model': results.get('best_model'),
                    'best_score': results.get('best_score'),
                    'models_trained': len(results.get('models', {})),
                    'dataset_records': results.get('dataset_info', {}).get('records', 0)
                }
            else:
                summary['technologies'][tech] = {'error': results['error']}

        summary_path = self.results_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def evaluate_model(
        self,
        technology: str,
        model_type: str = 'best',
        test_days: int = 30
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model on recent data.

        Args:
            technology: Name of the technology
            model_type: Type of model to evaluate
            test_days: Number of recent days to use for testing

        Returns:
            Dictionary with evaluation results
        """
        # Load model
        model_path = self.models_dir / f"{technology}_{model_type}_model.pkl"
        if not model_path.exists():
            raise ValueError(f"No trained model found for {technology} ({model_type})")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Prepare test data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=test_days)

        test_data = self.data_collector.collect_historical_data(
            technology,
            start_date,
            end_date
        )

        if test_data.empty:
            raise ValueError(f"No test data available for {technology}")

        # Engineer features
        from src.phase7.predictive_analytics.features import engineer_features
        features_df = engineer_features(test_data)

        # Prepare features (use same columns as training)
        feature_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        if feature_columns is None:
            # Try to load from training results
            results_path = self.results_dir / f"{technology}_training_results.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    training_results = json.load(f)
                # Find feature columns from the trained model
                for model_name, model_data in training_results.get('models', {}).items():
                    if 'feature_importance' in model_data:
                        feature_columns = list(model_data['feature_importance'].keys())
                        break

        if feature_columns is None:
            raise ValueError("Could not determine feature columns for evaluation")

        # Prepare features and make predictions
        X_test = features_df[feature_columns].fillna(0)
        if hasattr(model, 'scaler_'):
            X_test = model.scaler_.transform(X_test)

        y_test = features_df['commit_count'].values
        y_pred = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100

        return {
            'technology': technology,
            'model_type': model_type,
            'test_days': test_days,
            'test_samples': len(y_test),
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'actual_values': y_test.tolist(),
            'predicted_values': y_pred.tolist(),
            'evaluation_date': datetime.now().isoformat()
        }

    def retrain_model(
        self,
        technology: str,
        incremental: bool = False,
        lookback_days: int = 365
    ) -> Dict[str, Any]:
        """
        Retrain a model with new data.

        Args:
            technology: Name of the technology
            incremental: Whether to use incremental learning
            lookback_days: Historical data window

        Returns:
            Dictionary with retraining results
        """
        logger.info(f"Retraining model for {technology}")

        # Check if existing model exists
        model_path = self.models_dir / f"{technology}_best_model.pkl"
        existed = model_path.exists()

        # Retrain model
        results = self.train_technology_models(
            technology,
            lookback_days,
            optimize_hyperparams=True
        )

        # Compare performance if model existed before
        if existed:
            # Get previous performance
            results_path = self.results_dir / f"{technology}_training_results.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    old_results = json.load(f)

                if 'best_score' in old_results:
                    improvement = old_results['best_score'] - results['best_score']
                    results['performance_change'] = improvement
                    results['previous_best_score'] = old_results['best_score']

        results['retraining_date'] = datetime.now().isoformat()
        results['incremental'] = incremental

        return results

    def get_model_info(self, technology: str) -> Dict[str, Any]:
        """Get information about trained models for a technology."""
        info = {
            'technology': technology,
            'models': {}
        }

        # Check for model files
        for model_type in ['best'] + list(self.model_types.keys()):
            model_path = self.models_dir / f"{technology}_{model_type}_model.pkl"
            results_path = self.results_dir / f"{technology}_training_results.json"

            if model_path.exists():
                info['models'][model_type] = {
                    'model_file': str(model_path),
                    'file_size': model_path.stat().st_size,
                    'modified': datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
                }

                # Load additional info from results
                if results_path.exists():
                    with open(results_path, 'r') as f:
                        results = json.load(f)

                    if model_type in results.get('models', {}):
                        model_info = results['models'][model_type]
                        info['models'][model_type].update({
                            'cv_score': model_info.get('cv_score'),
                            'mae': model_info.get('mae'),
                            'rmse': model_info.get('rmse'),
                            'r2': model_info.get('r2')
                        })

        return info