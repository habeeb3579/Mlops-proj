"""
NYC Taxi Trip Duration Prediction - MLflow Production Framework
=============================================================

This module provides a production-grade framework for training, tracking,
and registering ML models for NYC Taxi trip duration prediction using MLflow.

Features:
- OOP design with configurable components
- Support for multiple ML models (LinearRegression, Ridge, Lasso, XGBoost, RandomForest, etc.)
- Scikit-learn Pipelines for combining preprocessing and modeling
- ColumnTransformer for flexible feature preprocessing options
- Hyperparameter tuning with hyperopt
- Comprehensive experiment tracking with MLflow
- Flexible storage options (local, SQLite, PostgreSQL, AWS, GCP)
- Model registration and versioning
- Production deployment with aliases and stages

Author: Habeeb Babatunde
Date: May 14, 2025
"""

import os
import yaml
import urllib.parse
import pickle
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime
import uuid
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoLarsCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from scripts.download_data import DownloadProgressBar, download_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type aliases for clarity
ModelType = Any  # Generic model type
ArrayLike = Union[List, np.ndarray, pd.Series]
DictConfig = Dict[str, Any]


class StorageConfig:
    """Configuration for various storage backends for MLflow"""

    @staticmethod
    def get_tracking_uri(storage_type: str, **kwargs) -> str:
        """
        Generate the appropriate tracking URI based on storage type.

        Args:
            storage_type: One of ("sqlite", "postgresql", "aws", "gcp", "local", "remote")
            **kwargs: Parameters like db_path, host, port, user, password, etc.

        Returns:
            str: MLflow tracking URI
        """
        if storage_type == "sqlite":
            return f"sqlite:///{kwargs.get('db_path', 'mlflow.db')}"

        if storage_type == "postgresql":
            return "postgresql://{user}:{password}@{host}:{port}/{database}".format(
                user=kwargs.get("user", "mlflow"),
                password=kwargs.get("password", "mlflow"),
                host=kwargs.get("host", "localhost"),
                port=kwargs.get("port", 5432),
                database=kwargs.get("database", "mlflow")
            )

        # Covers aws, gcp, remote, or custom server-based URIs
        return kwargs.get("tracking_uri", f"http://{kwargs.get('host', 'localhost')}:{kwargs.get('port', 5000)}")

    @staticmethod
    def get_artifact_location(storage_type: str, **kwargs) -> str:
        """
        Generate the appropriate artifact storage location based on storage type.

        Args:
            storage_type: One of ("local", "s3", "gcs")
            **kwargs: Parameters like bucket, s3_bucket, prefix, etc.

        Returns:
            str: Artifact location URI
        """
        def join_path(base: str, prefix: str | None) -> str:
            return f"{base}/{prefix}" if prefix else base

        if storage_type == "s3":
            return join_path(f"s3://{kwargs.get('s3_bucket', 'mlflow-artifacts')}", kwargs.get("prefix"))

        if storage_type == "gcs":
            return join_path(f"gs://{kwargs.get('bucket', 'mlflow-artifacts')}", kwargs.get("prefix"))

        return kwargs.get("artifact_location", "mlruns")


class FeatureProcessor:
    """Class for feature engineering operations"""
    
    def __init__(self):
        self.categorical_features = ['PU_DO']
        self.numerical_features = ['trip_distance']
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create necessary features from raw dataframe
        
        Args:
            df: Raw dataframe with taxi trip data
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        df = df.copy()
        # Create PU_DO combined feature
        df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
        return df


class PreprocessingOptions:
    """Class to define and create preprocessing options for different feature types"""
    
    @staticmethod
    def get_categorical_preprocessors() -> Dict[str, Any]:
        """
        Get available preprocessors for categorical features
        
        Returns:
            Dict of preprocessor name to transformer
        """
        return {
            'onehot': OneHotEncoder(handle_unknown='ignore'),
            'onehot_sparse': OneHotEncoder(handle_unknown='ignore', sparse_output=True),
            'dict_vectorizer': DictVectorizer()
        }
    
    @staticmethod
    def get_numerical_preprocessors() -> Dict[str, Any]:
        """
        Get available preprocessors for numerical features
        
        Returns:
            Dict of preprocessor name to transformer
        """
        return {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'none': 'passthrough'
        }
    
    @staticmethod
    def create_column_transformer(
        categorical_features: List[str], 
        numerical_features: List[str],
        categorical_transformer: str = 'onehot',
        numerical_transformer: str = 'standard'
    ) -> ColumnTransformer:
        """
        Create a column transformer for preprocessing
        
        Args:
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            categorical_transformer: Name of transformer to use for categorical features
            numerical_transformer: Name of transformer to use for numerical features
            
        Returns:
            ColumnTransformer: Configured preprocessor
        """
        categorical_preprocessors = PreprocessingOptions.get_categorical_preprocessors()
        numerical_preprocessors = PreprocessingOptions.get_numerical_preprocessors()
        
        if categorical_transformer not in categorical_preprocessors:
            raise ValueError(f"Unknown categorical transformer: {categorical_transformer}")
        
        if numerical_transformer not in numerical_preprocessors:
            raise ValueError(f"Unknown numerical transformer: {numerical_transformer}")
        
        transformers = []
        
        # Add categorical preprocessor if features exist
        if categorical_features:
            cat_transformer = categorical_preprocessors[categorical_transformer]
            # For DictVectorizer, we need special handling since it expects dict input
            if categorical_transformer == 'dict_vectorizer':
                # This will be handled separately
                pass
            else:
                transformers.append(
                    ('cat', Pipeline([
                        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                        ('encoder', cat_transformer)
                    ]), categorical_features)
                )
        
        # Add numerical preprocessor if features exist
        if numerical_features:
            num_transformer = numerical_preprocessors[numerical_transformer]
            transformers.append(
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', num_transformer)
                ]), numerical_features)
            )
        
        return ColumnTransformer(transformers=transformers)


class ModelTrainer(ABC):
    """Abstract base class for model trainers"""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model = None
        self.best_params = None
    
    @abstractmethod
    def create_pipeline(self, preprocessor: ColumnTransformer) -> Pipeline:
        """Create a pipeline with preprocessor and model"""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict:
        """Get the current model parameters"""
        pass
    
    @abstractmethod
    def set_params(self, params: Dict) -> None:
        """Set model parameters"""
        pass
    
    @abstractmethod
    def hyperopt_space(self) -> Dict:
        """Define hyperparameter search space"""
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the trained pipeline"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        y_pred = self.predict(X)
        
        return {
            'rmse': root_mean_squared_error(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }


class LinearRegressionTrainer(ModelTrainer):
    """Trainer for Linear Regression models"""
    
    def __init__(self, **kwargs):
        super().__init__("LinearRegression")
        self.params = kwargs
    
    def create_pipeline(self, preprocessor: ColumnTransformer) -> Pipeline:
        """Create a pipeline with preprocessor and linear regression model"""
        return Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression(**self.params))
        ])
    
    def get_params(self) -> Dict:
        """Get current model parameters"""
        return self.params
    
    def set_params(self, params: Dict) -> None:
        """Set model parameters"""
        self.params = params
    
    def hyperopt_space(self) -> Dict:
        """Linear regression has no hyperparameters to tune"""
        return {}


class RidgeTrainer(ModelTrainer):
    """Trainer for Ridge Regression models"""
    
    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__("Ridge")
        self.params = {'alpha': alpha, **kwargs}
    
    def create_pipeline(self, preprocessor: ColumnTransformer) -> Pipeline:
        """Create a pipeline with preprocessor and ridge regression model"""
        return Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', Ridge(**self.params))
        ])
    
    def get_params(self) -> Dict:
        """Get current model parameters"""
        return self.params
    
    def set_params(self, params: Dict) -> None:
        """Set model parameters"""
        self.params = params
    
    def hyperopt_space(self) -> Dict:
        """Define hyperparameter search space for Ridge"""
        return {
            'alpha': hp.loguniform('alpha', -5, 5)
        }


class LassoTrainer(ModelTrainer):
    """Trainer for Lasso Regression models"""
    
    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__("Lasso")
        self.params = {'alpha': alpha, **kwargs}
    
    def create_pipeline(self, preprocessor: ColumnTransformer) -> Pipeline:
        """Create a pipeline with preprocessor and lasso regression model"""
        return Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', Lasso(**self.params))
        ])
    
    def get_params(self) -> Dict:
        """Get current model parameters"""
        return self.params
    
    def set_params(self, params: Dict) -> None:
        """Set model parameters"""
        self.params = params
    
    def hyperopt_space(self) -> Dict:
        """Define hyperparameter search space for Lasso"""
        return {
            'alpha': hp.loguniform('alpha', -5, 5)
        }


class LassoLarsTrainer(ModelTrainer):
    """Trainer for LassoLarsCV models"""
    
    def __init__(self, **kwargs):
        super().__init__("LassoLarsCV")
        self.params = kwargs
    
    def create_pipeline(self, preprocessor: ColumnTransformer) -> Pipeline:
        """Create a pipeline with preprocessor and LassoLarsCV model"""
        return Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LassoLarsCV(cv=5, **self.params))
        ])
    
    def get_params(self) -> Dict:
        """Get current model parameters"""
        return self.params
    
    def set_params(self, params: Dict) -> None:
        """Set model parameters"""
        self.params = params
    
    def hyperopt_space(self) -> Dict:
        """LassoLarsCV handles its own CV internally"""
        return {}


class LinearSVRTrainer(ModelTrainer):
    """Trainer for Linear SVR models"""
    
    def __init__(self, C: float = 1.0, epsilon: float = 0.1, **kwargs):
        super().__init__("LinearSVR")
        self.params = {'C': C, 'epsilon': epsilon, 'max_iter': 10000, **kwargs}
    
    def create_pipeline(self, preprocessor: ColumnTransformer) -> Pipeline:
        """Create a pipeline with preprocessor and Linear SVR model"""
        return Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearSVR(**self.params))
        ])
    
    def get_params(self) -> Dict:
        """Get current model parameters"""
        return self.params
    
    def set_params(self, params: Dict) -> None:
        """Set model parameters"""
        self.params = params
    
    def hyperopt_space(self) -> Dict:
        """Define hyperparameter search space for LinearSVR"""
        return {
            'C': hp.loguniform('C', -3, 3),
            'epsilon': hp.loguniform('epsilon', -5, 0)
        }


class RandomForestTrainer(ModelTrainer):
    """Trainer for Random Forest models"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = None, **kwargs):
        super().__init__("RandomForest")
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': 42,
            **kwargs
        }
    
    def create_pipeline(self, preprocessor: ColumnTransformer) -> Pipeline:
        """Create a pipeline with preprocessor and Random Forest model"""
        return Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(**self.params))
        ])
    
    def get_params(self) -> Dict:
        """Get current model parameters"""
        return self.params
    
    def set_params(self, params: Dict) -> None:
        """Set model parameters"""
        for k in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
            if k in params and params[k] is not None:
                params[k] = int(params[k])
        self.params = params
    
    def hyperopt_space(self) -> Dict:
        """Define hyperparameter search space for Random Forest"""
        return {
            'n_estimators': scope.int(hp.quniform('n_estimators', 10, 300, 10)),
            'max_depth': scope.int(hp.quniform('max_depth', 3, 50, 1)),
            'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
            'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 10, 1))
        }


class XGBoostTrainer(ModelTrainer):
    """Trainer for XGBoost models"""
    
    def __init__(self, params: Dict = None):
        super().__init__("XGBoost")
        default_params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 6,
            'seed': 42
        }
        self.params = {**default_params, **(params or {})}
        self.num_boost_round = 100
    
    def create_pipeline(self, preprocessor: ColumnTransformer) -> Pipeline:
        """Create a pipeline with preprocessor and XGBoost model"""
        return Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(**self.params, n_estimators=self.num_boost_round))
        ])
    
    def get_params(self) -> Dict:
        """Get current model parameters"""
        return self.params
    
    def set_params(self, params: Dict) -> None:
        """Set model parameters"""
        int_params = ['max_depth']
        for key in int_params:
            if key in params:
                params[key] = int(params[key])
        self.params = params
    
    def hyperopt_space(self) -> Dict:
        """Define hyperparameter search space for XGBoost"""
        return {
            'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
            'learning_rate': hp.loguniform('learning_rate', -3, 0),
            'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
            'reg_lambda': hp.loguniform('reg_lambda', -6, -1)
        }


class ModelRegistry:
    """Manages model registration and versioning in MLflow"""
    
    def __init__(self, client: MlflowClient):
        self.client = client
    
    def register_model(self, run_id: str, model_uri: str, name: str) -> None:
        """
        Register a model with MLflow
        
        Args:
            run_id: ID of the run containing the model
            model_uri: URI pointing to the model
            name: Name to register the model under
        """
        result = mlflow.register_model(model_uri=model_uri, name=name)
        logger.info(f"Model registered as {name} v{result.version}")
        return result
    
    def transition_to_production(self, name: str, version: int, archive_existing: bool = True) -> None:
        """
        Transition a model version to production stage
        
        Args:
            name: Name of the registered model
            version: Version to transition
            archive_existing: Whether to archive existing production models
        """
        self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage="Production",
            archive_existing_versions=archive_existing
        )
        logger.info(f"Model {name} v{version} transitioned to Production stage")
    
    def set_alias(self, name: str, version: int, alias: str) -> None:
        """
        Set an alias for a model version
        
        Args:
            name: Name of the registered model
            version: Version to set alias for
            alias: Alias to set
        """
        self.client.set_registered_model_alias(name, alias, version)
        logger.info(f"Set alias '{alias}' for {name} v{version}")
    
    # def find_best_models(self, experiment_id: str, metric: str = "rmse", max_results: int = 5) -> List:
    #     """
    #     Find the best performing models for an experiment
        
    #     Args:
    #         experiment_id: ID of the experiment to search
    #         metric: Metric to sort by
    #         max_results: Maximum number of results to return
            
    #     Returns:
    #         List of run information for the best models
    #     """
    #     runs = self.client.search_runs(
    #         experiment_ids=[experiment_id],
    #         filter_string=f"tags.mlflow.parentRunId IS NULL AND metrics.{metric} IS NOT NULL",
    #         order_by=[f"metrics.{metric} ASC"],
    #         run_view_type=ViewType.ACTIVE_ONLY,
    #         max_results=max_results
    #     )
        
    #     return runs
    
    def find_best_models(self, experiment_id: str, metric: str = "rmse", max_results: int = 5) -> List:
        # Step 1: Retrieve a broad list of recent runs (e.g. 200)
        runs = self.client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="",  # No filtering here due to MLflow limitations
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=200
        )

        # Step 2: Python-side filtering
        valid_runs = []
        for run in runs:
            is_not_nested = "mlflow.parentRunId" not in run.data.tags
            has_metric = metric in run.data.metrics

            if is_not_nested and has_metric:
                valid_runs.append(run)

        # Step 3: Sort by metric value (ascending)
        valid_runs.sort(key=lambda r: r.data.metrics[metric])

        return valid_runs[:max_results]

    
    # def get_model_size(self, run_id: str, artifact_path: str) -> int:
    #     """
    #     Get the size of a model artifact in bytes
        
    #     Args:
    #         run_id: ID of the run containing the model
    #         artifact_path: Path to the artifact within the run
            
    #     Returns:
    #         Size of the model in bytes
    #     """
    #     # This is a simplified version; in practice you might need to download
    #     # and check sizes of all files in the artifact directory
    #     artifact_uri = self.client.get_run(run_id).info.artifact_uri
    #     # In a real implementation, this would involve filesystem operations
    #     # to calculate the size of the model artifact
    #     # For now, let's return a placeholder
    #     return 1000000  # 1MB placeholder

    def get_model_size(self, run_id: str, artifact_path: str = "model") -> float:
        """
        Get the size of a model artifact in kilobytes (KB).

        - First attempts to read `model_size_bytes` from MLmodel.
        - If not available, computes the total size of files in the model directory.

        Args:
            run_id (str): MLflow run ID
            artifact_path (str): Artifact path of the model (default: "model")

        Returns:
            float: Size in kilobytes (KB)
        """
        run = self.client.get_run(run_id)

        # Step 1: Try to read size from MLmodel
        artifact_uri = run.info.artifact_uri
        parsed = urllib.parse.urlparse(artifact_uri)
        model_dir = os.path.join(parsed.path, artifact_path)
        mlmodel_path = os.path.join(model_dir, "MLmodel")

        if os.path.exists(mlmodel_path):
            try:
                with open(mlmodel_path, "r") as f:
                    mlmodel = yaml.safe_load(f)
                size_bytes = mlmodel.get("model_size_bytes")
                if size_bytes is not None:
                    return int(size_bytes) / 1024.0
            except Exception as e:
                print(f"[Warning] Failed to read model_size_bytes from MLmodel: {e}")

        # Step 2: Fallback to manual calculation
        if parsed.scheme != "file":
            raise NotImplementedError("Manual size calculation only supported for local files.")

        total_size = 0
        for dirpath, _, filenames in os.walk(model_dir):
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                total_size += os.path.getsize(fpath)

        return total_size / 1024.0  # Return KB
    
    # def find_production_candidate(self, experiment_id: str, metric: str = "rmse",
    #                              size_weight: float = 0.2, 
    #                              performance_weight: float = 0.8) -> Dict:
    #     """
    #     Find the best candidate for production based on both performance and model size
        
    #     Args:
    #         experiment_id: ID of the experiment to search
    #         metric: Performance metric to consider
    #         size_weight: Weight to give to model size in decision
    #         performance_weight: Weight to give to performance in decision
            
    #     Returns:
    #         Dictionary with information about the best candidate
    #     """
    #     best_runs = self.find_best_models(experiment_id, metric, max_results=10)
        
    #     best_candidate = None
    #     best_score = float('-inf')
        
    #     for run in best_runs:
    #         # Get performance score (lower is better for RMSE)
    #         perf_score = -run.data.metrics[metric]  # Negative because lower RMSE is better
            
    #         # Get model size (smaller is better)
    #         try:
    #             model_path = [a.path for a in self.client.list_artifacts(run.info.run_id) 
    #                          if a.path.startswith('model')][0]
    #             size = self.get_model_size(run.info.run_id, model_path)
    #             size_score = -size  # Negative because smaller size is better
    #         except:
    #             # If we can't get size, assume it's neutral
    #             size_score = 0
            
    #         # Calculate combined score
    #         combined_score = (performance_weight * perf_score) + (size_weight * size_score)
            
    #         if combined_score > best_score:
    #             best_score = combined_score
    #             best_candidate = run
        
    #     if best_candidate:
    #         return {
    #             'run_id': best_candidate.info.run_id,
    #             'metrics': best_candidate.data.metrics,
    #             'params': best_candidate.data.params
    #         }
    #     return None
    

    def find_production_candidate(self, experiment_id: str, metric: str = "rmse",
                              size_weight: float = 0.2,
                              performance_weight: float = 0.8,
                              max_candidates: int = 10) -> Dict:
        """
        Select the best model run based on performance and size.

        Args:
            experiment_id (str): MLflow experiment ID
            metric (str): Performance metric (e.g., 'rmse')
            size_weight (float): Weight to assign to model size (smaller is better)
            performance_weight (float): Weight to assign to metric performance (better is lower for RMSE)
            max_candidates (int): Number of runs to evaluate

        Returns:
            Dict: Best run info or None
        """
        # Step 1: Get valid parent runs with the desired metric
        all_runs = self.client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=200
        )

        parent_runs = [
            run for run in all_runs
            if "mlflow.parentRunId" not in run.data.tags and metric in run.data.metrics
        ]

        if not parent_runs:
            logger.warning("No parent runs with the specified metric found.")
            return None

        # Step 2: Compute RMSE and model size for each run
        candidates = []
        for run in parent_runs[:max_candidates]:
            run_id = run.info.run_id
            try:
                rmse_val = run.data.metrics[metric]
                size_kb = self.get_model_size(run_id, artifact_path="model")
                candidates.append({
                    "run": run,
                    "rmse": rmse_val,
                    "size_kb": size_kb
                })
            except Exception as e:
                logger.warning(f"Skipping run {run_id} due to error: {e}")

        if not candidates:
            logger.warning("No candidates with valid model sizes and metrics.")
            return None

        # Step 3: Normalize RMSE and model size (min-max)
        rmses = [c["rmse"] for c in candidates]
        sizes = [c["size_kb"] for c in candidates]
        min_rmse, max_rmse = min(rmses), max(rmses)
        min_size, max_size = min(sizes), max(sizes)

        for c in candidates:
            c["rmse_score"] = 1 - (c["rmse"] - min_rmse) / (max_rmse - min_rmse + 1e-8)
            c["size_score"] = 1 - (c["size_kb"] - min_size) / (max_size - min_size + 1e-8)
            c["combined_score"] = (
                performance_weight * c["rmse_score"] + size_weight * c["size_score"]
            )

        # Step 4: Select best candidate
        best = max(candidates, key=lambda x: x["combined_score"])
        run = best["run"]

        return {
            "run_id": run.info.run_id,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "size_kb": best["size_kb"],
            "rmse": best["rmse"],
            "combined_score": best["combined_score"]
        }



class HyperparameterTuner:
    """Handles hyperparameter tuning for different models"""
    
    def __init__(self, trainer: ModelTrainer, max_evals: int = 20):
        self.trainer = trainer
        self.max_evals = max_evals
        self.best_params = None
    
    def get_objective(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                     X_val: pd.DataFrame, y_val: np.ndarray, 
                     preprocessor: ColumnTransformer) -> callable:
        """
        Create an objective function for hyperopt
        
        Args:
            X_train: Training features dataframe
            y_train: Training targets
            X_val: Validation features dataframe
            y_val: Validation targets
            preprocessor: Column transformer for preprocessing
            
        Returns:
            Callable objective function
        """
        def objective(params):
            # Log parameters with MLflow
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                
                # Set trainer parameters
                self.trainer.set_params(params)
                
                # Create and train pipeline
                pipeline = self.trainer.create_pipeline(preprocessor)
                pipeline.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = pipeline.predict(X_val)
                rmse = root_mean_squared_error(y_val, y_pred)
                
                # Log metrics
                mlflow.log_metric("rmse", rmse)
                
                return {'loss': rmse, 'status': STATUS_OK}
        
        return objective
    
    def tune(self, X_train: pd.DataFrame, y_train: np.ndarray, 
            X_val: pd.DataFrame, y_val: np.ndarray,
            preprocessor: ColumnTransformer) -> Dict:
        """
        Run hyperparameter tuning
        
        Args:
            X_train: Training features dataframe
            y_train: Training targets
            X_val: Validation features dataframe
            y_val: Validation targets
            preprocessor: Column transformer for preprocessing
            
        Returns:
            Dictionary with best parameters
        """
        # Get hyperparameter space from trainer
        space = self.trainer.hyperopt_space()
        
        # If there's nothing to tune, return empty dict
        if not space:
            logger.info(f"No hyperparameters to tune for {self.trainer.model_type}")
            return {}
        
        # Create the objective function
        objective = self.get_objective(X_train, y_train, X_val, y_val, preprocessor)
        
        # Run hyperopt
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials
        )
        
        # Store and return best parameters
        self.best_params = best
        logger.info(f"Best parameters for {self.trainer.model_type}: {best}")
        return best


class ExperimentManager:
    """Main class for managing ML experiments"""
    
    def __init__(self, 
                 tracking_config: Dict[str, Any] = None,
                 experiment_name: str = "nyc-taxi-experiment",
                 model_registry_name: str = "nyc-taxi-regressor",
                  models_dir: str = "./models"):
        """
        Initialize the experiment manager
        
        Args:
            tracking_config: Configuration for MLflow tracking
            experiment_name: Name for the MLflow experiment
            model_registry_name: Name for the model registry
        """
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

        self.tracking_config = tracking_config or {
            'tracking_store': 'sqlite',
            'db_path': 'mlflow.db'
        }
        
        # Set up MLflow tracking URI
        tracking_uri = StorageConfig.get_tracking_uri(
            self.tracking_config.get('tracking_store', 'sqlite'),
            **self.tracking_config
        )
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set up experiment
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        
        # Initialize MLflow client
        self.client = MlflowClient(tracking_uri)
        
        # Set up model registry
        self.model_registry_name = model_registry_name
        self.registry = ModelRegistry(self.client)
        
        # Set up feature processor
        self.feature_processor = FeatureProcessor()
    
    def _get_trainer(self, model_type: str, params: Dict = None) -> ModelTrainer:
        """
        Get the appropriate model trainer based on model type
        
        Args:
            model_type: Type of model to train
            params: Parameters for the model
            
        Returns:
            ModelTrainer: Trainer for the specified model
        """
        params = params or {}
        
        if model_type == "LinearRegression":
            return LinearRegressionTrainer(**params)
        elif model_type == "Ridge":
            return RidgeTrainer(**params)
        elif model_type == "Lasso":
            return LassoTrainer(**params)
        elif model_type == "LassoLarsCV":
            return LassoLarsTrainer(**params)
        elif model_type == "LinearSVR":
            return LinearSVRTrainer(**params)
        elif model_type == "RandomForest":
            return RandomForestTrainer(**params)
        elif model_type == "XGBoost":
            return XGBoostTrainer(params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train_model(self, 
                   model_type: str,
                   X_train: pd.DataFrame, 
                   y_train: np.ndarray,
                   X_val: pd.DataFrame, 
                   y_val: np.ndarray,
                   categorical_features: List[str],
                   numerical_features: List[str],
                   categorical_transformer: str = 'onehot',
                   numerical_transformer: str = 'standard',
                   params: Dict = None,
                   tags: Dict = None,
                   tune_hyperparams: bool = False,
                   max_evals: int = 20) -> Dict:
        """
        Train a model with MLflow tracking
        
        Args:
            model_type: Type of model to train
            X_train: Training features dataframe
            y_train: Training targets
            X_val: Validation features dataframe
            y_val: Validation targets
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            categorical_transformer: Type of transformer for categorical features
            numerical_transformer: Type of transformer for numerical features
            params: Parameters for the model
            tags: Tags to add to the MLflow run
            tune_hyperparams: Whether to tune hyperparameters
            max_evals: Maximum number of hyperparameter tuning evaluations
            
        Returns:
            Dictionary with run information
        """
        trainer = self._get_trainer(model_type, params)
        
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            
            # Log tags
            if tags:
                mlflow.set_tags(tags)

            # Log preprocessing config
            mlflow.log_param("categorical_transformer", categorical_transformer)
            mlflow.log_param("numerical_transformer", numerical_transformer)
            
            # Create column transformer for preprocessing
            preprocessor = PreprocessingOptions.create_column_transformer(
                categorical_features, numerical_features,
                categorical_transformer, numerical_transformer
            )
            
            # Log default parameters
            for param, value in (params or {}).items():
                mlflow.log_param(param, value)
            
            # Hyperparameter tuning if requested
            if tune_hyperparams:
                tuner = HyperparameterTuner(trainer, max_evals=max_evals)
                best_params = tuner.tune(X_train, y_train, X_val, y_val, preprocessor)
                
                # Update trainer with best parameters
                if best_params:
                    trainer.set_params(best_params)
            
            # Create pipeline with preprocessor and model
            pipeline = trainer.create_pipeline(preprocessor)
            
            # Train model
            logger.info(f"Training {model_type} model")
            pipeline.fit(X_train, y_train)
            trainer.model = pipeline
            
            # Log model with appropriate MLflow flavor
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            
            # Save model locally
            model_path = os.path.join(self.models_dir, f"model_{run_id}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(pipeline, f)
            
            # Evaluate model on validation data
            metrics = trainer.evaluate(X_val, y_val)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log the local model file as an artifact
            mlflow.log_artifact(model_path, artifact_path="model_pickle")
            
            logger.info(f"Run {run_id} completed with {model_type} model")
            logger.info(f"Validation RMSE: {metrics['rmse']:.4f}")
            
            return {
                'run_id': run_id, 
                'metrics': metrics,
                'model_type': model_type
            }
    
    def train_multiple_models(self,
                            X_train: pd.DataFrame, 
                            y_train: np.ndarray,
                            X_val: pd.DataFrame, 
                            y_val: np.ndarray,
                            categorical_features: List[str],
                            numerical_features: List[str],
                            model_types: List[str] = None,
                            categorical_transformer: str = 'onehot',
                            numerical_transformer: str = 'standard',
                            common_tags: Dict = None,
                            tune_hyperparams: bool = True) -> List[Dict]:
        """
        Train multiple model types and track with MLflow
        
        Args:
            X_train: Training features dataframe
            y_train: Training targets
            X_val: Validation features dataframe
            y_val: Validation targets
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            model_types: List of model types to train
            categorical_transformer: Type of transformer for categorical features
            numerical_transformer: Type of transformer for numerical features
            common_tags: Tags to add to all MLflow runs
            tune_hyperparams: Whether to tune hyperparameters
            
        Returns:
            List of dictionaries with run information
        """
        # Default to all supported model types if not specified
        if model_types is None:
            model_types = [
                "LinearRegression", "Ridge", "Lasso", "LassoLarsCV",
                "LinearSVR", "RandomForest", "XGBoost"
            ]
        
        # Initialize results list
        results = []
        
        # Train each model type
        for model_type in model_types:
            logger.info(f"Training {model_type} model")
            
            # Default parameters for different model types
            params = {}
            if model_type == "XGBoost":
                params = {
                    'objective': 'reg:squarederror',
                    'seed': 42
                }
            
            # Set up tags
            tags = common_tags.copy() if common_tags else {}
            tags['model_type'] = model_type
            
            # Train the model
            result = self.train_model(
                model_type=model_type,
                X_train=X_train, 
                y_train=y_train,
                X_val=X_val, 
                y_val=y_val,
                categorical_features=categorical_features,
                numerical_features=numerical_features,
                categorical_transformer=categorical_transformer,
                numerical_transformer=numerical_transformer,
                params=params,
                tags=tags,
                tune_hyperparams=tune_hyperparams
            )
            
            results.append(result)
        
        return results
    
    def register_best_model(self, metric: str = "rmse", size_weight: float = 0.2) -> Dict:
        """
        Find and register the best model for production
        
        Args:
            metric: Metric to use for finding the best model
            size_weight: Weight to give to model size in decision
            
        Returns:
            Dictionary with information about the registered model
        """
        # Find the best candidate
        best_candidate = self.registry.find_production_candidate(
            experiment_id=self.experiment.experiment_id,
            metric=metric,
            size_weight=size_weight
        )
        #print(f"best_cand are {best_candidate}")

        if not best_candidate:
            logger.warning("No suitable model found for production")
            return None
        
        # Get run information
        run_id = best_candidate['run_id']
        run = self.client.get_run(run_id)
        #print(f"runid and run are {run_id} and {run}")
        # Find model artifact path
        #artifacts = self.client.list_artifacts(run_id)
        artifacts = self.client.list_artifacts(run_id, "model")
        #print(f"artifacts are {artifacts}")
        #model_path = [a.path for a in artifacts if a.path.startswith('model')][0]
        model_path = [a.path for a in artifacts][0]
        
        # Register model
        model_uri = f"runs:/{run_id}/{model_path}"
        result = self.registry.register_model(
            run_id=run_id,
            model_uri=model_uri,
            name=self.model_registry_name
        )
        
        # Set to production stage
        self.registry.transition_to_production(
            name=self.model_registry_name,
            version=result.version
        )
        
        # Set alias
        self.registry.set_alias(
            name=self.model_registry_name,
            version=result.version,
            alias="production"
        )
        
        logger.info(f"Model from run {run_id} registered as {self.model_registry_name} v{result.version}")
        logger.info(f"Model type: {run.data.tags.get('model_type')}")
        logger.info(f"Performance: {best_candidate['metrics']}")
        
        return {
            'run_id': run_id,
            'model_name': self.model_registry_name,
            'version': result.version,
            'metrics': best_candidate['metrics']
        }
    
    def load_production_model(self) -> Any:
        """
        Load the current production model
        
        Returns:
            The loaded production model
        """
        model = mlflow.pyfunc.load_model(f"models:/{self.model_registry_name}@production")
        logger.info(f"Loaded production model: {self.model_registry_name}")
        return model


class DataProcessor:
    """Class for data processing operations"""
    
    def __init__(self):
        self.feature_processor = FeatureProcessor()
        self.categorical_preprocessor = None
        self.numerical_preprocessor = None
        
    def prepare_data(self, df: pd.DataFrame, fit: bool = False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Prepare data for model training or prediction
        
        Args:
            df: Raw dataframe with taxi trip data
            fit: Whether to fit preprocessors on this data
            
        Returns:
            Tuple with processed features and optionally target values
        """
        # Apply feature engineering
        df = self.feature_processor.create_features(df)
        
        # Extract features
        features = df[self.feature_processor.categorical_features + self.feature_processor.numerical_features].copy()
        
        return features, None
    
    def save(self, filepath: str) -> None:
        """
        Save the data processor to a file
        
        Args:
            filepath: Path to save the processor
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath: str) -> 'DataProcessor':
        """
        Load a data processor from a file
        
        Args:
            filepath: Path to the saved processor
            
        Returns:
            Loaded DataProcessor instance
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class NYCTaxiDurationExperiment:
    """Main experiment class for NYC Taxi Duration Prediction"""
    
    def __init__(self, 
                 tracking_config: Dict[str, Any] = None,
                 experiment_name: str = "nyc-taxi-experiment",
                 model_registry_name: str = "nyc-taxi-regressor",
                 data_dir: str = "./data",
                 models_dir: str = "./models"):
        """
        Initialize the NYC Taxi Duration Experiment
        
        Args:
            tracking_config: Configuration for MLflow tracking
            experiment_name: Name for the MLflow experiment
            model_registry_name: Name for the model registry
            data_dir: Directory for data storage
            models_dir: Directory for model storage
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize experiment manager
        self.manager = ExperimentManager(
            tracking_config=tracking_config,
            experiment_name=experiment_name,
            model_registry_name=model_registry_name,
            models_dir=models_dir
        )
        
        # Initialize feature processor
        self.feature_processor = FeatureProcessor()
    
    def download_data(self, year: int, month: int, taxi: str = 'green') -> pd.DataFrame:
        """
        Download NYC Taxi trip data
        
        Args:
            year: Year of data to download
            month: Month of data to download
            taxi: Taxi type ('green' or 'yellow')
            
        Returns:
            pd.DataFrame: Raw taxi trip data
        """
        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi}_tripdata_{year}-{month:02d}.parquet"
        local_path = os.path.join(self.data_dir, f"{taxi}_tripdata_{year}-{month:02d}.parquet")
        
        # Download if not already available
        if not os.path.exists(local_path):
            logger.info(f"Downloading data from {url}")

            try:
                download_file(url, local_path)
            except Exception as e:
                print(f"Error downloading data: {e}")
                raise FileNotFoundError(f"Failed to download data from {url}") from e

        
        # Read and return data
        df = pd.read_parquet(local_path)
        logger.info(f"Loaded {len(df)} records from {local_path}")
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare raw taxi data for modeling
        
        Args:
            df: Raw taxi trip data
            
        Returns:
            Tuple with processed dataframe and target values
        """
        # Create copy to avoid modifying the original
        df = df.copy()
        
        # Convert datetime columns
        df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
        df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
        
        # Calculate trip duration in minutes
        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
        
        # Filter trips with reasonable duration (1-60 minutes)
        df = df[(df.duration >= 1) & (df.duration <= 60)]
        
        # Convert categorical columns to strings
        categorical_cols = ['PULocationID', 'DOLocationID']
        df[categorical_cols] = df[categorical_cols].astype(str)
        
        # Extract target variable
        target = df['duration'].values
        
        # Create features using feature processor
        df = self.feature_processor.create_features(df)
        
        return df, target
    
    def run_experiment(self, 
                     train_year: int, train_month: int,
                     val_year: int, val_month: int,
                     taxi: str = 'green',
                     model_types: List[str] = None,
                     categorical_transformer: str = 'onehot',
                     numerical_transformer: str = 'standard',
                     register_model: bool = True,
                     tune_hyperparams: bool = True) -> Dict:
        """
        Run a complete experiment with multiple models
        
        Args:
            train_year: Year of training data
            train_month: Month of training data
            val_year: Year of validation data
            val_month: Month of validation data
            taxi: Taxi type ('green' or 'yellow')
            model_types: List of model types to train
            categorical_transformer: Type of transformer for categorical features
            numerical_transformer: Type of transformer for numerical features
            register_model: Whether to register the best model
            tune_hyperparams: Whether to tune hyperparameters
            
        Returns:
            Dictionary with experiment results
        """
        # Download and prepare data
        logger.info(f"Downloading training data: {train_year}-{train_month}")
        df_train = self.download_data(train_year, train_month, taxi)
        df_train, y_train = self.prepare_data(df_train)
        
        logger.info(f"Downloading validation data: {val_year}-{val_month}")
        df_val = self.download_data(val_year, val_month, taxi)
        df_val, y_val = self.prepare_data(df_val)
        
        # Get feature lists
        categorical_features = self.feature_processor.categorical_features
        numerical_features = self.feature_processor.numerical_features
        
        # Set up common tags
        common_tags = {
            'train_data': f"{train_year}-{train_month}-{taxi}",
            'val_data': f"{val_year}-{val_month}-{taxi}",
            'developer': os.environ.get('USER', 'unknown')
        }
        
        # Train models
        logger.info("Training models")
        results = self.manager.train_multiple_models(
            X_train=df_train,
            y_train=y_train,
            X_val=df_val,
            y_val=y_val,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            model_types=model_types,
            categorical_transformer=categorical_transformer,
            numerical_transformer=numerical_transformer,
            common_tags=common_tags,
            tune_hyperparams=tune_hyperparams
        )
        
        # Register best model if requested
        registered_model = None
        if register_model:
            logger.info("Registering best model")
            registered_model = self.manager.register_best_model()
        
        return {
            'training_results': results,
            'registered_model': registered_model
        }
    
    def load_production_model(self) -> Any:
        """
        Load the current production model
        
        Returns:
            The loaded production model
        """
        return self.manager.load_production_model()


def run_nyc_taxi_experiment(
    tracking_config: Dict[str, Any] = None,
    experiment_name: str = "nyc-taxi-experiment",
    model_registry_name: str = "nyc-taxi-regressor",
    train_year: int = 2021, 
    train_month: int = 1,
    val_year: int = 2021, 
    val_month: int = 2,
    taxi: str = 'green',
    model_types: List[str] = None,
    categorical_transformer: str = 'onehot',
    numerical_transformer: str = 'standard',
    register_model: bool = True,
    tune_hyperparams: bool = True
) -> Dict:
    """
    Convenience function to run a complete NYC Taxi Duration experiment
    
    Args:
        tracking_config: Configuration for MLflow tracking
        experiment_name: Name for the MLflow experiment
        model_registry_name: Name for the model registry
        train_year: Year of training data
        train_month: Month of training data
        val_year: Year of validation data
        val_month: Month of validation data
        taxi: Taxi type ('green' or 'yellow')
        model_types: List of model types to train
        categorical_transformer: Type of transformer for categorical features
        numerical_transformer: Type of transformer for numerical features
        register_model: Whether to register the best model
        tune_hyperparams: Whether to tune hyperparameters
        
    Returns:
        Dictionary with experiment results
    """
    experiment = NYCTaxiDurationExperiment(
        tracking_config=tracking_config,
        experiment_name=experiment_name,
        model_registry_name=model_registry_name
    )
    
    return experiment.run_experiment(
        train_year=train_year,
        train_month=train_month,
        val_year=val_year,
        val_month=val_month,
        taxi=taxi,
        model_types=model_types,
        categorical_transformer=categorical_transformer,
        numerical_transformer=numerical_transformer,
        register_model=register_model,
        tune_hyperparams=tune_hyperparams
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run NYC Taxi Duration Prediction Experiment")
    
    # Data parameters
    parser.add_argument("--train-year", type=int, default=2021)
    parser.add_argument("--train-month", type=int, default=1)
    parser.add_argument("--val-year", type=int, default=2021)
    parser.add_argument("--val-month", type=int, default=2)
    parser.add_argument("--taxi", type=str, default="green", choices=["green", "yellow"])

    # MLflow tracking parameters
    parser.add_argument("--tracking-store", type=str, default="sqlite", 
                        choices=["sqlite", "postgresql", "aws", "gcp", "local", "remote"])
    parser.add_argument("--db-path", type=str, default="mlflow.db")
    parser.add_argument("--host", type=str, help="Tracking server host (for cloud/remote)", default="localhost")
    parser.add_argument("--port", type=int, help="Tracking server port (for cloud/remote)", default=5000)
    parser.add_argument("--tracking-uri", type=str, help="Explicit tracking URI (for remote)")
    
    # Artifact storage
    parser.add_argument("--artifact-store", type=str, default="local", 
                        choices=["local", "s3", "gcs"])
    parser.add_argument("--artifact-location", type=str, help="Local artifact path")
    parser.add_argument("--bucket", type=str, help="GCS or S3 bucket name")
    parser.add_argument("--prefix", type=str, help="Artifact path prefix")

    # MLflow experiment/model info
    parser.add_argument("--experiment-name", type=str, default="nyc-taxi-experiment")
    parser.add_argument("--model-name", type=str, default="nyc-taxi-regressor")

    # Preprocessing
    parser.add_argument("--categorical-transformer", type=str, default="onehot",
                        choices=["onehot", "onehot_sparse", "dict_vectorizer"])
    parser.add_argument("--numerical-transformer", type=str, default="standard",
                        choices=["standard", "minmax", "robust", "none"])
    
    # Model parameters
    parser.add_argument("--models", type=str, nargs="+", 
                        choices=["LinearRegression", "Ridge", "Lasso", "LassoLarsCV", 
                                 "LinearSVR", "RandomForest", "XGBoost", "all"],
                        default=["all"])
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--max-evals", type=int, default=20)

    args = parser.parse_args()

    # Resolve model list
    model_types = None if "all" in args.models else args.models

    # Compute tracking URI
    tracking_uri = StorageConfig.get_tracking_uri(
        storage_type=args.tracking_store,
        db_path=args.db_path,
        host=args.host,
        port=args.port,
        tracking_uri=args.tracking_uri
    )

    artifact_uri = StorageConfig.get_artifact_location(
        storage_type=args.artifact_store,
        bucket=args.bucket,
        s3_bucket=args.bucket,  # handles both cases
        prefix=args.prefix,
        artifact_location=args.artifact_location
    )

    tracking_config = {
        'tracking_uri': tracking_uri,
        'artifact_location': artifact_uri
    }

    # Run the experiment
    results = run_nyc_taxi_experiment(
        tracking_config=tracking_config,
        experiment_name=args.experiment_name,
        model_registry_name=args.model_name,
        train_year=args.train_year,
        train_month=args.train_month,
        val_year=args.val_year,
        val_month=args.val_month,
        taxi=args.taxi,
        model_types=model_types,
        categorical_transformer=args.categorical_transformer,
        numerical_transformer=args.numerical_transformer,
        register_model=not args.no_register,
        tune_hyperparams=not args.no_tune,
        max_evals=args.max_evals
    )

    # Output summary
    print(f"Experiment completed with {len(results['training_results'])} models trained")
    if results['registered_model']:
        print(f"Best model registered as {args.model_name}")
        print(f"Performance: RMSE = {results['registered_model']['metrics']['rmse']:.4f}")