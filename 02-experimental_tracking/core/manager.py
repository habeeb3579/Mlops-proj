import numpy as np
import os
from typing import Dict, List, Tuple, Union, Optional, Any
import pandas as pd
import pickle

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from utils.logger import get_logger
from pipeline.features import FeatureProcessor
from core.registry import ModelRegistry
from pipeline.preprocessing import PreprocessingOptions
from models_files.base import ModelTrainer
from tuning.tuner import HyperparameterTuner
from models_files.linear import (
    LinearRegressionTrainer, RidgeTrainer, 
    LassoTrainer, LassoLarsTrainer, LinearSVRTrainer)
from models_files.tree import RandomForestTrainer, XGBoostTrainer
#from core.storage import StorageConfig
from core.storage_new import StorageConfig

# Type aliases for clarity
ModelType = Any  # Generic model type
ArrayLike = Union[List, np.ndarray, pd.Series]
DictConfig = Dict[str, Any]

logger = get_logger(__name__)

class ExperimentManager:
    """Main class for managing ML experiments"""
    
    def __init__(self, 
                 tracking_config: Dict[str, Any] = None,
                 experiment_name: str = "nyc-taxi-experiment",
                 model_registry_name: str = "nyc-taxi-regressor",
                 models_dir: str = "./models_newest"):
        """
        Initialize the experiment manager
        
        Args:
            tracking_config: Configuration for MLflow tracking
            experiment_name: Name for the MLflow experiment
            model_registry_name: Name for the model registry
        """
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

        self.tracking_config = tracking_config 
        
        # Set up MLflow tracking URI
        # tracking_uri = StorageConfig.get_tracking_uri(
        #     self.tracking_config.get('tracking_store', 'sqlite'),
        #     **self.tracking_config
        # )

        tracking_uri = self.tracking_config['tracking_uri']

        logger.info(f"tracking uri is: {tracking_uri}")

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
        
        with mlflow.start_run(run_name=model_type) as run:
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
                            tune_hyperparams: bool = True,
                            max_evals: int = 20) -> List[Dict]:
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
                tune_hyperparams=tune_hyperparams,
                max_evals=max_evals
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
        #model_uri = f"runs:/{run_id}/{model_path}"
        model_uri = f"runs:/{run_id}/model"
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
