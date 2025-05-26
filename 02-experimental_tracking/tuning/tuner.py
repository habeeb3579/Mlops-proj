import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import pandas as pd


import mlflow

from sklearn.metrics import root_mean_squared_error
from sklearn.compose import ColumnTransformer
from hyperopt import fmin, tpe, STATUS_OK, Trials

from utils.logger import get_logger
from models_files.base import ModelTrainer


# Type aliases for clarity
ModelType = Any  # Generic model type
ArrayLike = Union[List, np.ndarray, pd.Series]
DictConfig = Dict[str, Any]

logger = get_logger(__name__)

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
