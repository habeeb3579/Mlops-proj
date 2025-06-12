from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_error


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

