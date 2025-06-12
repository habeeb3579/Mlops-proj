from typing import Dict
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoLarsCV
from sklearn.svm import LinearSVR
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from hyperopt import hp

from models_files.base import ModelTrainer





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
