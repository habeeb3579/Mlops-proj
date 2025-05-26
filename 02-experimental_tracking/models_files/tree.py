from typing import Dict
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from hyperopt import hp
from hyperopt.pyll import scope

from models_files.base import ModelTrainer



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
