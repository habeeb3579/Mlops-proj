import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Dict, List, Union, Any

from utils.logger import get_logger
from pipeline.dict_transform import DictTransformer

# Type aliases for clarity
ModelType = Any  # Generic model type
ArrayLike = Union[List, np.ndarray, pd.Series]
DictConfig = Dict[str, Any]

logger = get_logger(__name__)

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
            'onehot': OneHotEncoder(handle_unknown='ignore', sparse_output=True),
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
                # transformers.append(
                #     ('cat', Pipeline([
                #         ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                #         ('dict_transformer', DictTransformer()),  
                #         ('encoder', cat_transformer)
                #     ]), categorical_features)
                # )
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
