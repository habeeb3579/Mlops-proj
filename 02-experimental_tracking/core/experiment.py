import numpy as np
import os
from typing import Dict, List, Tuple, Union, Optional, Any
import pandas as pd

from utils.logger import get_logger
from core.manager import ExperimentManager
from pipeline.features import FeatureProcessor
from scripts.download_data import download_file

# Type aliases for clarity
ModelType = Any  # Generic model type
ArrayLike = Union[List, np.ndarray, pd.Series]
DictConfig = Dict[str, Any]

logger = get_logger(__name__)

class NYCTaxiDurationExperiment:
    """Main experiment class for NYC Taxi Duration Prediction"""
    
    def __init__(self, 
                 tracking_config: Dict[str, Any] = None,
                 experiment_name: str = "nyc-taxi-experiment",
                 model_registry_name: str = "nyc-taxi-regressor",
                 data_dir: str = "./data",
                 models_dir: str = "./models_newest"):
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
                     tune_hyperparams: bool = True,
                     max_evals: int = 20) -> Dict:
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
            tune_hyperparams=tune_hyperparams,
            max_evals=max_evals
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
    tune_hyperparams: bool = True,
    max_evals: int = 20
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
        tune_hyperparams=tune_hyperparams,
        max_evals=max_evals
    )

