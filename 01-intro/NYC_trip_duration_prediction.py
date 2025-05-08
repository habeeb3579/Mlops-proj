import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from typing import List, Optional, Dict, Union, Tuple
from datetime import datetime
import logging

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class NYCTaxiDurationPredictor:
    """
    A pipeline for predicting NYC taxi ride durations.
    
    This class handles downloading data, feature engineering, model training,
    evaluation, visualization, and model persistence for NYC taxi ride duration prediction.
    """
    
    def __init__(
        self,
        feature_columns: List[str] = None,
        categorical_columns: List[str] = None,
        numerical_columns: List[str] = None,
        target_transformer: str = None,
        model_name: str = "linear_regression",
        random_state: int = 42,
        model_path: str = None
    ):
        """
        Initialize the NYC Taxi Duration Predictor.
        
        Args:
            feature_columns: List of columns to use as features (default: uses PU_location_id, DO_location_id, trip_distance)
            categorical_columns: List of categorical feature columns
            numerical_columns: List of numerical feature columns
            target_transformer: Transformation to apply to target variable ("log", "power", None)
            model_name: Name of the model to use
            random_state: Random seed for reproducibility
            model_path: Path to a saved model to load (if provided, other parameters are ignored)
        """
        # Check if we're loading a saved model
        if model_path:
            logging.info(f"Loading model from {model_path}")
            self._load_model(model_path)
        else:
            self.random_state = random_state
            
            # Default feature columns if none provided
            self.feature_columns = feature_columns or ["PULocationID", "DOLocationID", "trip_distance"]
            
            # Default categorical and numerical features if none provided
            self.categorical_columns = categorical_columns or ["PULocationID", "DOLocationID"]
            self.numerical_columns = numerical_columns or ["trip_distance"]
            
            # Target transformation
            self.target_transformer = target_transformer
            self.target_transformer_obj = None
            
            # Model selection
            self.model_name = model_name
            self.model = self._get_model(model_name)
            
            # Prepare pipeline
            self.preprocessor = None
            self.pipeline = None
            self._build_pipeline()
        
        # Results storage
        self.metrics = {}
        self.predictions = None
        self.actual_values = None
        self.transformed_target = None
    
    def _get_model(self, model_name: str):
        """
        Get the specified scikit-learn model.
        
        Args:
            model_name: Name of the model to use
            
        Returns:
            A scikit-learn model instance
        """
        models = {
            "linear_regression": LinearRegression(),
            "ridge": Ridge(random_state=self.random_state),
            "lasso": Lasso(random_state=self.random_state),
            "random_forest": RandomForestRegressor(random_state=self.random_state),
            "gradient_boosting": GradientBoostingRegressor(random_state=self.random_state)
        }
        
        if model_name not in models:
            raise ValueError(f"Model {model_name} not supported. Choose from: {list(models.keys())}")
        
        return models[model_name]
    
    def _build_pipeline(self):
        """Build the preprocessing and model pipeline."""
        # Categorical features preprocessor with DictVectorizer
        categorical_transformer = Pipeline(steps=[
            ('dictifier', DictVectorizingTransformer(self.categorical_columns)),
        ])
        
        # Numerical features preprocessor
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Column transformer that applies the appropriate preprocessing to each column type
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, self.categorical_columns),
                ('num', numerical_transformer, self.numerical_columns)
            ]
        )
        
        # Full pipeline with preprocessing and model
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', self.model)
        ])
    
    def download_data(self, year: int, month: int) -> pd.DataFrame:
        """
        Download NYC taxi trip data.
        
        Args:
            year: Year of the data
            month: Month of the data
            
        Returns:
            DataFrame with the loaded data
        """
        # URL template for NYC TLC data
        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
        
        print(f"Downloading data from {url}")
        try:
            df = pd.read_parquet(url)
            print(f"Downloaded {len(df)} records")
            return df
        except Exception as e:
            print(f"Error downloading data: {e}")
            raise
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for modeling by computing duration and extracting features.
        
        Args:
            df: DataFrame with taxi trip data
            
        Returns:
            Tuple of (X, y) where X contains features and y contains duration in minutes
        """
        # Calculate ride duration in minutes
        df['duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
        
        # Filter out unreasonable durations (less than 1 minute or more than 2 hours)
        df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]

        df[self.categorical_columns] = df[self.categorical_columns].astype(str)
        
        # Extract features and target
        X = df[self.feature_columns]
        y = df['duration']
        
        # Transform target if specified
        if self.target_transformer == "log":
            y = np.log1p(y)
            self.target_transformer_obj = "log"
        elif self.target_transformer == "power":
            self.target_transformer_obj = PowerTransformer(method='yeo-johnson')
            y = self.target_transformer_obj.fit_transform(y.values.reshape(-1, 1)).ravel()
        
        self.transformed_target = self.target_transformer is not None
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the model on the provided data.
        
        Args:
            X: Features
            y: Target
        """
        print(f"Training {self.model_name} model...")
        self.pipeline.fit(X, y)
        print("Training complete!")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features
            
        Returns:
            Array of predictions
        """
        predictions = self.pipeline.predict(X)
        
        # Inverse transform predictions if needed
        if self.transformed_target:
            if self.target_transformer_obj == "log":
                predictions = np.expm1(predictions)
            elif self.target_transformer_obj:
                predictions = self.target_transformer_obj.inverse_transform(
                    predictions.reshape(-1, 1)
                ).ravel()
        
        return predictions
    
    def evaluate(self, X: pd.DataFrame, y_true: pd.Series):
        """
        Evaluate the model and store metrics.
        
        Args:
            X: Features
            y_true: True target values
        """
        # Get predictions
        y_pred = self.predict(X)
        
        # Inverse transform y_true if needed
        if self.transformed_target:
            if self.target_transformer_obj == "log":
                y_true_original = np.expm1(y_true)
            elif self.target_transformer_obj:
                y_true_original = self.target_transformer_obj.inverse_transform(
                    y_true.values.reshape(-1, 1)
                ).ravel()
        else:
            y_true_original = y_true
        
        # Store predictions and actual values for visualization
        self.predictions = y_pred
        self.actual_values = y_true_original
        
        # Calculate metrics
        self.metrics = {
            "rmse": root_mean_squared_error(y_true_original, y_pred),
            "mae": mean_absolute_error(y_true_original, y_pred),
            "r2": r2_score(y_true_original, y_pred)
        }
        
        # Print metrics
        print(f"RMSE: {self.metrics['rmse']:.2f}")
        print(f"MAE: {self.metrics['mae']:.2f}")
        print(f"R²: {self.metrics['r2']:.2f}")
    
    def save_model(self, path: str, format: str = "joblib"):
        """
        Save the trained model pipeline to a file.
        
        Args:
            path: Path where to save the model
            format: Format to save the model ("joblib" or "pickle")
        """
        if self.pipeline is None:
            logging.error("No model to save. Train the model first.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Save model metadata
        metadata = {
            "feature_columns": self.feature_columns,
            "categorical_columns": self.categorical_columns,
            "numerical_columns": self.numerical_columns,
            "target_transformer": self.target_transformer,
            "target_transformer_obj": self.target_transformer_obj,
            "model_name": self.model_name,
            "transformed_target": self.transformed_target,
            "metrics": self.metrics
        }
        
        # Save model and metadata
        if format.lower() == "joblib":
            model_data = {"pipeline": self.pipeline, "metadata": metadata}
            joblib.dump(model_data, path)
            logging.info(f"Model saved to {path} using joblib")
        elif format.lower() == "pickle":
            model_data = {"pipeline": self.pipeline, "metadata": metadata}
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            logging.info(f"Model saved to {path} using pickle")
        else:
            logging.error(f"Unknown format: {format}. Use 'joblib' or 'pickle'.")
    
    def _load_model(self, path: str):
        """
        Load a trained model from a file.
        
        Args:
            path: Path to the saved model
        """
        # Determine the format based on file extension
        if path.endswith('.joblib'):
            model_data = joblib.load(path)
        elif path.endswith('.pkl') or path.endswith('.pickle'):
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
        else:
            try:
                # Try joblib first
                model_data = joblib.load(path)
            except:
                # Fall back to pickle
                with open(path, 'rb') as f:
                    model_data = pickle.load(f)
        
        # Load pipeline and metadata
        self.pipeline = model_data["pipeline"]
        metadata = model_data["metadata"]
        
        # Set object attributes from metadata
        self.feature_columns = metadata["feature_columns"]
        self.categorical_columns = metadata["categorical_columns"]
        self.numerical_columns = metadata["numerical_columns"]
        self.target_transformer = metadata["target_transformer"]
        self.target_transformer_obj = metadata["target_transformer_obj"]
        self.model_name = metadata["model_name"]
        self.transformed_target = metadata["transformed_target"]
        self.metrics = metadata.get("metrics", {})
        
        logging.info(f"Model loaded from {path}")
        logging.info(f"Model: {self.model_name}")
        logging.info(f"Features: {', '.join(self.feature_columns)}")
    
    def visualize_predictions(self, save_path: str = None):
        """
        Create an overlaid distribution plot of actual vs predicted values.
        
        Args:
            save_path: Path to save the plot (if None, plot is displayed)
        """
        if self.predictions is None or self.actual_values is None:
            logging.warning("No predictions available. Run evaluate() first.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Create KDE plots
        sns.kdeplot(self.actual_values, label='Actual', color='blue', fill=True, alpha=0.3)
        sns.kdeplot(self.predictions, label='Predicted', color='red', fill=True, alpha=0.3)
        
        # Add labels and title
        plt.xlabel('Duration (minutes)')
        plt.ylabel('Density')
        plt.title(f'NYC Taxi Ride Duration: Actual vs Predicted\nModel: {self.model_name}')
        
        # Add metrics to the plot
        metrics_text = f"RMSE: {self.metrics['rmse']:.2f}\nMAE: {self.metrics['mae']:.2f}\nR²: {self.metrics['r2']:.2f}"
        plt.annotate(metrics_text, xy=(0.05, 0.85), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        plt.legend()
        plt.tight_layout()
        
        # Save or display the plot
        if save_path:
            plt.savefig(save_path)
            logging.info(f"Plot saved to {save_path}")
        else:
            plt.show()


class DictVectorizingTransformer:
    """
    Transformer that converts categorical features to dicts for DictVectorizer.
    """
    
    def __init__(self, feature_names):
        """
        Initialize the transformer.
        
        Args:
            feature_names: List of feature names to convert to dicts
        """
        self.feature_names = feature_names
        self.dict_vectorizer = DictVectorizer(sparse=False)
    
    def fit(self, X, y=None):
        """
        Fit the DictVectorizer.
        
        Args:
            X: Input features
            y: Target (not used)
            
        Returns:
            self
        """
        dicts = self._convert_to_dicts(X)
        self.dict_vectorizer.fit(dicts)
        return self
    
    def transform(self, X):
        """
        Transform categorical features using DictVectorizer.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        dicts = self._convert_to_dicts(X)
        return self.dict_vectorizer.transform(dicts)
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step.
        
        Args:
            X: Input features
            y: Target (not used)
            
        Returns:
            Transformed features
        """
        self.fit(X, y)
        return self.transform(X)
    
    def _convert_to_dicts(self, X):
        """
        Convert DataFrame to list of dicts for DictVectorizer.
        
        Args:
            X: DataFrame with features
            
        Returns:
            List of dicts
        """
        X_subset = X[self.feature_names].copy()
        return X_subset.to_dict(orient='records')


def train_model(args):
    """
    Train a new model using the specified arguments.
    
    Args:
        args: Command line arguments from argparse
    """
    logging.info(f"Starting NYC Taxi Duration Prediction - Training Pipeline")
    logging.info(f"Model: {args.model}, Target Transform: {args.target_transform}")
    
    # Initialize predictor
    predictor = NYCTaxiDurationPredictor(
        feature_columns=args.features,
        categorical_columns=args.categorical_features,
        numerical_columns=args.numerical_features,
        target_transformer=args.target_transform,
        model_name=args.model,
        random_state=args.random_state
    )
    
    # Download and prepare training data
    train_df = predictor.download_data(args.train_year, args.train_month)
    X_train_full, y_train_full = predictor.prepare_data(train_df)
    
    # Handle test data
    if args.test_year and args.test_month:
        # Use specified test data
        test_df = predictor.download_data(args.test_year, args.test_month)
        X_test, y_test = predictor.prepare_data(test_df)
        
        # Use all training data for training
        X_train = X_train_full
        y_train = y_train_full
    else:
        # Split training data into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_full, y_train_full, test_size=args.test_size, random_state=args.random_state
        )
    
    # Train the model
    predictor.train(X_train, y_train)
    
    # Save the model if requested
    if args.save_model:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Create model filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = ".joblib" if args.model_format.lower() == "joblib" else ".pkl"
        model_path = f"models/nyc_taxi_duration_{args.model}_{timestamp}{extension}"
        
        # Save the model
        predictor.save_model(model_path, format=args.model_format)
        logging.info(f"Model saved to {model_path}")
    
    # Evaluate the model
    predictor.evaluate(X_test, y_test)
    
    # Visualize results
    if args.save_plot:
        # Create plots directory if it doesn't exist
        os.makedirs("plots", exist_ok=True)
        
        # Create plot filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"plots/nyc_taxi_duration_{predictor.model_name}_{timestamp}.png"
        
        predictor.visualize_predictions(save_path=plot_path)
    else:
        predictor.visualize_predictions()
    
    return predictor


def predict_with_model(args):
    """
    Make predictions using a pre-trained model.
    
    Args:
        args: Command line arguments from argparse
    """
    if not args.model_path:
        logging.error("Model path is required for prediction mode")
        return
    
    logging.info(f"Starting NYC Taxi Duration Prediction - Prediction Pipeline")
    logging.info(f"Loading model from {args.model_path}")
    
    # Load the model
    predictor = NYCTaxiDurationPredictor(model_path=args.model_path)
    
    # Download and prepare test data
    test_df = predictor.download_data(args.year, args.month)
    X_test, y_test = predictor.prepare_data(test_df)
    
    # Evaluate the model
    predictor.evaluate(X_test, y_test)
    
    # Visualize results
    if args.save_plot:
        # Create plots directory if it doesn't exist
        os.makedirs("plots", exist_ok=True)
        
        # Create plot filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"plots/nyc_taxi_duration_{predictor.model_name}_prediction_{timestamp}.png"
        
        predictor.visualize_predictions(save_path=plot_path)
    else:
        predictor.visualize_predictions()
    
    return predictor


def main():
    # Create the main parser
    parser = argparse.ArgumentParser(
        description='NYC Taxi Ride Duration Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for the different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operating mode')
    
    # Train mode subparser
    train_parser = subparsers.add_parser('train', help='Train a new model',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Data selection arguments for train mode
    train_parser.add_argument('--train-year', type=int, default=2021, help='Year for training data')
    train_parser.add_argument('--train-month', type=int, default=1, help='Month for training data')
    train_parser.add_argument('--test-year', type=int, help='Year for test data (optional)')
    train_parser.add_argument('--test-month', type=int, help='Month for test data (optional)')
    train_parser.add_argument('--test-size', type=float, default=0.2,
                              help='Test size for train/test split if test year/month not provided')
    
    # Feature selection arguments for train mode
    train_parser.add_argument('--features', type=str, nargs='+', 
                           help='List of feature columns to use')
    train_parser.add_argument('--categorical-features', type=str, nargs='+',
                           help='List of categorical feature columns')
    train_parser.add_argument('--numerical-features', type=str, nargs='+',
                           help='List of numerical feature columns')
    
    # Model selection arguments for train mode
    train_parser.add_argument('--model', type=str, default='linear_regression',
                          choices=['linear_regression', 'ridge', 'lasso', 'random_forest', 'gradient_boosting'],
                          help='Machine learning model to use')
    
    # Preprocessing arguments for train mode
    train_parser.add_argument('--target-transform', type=str, choices=['log', 'power', None],
                          help='Transformation to apply to the target variable')
    
    # Model saving arguments for train mode
    train_parser.add_argument('--save-model', action='store_true', default=True,
                          help='Save the trained model')
    train_parser.add_argument('--no-save-model', dest='save_model', action='store_false',
                          help='Do not save the trained model')
    train_parser.add_argument('--model-format', type=str, default='joblib',
                          choices=['joblib', 'pickle'],
                          help='Format to save the model')
    
    # Other arguments for train mode
    train_parser.add_argument('--random-state', type=int, default=42,
                          help='Random seed for reproducibility')
    train_parser.add_argument('--save-plot', action='store_true', default=True,
                          help='Save the prediction plot')
    train_parser.add_argument('--no-save-plot', dest='save_plot', action='store_false',
                          help='Do not save the prediction plot')
    
    # Set the train function as the default function for the train subparser
    train_parser.set_defaults(func=train_model)
    
    # Predict mode subparser
    predict_parser = subparsers.add_parser('predict', help='Make predictions with an existing model',
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Model loading arguments for predict mode
    predict_parser.add_argument('--model-path', type=str, required=True,
                             help='Path to a saved model to load')
    
    # Data selection arguments for predict mode
    predict_parser.add_argument('--year', type=int, default=2021, help='Year for test data')
    predict_parser.add_argument('--month', type=int, default=1, help='Month for test data')
    
    # Other arguments for predict mode
    predict_parser.add_argument('--save-plot', action='store_true', default=True,
                             help='Save the prediction plot')
    predict_parser.add_argument('--no-save-plot', dest='save_plot', action='store_false',
                             help='Do not save the prediction plot')
    
    # Set the predict function as the default function for the predict subparser
    predict_parser.set_defaults(func=predict_with_model)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate function based on the subparser
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()