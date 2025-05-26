#!/usr/bin/env python
"""
NYC Taxi Trip Duration Prediction - Experiment Runner
=====================================================

Command-line interface for running NYC Taxi trip duration prediction experiments
with MLflow tracking and model management.

Examples:
    # Run an experiment with default settings (all models, with tuning)
    python run_experiment.py

    # Train only XGBoost and RandomForest models
    python run_experiment.py --models XGBoost RandomForest

    # Disable hyperparameter tuning for faster iterations
    python run_experiment.py --no-tune

    # Use different train/validation months
    python run_experiment.py --train-month 3 --val-month 4

    # Use yellow taxi data instead of green
    python run_experiment.py --taxi yellow

    # Set up MLflow to use PostgreSQL backend
    python run_experiment.py --tracking-store postgresql --host localhost --port 5432 --database mlflow --user mlflow --password mlflow_secret

    # Use AWS S3 for artifact storage
    python run_experiment.py --tracking-store aws --s3-bucket my-mlflow-bucket --region us-west-2

Author: Habeeb Babatunde
Date: May 14, 2025
"""

import argparse
import sys
import os
import logging
from typing import Dict, Any, List, Optional

from mlflowpy import run_nyc_taxi_experiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the experiment runner."""

    parser = argparse.ArgumentParser(
        description="Run NYC Taxi Duration Prediction Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    data_group = parser.add_argument_group("Data options")
    data_group.add_argument("--train-year", type=int, default=2021, help="Year of training data")
    data_group.add_argument("--train-month", type=int, default=1, help="Month of training data")
    data_group.add_argument("--val-year", type=int, default=2021, help="Year of validation data")
    data_group.add_argument("--val-month", type=int, default=2, help="Month of validation data")
    data_group.add_argument("--taxi", type=str, default="green", choices=["green", "yellow"], 
                          help="Taxi type")
    data_group.add_argument("--data-dir", type=str, default="./data", 
                          help="Directory for data storage")
    
    # MLflow parameters
    mlflow_group = parser.add_argument_group("MLflow tracking options")
    mlflow_group.add_argument("--tracking-store", type=str, default="sqlite", 
                        choices=["sqlite", "postgresql", "aws", "gcp", "local"], 
                        help="MLflow tracking store type")
    mlflow_group.add_argument("--experiment-name", type=str, default="nyc-taxi-experiment", 
                        help="MLflow experiment name")
    mlflow_group.add_argument("--model-name", type=str, default="nyc-taxi-regressor", 
                        help="Model registry name")
    
    # SQLite specific options
    mlflow_group.add_argument("--db-path", type=str, default="mlflow.db", 
                        help="SQLite database path")
    
    # PostgreSQL specific options
    mlflow_group.add_argument("--host", type=str, default="localhost", 
                        help="PostgreSQL host (for postgresql tracking)")
    mlflow_group.add_argument("--port", type=int, default=5432, 
                        help="PostgreSQL port (for postgresql tracking)")
    mlflow_group.add_argument("--database", type=str, default="mlflow", 
                        help="PostgreSQL database name (for postgresql tracking)")
    mlflow_group.add_argument("--user", type=str, default="mlflow", 
                        help="PostgreSQL username (for postgresql tracking)")
    mlflow_group.add_argument("--password", type=str, default="mlflow", 
                        help="PostgreSQL password (for postgresql tracking)")
    
    # AWS specific options
    mlflow_group.add_argument("--s3-bucket", type=str, default="mlflow-artifacts", 
                        help="S3 bucket name (for aws tracking)")
    mlflow_group.add_argument("--region", type=str, default="us-east-1", 
                        help="AWS region (for aws tracking)")
    
    # GCP specific options
    mlflow_group.add_argument("--project", type=str, default="mlflow-project", 
                        help="GCP project (for gcp tracking)")
    mlflow_group.add_argument("--bucket", type=str, default="mlflow-artifacts", 
                        help="GCS bucket name (for gcp tracking)")
    
    # Preprocessing parameters
    preproc_group = parser.add_argument_group("Preprocessing options")
    preproc_group.add_argument("--categorical-transformer", type=str, default="onehot",
                        choices=["onehot", "onehot_sparse", "dict_vectorizer"],
                        help="Transformer for categorical features")
    preproc_group.add_argument("--numerical-transformer", type=str, default="standard",
                        choices=["standard", "minmax", "robust", "none"],
                        help="Transformer for numerical features")
    
    # Model parameters
    model_group = parser.add_argument_group("Model options")
    model_group.add_argument("--models", type=str, nargs="+", 
                        choices=["LinearRegression", "Ridge", "Lasso", "LassoLarsCV", 
                                "LinearSVR", "RandomForest", "XGBoost", "all"],
                        default=["all"], help="Models to train")
    model_group.add_argument("--no-tune", action="store_true", 
                        help="Disable hyperparameter tuning")
    model_group.add_argument("--no-register", action="store_true", 
                        help="Don't register the best model")
    model_group.add_argument("--max-evals", type=int, default=20,
                        help="Maximum hyperparameter tuning evaluations")
    model_group.add_argument("--models-dir", type=str, default="./models", 
                        help="Directory for model storage")
    
    # Output/logging parameters
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    output_group.add_argument("--summary", action="store_true",
                        help="Print summary of results")
    
    return parser.parse_args()


def build_tracking_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build MLflow tracking configuration based on command-line arguments
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dict: Configuration for MLflow tracking
    """
    tracking_config = {
        'tracking_store': args.tracking_store,
    }
    
    # Add store-specific configuration
    if args.tracking_store == "sqlite":
        tracking_config['db_path'] = args.db_path
    
    elif args.tracking_store == "postgresql":
        tracking_config.update({
            'host': args.host,
            'port': args.port,
            'database': args.database,
            'user': args.user,
            'password': args.password
        })
    
    elif args.tracking_store == "aws":
        tracking_config.update({
            's3_bucket': args.s3_bucket,
            'region': args.region
        })
    
    elif args.tracking_store == "gcp":
        tracking_config.update({
            'project': args.project,
            'bucket': args.bucket
        })
    
    return tracking_config


def main() -> int:
    """
    Main entry point for the experiment runner
    
    Returns:
        int: Exit status code
    """
    args = parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Handle 'all' model option
    if "all" in args.models:
        model_types = None  # Will use default list of all models
    else:
        model_types = args.models
    
    # Build tracking configuration
    tracking_config = build_tracking_config(args)
    
    # Ensure directories exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    
    # Log execution settings
    logger.info(f"Running experiment with the following settings:")
    logger.info(f"  Data: {args.taxi} taxi data from {args.train_year}-{args.train_month} (train) and {args.val_year}-{args.val_month} (val)")
    logger.info(f"  Models: {model_types or 'all available models'}")
    logger.info(f"  Preprocessing: categorical={args.categorical_transformer}, numerical={args.numerical_transformer}")
    logger.info(f"  Hyperparameter tuning: {'disabled' if args.no_tune else 'enabled'}")
    logger.info(f"  MLflow tracking store: {args.tracking_store}")
    
    try:
        # Run experiment
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
            tune_hyperparams=not args.no_tune
        )
        
        # Print summary if requested
        if args.summary:
            print("\n" + "="*80)
            print(f"EXPERIMENT SUMMARY")
            print("="*80)
            
            print(f"\nTrained {len(results['training_results'])} models:")
            for i, result in enumerate(results['training_results'], 1):
                model_type = result.get('model_type', 'Unknown')
                rmse = result.get('metrics', {}).get('rmse', float('nan'))
                r2 = result.get('metrics', {}).get('r2', float('nan'))
                print(f"  {i}. {model_type}: RMSE = {rmse:.4f}, R² = {r2:.4f}")
            
            if results['registered_model']:
                print("\nBest model registered to production:")
                print(f"  Name: {args.model_name}")
                metrics = results['registered_model'].get('metrics', {})
                print(f"  Performance: RMSE = {metrics.get('rmse', float('nan')):.4f}, "
                      f"R² = {metrics.get('r2', float('nan')):.4f}")
        
        logger.info("Experiment completed successfully")
        return 0
    
    except Exception as e:
        logger.exception(f"Error running experiment: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())