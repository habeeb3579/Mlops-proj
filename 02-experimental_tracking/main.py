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
from core.experiment import run_nyc_taxi_experiment

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run NYC Taxi Duration Prediction Experiment")
    
    # Data parameters
    parser.add_argument("--train-year", type=int, default=2021, help="Year of training data")
    parser.add_argument("--train-month", type=int, default=1, help="Month of training data")
    parser.add_argument("--val-year", type=int, default=2021, help="Year of validation data")
    parser.add_argument("--val-month", type=int, default=2, help="Month of validation data")
    parser.add_argument("--taxi", type=str, default="green", choices=["green", "yellow"], help="Taxi type")
    
    # MLflow parameters
    parser.add_argument("--tracking-store", type=str, default="sqlite", 
                       choices=["sqlite", "postgresql", "aws", "gcp", "local"], 
                       help="MLflow tracking store type")
    parser.add_argument("--db-path", type=str, default="mlflow.db", help="SQLite database path")
    parser.add_argument("--experiment-name", type=str, default="nyc-taxi-experiment", help="MLflow experiment name")
    parser.add_argument("--model-name", type=str, default="nyc-taxi-regressor", help="Model registry name")
    
    # Preprocessing parameters
    parser.add_argument("--categorical-transformer", type=str, default="onehot",
                      choices=["onehot", "onehot_sparse", "dict_vectorizer"],
                      help="Transformer for categorical features")
    parser.add_argument("--numerical-transformer", type=str, default="standard",
                      choices=["standard", "minmax", "robust", "none"],
                      help="Transformer for numerical features")
    
    # Model parameters
    parser.add_argument("--models", type=str, nargs="+", 
                       choices=["LinearRegression", "Ridge", "Lasso", "LassoLarsCV", 
                                "LinearSVR", "RandomForest", "XGBoost", "all"],
                       default=["all"], help="Models to train")
    parser.add_argument("--no-tune", action="store_true", help="Disable hyperparameter tuning")
    parser.add_argument("--no-register", action="store_true", help="Don't register the best model")
    parser.add_argument("--max-evals", type=int, default=20, help="Maximum number of evaluations during hyperparameter tuning")
    
    args = parser.parse_args()
    
    # Handle 'all' model option
    if "all" in args.models:
        model_types = None  # Will use default list of all models
    else:
        model_types = args.models
    
    # Set up tracking configuration
    tracking_config = {
        'tracking_store': args.tracking_store,
        'db_path': args.db_path
    }
    
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
        tune_hyperparams=not args.no_tune,
        max_evals= args.max_evals
    )
    
    print(f"Experiment completed with {len(results['training_results'])} models trained")
    if results['registered_model']:
        print(f"Best model registered as {args.model_name}")
        print(f"Performance: RMSE = {results['registered_model']['metrics']['rmse']:.4f}")