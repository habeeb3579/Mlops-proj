"""
NYC Taxi Trip Duration Prediction - MLflow Production Framework
=============================================================

This module provides a production-grade framework for training, tracking,
and registering ML models for NYC Taxi trip duration prediction using MLflow.

Author: Habeeb Babatunde
Date: May 14, 2025
"""
from core.experiment import run_nyc_taxi_experiment
from core.storage_new import StorageConfig  # assumes you saved the class here

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run NYC Taxi Duration Prediction Experiment")
    
    # Data parameters
    parser.add_argument("--train-year", type=int, default=2021)
    parser.add_argument("--train-month", type=int, default=1)
    parser.add_argument("--val-year", type=int, default=2021)
    parser.add_argument("--val-month", type=int, default=2)
    parser.add_argument("--taxi", type=str, default="green", choices=["green", "yellow"])

    # MLflow tracking parameters
    parser.add_argument("--tracking-store", type=str, default="sqlite", 
                        choices=["sqlite", "postgresql", "aws", "gcp", "local", "remote"])
    parser.add_argument("--db-path", type=str, default="mlflow.db")
    parser.add_argument("--host", type=str, help="Tracking server host (for cloud/remote)", default="localhost")
    parser.add_argument("--port", type=int, help="Tracking server port (for cloud/remote)", default=5000)
    parser.add_argument("--tracking-uri", type=str, help="Explicit tracking URI (for remote)")
    
    # Artifact storage
    parser.add_argument("--artifact-store", type=str, default="gcs", 
                        choices=["local", "s3", "gcs"])
    parser.add_argument("--artifact-location", type=str, default="gs://dezoomfinal-mlflow-artifacts",  help="Artifact path")
    parser.add_argument("--bucket", type=str, default="dezoomfinal-mlflow-artifacts",  help="GCS or S3 bucket name")
    parser.add_argument("--prefix", type=str, help="Artifact path prefix")

    # MLflow experiment/model info
    parser.add_argument("--experiment-name", type=str, default="nyc-taxi-experiment")
    parser.add_argument("--model-name", type=str, default="nyc-taxi-regressor")

    # Preprocessing
    parser.add_argument("--categorical-transformer", type=str, default="onehot",
                        choices=["onehot", "onehot_sparse", "dict_vectorizer"])
    parser.add_argument("--numerical-transformer", type=str, default="standard",
                        choices=["standard", "minmax", "robust", "none"])
    
    # Model parameters
    parser.add_argument("--models", type=str, nargs="+", 
                        choices=["LinearRegression", "Ridge", "Lasso", "LassoLarsCV", 
                                 "LinearSVR", "RandomForest", "XGBoost", "all"],
                        default=["all"])
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--max-evals", type=int, default=20)

    args = parser.parse_args()

    # Resolve model list
    model_types = None if "all" in args.models else args.models

    # Compute tracking URI
    tracking_uri = StorageConfig.get_tracking_uri(
        storage_type=args.tracking_store,
        db_path=args.db_path,
        host=args.host,
        port=args.port,
        tracking_uri=args.tracking_uri
    )

    artifact_uri = StorageConfig.get_artifact_location(
        storage_type=args.artifact_store,
        bucket=args.bucket,
        s3_bucket=args.bucket,  # handles both cases
        prefix=args.prefix,
        artifact_location=args.artifact_location
    )

    tracking_config = {
        'tracking_uri': tracking_uri,
        'artifact_location': artifact_uri
    }

    # Run the experiment
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
        max_evals=args.max_evals
    )

    # Output summary
    print(f"Experiment completed with {len(results['training_results'])} models trained")
    if results['registered_model']:
        print(f"Best model registered as {args.model_name}")
        print(f"Performance: RMSE = {results['registered_model']['metrics']['rmse']:.4f}")