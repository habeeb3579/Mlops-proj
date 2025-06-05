# NYC Taxi Trip Duration Prediction with MLflow

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository provides a production-grade framework for training, tracking, and registering machine learning models for NYC Taxi trip duration prediction using MLflow. The implementation offers a comprehensive approach to the entire ML lifecycle, from data preparation to model deployment.

## Features

- **Object-Oriented Design**: Modular, configurable architecture for flexibility and extensibility
- **Multi-Model Support**: Train and compare various regression models
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - LassoLarsCV
  - Linear SVR
  - Random Forest Regressor
  - XGBoost Regressor
- **Scikit-learn Pipelines**: Combined preprocessing and modeling workflows
- **ColumnTransformer**: Flexible feature preprocessing for different data types
- **Hyperparameter Tuning**: Automated tuning with Hyperopt
- **Experiment Tracking**: Comprehensive logging with MLflow
- **Storage Options**: Support for various backends
  - Local filesystem
  - SQLite
  - PostgreSQL
  - AWS S3
  - Google Cloud Storage
- **Model Registry**: Model versioning with staging transitions and aliases
- **Production Deployment**: Ready for deployment with clean interfaces

## Requirements

- Python 3.8+
- MLflow 2.0+
- Scikit-learn 1.0+
- XGBoost 1.5+
- Pandas 1.3+
- NumPy 1.20+
- Hyperopt 0.2+

## Installation

```bash
# Clone the repository
git clone https://github.com/habeeb3579/mlops-proj.git
cd 02-experimental_tracking

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
02-experimental_tracking/
├── mlflowpy.py         # Main implementation file
├── run_experiment.py   # CLI wrapper for running experiments
├── Makefile            # Command shortcuts for common operations
├── requirements.txt    # Project dependencies
├── data/               # Data storage directory
├── models/             # Model storage directory
├── notebooks/          # Jupyter notebooks for exploration
└── scripts/            # Utility scripts
```

## Data Preparation

This project uses the NYC Taxi trip dataset available in parquet format. The data is automatically downloaded if not already present in the `data` directory.

The following preprocessing steps are applied:

1. Convert datetime columns to proper types
2. Calculate trip duration in minutes
3. Filter for reasonable trip durations (1-60 minutes)
4. Create engineered features like PU_DO (combined pickup/dropoff location ID)

## Usage Examples

### Basic Usage

The simplest way to run an experiment is:

```python
from mlflowpy import run_nyc_taxi_experiment

results = run_nyc_taxi_experiment(
    train_year=2021,
    train_month=1,
    val_year=2021,
    val_month=2,
    model_types=["LinearRegression", "XGBoost"]
)
```

### Command Line Interface

The package provides a command-line interface through the `run_experiment.py` script:

```bash
# Run experiment with default settings
python run_experiment.py

# Run specific models with custom parameters
python run_experiment.py --models Ridge XGBoost --train-year 2021 --train-month 3 --val-year 2021 --val-month 4
```

### Using the Makefile

The included Makefile provides shortcuts for common operations:

```bash
# Run full experiment with all models
make run-all

# Run specific models
make run-models MODELS="Ridge XGBoost"

# Run with custom dataset
make run-custom TRAIN_YEAR=2021 TRAIN_MONTH=5 VAL_YEAR=2021 VAL_MONTH=6

# Register the best model without training
make register-models

# Clean up MLflow tracking database
make clean
```

### Advanced Configuration

For advanced usage, you can customize the tracking configuration:

```python
from mlflowpy import NYCTaxiDurationExperiment

# Configure experiment with PostgreSQL backend
tracking_config = {
    'tracking_store': 'postgresql',
    'host': 'localhost',
    'port': 5432,
    'database': 'mlflow',
    'user': 'mlflow_user',
    'password': 'mlflow_pass'
}

experiment = NYCTaxiDurationExperiment(
    tracking_config=tracking_config,
    experiment_name="custom-nyc-taxi-experiment",
    model_registry_name="custom-taxi-regressor"
)

# Run the experiment
results = experiment.run_experiment(
    train_year=2021,
    train_month=1,
    val_year=2021,
    val_month=2,
    model_types=["RandomForest", "XGBoost"],
    tune_hyperparams=True
)
```

## Model Registry and Deployment

After training, the best model is automatically registered in the MLflow Model Registry:

```python
# Load the production model
from mlflowpy import ExperimentManager

manager = ExperimentManager(
    model_registry_name="nyc-taxi-regressor"
)

# Load the production model
model = manager.load_production_model()

# Make predictions
predictions = model.predict(test_data)
```

## Working with MLflow UI

You can view experiments in the MLflow UI:

```bash
# Launch MLflow UI with default SQLite backend
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Launch with PostgreSQL backend
mlflow ui --backend-store-uri postgresql://user:password@localhost:5432/mlflow
```

## Hyperparameter Tuning

The framework uses Hyperopt for intelligent hyperparameter tuning. Each model has a predefined search space suitable for its algorithm:

- **Ridge/Lasso**: alpha parameter
- **SVR**: C and epsilon parameters
- **Random Forest**: n_estimators, max_depth, min_samples_split, min_samples_leaf
- **XGBoost**: learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, reg_alpha, reg_lambda

To customize the hyperparameter search:

```python
from mlflowpy import HyperparameterTuner, RandomForestTrainer

# Create trainer and tuner
trainer = RandomForestTrainer()
tuner = HyperparameterTuner(trainer, max_evals=50)  # Increase evaluations for better results

# Run tuning
best_params = tuner.tune(X_train, y_train, X_val, y_val, preprocessor)
```

## Cloud Storage Integration

The framework supports cloud storage for artifacts and tracking:

### AWS S3

```python
tracking_config = {
    'tracking_store': 'aws',
    's3_bucket': 'my-mlflow-bucket',
    'region': 'us-west-2'
}

# Run experiment with AWS S3 storage
results = run_nyc_taxi_experiment(tracking_config=tracking_config)
```

### Google Cloud Storage

```python
tracking_config = {
    'tracking_store': 'gcp',
    'project': 'my-gcp-project',
    'bucket': 'my-mlflow-bucket'
}

# Run experiment with GCP storage
results = run_nyc_taxi_experiment(tracking_config=tracking_config)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The NYC Taxi and Limousine Commission for providing the dataset
- The MLflow team for their excellent model tracking framework

## Makefiles

Makefile (if using main.py to run the code)
Makefile2 (if using run_experiment.py to run the code)
Makefile1 (if using mlflowpy.py to run the code)

## Configure mlflow on GCP

[watch](https://www.youtube.com/watch?v=MWfKAgEHsHo)
create a vm, bucket, postgresdb
in the vm, sudo apt update, pip install mlflow psycopg2-binary, mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://DB_USER:DB_PASSWORD@DB_ENDPOINT:5432/DB_NAME --default-artifact-root gs://GS_BUCKET_NAME

go to compute engine, copy external api, add :5000 to it
