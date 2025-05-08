# NYC Taxi Duration Predictor

This project provides a pipeline for predicting NYC taxi ride durations using various machine learning models. The implementation includes data downloading, feature engineering, model training, evaluation, visualization, and model persistence.

## Features

- Download NYC TLC taxi trip data
- Calculate and predict ride durations
- Support for multiple ML models (Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting)
- Feature preprocessing with categorical and numerical transformers
- Target variable transformation (log, power)
- Model evaluation with standard metrics (RMSE, MAE, R²)
- Visualization of actual vs. predicted values
- Model saving and loading capabilities
- Comprehensive CLI with separate train and predict modes
- Convenient Makefile for common operations

## Requirements

- Python 3.6+
- Required packages: pandas, numpy, matplotlib, seaborn, scikit-learn, pyarrow

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn pyarrow joblib
   ```

## Quick Start

### Using the Command Line Interface

The script provides two main modes of operation: `train` and `predict`.

#### Training a new model

```bash
python taxi_duration_predictor.py train \
    --train-year 2021 \
    --train-month 1 \
    --model random_forest \
    --target-transform log
```

#### Making predictions with an existing model

```bash
python taxi_duration_predictor.py predict \
    --model-path models/nyc_taxi_duration_random_forest_20250507_000000.joblib \
    --year 2021 \
    --month 2
```

### Using the Makefile

For convenience, a Makefile is provided with common operations:

```bash
Add models and plots folders
make setup
```

```bash
Install requirements in current env
make install-requirements
```

```bash
create conda env, activate it and install requirements

# Create env with default settings
make create-env

# Create env with Jupyter kernel
make create-env WITH_IPYKERNEL=true

# Activate the env
conda activate nyc-taxi-env

# Install dependencies
make install-requirements

# OR

# Full setup with dependencies
make install-conda-requirements
```

```bash
# Train a linear regression model with default settings
make train-linear

# Train a random forest model with custom data
make train-custom DATA_YEAR=2022 DATA_MONTH=3 MODEL=random_forest

# Make predictions using a trained model
make predict MODEL_PATH=models/nyc_taxi_duration_random_forest_20250507_000000.joblib DATA_YEAR=2021 DATA_MONTH=2

# Train all supported model types
make train-all

# Display help
make help
```

```
Data talks club used linear regression, 2021-01 data and predict on 2021-02
make train-custom DATA_YEAR=2021 DATA_MONTH=1 MODEL=linear_regression

## Command Line Arguments

### Train Mode

- `--train-year`: Year for training data (default: 2021)
- `--train-month`: Month for training data (default: 1)
- `--test-year`: Year for test data (optional)
- `--test-month`: Month for test data (optional)
- `--test-size`: Test size for train/test split if test year/month not provided (default: 0.2)
- `--features`: List of feature columns to use
- `--categorical-features`: List of categorical feature columns
- `--numerical-features`: List of numerical feature columns
- `--model`: Machine learning model to use (default: linear_regression)
- `--target-transform`: Transformation to apply to the target variable
- `--save-model`: Save the trained model (default: True)
- `--no-save-model`: Do not save the trained model
- `--model-format`: Format to save the model (default: joblib)
- `--random-state`: Random seed for reproducibility (default: 42)
- `--save-plot`: Save the prediction plot (default: True)
- `--no-save-plot`: Do not save the prediction plot

### Predict Mode

- `--model-path`: Path to a saved model to load (required)
- `--year`: Year for test data (default: 2021)
- `--month`: Month for test data (default: 1)
- `--save-plot`: Save the prediction plot (default: True)
- `--no-save-plot`: Do not save the prediction plot

## Output Structure

The pipeline creates the following directory structure:

```
01-intro/
├── models/          # Saved models
│   └── nyc_taxi_duration_model_timestamp.joblib
├── plots/           # Saved visualization plots
│   └── nyc_taxi_duration_model_timestamp.png
├── NYC_trip_duration_prediction.py  # Main script
├── Makefile
└── README.md
```