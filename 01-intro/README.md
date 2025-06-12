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
- Cross-validation experiments

## Requirements

- Python 3.9+
- Required packages: pandas, numpy, matplotlib, seaborn, scikit-learn, pyarrow, joblib

## Installation

1. Clone this repository and cd into 01-intro
2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

   Or use provided Makefile targets:

   ```
   # Install requirements in current environment
   make install-requirements

   # OR create and setup a conda environment
   make create-env
   make install-conda-requirements
   ```

## Quick Start

### Using the Command Line Interface

The script provides two main modes of operation: `train` and `predict`.

#### Training a new model

```bash
python NYC_trip_duration_prediction.py train \
    --train-year 2021 \
    --train-month 1 \
    --test-year 2021 \
    --test-month 2 \
    --model random_forest \
    --target-transform log \
    --features PULocationID DOLocationID trip_distance pickup_hour \
    --categorical-features PULocationID DOLocationID \
    --numerical-features trip_distance pickup_hour \
    --cat-preprocessor onehot \
    --num-preprocessor standard \
    --save-model \
    --save-plot
```

#### With cat_preprocessor_params

```bash
predictor = NYCTaxiDurationPredictor(
    cat_preprocessor='onehot',
    cat_preprocessor_params={'drop': 'first', 'sparse_output': False}
)
```

```bash
python script.py train --cat-preprocessor onehot --cat-preprocessor-params drop=first sparse_output=False
```

#### Making predictions with an existing model

```bash
python NYC_trip_duration_prediction.py predict \
    --model-path models/nyc_taxi_duration_random_forest_20250512_000000.joblib \
    --year 2021 \
    --month 2 \
    --save-plot
```

### Using the Makefile

For convenience, a Makefile is provided with common operations:

```bash
# Create required directories
make setup
```

```bash
# Install requirements in current env
make install-requirements
```

```bash
# Create conda env, activate it and install requirements

# Create env with default settings
make create-env

# Create env with Jupyter kernel
make create-env WITH_IPYKERNEL=true

# Install dependencies in the conda environment
make install-conda-requirements
```

```bash
# Train a linear regression model with default settings
make train-linear

# Train a ridge regression model
make train-ridge

# Train a lasso regression model
make train-lasso

# Train a random forest model
make train-rf

# Train a gradient boosting model
make train-gb

# Train a custom model with specific parameters
make train-custom DATA_YEAR=2021 DATA_MONTH=1 MODEL=random_forest FEATURES="PULocationID DOLocationID trip_distance pickup_hour" TARGET_TRANSFORM=log

make train-custom \
  DATA_YEAR=2021 \
  DATA_MONTH=1 \
  TEST_YEAR=2021 \
  TEST_MONTH=1 \
  MODEL=linear_regression \
  TARGET_TRANSFORM=none \
  FEATURES="PULocationID DOLocationID trip_distance" \
  CAT_FEATURES="PULocationID DOLocationID" \
  NUM_FEATURES="trip_distance" \
  CAT_PREPROCESSOR=dictvectorizer \
  NUM_PREPROCESSOR=none

# Make predictions using a trained model
make predict MODEL_PATH=models/nyc_taxi_duration_random_forest_20250512_000000.joblib DATA_YEAR=2021 DATA_MONTH=2

# Train all supported model types
make train-all

# Run feature engineering experiments
make train-feature-experiment

# Run preprocessing experiments
make train-preproc-experiment

# Run cross-validation experiments
make cv-experiment

# Download data only
make download-data DATA_YEAR=2021 DATA_MONTH=1 TAXI_TYPE=yellow

# Clean up cache files
make clean

# Clean up all generated files
make clean-all

# Display help
make help
```

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
- `--cat-preprocessor`: Categorical feature preprocessor (default: onehot)
- `--num-preprocessor`: Numerical feature preprocessor (default: standard)
- `--model`: Machine learning model to use (default: linear_regression)
- `--target-transform`: Transformation to apply to the target variable (default: log)
- `--save-model`: Save the trained model (default: True)
- `--no-save-model`: Do not save the trained model
- `--save-plot`: Save the prediction plot (default: True)
- `--no-save-plot`: Do not save the prediction plot
- `--random-state`: Random seed for reproducibility (default: 42)
- `--cv`: Number of cross-validation folds (optional)

### Predict Mode

- `--model-path`: Path to a saved model to load (required)
- `--year`: Year for test data (default: 2021)
- `--month`: Month for test data (default: 1)
- `--taxi-type`: Type of taxi data to use (optional)
- `--save-plot`: Save the prediction plot (default: True)
- `--no-save-plot`: Do not save the prediction plot

## Output Structure

The pipeline creates the following directory structure:

```
./
├── models/          # Saved models
│   └── nyc_taxi_duration_model_timestamp.joblib
├── plots/           # Saved visualization plots
│   └── nyc_taxi_duration_model_timestamp.png
├── data/            # Downloaded and processed data
│   └── nyc_taxi_type_year_month.parquet
├── logs/            # Log files
├── NYC_trip_duration_pred.py  # Main script
├── Makefile
├── requirements.txt
└── README.md
```

## Example Workflows

### Basic Training and Prediction

```bash
# Train a linear regression model on January 2021 data
make train-linear

# Use the trained model to predict on February 2021 data
make predict-example
```

### Data Talks Club Example

```bash
# Train a linear regression model on January 2021 data and test on February 2021
!make train-custom \
  DATA_YEAR=2021 \
  DATA_MONTH=1 \
  TEST_YEAR=2021 \
  TEST_MONTH=2 \
  MODEL=linear_regression \
  TARGET_TRANSFORM=none \
  FEATURES="PULocationID DOLocationID trip_distance" \
  CAT_FEATURES="PULocationID DOLocationID" \
  NUM_FEATURES="trip_distance" \
  CAT_PREPROCESSOR=dictvectorizer \
  NUM_PREPROCESSOR=none
```
