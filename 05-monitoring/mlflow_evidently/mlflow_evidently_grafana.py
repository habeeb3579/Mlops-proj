import datetime
import time
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import psycopg2
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pickle
from loguru import logger
from tqdm import tqdm
import pytz
from prefect import flow, task

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Evidently AI
from evidently import DataDefinition
from evidently import Dataset
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.metrics import ValueDrift, DriftedColumnsCount, MissingValueCount

#mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db

# Configuration
class Config:
    # Database settings
    DB_HOST = "localhost"
    DB_PORT = 5433
    DB_NAME = "taxi_monitoring"
    DB_USER = "postgres"
    DB_PASSWORD = "example"
    
    # Data settings
    TAXI_TYPE = "green"
    TARGET_COLUMN = "duration_min"
    NUM_FEATURES = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
    CAT_FEATURES = ["PULocationID", "DOLocationID"]
    
    # Training configuration
    TRAIN_MONTHS = [(2022, 1), (2022, 2)]  # Months for training
    VAL_MONTHS = [(2022, 3)]  # Months for validation (optional)
    TEST_MONTHS = [(2022, 4)]  # Months for testing (optional)
    
    # MLflow settings
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    EXPERIMENT_NAME = "taxi_duration_prediction"
    
    # Paths
    DATA_DIR = Path("data")
    LOGS_DIR = Path("logs")
    MODELS_DIR = Path("models")
    
    # Processing settings
    TIMEZONE = pytz.timezone('America/New_York')

def generate_data_identifier(taxi_type: str, year: int, month: int) -> str:
    """Generate standardized data identifier"""
    return f"{taxi_type}-{year}-{month:02d}"

def generate_model_name(taxi_type: str, train_months: List[Tuple[int, int]]) -> str:
    """Generate standardized model name based on training data"""
    if len(train_months) == 1:
        year, month = train_months[0]
        return f"{taxi_type}-{year}-{month:02d}"
    else:
        start_year, start_month = train_months[0]
        end_year, end_month = train_months[-1]
        return f"{taxi_type}-{start_year}-{start_month:02d}_to_{end_year}-{end_month:02d}"

# Setup directories and logging
def setup_environment():
    """Setup directories and logging configuration"""
    for directory in [Config.DATA_DIR, Config.LOGS_DIR, Config.MODELS_DIR]:
        directory.mkdir(exist_ok=True)
    
    # Configure loguru with data-aware naming
    data_id = generate_data_identifier(Config.TAXI_TYPE, 
                                      Config.TRAIN_MONTHS[0][0], 
                                      Config.TRAIN_MONTHS[0][1])
    log_file = Config.LOGS_DIR / f"taxi_monitoring_{data_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(
        log_file,
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
        level="INFO"
    )
    
    # Setup MLflow
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.EXPERIMENT_NAME)

# Database operations
def get_connection(dbname: Optional[str] = None, autocommit: bool = False) -> psycopg2.extensions.connection:
    """Create database connection"""
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        dbname=dbname or "postgres"
    )
    if autocommit:
        conn.autocommit = True
    return conn

@task
def setup_database():
    """Setup database and create tables"""
    try:
        # Create database if it doesn't exist
        conn = get_connection(autocommit=True)
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (Config.DB_NAME,))
            if not cur.fetchone():
                cur.execute(f"CREATE DATABASE {Config.DB_NAME}")
                logger.info(f"✅ Database '{Config.DB_NAME}' created.")
            else:
                logger.info(f"ℹ️ Database '{Config.DB_NAME}' already exists.")
            cur.close()
        finally:
            conn.close()
        
        # Create tables
        create_table_sql = """
        DROP TABLE IF EXISTS taxi_drift_metrics;
        CREATE TABLE taxi_drift_metrics (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP WITH TIME ZONE,
            date_processed DATE,
            prediction_drift FLOAT,
            num_drifted_columns INTEGER,
            share_missing_values FLOAT,
            model_name VARCHAR(100),
            model_version VARCHAR(50),
            data_points_count INTEGER,
            data_identifier VARCHAR(50),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_taxi_drift_date ON taxi_drift_metrics(date_processed);
        CREATE INDEX IF NOT EXISTS idx_taxi_drift_timestamp ON taxi_drift_metrics(timestamp);
        CREATE INDEX IF NOT EXISTS idx_taxi_drift_data_id ON taxi_drift_metrics(data_identifier);
        """
        
        with get_connection(Config.DB_NAME, autocommit=False) as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
                conn.commit()
                logger.info("✅ taxi_drift_metrics table created.")
    
    except Exception as e:
        logger.error(f"❌ Error setting up database: {e}")
        raise

@task
def download_data(year: int, month: int) -> pd.DataFrame:
    """Download NYC taxi data for a specific month"""
    data_id = generate_data_identifier(Config.TAXI_TYPE, year, month)
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{Config.TAXI_TYPE}_tripdata_{year}-{month:02d}.parquet"
    logger.info(f"Downloading data for {data_id} from {url}")
    
    try:
        df = pd.read_parquet(url)
        logger.info(f"✅ Downloaded {len(df)} records for {data_id}")
        return df
    except Exception as e:
        logger.error(f"❌ Error downloading data for {data_id}: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess taxi data with flexible datetime column handling"""
    
    # Handle different datetime column names
    pickup_col = None
    dropoff_col = None
    
    if 'lpep_pickup_datetime' in df.columns:
        pickup_col = 'lpep_pickup_datetime'
        dropoff_col = 'lpep_dropoff_datetime'
    elif 'tpep_pickup_datetime' in df.columns:
        pickup_col = 'tpep_pickup_datetime'
        dropoff_col = 'tpep_dropoff_datetime'
    else:
        raise ValueError("No recognized pickup/dropoff datetime columns found")
    
    logger.info(f"Using datetime columns: {pickup_col}, {dropoff_col}")
    
    # Create duration target variable
    df['duration_min'] = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60
    
    # Filter reasonable durations (1 minute to 3 hours)
    df = df[(df['duration_min'] >= 1) & (df['duration_min'] <= 180)]
    
    # Handle missing values for numerical features
    for col in Config.NUM_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Handle missing values for categorical features
    for col in Config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    # Store pickup datetime for daily processing
    df['pickup_datetime'] = df[pickup_col]
    
    logger.info(f"✅ Preprocessed data: {len(df)} records")
    return df

def create_preprocessing_pipeline():
    """Create preprocessing pipeline with ColumnTransformer"""
    
    # Numerical preprocessing
    num_transformer = StandardScaler()
    
    # Categorical preprocessing
    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, Config.NUM_FEATURES),
            ('cat', cat_transformer, Config.CAT_FEATURES)
        ]
    )
    
    return preprocessor

def get_hyperparameter_space(model_name: str) -> Dict:
    """Define hyperparameter search spaces for different models"""
    
    spaces = {
        'LinearRegression': {},
        
        'Ridge': {
            'model__alpha': hp.loguniform('alpha', np.log(0.01), np.log(100))
        },
        
        'RandomForest': {
            'model__n_estimators': hp.quniform('n_estimators', 50, 200, 50),  # 50, 100, 150, 200
            'model__max_depth': hp.choice('max_depth', [5, 10, 15, None]),
            'model__min_samples_split': hp.quniform('min_samples_split', 2, 10, 2),  # 2, 4, 6, 8, 10
            'model__min_samples_leaf': hp.quniform('min_samples_leaf', 1, 4, 1)  # 1, 2, 3, 4
        },
        
        'XGBoost': {
            'model__n_estimators': hp.quniform('n_estimators', 50, 200, 50),  # 50, 100, 150, 200
            'model__max_depth': hp.quniform('max_depth', 3, 7, 2),  # 3, 5, 7
            'model__learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'model__subsample': hp.uniform('subsample', 0.6, 1.0),
            'model__colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
        },
        
        'GradientBoosting': {
            'model__n_estimators': hp.quniform('n_estimators', 50, 200, 50),  # 50, 100, 150, 200
            'model__max_depth': hp.quniform('max_depth', 3, 7, 2),  # 3, 5, 7
            'model__learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'model__subsample': hp.uniform('subsample', 0.6, 1.0)
        },
        
        'SVR': {
            'model__C': hp.loguniform('C', np.log(0.1), np.log(100)),
            'model__gamma': hp.choice('gamma', ['scale', 'auto']),
            'model__epsilon': hp.uniform('epsilon', 0.01, 0.2)
        },
        
        'KNNRegressor': {
            'model__n_neighbors': hp.quniform('n_neighbors', 3, 11, 2),  # 3, 5, 7, 9, 11
            'model__weights': hp.choice('weights', ['uniform', 'distance'])
        }
    }
    
    return spaces.get(model_name, {})

def convert_hyperparams_to_native_types(params, model_name):
    """Convert numpy types to native Python types and apply model-specific fixes"""
    converted = {}
    
    for key, value in params.items():
        # Convert numpy types to native Python types
        if isinstance(value, np.integer):
            converted[key] = int(value)
        elif isinstance(value, np.floating):
            converted[key] = float(value)
        else:
            converted[key] = value
    
    # Model-specific parameter validation and fixes
    if model_name == 'RandomForest':
        # Ensure min_samples_split is at least 2
        if 'min_samples_split' in converted and converted['min_samples_split'] < 2:
            converted['min_samples_split'] = 2
        # Convert n_estimators to int
        if 'n_estimators' in converted:
            converted['n_estimators'] = max(1, int(converted['n_estimators']))
        # Handle max_depth - keep None or convert to int
        if 'max_depth' in converted and converted['max_depth'] is not None:
            converted['max_depth'] = max(1, int(converted['max_depth']))
        # Convert other integer parameters
        if 'min_samples_leaf' in converted:
            converted['min_samples_leaf'] = max(1, int(converted['min_samples_leaf']))
            
    elif model_name == 'XGBoost':
        # XGBoost specific parameter handling
        if 'n_estimators' in converted:
            converted['n_estimators'] = max(1, int(converted['n_estimators']))
        if 'max_depth' in converted:
            converted['max_depth'] = max(1, int(converted['max_depth']))
    
    elif model_name == 'GradientBoosting':
        # GradientBoosting specific parameter handling
        if 'n_estimators' in converted:
            converted['n_estimators'] = max(1, int(converted['n_estimators']))
        if 'max_depth' in converted:
            converted['max_depth'] = max(1, int(converted['max_depth']))
    
    elif model_name == 'SVR':
        # Handle gamma parameter for SVR - hp.choice returns index, convert to actual value
        if 'gamma' in converted:
            gamma_options = ['scale', 'auto']
            if isinstance(converted['gamma'], (int, float)) and 0 <= converted['gamma'] < len(gamma_options):
                converted['gamma'] = gamma_options[int(converted['gamma'])]
    
    elif model_name == 'KNNRegressor':
        # Handle weights parameter for KNN - hp.choice returns index, convert to actual value
        if 'weights' in converted:
            weights_options = ['uniform', 'distance']
            if isinstance(converted['weights'], (int, float)) and 0 <= converted['weights'] < len(weights_options):
                converted['weights'] = weights_options[int(converted['weights'])]
        # Convert n_neighbors to int
        if 'n_neighbors' in converted:
            converted['n_neighbors'] = max(1, int(converted['n_neighbors']))
    
    return converted

def hyperparameter_objective(params, model_class, X_train, y_train, X_val, y_val, model_name):
    """Objective function for hyperparameter optimization with type conversion"""
    try:
        # Create preprocessing pipeline
        preprocessor = create_preprocessing_pipeline()
        
        # Convert hyperparameters and remove model__ prefix
        clean_params = {k.replace('model__', ''): v for k, v in params.items()}
        clean_params = convert_hyperparams_to_native_types(clean_params, model_name)
        
        # Debug logging
        logger.debug(f"Raw params for {model_name}: {params}")
        logger.debug(f"Clean params for {model_name}: {clean_params}")
        
        # Create model with hyperparameters
        model = model_class(**clean_params)
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train and evaluate
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        
        return {'loss': mae, 'status': STATUS_OK}
    
    except Exception as e:
        logger.error(f"Error in hyperparameter optimization for {model_name}: {e}")
        logger.error(f"Problematic params: {clean_params}")
        return {'loss': float('inf'), 'status': STATUS_OK}


@task
def load_training_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and combine training, validation, and test data"""
    
    train_data_list = []
    val_data_list = []
    test_data_list = []
    
    # Load training data
    for year, month in Config.TRAIN_MONTHS:
        try:
            df = download_data(year, month)
            df = preprocess_data(df)
            train_data_list.append(df)
            data_id = generate_data_identifier(Config.TAXI_TYPE, year, month)
            logger.info(f"✅ Loaded training data for {data_id}: {len(df)} records")
        except Exception as e:
            data_id = generate_data_identifier(Config.TAXI_TYPE, year, month)
            logger.error(f"❌ Failed to load training data for {data_id}: {e}")
    
    # Load validation data
    for year, month in Config.VAL_MONTHS:
        try:
            df = download_data(year, month)
            df = preprocess_data(df)
            val_data_list.append(df)
            data_id = generate_data_identifier(Config.TAXI_TYPE, year, month)
            logger.info(f"✅ Loaded validation data for {data_id}: {len(df)} records")
        except Exception as e:
            data_id = generate_data_identifier(Config.TAXI_TYPE, year, month)
            logger.error(f"❌ Failed to load validation data for {data_id}: {e}")
    
    # Load test data
    for year, month in Config.TEST_MONTHS:
        try:
            df = download_data(year, month)
            df = preprocess_data(df)
            test_data_list.append(df)
            data_id = generate_data_identifier(Config.TAXI_TYPE, year, month)
            logger.info(f"✅ Loaded test data for {data_id}: {len(df)} records")
        except Exception as e:
            data_id = generate_data_identifier(Config.TAXI_TYPE, year, month)
            logger.error(f"❌ Failed to load test data for {data_id}: {e}")
    
    # Combine data
    train_data = pd.concat(train_data_list, ignore_index=True) if train_data_list else pd.DataFrame()
    val_data = pd.concat(val_data_list, ignore_index=True) if val_data_list else pd.DataFrame()
    test_data = pd.concat(test_data_list, ignore_index=True) if test_data_list else pd.DataFrame()
    
    logger.info(f"📊 Data loaded - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data

@task
def train_models_with_hyperopt(train_data: pd.DataFrame, val_data: pd.DataFrame, 
                              test_data: pd.DataFrame) -> Tuple[str, str, Dict]:
    """Train multiple models with hyperparameter optimization and structured MLflow logging"""
    
    model_name_base = generate_model_name(Config.TAXI_TYPE, Config.TRAIN_MONTHS)
    logger.info(f"🚀 Starting model training with hyperparameter optimization for {model_name_base}")
    
    # Prepare features and target
    feature_cols = Config.NUM_FEATURES + Config.CAT_FEATURES
    X_train = train_data[feature_cols].copy()
    y_train = train_data[Config.TARGET_COLUMN]
    
    # Handle validation data
    if not val_data.empty:
        X_val = val_data[feature_cols].copy()
        y_val = val_data[Config.TARGET_COLUMN]
    else:
        # Split training data if no separate validation data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        logger.info("📊 Using train split for validation")
    
    # Prepare test data
    if not test_data.empty:
        X_test = test_data[feature_cols].copy()
        y_test = test_data[Config.TARGET_COLUMN]
    else:
        X_test, y_test = X_val, y_val
        logger.info("📊 Using validation data as test data")
    
    # Define model classes
    model_classes = {
        'LinearRegression': LinearRegression,
        'Ridge': Ridge,
        #'RandomForest': RandomForestRegressor,
        'XGBoost': xgb.XGBRegressor,
        'GradientBoosting': GradientBoostingRegressor,
        #'SVR': SVR,
        #'KNNRegressor': KNeighborsRegressor
    }
    
    best_model = None
    best_score = float('inf')
    best_model_name = None
    best_params = None
    model_results = {}
    model_run_ids = {}
    
    # Start parent run for model comparison
    # Start parent run for model comparison
    with mlflow.start_run(run_name=f"ModelComparison_{model_name_base}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        parent_run_id = mlflow.active_run().info.run_id
        
        for name, model_class in tqdm(model_classes.items(), desc="Training models"):
            try:
                logger.info(f"🔧 Training {name} with hyperparameter optimization")
                
                # Start model-specific parent run
                with mlflow.start_run(run_name=f"{name}_{model_name_base}", nested=True):
                    model_parent_run_id = mlflow.active_run().info.run_id
                    
                    # Get hyperparameter space
                    param_space = get_hyperparameter_space(name)
                    
                    if param_space:
                        # Hyperparameter optimization with nested runs
                        trials = Trials()
                        best_hp = None
                        
                        def objective(params):
                            # Start nested run for each hyperparameter trial
                            with mlflow.start_run(nested=True):
                                trial_result = hyperparameter_objective(
                                    params, model_class, X_train, y_train, X_val, y_val, name
                                )
                                
                                # Log trial parameters and metrics
                                clean_params = {k.replace('model__', ''): v for k, v in params.items()}
                                clean_params = convert_hyperparams_to_native_types(clean_params, name)
                                mlflow.log_params(clean_params)
                                mlflow.log_metric("val_mae", trial_result['loss'])
                                
                                return trial_result
                        
                        best_hp = fmin(
                            fn=objective,
                            space=param_space,
                            algo=tpe.suggest,
                            max_evals=5,
                            trials=trials,
                            verbose=False
                        )
                        
                        # Convert hyperparameters
                        model_params = {k.replace('model__', ''): v for k, v in best_hp.items()}
                        model_params = convert_hyperparams_to_native_types(model_params, name)
                    else:
                        model_params = {}
                    
                    # Create preprocessing pipeline
                    preprocessor = create_preprocessing_pipeline()
                    
                    # Create final model with best hyperparameters
                    if name == 'XGBoost':
                        model = model_class(random_state=42, **model_params)
                    elif name in ['RandomForest', 'GradientBoosting']:
                        model = model_class(random_state=42, **model_params)
                    else:
                        model = model_class(**model_params)
                    
                    # Create pipeline
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])
                    
                    # Train final model
                    pipeline.fit(X_train, y_train)
                    
                    # Evaluate on validation and test sets
                    y_val_pred = pipeline.predict(X_val)
                    y_test_pred = pipeline.predict(X_test)
                    
                    # Calculate metrics
                    val_mae = mean_absolute_error(y_val, y_val_pred)
                    val_mse = root_mean_squared_error(y_val, y_val_pred)
                    val_r2 = r2_score(y_val, y_val_pred)
                    
                    test_mae = mean_absolute_error(y_test, y_test_pred)
                    test_mse = root_mean_squared_error(y_test, y_test_pred)
                    test_r2 = r2_score(y_test, y_test_pred)
                    
                    # Store results
                    model_results[name] = {
                        'val_mae': val_mae,
                        'val_rmse': val_mse,
                        'val_r2': val_r2,
                        'test_mae': test_mae,
                        'test_rmse': test_mse,
                        'test_r2': test_r2,
                        'best_params': model_params,
                        'model_pipeline': pipeline,
                        'run_id': model_parent_run_id  # This is the run where the model is logged
                    }
                    
                    model_run_ids[name] = model_parent_run_id
                    
                    # Log optimal model parameters and metrics
                    mlflow.log_param("model_type", name)
                    mlflow.log_param("data_identifier", model_name_base)
                    mlflow.log_params(model_params)
                    mlflow.log_metrics({
                        "val_mae": val_mae,
                        "val_mse": val_mse,
                        "val_r2": val_r2,
                        "test_mae": test_mae,
                        "test_mse": test_mse,
                        "test_r2": test_r2
                    })
                    
                    # Log the optimized model - MAKE SURE THIS IS IN THE CORRECT RUN CONTEXT
                    mlflow.sklearn.log_model(pipeline, "model")
                    #mlflow.sklearn.log_model(pipeline, f"optimal_{name}_model")
                    
                    logger.info(f"✅ {name} - Val MAE: {val_mae:.4f}, Test MAE: {test_mae:.4f}, R2: {test_r2:.4f}")
                    
                    # Update best model based on validation MAE
                    if val_mae < best_score:
                        best_score = val_mae
                        best_model = pipeline
                        best_model_name = name
                        best_params = model_params
                        
            except Exception as e:
                logger.error(f"❌ Error training {name}: {e}")
        
        # Log comparison metrics in parent run
        if model_results:
            comparison_metrics = {}
            for name, results in model_results.items():
                comparison_metrics[f"{name}_val_mae"] = results['val_mae']
                comparison_metrics[f"{name}_test_mae"] = results['test_mae']
                comparison_metrics[f"{name}_test_r2"] = results['test_r2']
            
            mlflow.log_metrics(comparison_metrics)
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_param("data_identifier", model_name_base)
            mlflow.log_metric("best_val_mae", best_score)
        
        # Save best model with data-aware naming
        model_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = Config.MODELS_DIR / f"best_model_{model_name_base}_{model_timestamp}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        # FIXED: Get the correct run ID for the best model
        best_model_run_id = model_run_ids[best_model_name]
        model_uri = f"runs:/{best_model_run_id}/model"
        #model_uri = f"runs:/{model_run_ids[best_model_name]}/optimal_{best_model_name}_model"
        
        logger.info(f"model uri is: {model_uri}")
        logger.info(f"best model name: {best_model_name}")
        logger.info(f"best model run id: {best_model_run_id}")
        
        # Create or get registered model
        client = MlflowClient()
        registered_model_name = f"taxi_duration_{Config.TAXI_TYPE}"
        
        try:
            # Check if model URI exists before registering
            try:
                run = client.get_run(best_model_run_id)
                artifacts = client.list_artifacts(best_model_run_id)
                logger.info(f"Available artifacts in run {best_model_run_id}: {[a.path for a in artifacts]}")
            except Exception as e:
                logger.error(f"Error checking run artifacts: {e}")
                
            # Try to get registered model, create if doesn't exist
            client.get_registered_model(registered_model_name)
            logger.info(f"Registered model '{registered_model_name}' already exists")
        except Exception:
            logger.info(f"Creating new registered model: {registered_model_name}")
            client.create_registered_model(
                registered_model_name, 
                description=f"Best taxi duration prediction model for {Config.TAXI_TYPE} taxi data"
            )

        try:
            # Register the model version
            logger.info(f"Registering model version from URI: {model_uri}")
            model_version = client.create_model_version(
                name=registered_model_name,
                source=model_uri,
                description=f"Best model: {best_model_name} trained on {model_name_base} with MAE: {best_score:.4f}"
            )
            
            logger.info(f"Successfully registered model version: {model_version.version}")
            
            # Set alias for the model version
            client.set_registered_model_alias(
                name=registered_model_name,
                alias="production",
                version=model_version.version
            )
            
            # Also set a training-specific alias
            training_alias = f"training_{model_name_base.replace('-', '_')}"
            client.set_registered_model_alias(
                name=registered_model_name,
                alias=training_alias,
                version=model_version.version
            )

            # TEST: Load the model using the production alias to verify registration worked
            try:
                logger.info("🔍 Testing model loading from registry...")
                
                # Method 1: Load using production alias
                production_model_uri = f"models:/{registered_model_name}@production"
                loaded_production_model = mlflow.sklearn.load_model(production_model_uri)
                logger.info(f"✅ Successfully loaded model using production alias: {production_model_uri}")
                
                # Method 2: Load using training-specific alias
                training_model_uri = f"models:/{registered_model_name}@{training_alias}"
                loaded_training_model = mlflow.sklearn.load_model(training_model_uri)
                logger.info(f"✅ Successfully loaded model using training alias: {training_model_uri}")
                
                # Method 3: Load using version number
                version_model_uri = f"models:/{registered_model_name}/{model_version.version}"
                loaded_version_model = mlflow.sklearn.load_model(version_model_uri)
                logger.info(f"✅ Successfully loaded model using version number: {version_model_uri}")
                
                # Optional: Test prediction to ensure model works
                if len(X_test) > 0:
                    sample_prediction = loaded_production_model.predict(X_test[:1])
                    logger.info(f"✅ Sample prediction test successful: {sample_prediction[0]:.4f}")
                
            except Exception as load_error:
                logger.error(f"❌ Error loading registered model: {load_error}")
                logger.error("This indicates the model registration may have failed")
            
        except Exception as e:
            logger.error(f"Error during model registration: {e}")
            logger.error(f"Model URI used: {model_uri}")
            raise
        
        logger.info(f"🏆 Best model: {best_model_name} with validation MAE: {best_score:.4f}")
        logger.info(f"📊 Best model test MAE: {model_results[best_model_name]['test_mae']:.4f}")
        logger.info(f"🏷️ Model registered as '{registered_model_name}' v{model_version.version} with aliases: production, {training_alias}")
        
        return str(model_path), model_version.version, model_results

def load_model(model_path: str) -> Pipeline:
    """Load trained model pipeline"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def generate_predictions(data: pd.DataFrame, model_pipeline: Pipeline) -> pd.DataFrame:
    """Generate predictions using the trained pipeline"""
    feature_cols = Config.NUM_FEATURES + Config.CAT_FEATURES
    X = data[feature_cols].copy()
    
    # Generate predictions
    predictions = model_pipeline.predict(X)
    data_with_predictions = data.copy()
    data_with_predictions['prediction'] = predictions
    
    return data_with_predictions

@task
def calculate_drift_metrics(reference_data: pd.DataFrame, current_data: pd.DataFrame, 
                          model_name: str, model_version: str) -> Dict[str, Any]:
    """Calculate drift metrics using Evidently AI"""
    try:
        # Get feature names after preprocessing
        feature_cols = Config.NUM_FEATURES + Config.CAT_FEATURES
        
        # Define data structure
        data_definition = DataDefinition(
            numerical_columns=Config.NUM_FEATURES + ['prediction'],
            categorical_columns=Config.CAT_FEATURES
        )
        
        # Create datasets
        train_dataset = Dataset.from_pandas(reference_data, data_definition)
        val_dataset = Dataset.from_pandas(current_data, data_definition)
        
        # Create and run report
        report = Report(metrics=[
            ValueDrift(column='prediction'),
            DriftedColumnsCount(),
            MissingValueCount(column='prediction'),
        ])
        
        snapshot = report.run(reference_data=train_dataset, current_data=val_dataset)
        result = snapshot.dict()
        
        # Extract metrics
        prediction_drift = next(
            (item['value'] for item in result['metrics'] 
             if item['metric_id'] == 'ValueDrift(column=prediction)'), 
            None
        )
        
        num_drifted_columns = next(
            (item['value']['count'] for item in result['metrics'] 
             if item['metric_id'] == 'DriftedColumnsCount(drift_share=0.5)'), 
            None
        )
        
        missing_values_share = next(
            (item['value']['count'] for item in result['metrics'] 
             if item['metric_id'] == 'MissingValueCount(column=prediction)'), 
            None
        )
        
        metrics = {
            'prediction_drift': float(prediction_drift),
            'num_drifted_columns': int(num_drifted_columns),
            'share_missing_values': float(missing_values_share) / len(current_data) if missing_values_share else 0.0,
            'model_name': model_name,
            'model_version': model_version,
            'data_points_count': len(current_data)
        }
        
        logger.info(f"📊 Drift metrics calculated: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Error calculating drift metrics: {e}")
        return {
            'prediction_drift': None,
            'num_drifted_columns': None,
            'share_missing_values': None,
            'model_name': model_name,
            'model_version': model_version,
            'data_points_count': len(current_data)
        }

@task
def store_metrics(metrics: Dict[str, Any], timestamp: datetime.datetime, date_processed: datetime.date, data_identifier: str):
    """Store drift metrics in database with data identifier"""
    try:
        with get_connection(Config.DB_NAME) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO taxi_drift_metrics 
                    (timestamp, date_processed, prediction_drift, num_drifted_columns, 
                     share_missing_values, model_name, model_version, data_points_count, data_identifier)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    timestamp,
                    date_processed,
                    metrics['prediction_drift'],
                    metrics['num_drifted_columns'],
                    metrics['share_missing_values'],
                    metrics['model_name'],
                    metrics['model_version'],
                    metrics['data_points_count'],
                    data_identifier
                ))
                conn.commit()
                logger.info(f"✅ Metrics stored for {date_processed} with data identifier: {data_identifier}")
    except Exception as e:
        logger.error(f"❌ Error storing metrics: {e}")
        raise

@task
def process_daily_data(monthly_data: pd.DataFrame, reference_data: pd.DataFrame, 
                      model_pipeline: Pipeline, model_name: str, 
                      model_version: str, year: int, month: int) -> List[Dict]:
    """Process data day by day for drift monitoring"""
    
    # Generate data identifier for this month
    data_identifier = generate_data_identifier(Config.TAXI_TYPE, year, month)
    
    # Get unique dates in the month
    monthly_data['date'] = monthly_data['pickup_datetime'].dt.date
    unique_dates = sorted(monthly_data['date'].unique())
    
    results = []
    
    for date in tqdm(unique_dates, desc=f"Processing {year}-{month:02d}"):
        try:
            # Filter data for the specific date
            daily_data = monthly_data[monthly_data['date'] == date].copy()
            
            if len(daily_data) == 0:
                logger.warning(f"⚠️ No data available for {date}")
                continue
            
            logger.info(f"📅 Processing {date} - {len(daily_data)} records")
            
            # Generate predictions
            daily_data_with_predictions = generate_predictions(daily_data, model_pipeline)
            
            # Calculate drift metrics
            metrics = calculate_drift_metrics(
                reference_data, 
                daily_data_with_predictions, 
                model_name, 
                model_version
            )
            
            # Store metrics with data identifier
            timestamp = datetime.datetime.combine(date, datetime.time.min)
            timestamp = Config.TIMEZONE.localize(timestamp)
            
            store_metrics(metrics, timestamp, date, data_identifier)
            
            results.append({
                'date': date,
                'metrics': metrics,
                'data_points': len(daily_data),
                'data_identifier': data_identifier
            })
            
        except Exception as e:
            logger.error(f"❌ Error processing data for {date}: {e}")
            continue
    
    return results


@flow()
def taxi_monitoring_pipeline(year: int, month: int, retrain_model: bool = True):
    """Main pipeline for taxi data monitoring"""
    
    logger.info(f"🚀 Starting taxi monitoring pipeline for {year}-{month:02d}")
    
    # Setup
    setup_environment()
    setup_database()
    
    # Download and prepare monitoring data
    monthly_data = download_data(year, month)
    monthly_data = preprocess_data(monthly_data)
    
    if len(monthly_data) == 0:
        logger.error(f"❌ No valid data for {year}-{month:02d}")
        return
    
    # Train model or load existing one
    if retrain_model:
        # Load training, validation, and test data
        train_data, val_data, test_data = load_training_data()
        
        if train_data.empty:
            logger.error("❌ No training data available")
            return
        
        # Train models with hyperparameter optimization
        model_path, model_version, model_results = train_models_with_hyperopt(
            train_data, val_data, test_data
        )
        
        # Save reference data (use training data as reference)
        reference_path = Config.DATA_DIR / f"reference_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        train_data.to_parquet(reference_path)
        
        # Load model and generate predictions for reference data
        model_pipeline = load_model(model_path)
        reference_data = generate_predictions(train_data, model_pipeline)
        model_name = "retrained_model"
        
        # Log model results summary
        logger.info("📊 Model Training Results Summary:")
        for name, results in model_results.items():
            logger.info(f"   {name}: Val MAE={results['val_mae']:.4f}, Test MAE={results['test_mae']:.4f}")
        
    else:
        # Load existing model and reference data
        model_files = list(Config.MODELS_DIR.glob("best_model_*.pkl"))
        if not model_files:
            logger.error("❌ No trained model found. Please run with retrain_model=True first.")
            return
        
        model_path = str(sorted(model_files)[-1])  # Get latest model
        model_pipeline = load_model(model_path)
        model_name = "existing_model"
        model_version = "1"
        
        # Load reference data
        reference_files = list(Config.DATA_DIR.glob("reference_data_*.parquet"))
        if not reference_files:
            logger.error("❌ No reference data found.")
            return
        
        reference_data = pd.read_parquet(sorted(reference_files)[-1])
        reference_data = generate_predictions(reference_data, model_pipeline)
    
    # Process daily data for drift monitoring
    results = process_daily_data(
        monthly_data, 
        reference_data, 
        model_pipeline, 
        model_name, 
        model_version, 
        year, 
        month
    )
    
    # Summary logging
    processed_days = len(results)
    total_points = sum(r['data_points'] for r in results)
    
    logger.info(f"🎉 Pipeline completed for {year}-{month:02d}")
    logger.info(f"📊 Processed {processed_days} days with {total_points} total data points")
    
    # Calculate summary statistics
    if results:
        drift_values = [r['metrics']['prediction_drift'] for r in results 
                       if r['metrics']['prediction_drift'] is not None]
        drifted_cols_values = [r['metrics']['num_drifted_columns'] for r in results 
                              if r['metrics']['num_drifted_columns'] is not None]
        
        if drift_values:
            avg_drift = np.mean(drift_values)
            logger.info(f"📈 Average prediction drift: {avg_drift:.4f}")
        
        if drifted_cols_values:
            avg_drifted_cols = np.mean(drifted_cols_values)
            logger.info(f"📈 Average drifted columns: {avg_drifted_cols:.2f}")

@flow()
def train_only_pipeline():
    """Pipeline for training models only"""
    
    logger.info("🚀 Starting training-only pipeline")
    
    # Setup
    setup_environment()
    
    # Load training, validation, and test data
    train_data, val_data, test_data = load_training_data()
    
    if train_data.empty:
        logger.error("❌ No training data available")
        return
    
    # Train models with hyperparameter optimization
    model_path, model_version, model_results = train_models_with_hyperopt(
        train_data, val_data, test_data
    )
    
    # Save reference data
    reference_path = Config.DATA_DIR / f"reference_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    train_data.to_parquet(reference_path)
    
    # Log model results summary
    logger.info("📊 Model Training Results Summary:")
    logger.info("="*60)
    for name, results in model_results.items():
        logger.info(f"{name:>15}: Val MAE={results['val_mae']:.4f} | Test MAE={results['test_mae']:.4f} | Test R²={results['test_r2']:.4f}")
        if results['best_params']:
            logger.info(f"{'':>15}  Best params: {results['best_params']}")
    logger.info("="*60)
    
    logger.info(f"🎉 Training completed. Best model saved to: {model_path}")
    
    return model_path, model_version, model_results


def create_model_comparison_report(model_results: Dict) -> str:
    """Create a detailed model comparison report"""
    
    report = []
    report.append("="*80)
    report.append("MODEL COMPARISON REPORT")
    report.append("="*80)
    report.append("")
    
    # Sort models by validation MAE
    sorted_models = sorted(model_results.items(), key=lambda x: x[1]['val_mae'])
    
    report.append("RANKING BY VALIDATION MAE:")
    report.append("-" * 40)
    for i, (name, results) in enumerate(sorted_models, 1):
        report.append(f"{i:2d}. {name:<20} | Val MAE: {results['val_mae']:.4f}")
    report.append("")
    
    # Detailed results table
    report.append("DETAILED RESULTS:")
    report.append("-" * 80)
    report.append(f"{'Model':<20} | {'Val MAE':<8} | {'Val R²':<8} | {'Test MAE':<8} | {'Test R²':<8}")
    report.append("-" * 80)
    
    for name, results in sorted_models:
        report.append(f"{name:<20} | {results['val_mae']:<8.4f} | {results['val_r2']:<8.4f} | "
                     f"{results['test_mae']:<8.4f} | {results['test_r2']:<8.4f}")
    
    report.append("")
    report.append("HYPERPARAMETER DETAILS:")
    report.append("-" * 40)
    
    for name, results in sorted_models:
        if results['best_params']:
            report.append(f"\n{name}:")
            for param, value in results['best_params'].items():
                report.append(f"  {param}: {value}")
        else:
            report.append(f"\n{name}: No hyperparameters tuned")
    
    report.append("")
    report.append("="*80)
    
    return "\n".join(report)


# Additional utility function for querying with data identifier
def analyze_drift_trends_by_identifier(data_identifier: str = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Analyze drift trends over time, optionally filtered by data identifier"""
    
    base_query = """
    SELECT 
        date_processed,
        prediction_drift,
        num_drifted_columns,
        share_missing_values,
        model_name,
        model_version,
        data_points_count,
        data_identifier
    FROM taxi_drift_metrics 
    WHERE 1=1
    """
    
    params = []
    
    if data_identifier:
        base_query += " AND data_identifier = %s"
        params.append(data_identifier)
    
    if start_date and end_date:
        base_query += " AND date_processed BETWEEN %s AND %s"
        params.extend([start_date, end_date])
    elif start_date:
        base_query += " AND date_processed >= %s"
        params.append(start_date)
    elif end_date:
        base_query += " AND date_processed <= %s"
        params.append(end_date)
    
    base_query += " ORDER BY date_processed"
    
    try:
        with get_connection(Config.DB_NAME) as conn:
            df = pd.read_sql_query(base_query, conn, params=params)
            
        logger.info(f"📊 Retrieved {len(df)} drift records")
        if data_identifier:
            logger.info(f"📋 Filtered by data identifier: {data_identifier}")
        return df
        
    except Exception as e:
        logger.error(f"❌ Error analyzing drift trends: {e}")
        return pd.DataFrame()

def get_drift_summary_by_month(year: int, month: int) -> Dict:
    """Get summary statistics for drift metrics for a specific month"""
    
    data_identifier = generate_data_identifier(Config.TAXI_TYPE, year, month)
    start_date = f"{year}-{month:02d}-01"
    
    # Calculate end date for the month
    if month == 12:
        end_date = f"{year + 1}-01-01"
    else:
        end_date = f"{year}-{month + 1:02d}-01"
    
    df = analyze_drift_trends_by_identifier(data_identifier, start_date, end_date)
    
    if df.empty:
        return {'error': f'No data found for {data_identifier}'}
    
    summary = {
        'data_identifier': data_identifier,
        'period': f"{year}-{month:02d}",
        'total_days': len(df),
        'date_range': f"{df['date_processed'].min()} to {df['date_processed'].max()}",
        'avg_prediction_drift': df['prediction_drift'].mean() if df['prediction_drift'].notna().any() else None,
        'max_prediction_drift': df['prediction_drift'].max() if df['prediction_drift'].notna().any() else None,
        'min_prediction_drift': df['prediction_drift'].min() if df['prediction_drift'].notna().any() else None,
        'avg_drifted_columns': df['num_drifted_columns'].mean() if df['num_drifted_columns'].notna().any() else None,
        'max_drifted_columns': df['num_drifted_columns'].max() if df['num_drifted_columns'].notna().any() else None,
        'avg_missing_values': df['share_missing_values'].mean() if df['share_missing_values'].notna().any() else None,
        'total_data_points': df['data_points_count'].sum(),
        'model_name': df['model_name'].iloc[0] if len(df) > 0 else None,
        'model_version': df['model_version'].iloc[0] if len(df) > 0 else None
    }
    
    # Calculate trend (comparing first half vs second half of month)
    if len(df) >= 4:
        mid_point = len(df) // 2
        first_half_drift = df['prediction_drift'].iloc[:mid_point].mean()
        second_half_drift = df['prediction_drift'].iloc[mid_point:].mean()
        
        if pd.notna(first_half_drift) and pd.notna(second_half_drift):
            if second_half_drift > first_half_drift * 1.1:
                summary['drift_trend'] = 'increasing'
            elif second_half_drift < first_half_drift * 0.9:
                summary['drift_trend'] = 'decreasing'
            else:
                summary['drift_trend'] = 'stable'
        else:
            summary['drift_trend'] = 'insufficient_data'
    else:
        summary['drift_trend'] = 'insufficient_data'
    
    return summary

# Enhanced main execution with better examples
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Taxi Duration Prediction and Monitoring Pipeline')
    parser.add_argument('--mode', choices=['train', 'monitor', 'batch'], default='train',
                       help='Pipeline mode: train (train only), monitor (single month), batch (multiple months)')
    parser.add_argument('--year', type=int, default=2022, help='Year to process')
    parser.add_argument('--month', type=int, help='Month to process (required for monitor mode)')
    parser.add_argument('--start-month', type=int, help='Start month for batch processing')
    parser.add_argument('--end-month', type=int, help='End month for batch processing')
    parser.add_argument('--retrain', action='store_true', help='Retrain model before monitoring')
    parser.add_argument('--analyze', action='store_true', help='Run analysis after processing')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("🚀 Starting training-only pipeline...")
        model_path, model_version, model_results = train_only_pipeline()
        
        # Create and save detailed report
        report = create_model_comparison_report(model_results)
        print(f"\n{report}")
        
        report_path = Config.LOGS_DIR / f"model_comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"📄 Detailed report saved to: {report_path}")
        
    elif args.mode == 'monitor':
        if not args.month:
            print("❌ Month is required for monitor mode")
            exit(1)
            
        print(f"🚀 Starting monitoring pipeline for {args.year}-{args.month:02d}...")
        taxi_monitoring_pipeline(year=args.year, month=args.month, retrain_model=args.retrain)
        
        if args.analyze:
            print(f"\n📊 Analysis for {args.year}-{args.month:02d}:")
            summary = get_drift_summary_by_month(args.year, args.month)
            for key, value in summary.items():
                print(f"  {key}: {value}")
                
    elif args.mode == 'batch':
        if not args.start_month or not args.end_month:
            print("❌ Start month and end month are required for batch mode")
            exit(1)
            
        print(f"🚀 Starting batch processing from {args.year}-{args.start_month:02d} to {args.year}-{args.end_month:02d}...")
        
        for month in range(args.start_month, args.end_month + 1):
            print(f"\n📅 Processing {args.year}-{month:02d}...")
            try:
                taxi_monitoring_pipeline(year=args.year, month=month, retrain_model=False)
                
                if args.analyze:
                    summary = get_drift_summary_by_month(args.year, month)
                    print(f"📊 Summary for {args.year}-{month:02d}: "
                          f"Avg Drift: {summary.get('avg_prediction_drift', 'N/A'):.4f}, "
                          f"Days: {summary.get('total_days', 'N/A')}, "
                          f"Trend: {summary.get('drift_trend', 'N/A')}")
            except Exception as e:
                print(f"❌ Error processing {args.year}-{month:02d}: {e}")
                continue
    
    # Example usage without arguments:
    print("\n" + "="*60)
    print("EXAMPLE USAGE:")
    print("="*60)
    print("# Train models only:")
    print("python mlflow_evidently_grafana.py --mode train")
    print()
    print("# Monitor single month with existing model:")
    print("python mlflow_evidently_grafana.py --mode monitor --year 2022 --month 5")
    print()
    print("# Monitor single month with retraining:")
    print("python mlflow_evidently_grafana.py --mode monitor --year 2022 --month 5 --retrain")
    print()
    print("# Batch process multiple months:")
    print("python mlflow_evidently_grafana.py --mode batch --year 2022 --start-month 5 --end-month 8 --analyze")
    print("="*60)