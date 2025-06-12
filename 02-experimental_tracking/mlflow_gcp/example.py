import mlflow
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a model with MLflow tracking')
parser.add_argument('--tracking-uri', type=str, help='MLflow tracking URI')
parser.add_argument('--experiment-name', type=str, default='GCP-MLflow-Demo',
                    help='MLflow experiment name')
args = parser.parse_args()

# Set MLflow tracking URI if provided
if args.tracking_uri:
    mlflow.set_tracking_uri(args.tracking_uri)
    print(f"Using MLflow tracking URI: {args.tracking_uri}")
else:
    print(f"Using default MLflow tracking URI: {mlflow.get_tracking_uri()}")

# Set experiment
mlflow.set_experiment(args.experiment_name)

# Sample data generation
def generate_data(n_samples=1000):
    np.random.seed(42)
    X = np.random.rand(n_samples, 4)
    y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + 0.5 * X[:, 3] + np.random.normal(0, 0.1, n_samples)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])
    df['target'] = y
    return df

print("Generating sample data...")
data = generate_data()

# Split the data
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model parameters
n_estimators = 100
max_depth = 5

# Start MLflow run
with mlflow.start_run() as run:
    print(f"MLflow Run ID: {run.info.run_id}")
    print(f"Training model with n_estimators={n_estimators}, max_depth={max_depth}...")
    
    # Train model
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)
    
    # Make predictions
    predictions = rf.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print(f"Model performance metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    
    # Log model
    mlflow.sklearn.log_model(rf, "random_forest_model")
    
    # Log feature importance plot
    feature_importance = pd.DataFrame(
        {'feature': X.columns, 'importance': rf.feature_importances_}
    ).sort_values('importance', ascending=False)
    
    # Save feature importances as CSV
    feature_importance.to_csv("feature_importances.csv", index=False)
    mlflow.log_artifact("feature_importances.csv")
    
    print(f"Feature importances:")
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print(f"\nRun completed. Check MLflow UI for details.")
    print(f"Run URL: {mlflow.get_tracking_uri()}/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{mlflow.active_run().info.run_id}")