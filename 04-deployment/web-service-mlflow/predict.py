import pickle
from flask import Flask, request, jsonify 
import mlflow #.pyfunc
import pandas as pd
from typing import Union, Tuple, List, Dict
import numpy as np
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS']="/home/habeeb/dprof-dezoomfinal-b4d188529d18.json"

TRACKING_SERVER_HOST = "35.224.212.79" # fill in with the public IP
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")

model_name = "nyc-taxi-regressor-weighted-main9"
model = mlflow.pyfunc.load_model(f"models:/{model_name}@production")
#model = mlflow.sklearn.load_model(f"models:/{model_name}@production")

def prepare_df(data: Union[pd.DataFrame, List[Dict], Dict]) -> Tuple[pd.DataFrame, Union[np.ndarray, None]]:
    """
    Prepare taxi trip data for model input.
    
    Supports:
      - DataFrame input
      - List of dictionaries
      - Single dictionary input
    
    Handles both lpep and tpep datetime columns.
    
    Args:
        data: Input data
    
    Returns:
        Tuple of (processed DataFrame, target if available else None)
    """
    # Normalize input to DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input must be a DataFrame, list of dicts, or single dict.")

    # Determine pickup/dropoff column names
    if 'lpep_pickup_datetime' in df.columns and 'lpep_dropoff_datetime' in df.columns:
        pickup_col, dropoff_col = 'lpep_pickup_datetime', 'lpep_dropoff_datetime'
    elif 'tpep_pickup_datetime' in df.columns and 'tpep_dropoff_datetime' in df.columns:
        pickup_col, dropoff_col = 'tpep_pickup_datetime', 'tpep_dropoff_datetime'
    else:
        pickup_col, dropoff_col = None, None

    # Handle datetime conversion and duration calculation
    if pickup_col and dropoff_col:
        if not np.issubdtype(df[pickup_col].dtype, np.datetime64):
            df[pickup_col] = pd.to_datetime(df[pickup_col])
        if not np.issubdtype(df[dropoff_col].dtype, np.datetime64):
            df[dropoff_col] = pd.to_datetime(df[dropoff_col])

        df["duration"] = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60
        df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    else:
        df["duration"] = None

    # Convert categorical columns to string if present
    for col in ["PULocationID", "DOLocationID"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Create combined PU_DO feature
    if "PULocationID" in df.columns and "DOLocationID" in df.columns:
        df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    # Return target if fully computed
    target = df["duration"].values if df["duration"].notna().all() else None
    return df, target

def predict(features):
    preds = model.predict(features)
    return preds

app = Flask('duration-prediction')

@app.route('/predict', methods=["POST"])
def predict_endpoint():
    ride = request.get_json()
    features, _ = prepare_df(ride)
    pred = predict(features)
    result = {
        'duration': pred.tolist(),
        'model_version': model.metadata.run_id
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)