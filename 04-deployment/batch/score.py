from prefect import flow, task
import os
import uuid
import pandas as pd
import numpy as np
import mlflow
from typing import Union, Tuple, List, Dict
import typer

app = typer.Typer()

# Set GCP credentials and MLflow tracking
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/habeeb/dprof-dezoomfinal-b4d188529d18.json"
TRACKING_SERVER_HOST = "35.224.212.79"
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
model_name = "nyc-taxi-regressor-weighted-main9"
model = mlflow.pyfunc.load_model(f"models:/{model_name}@production")

@task
def generate_uuids(n):
    return [str(uuid.uuid4()) for _ in range(n)]

@task
def download_data(taxi: str, year: int, month: int) -> pd.DataFrame:
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi}_tripdata_{year}-{month:02d}.parquet"
    print(f"Downloading data from {url}")
    return pd.read_parquet(url)

@task
def prepare_df(data: Union[pd.DataFrame, List[Dict], Dict]) -> Tuple[pd.DataFrame, Union[np.ndarray, None]]:
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input must be a DataFrame, list of dicts, or single dict.")

    if 'lpep_pickup_datetime' in df.columns and 'lpep_dropoff_datetime' in df.columns:
        pickup_col, dropoff_col = 'lpep_pickup_datetime', 'lpep_dropoff_datetime'
    elif 'tpep_pickup_datetime' in df.columns and 'tpep_dropoff_datetime' in df.columns:
        pickup_col, dropoff_col = 'tpep_pickup_datetime', 'tpep_dropoff_datetime'
    else:
        pickup_col = dropoff_col = None

    if pickup_col and dropoff_col:
        if not np.issubdtype(df[pickup_col].dtype, np.datetime64):
            df[pickup_col] = pd.to_datetime(df[pickup_col])
        if not np.issubdtype(df[dropoff_col].dtype, np.datetime64):
            df[dropoff_col] = pd.to_datetime(df[dropoff_col])
        df["duration"] = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60
        df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    else:
        df["duration"] = None

    for col in ["PULocationID", "DOLocationID"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    if "PULocationID" in df.columns and "DOLocationID" in df.columns:
        df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    df["ride_id"] = generate_uuids.fn(len(df))

    target = df["duration"].values if df["duration"].notna().all() else None
    return df, target

@task
def predict(features: pd.DataFrame) -> np.ndarray:
    return model.predict(features)

@task
def save_predictions(dfs: pd.DataFrame, targ: np.ndarray, preds: np.ndarray, taxi: str, year: int, month: int):
    df_result = pd.DataFrame({
        'ride_id': dfs['ride_id'],
        'lpep_pickup_datetime': dfs['lpep_pickup_datetime'],
        'PULocationID': dfs['PULocationID'],
        'DOLocationID': dfs['DOLocationID'],
        'actual_duration': targ,
        'predicted_duration': preds,
        'diff': targ - preds,
        'model_version': model.metadata.run_id
    })
    output_file = f'output/{taxi}/{year:04d}-{month:02d}.parquet'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_result.to_parquet(output_file, index=False)
    print(f"✅ Results saved to {output_file}")

@flow
def apply_model_flow(taxi: str = "green", year: int = 2021, month: int = 2):
    print(f"▶️ Running apply_model for {taxi} - {year}-{month}")
    df = download_data(taxi, year, month)
    dfs, targ = prepare_df(df)
    preds = predict(dfs)
    save_predictions(dfs, targ, preds, taxi, year, month)

@app.command()
def run(
    taxi: str = typer.Option("green", help="Taxi type (green or yellow)"),
    year: int = typer.Option(2021, help="Year of data"),
    month: int = typer.Option(2, help="Month of data"),
    verbose: bool = typer.Option(True, help="Enable verbose output")
):
    apply_model_flow(taxi=taxi, year=year, month=month, verbose=verbose)

if __name__ == "__main__":
    app()

# python your_script.py run --taxi green --year 2021 --month 2 --verbose