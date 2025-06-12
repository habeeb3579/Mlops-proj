import pathlib
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from prefect import flow, task


@task(retries=3, retry_delay_seconds=2)
def download_data(year: int, month: int, taxi: str = "green") -> pd.DataFrame:
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi}_tripdata_{year}-{month:02d}.parquet"
    print(f"Downloading data from {url}")
    try:
        df = pd.read_parquet(url)
        print(f"Downloaded {len(df)} records")
        return df
    except Exception as e:
        print(f"Error downloading data: {e}")
        raise


@task
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # Check and use green or yellow trip schema
    if 'lpep_dropoff_datetime' in df.columns and 'lpep_pickup_datetime' in df.columns:
        pickup_col = 'lpep_pickup_datetime'
        dropoff_col = 'lpep_dropoff_datetime'
    elif 'tpep_dropoff_datetime' in df.columns and 'tpep_pickup_datetime' in df.columns:
        pickup_col = 'tpep_pickup_datetime'
        dropoff_col = 'tpep_dropoff_datetime'
    else:
        raise ValueError("No recognized pickup/dropoff datetime columns found.")

    # Convert to datetime only if not already
    if not np.issubdtype(df[pickup_col].dtype, np.datetime64):
        df[pickup_col] = pd.to_datetime(df[pickup_col])
    if not np.issubdtype(df[dropoff_col].dtype, np.datetime64):
        df[dropoff_col] = pd.to_datetime(df[dropoff_col])

    # Calculate trip duration in minutes
    df["duration"] = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60

    # Filter out trips with implausible durations
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    # Ensure location IDs are treated as categorical strings
    df["PULocationID"] = df["PULocationID"].astype(str)
    df["DOLocationID"] = df["DOLocationID"].astype(str)

    print(f"df after preprocessing is {df.shape}")

    return df


@task
def split_features(df: pd.DataFrame, fit_dv: bool = True, dv: DictVectorizer = None):
    """
    Transforms categorical features into vectors using DictVectorizer.

    Args:
        df: Preprocessed DataFrame
        fit_dv: Whether to fit a new DictVectorizer (True) or use an existing one (False)
        dv: Existing DictVectorizer to use when fit_dv is False

    Returns:
        X: Transformed feature matrix
        y: Target vector
        dv: DictVectorizer (fitted)
    """
    categorical = ["PULocationID", "DOLocationID"]
    dicts = df[categorical].to_dict(orient="records")

    if fit_dv:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        if dv is None:
            raise ValueError("DictVectorizer must be provided when fit_dv=False")
        X = dv.transform(dicts)

    y = df["duration"].values
    return X, y, dv



@task(log_prints=True)
def train_and_register_model(X, y, dv):
    X_train, X_val, y_train, y_val = X, X, y, y

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)

        print(f"Intercept of the model: {model.intercept_}")
        print(f"RMSE: {rmse}")

        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="homework-model")

        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/dv.pkl", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/dv.pkl", artifact_path="preprocessor")


@flow
def homework_flow(year: int = 2023, month: int = 3, taxi: str = "yellow"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-linear-model")

    df = download_data(year, month, taxi)
    df_clean = prepare_data(df)
    X, y, dv = split_features(df_clean)
    train_and_register_model(X, y, dv)


if __name__ == "__main__":
    homework_flow()