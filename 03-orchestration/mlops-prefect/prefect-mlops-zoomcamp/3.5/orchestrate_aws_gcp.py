import pathlib
import pickle
import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import mlflow
import xgboost as xgb
from prefect import flow, task
from prefect_aws import S3Bucket
from prefect_gcp import GcsBucket
from prefect.artifacts import create_markdown_artifact
from datetime import date
import typer

app = typer.Typer()


@task(retries=3, retry_delay_seconds=2)
def read_data(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)
    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    return df


@task
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    dv = DictVectorizer()
    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)
    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)
    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, X_val, y_train, y_val, dv


@task(log_prints=True)
def train_best_model(X_train, X_val, y_train, y_val, dv) -> None:
    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)
        best_params = {
            "learning_rate": 0.09585355369315604,
            "max_depth": 30,
            "min_child_weight": 1.060597050922164,
            "objective": "reg:linear",
            "reg_alpha": 0.018060244040060163,
            "reg_lambda": 0.011658731377413597,
            "seed": 42,
        }
        mlflow.log_params(best_params)
        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, "validation")],
            early_stopping_rounds=20,
        )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
        markdown_report = f"""# RMSE Report

## Summary

Duration Prediction 

## RMSE XGBoost Model

| Region    | RMSE |
|:----------|-------:|
| {date.today()} | {rmse:.2f} |
"""
        create_markdown_artifact(
            key="duration-model-report", markdown=markdown_report
        )


@flow
def main_flow(use_gcp: bool = False, upload_to_cloud: bool = False):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")

    data_path = "/home/habeeb/Mlops-proj/03-orchestration/mlops-prefect/prefect-mlops-zoomcamp/data"
    data_path2 = "/home/habeeb/Mlops-proj/03-orchestration/mlops-prefect/prefect-mlops-zoomcamp/data_from_gcp"

    if upload_to_cloud:
        if use_gcp:
            gcs = GcsBucket.load("gcs-bucket-example")
            gcs.upload_from_folder(from_folder=data_path, to_folder="data")
            print("✅ Uploaded data to GCS bucket")
        else:
            s3 = S3Bucket.load("s3-bucket-block")
            s3.upload_from_folder(from_folder=data_path, to_folder="data")
            print("✅ Uploaded data to S3 bucket")

    if use_gcp:
        bucket = GcsBucket.load("gcs-bucket-example")
    else:
        bucket = S3Bucket.load("s3-bucket-block")

    bucket.download_folder_to_path(from_folder="data", to_folder=data_path2)

    df_train = read_data(f"{data_path2}/green_tripdata_2023-01.parquet")
    df_val = read_data(f"{data_path2}/green_tripdata_2023-02.parquet")

    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)
    train_best_model(X_train, X_val, y_train, y_val, dv)


@app.command()
def run_pipeline(
    use_gcp: bool = typer.Option(False, help="Use GCP instead of AWS"),
    upload_to_cloud: bool = typer.Option(False, help="Upload local data folder to cloud")
):
    """
    Run the ML pipeline, optionally using GCP and/or uploading data to cloud
    """
    main_flow(use_gcp=use_gcp, upload_to_cloud=upload_to_cloud)


if __name__ == "__main__":
    app()

#help
# python /home/habeeb/Mlops-proj/03-orchestration/mlops-prefect/prefect-mlops-zoomcamp/3.5/orchestrate_aws_gcp.py --help
#gcp
# python /home/habeeb/Mlops-proj/03-orchestration/mlops-prefect/prefect-mlops-zoomcamp/3.5/orchestrate_aws_gcp.py --use-gcp --upload-to-cloud
#aws
# python /home/habeeb/Mlops-proj/03-orchestration/mlops-prefect/prefect-mlops-zoomcamp/3.5/orchestrate_aws_gcp.py --upload-to-cloud 
