import os
import pickle
import click
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

#mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment-homework2")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="/home/habeeb/Mlops-proj/02-experimental_tracking/output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    print(mlflow.search_experiments())
    # Enable autologging for sklearn models
    mlflow.sklearn.autolog()

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run() as run: 
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)

        print(f"RMSE: {rmse}")
        print(f"Run ID: {run.info.run_id}")


        # Log metric manually if desired (autolog may already log it)
        mlflow.log_metric("rmse", rmse)

    return run.info.run_id


if __name__ == '__main__':
    run_train()