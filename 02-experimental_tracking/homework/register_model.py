import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# Constants
HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
MODEL_NAME = "rf-best-model"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

# Configure MLflow
#mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog(disable=True)  # Disable autologging to avoid conflict with manual logs


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        # Convert Hyperopt string params to int
        for param in RF_PARAMS:
            params[param] = int(params[param])

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        val_rmse = root_mean_squared_error(y_val, rf.predict(X_val))
        test_rmse = root_mean_squared_error(y_test, rf.predict(X_test))

        mlflow.log_params(params)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("test_rmse", test_rmse)

        # Log model artifact
        mlflow.sklearn.log_model(rf, artifact_path="model")


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):
    client = MlflowClient()

    # 1. Get all child runs from the HPO experiment with 'rmse' metric
    hpo_experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    all_runs = client.search_runs(
        experiment_ids=hpo_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1000
    )

    valid_runs = [r for r in all_runs if "rmse" in r.data.metrics]
    top_runs = sorted(valid_runs, key=lambda r: r.data.metrics["rmse"])[:top_n]

    print(f"Evaluating top {top_n} Hyperopt runs...")

    # 2. Retrain & log each top model under a new experiment
    for run in top_runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # 3. Find best model from new experiment based on test_rmse
    best_experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=best_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"]
    )[0]

    # 4. Register the best model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    print(f"Registering model from run_id={run_id}")
    mlflow.register_model(model_uri, name=MODEL_NAME)


if __name__ == '__main__':
    run_register_model()