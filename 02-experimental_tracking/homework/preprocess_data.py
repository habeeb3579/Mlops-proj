import os
import pickle
import click
import pandas as pd

from sklearn.feature_extraction import DictVectorizer


def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def read_dataframe(filename: str = None, df: pd.DataFrame = None, taxi_type: str = "green"):
    # Use provided DataFrame if available, otherwise load from file
    if df is not None:
        df = df.copy()  # Create a copy to avoid modifying the original
    elif filename is not None:
        df = pd.read_parquet(filename)
    else:
        raise ValueError("Either 'filename' or 'df' must be provided")
    
    # Set column names based on taxi type
    if taxi_type == "green":
        pickup_col = 'lpep_pickup_datetime'
        dropoff_col = 'lpep_dropoff_datetime'
    elif taxi_type == "yellow":
        pickup_col = 'tpep_pickup_datetime'
        dropoff_col = 'tpep_dropoff_datetime'
    else:
        raise ValueError("taxi_type must be either 'green' or 'yellow'")

    df['duration'] = df[dropoff_col] - df[pickup_col]
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df


def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv


@click.command()
@click.option(
    "--raw_data_path",
    help="Location where the raw NYC taxi trip data was saved"
)
@click.option(
    "--dest_path",
    help="Location where the resulting files will be saved"
)
@click.option(
    "--taxi_type",
    default="green",
    help="Type of taxi data: 'green' or 'yellow' (default: green)"
)
def run_data_prep(raw_data_path: str = None, dest_path: str = None, taxi_type: str = "green", 
                  df_train: pd.DataFrame = None, df_val: pd.DataFrame = None, df_test: pd.DataFrame = None):
    # Load parquet files or use provided DataFrames
    if df_train is not None:
        df_train = read_dataframe(df=df_train, taxi_type=taxi_type)
    else:
        if raw_data_path is None:
            raise ValueError("Either 'raw_data_path' or 'df_train' must be provided")
        df_train = read_dataframe(
            filename=os.path.join(raw_data_path, f"{taxi_type}_tripdata_2023-01.parquet"),
            taxi_type=taxi_type
        )
    
    if df_val is not None:
        df_val = read_dataframe(df=df_val, taxi_type=taxi_type)
    else:
        if raw_data_path is None:
            raise ValueError("Either 'raw_data_path' or 'df_val' must be provided")
        df_val = read_dataframe(
            filename=os.path.join(raw_data_path, f"{taxi_type}_tripdata_2023-02.parquet"),
            taxi_type=taxi_type
        )
    
    if df_test is not None:
        df_test = read_dataframe(df=df_test, taxi_type=taxi_type)
    else:
        if raw_data_path is None:
            raise ValueError("Either 'raw_data_path' or 'df_test' must be provided")
        df_test = read_dataframe(
            filename=os.path.join(raw_data_path, f"{taxi_type}_tripdata_2023-03.parquet"),
            taxi_type=taxi_type
        )

    if dest_path is None:
        raise ValueError("'dest_path' must be provided")
    
    # Extract the target
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    # Fit the DictVectorizer and preprocess data
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv, fit_dv=False)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)

    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save DictVectorizer and datasets
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))


if __name__ == '__main__':
    run_data_prep()


# For green taxi (default)
#python script.py --raw_data_path /path/to/data --dest_path /path/to/output

# For yellow taxi
#python script.py --raw_data_path /path/to/data --dest_path /path/to/output --taxi_type yellow