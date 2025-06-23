import datetime
import time
import logging
import pandas as pd
import psycopg2
import joblib

from prefect import task, flow

from evidently import Report, DataDefinition, Dataset
from evidently.metrics import ValueDrift, DriftedColumnsCount, MissingValueCount

# ------------------------- Configuration ------------------------- #
DB_HOST = "localhost"
DB_PORT = 5433
DB_NAME = "test"
DB_USER = "postgres"
DB_PASSWORD = "example"
SEND_TIMEOUT = 10

NUMERICAL_FEATURES = ['passenger_count', 'trip_distance', 'fare_amount', 'total_amount']
CATEGORICAL_FEATURES = ['PULocationID', 'DOLocationID']

BEGIN_DATE = datetime.datetime(2022, 2, 1)
REFERENCE_DATA_PATH = 'data/reference.parquet'
CURRENT_DATA_PATH = 'data/green_tripdata_2022-02.parquet'
MODEL_PATH = 'models/lin_reg.bin'

CREATE_METRICS_TABLE_SQL = """
DROP TABLE IF EXISTS dummy_metrics;
CREATE TABLE dummy_metrics (
    timestamp TIMESTAMP,
    prediction_drift FLOAT,
    num_drifted_columns INTEGER,
    share_missing_values FLOAT
);
"""

# ------------------------- Logger Setup ------------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)

# ------------------------- Evidently Setup ------------------------- #
data_definition = DataDefinition(
    numerical_columns=NUMERICAL_FEATURES + ['prediction'],
    categorical_columns=CATEGORICAL_FEATURES,
)

report = Report(metrics=[
    ValueDrift(column='prediction'),
    DriftedColumnsCount(),
    MissingValueCount(column='prediction'),
])


# ------------------------- Utility Functions ------------------------- #
def get_connection(dbname=None, autocommit=False):
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=dbname or "postgres"
    )
    conn.autocommit = autocommit
    return conn


# ------------------------- Prefect Tasks ------------------------- #

@task
def prepare_database():
    """Create the test database and dummy_metrics table."""
    try:
        # Connect to the default "postgres" DB (no context manager here)
        conn = get_connection(autocommit=True)
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (DB_NAME,))
            if not cur.fetchone():
                cur.execute(f"CREATE DATABASE {DB_NAME}")
                logger.info(f"✅ Database '{DB_NAME}' created.")
            else:
                logger.info(f"ℹ️ Database '{DB_NAME}' already exists.")
            cur.close()
        finally:
            conn.close()  # Must manually close since no context manager

        # Now connect to the newly created DB to create the table
        with get_connection(DB_NAME, autocommit=False) as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_METRICS_TABLE_SQL)
                conn.commit()
                logger.info("✅ dummy_metrics table created.")

    except Exception as e:
        logger.error(f"❌ Failed to prepare database: {e}")


@task
def load_assets():
    """Load reference data, model and current month data."""
    reference_df = pd.read_parquet(REFERENCE_DATA_PATH)
    current_df = pd.read_parquet(CURRENT_DATA_PATH)
    with open(MODEL_PATH, 'rb') as f_in:
        model = joblib.load(f_in)
    return reference_df, current_df, model


@task
def insert_metrics(cur, timestamp, drift, drifted_cols, missing_share):
    """Insert calculated metrics into the database."""
    cur.execute(
        """
        INSERT INTO dummy_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values)
        VALUES (%s, %s, %s, %s)
        """,
        (timestamp, drift, drifted_cols, missing_share)
    )


@task
def calculate_and_store_metrics(index, reference_df, current_df, model):
    """Calculate metrics for a given day and insert into the DB."""
    day_start = BEGIN_DATE + datetime.timedelta(index)
    day_end = BEGIN_DATE + datetime.timedelta(index + 1)

    daily_data = current_df[
        (current_df.lpep_pickup_datetime >= day_start) &
        (current_df.lpep_pickup_datetime < day_end)
    ].copy()

    if daily_data.empty:
        logger.warning(f"⚠️ No data for day {day_start.date()}, skipping.")
        return

    daily_data['prediction'] = model.predict(daily_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].fillna(0))

    ref_dataset = Dataset.from_pandas(reference_df, data_definition=data_definition)
    curr_dataset = Dataset.from_pandas(daily_data, data_definition=data_definition)

    run = report.run(reference_data=ref_dataset, current_data=curr_dataset)
    result = run.dict()

    prediction_drift = float(result['metrics'][0]['value'])
    num_drifted_columns = int(result['metrics'][1]['value']['count'])
    share_missing_values = float(result['metrics'][2]['value']['share'])


    with get_connection(DB_NAME) as conn:
        with conn.cursor() as cur:
            insert_metrics.fn(cur, day_start, prediction_drift, num_drifted_columns, share_missing_values)
            conn.commit()

    logger.info(f"✅ Metrics stored for {day_start.date()}.")


# ------------------------- Prefect Flow ------------------------- #
@flow(name="Daily Dummy Metrics Workflow")
def daily_metrics_flow():
    prepare_database()
    reference_df, current_df, model = load_assets()

    last_send_time = time.monotonic() - SEND_TIMEOUT

    for day_index in range(27):
        calculate_and_store_metrics(day_index, reference_df, current_df, model)

        elapsed = time.monotonic() - last_send_time
        if elapsed < SEND_TIMEOUT:
            time.sleep(SEND_TIMEOUT - elapsed)
        last_send_time += SEND_TIMEOUT


# ------------------------- Entry Point ------------------------- #
if __name__ == "__main__":
    daily_metrics_flow()
