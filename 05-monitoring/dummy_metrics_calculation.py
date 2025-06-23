import datetime
import time
import random
import logging
import uuid
import pytz
import psycopg2

# Configuration
DB_HOST = "localhost"
DB_PORT = 5433
DB_NAME = "test"
DB_USER = "postgres" #default as it was not set in the docker-compose env
DB_PASSWORD = "example"
SEND_TIMEOUT = 10

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

# Random generator
rand = random.Random()

# SQL statements
CREATE_TABLE_SQL = """
DROP TABLE IF EXISTS dummy_metrics;
CREATE TABLE dummy_metrics (
    timestamp TIMESTAMP,
    value1 INTEGER,
    value2 VARCHAR,
    value3 FLOAT
);
"""

def get_connection(dbname=None, autocommit=False):
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=dbname or "postgres"
    )
    if autocommit:
        #conn.set_session(autocommit=True)
        # Enable autocommit
        conn.autocommit = True
    return conn

def prep_db():
    try:
        # Step 1: Create the database if it doesn't exist
        conn = get_connection(autocommit=True)
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (DB_NAME,))
            if not cur.fetchone():
                cur.execute(f"CREATE DATABASE {DB_NAME}")  # Removed semicolon
                logging.info(f"✅ Database '{DB_NAME}' created.")
            else:
                logging.info(f"ℹ️ Database '{DB_NAME}' already exists.")
            cur.close()
        finally:
            conn.close()

        # Step 2: Create the dummy_metrics table inside that DB
        # For table creation, you can use context manager since it works in transactions
        with get_connection(DB_NAME, autocommit=False) as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
                conn.commit()  # Explicit commit since autocommit=False
                logging.info("✅ dummy_metrics table created.")

    except Exception as e:
        logging.error(f"❌ Error preparing DB: {e}")

def insert_dummy_metrics(cur):
    now = datetime.datetime.now(pytz.timezone('Europe/London'))
    value1 = rand.randint(0, 1000)
    value2 = str(uuid.uuid4())
    value3 = rand.random()

    cur.execute(
        """
        INSERT INTO dummy_metrics(timestamp, value1, value2, value3)
        VALUES (%s, %s, %s, %s)
        """,
        (now, value1, value2, value3)
    )

def main():
    prep_db()
    last_send = time.monotonic() - SEND_TIMEOUT

    with get_connection(DB_NAME) as conn:
        for _ in range(100):
            with conn.cursor() as cur:
                insert_dummy_metrics(cur)

            elapsed = time.monotonic() - last_send
            if elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - elapsed)
            last_send += SEND_TIMEOUT
            logging.info("Dummy metric inserted.")

if __name__ == '__main__':
    main()
