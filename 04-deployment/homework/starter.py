import pickle
import pandas as pd
import numpy as np

# Load the model and DictVectorizer
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Categorical columns used for feature engineering
categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def predict(year: int = 2023, month: int = 3):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    print(f"📥 Reading data from: {url}")
    
    df = read_data(url)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f"✅ Done. Number of predictions: {len(y_pred)}")
    print(f"📊 Predictions mean: {np.mean(y_pred)}")
    
    return y_pred

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NYC Taxi Duration Prediction Script")
    parser.add_argument("--year", type=int, default=2023, help="Year of the dataset")
    parser.add_argument("--month", type=int, default=3, help="Month of the dataset")

    args = parser.parse_args()
    predict(year=args.year, month=args.month)

#python starter.py --year 2023 --month 3

#jupyter nbconvert --to script starter.ipynb


