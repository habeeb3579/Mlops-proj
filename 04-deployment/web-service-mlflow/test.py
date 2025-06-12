#import predict
import requests
import pandas as pd


taxi = "green"
year = 2021
month = 3
url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi}_tripdata_{year}-{month:02d}.parquet"        
print(f"Downloading data from {url}")
ride = pd.read_parquet(url)

for col in ride.select_dtypes(include=["datetime64[ns]"]).columns:
    ride[col] = ride[col].astype(str)

ride = ride.to_dict(orient='records')[:5]

# you can also send a single instance as a dict
# ride = {
#     "PULocationID": 10,
#     "DOLocationID": 50,
#     "trip_distance": 40
# }


url = "http://localhost:9696/predict"
response = requests.post(url, json=ride)
print(response.json())
