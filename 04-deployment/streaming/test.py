# test.py
import requests
#import json

url = "https://5kmy249uje.execute-api.us-east-1.amazonaws.com/dev/stream" # Replace with your API Gateway URL

payload = {
    "data": {
        "lpep_pickup_datetime": "2021-02-01T08:00:00",
        "lpep_dropoff_datetime": "2021-02-01T08:15:00",
        "PULocationID": 132,
        "DOLocationID": 138,
        "trip_distance": 50
    },
    "useFifo": False
}

response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())
