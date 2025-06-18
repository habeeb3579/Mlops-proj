import requests

url = "http://localhost:9696/predict"
params = {"year": 2023, "month": 5}
response = requests.get(url, params=params)
print(response.json())
