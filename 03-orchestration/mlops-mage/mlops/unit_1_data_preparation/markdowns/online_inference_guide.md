# Sample online inference

Use the following CURL command to get real-time predictions:

```curl
curl --location 'http://localhost:6789/api/runs' \
--header 'Authorization: Bearer b43f02fdb4364004a48ef7009bdf69bc' \
--header 'Content-Type: application/json' \
--header 'Cookie: lng=en' \
--data '{
    "run": {
        "pipeline_uuid": "predict_new",
        "block_uuid": "inference",
        "variables": {
            "inputs": [
                {
                    "DOLocationID": "239",
                    "PULocationID": "236",
                    "trip_distance": 1.98
                },
                {
                    "DOLocationID": "170",
                    "PULocationID": "65",
                    "trip_distance": 6.54
                }
            ]
        }
    }
}'
```

## Note

The `Authorization` header is using this pipeline’s API trigger’s token value.
The token value is set to `fire` for this project.
If you create a new trigger, that token will change.
Only use a fixed token for testing or demonstration purposes.