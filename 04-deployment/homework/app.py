from fastapi import FastAPI
from starter import predict  
import numpy as np

app = FastAPI()

@app.get("/predict")
def predict_endpoint(year: int = 2023, month: int = 3):
    preds = predict(year, month)
    return {"predictions mean": np.mean(preds)}
