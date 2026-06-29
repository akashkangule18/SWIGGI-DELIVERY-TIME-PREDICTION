from pydantic import BaseModel
import dagshub
import mlflow
import pandas as pd

from fastapi import FastAPI
from schema import DeliveryInput


class DeliveryInput(BaseModel):
    age: float
    rating: float

    weather_cond: str
    traffic: str

    vehicle_cond: int
    vehicle_type: str

    multiple_orders: float

    festival: str
    city_type: str
    city_name: str

    month: int
    weekend: int
    ordered_hour: float

    distance_km: float




import os
import mlflow

import os

import os
import mlflow

mlflow.set_tracking_uri(
    "https://dagshub.com/akashkangule18/SWIGGI-DELIVERY-TIME-PREDICTION.mlflow"
)

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

app = FastAPI(
    title="Swiggy Delivery Time Prediction API"
)

model = mlflow.sklearn.load_model(
    "models:/LightGBMRegressor@staging"
)


@app.get("/")
def home():
    return {"message": "API is Running"}


@app.post("/predict")
def predict(data: DeliveryInput):

    df = pd.DataFrame([data.model_dump()])

    try:
        prediction = model.predict(df)

        return {
            "Predicted Delivery Time": float(prediction[0])
        }

    except Exception as e:
        return {
            "error": str(e)
        }

    return {
        "Predicted Delivery Time": float(prediction[0])
    }