import os

import mlflow
import pandas as pd
from fastapi import FastAPI
from schema import DeliveryInput

# ----------------------------------------------------
# MLflow Authentication
# ----------------------------------------------------

username = os.getenv("DAGSHUB_USERNAME")
token = os.getenv("DAGSHUB_TOKEN")

if not username or not token:
    raise ValueError(
        "DAGSHUB_USERNAME or DAGSHUB_TOKEN environment variable is missing."
    )

os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

# ----------------------------------------------------
# MLflow Tracking URI
# ----------------------------------------------------

mlflow.set_tracking_uri(
    "https://dagshub.com/akashkangule18/SWIGGI-DELIVERY-TIME-PREDICTION.mlflow"
)

# ----------------------------------------------------
# Load Registered Model
# ----------------------------------------------------

model = mlflow.sklearn.load_model(
    "models:/LightGBMRegressor@staging"
)

# ----------------------------------------------------
# FastAPI
# ----------------------------------------------------

app = FastAPI(
    title="Swiggy Delivery Time Prediction API",
    version="1.0",
    description="Predict Swiggy Delivery Time using a model stored in DagsHub Model Registry."
)


@app.get("/")
def home():
    return {
        "message": "Swiggy Delivery Time Prediction API is Running!"
    }


@app.post("/predict")
def predict(data: DeliveryInput):

    try:
        input_df = pd.DataFrame([data.model_dump()])

        prediction = model.predict(input_df)

        return {
            "Predicted Delivery Time": float(prediction[0])
        }

    except Exception as e:
        return {
            "error": str(e)
        }