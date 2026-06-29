import os

import dagshub
import mlflow
import pandas as pd
from fastapi import FastAPI
from schema import DeliveryInput

# ----------------------------------------------------
# DagsHub Initialization
# ----------------------------------------------------

username = os.getenv("DAGSHUB_USERNAME")
token = os.getenv("DAGSHUB_TOKEN")

if not username or not token:
    raise ValueError(
        "DAGSHUB_USERNAME or DAGSHUB_TOKEN environment variable is missing."
    )

# Set MLflow credentials
os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

# Initialize DagsHub
dagshub.init(
    repo_owner="akashkangule18",
    repo_name="SWIGGI-DELIVERY-TIME-PREDICTION",
    mlflow=True
)

# ----------------------------------------------------
# Load Registered Model
# ----------------------------------------------------

model = mlflow.sklearn.load_model(
    "models:/LightGBMRegressor@staging"
)

# ----------------------------------------------------
# FastAPI App
# ----------------------------------------------------

app = FastAPI(
    title="Swiggy Delivery Time Prediction API",
    version="1.0",
    description="Predict Swiggy Delivery Time using a model stored in DagsHub Model Registry."
)

# ----------------------------------------------------
# Home Endpoint
# ----------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Swiggy Delivery Time Prediction API is Running!"
    }

# ----------------------------------------------------
# Prediction Endpoint
# ----------------------------------------------------

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