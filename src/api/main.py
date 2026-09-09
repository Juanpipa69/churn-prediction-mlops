"""API REST para predicción de churn, sirve el modelo campeón desde el Model Registry de MLflow."""

import os
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import ChurnFeatures, ChurnPrediction

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_URI = os.environ.get("MODEL_URI", "models:/churn-prediction-model@champion")

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(MODEL_URI)
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Predice la probabilidad de que un cliente cancele el servicio.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=ChurnPrediction)
def predict(features: ChurnFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo aún no está cargado")

    input_df = pd.DataFrame([features.model_dump(by_alias=True)])

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    return ChurnPrediction(churn_prediction=prediction, churn_probability=probability)
