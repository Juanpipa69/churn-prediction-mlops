"""Exporta el modelo campeón desde el Model Registry a una carpeta autocontenida para Docker."""

import shutil
from pathlib import Path

import mlflow

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_URI = "models:/churn-prediction-model@champion"
EXPORT_PATH = Path("models/champion")


def export_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(MODEL_URI)

    if EXPORT_PATH.exists():
        shutil.rmtree(EXPORT_PATH)
    mlflow.sklearn.save_model(
    model,
    str(EXPORT_PATH),
    skops_trusted_types=["numpy.dtype", "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
    )
    print(f"Modelo exportado a {EXPORT_PATH}")


if __name__ == "__main__":
    export_model()
