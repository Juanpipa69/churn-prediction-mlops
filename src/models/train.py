"""Entrenamiento, evaluación y logueo en MLflow de los modelos candidatos."""

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.preprocessing import build_preprocessor

RAW_DATA_PATH = "data/raw/churn.csv"


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Carga el dataset de churn desde disco."""
    return pd.read_csv(path)


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Separa features/target y hace el split train/test estratificado."""
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def get_candidate_models(y_train) -> dict:
    """Define los modelos candidatos con sus hiperparámetros (igual que en el notebook 03)."""
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    return {
        "logreg_baseline": {
            "estimator": LogisticRegression(random_state=42, max_iter=1000),
            "params": {"model_type": "LogisticRegression", "max_iter": 1000},
            "trusted_types": ["numpy.dtype"],
        },
        "random_forest_balanced": {
            "estimator": RandomForestClassifier(
                n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
            ),
            "params": {
                "model_type": "RandomForestClassifier",
                "n_estimators": 200,
                "max_depth": 10,
                "class_weight": "balanced",
            },
            "trusted_types": ["numpy.dtype"],
        },
        "xgboost_weighted": {
            "estimator": XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42,
            ),
            "params": {
                "model_type": "XGBClassifier",
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.1,
                "scale_pos_weight": scale_pos_weight,
            },
            "trusted_types": ["numpy.dtype", "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
        },
    }


def train_and_evaluate(estimator, X_train, y_train, X_test, y_test) -> tuple[Pipeline, dict]:
    """Entrena un pipeline (preprocesador + estimador) y calcula sus métricas."""
    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", estimator),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    return pipeline, metrics


def log_run(run_name: str, pipeline: Pipeline, params: dict, metrics: dict, trusted_types: list) -> str:
    """Loguea un run completo en MLflow: parámetros, métricas y el modelo. Devuelve el run_id."""
    with mlflow.start_run(run_name=run_name) as run:
        for key, value in params.items():
            mlflow.log_param(key, value)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.sklearn.log_model(pipeline, name="model", skops_trusted_types=trusted_types)
        return run.info.run_id
