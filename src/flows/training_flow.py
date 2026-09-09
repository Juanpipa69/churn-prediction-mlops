"""Flow de Prefect que orquesta el entrenamiento, comparación y registro del mejor modelo."""

import mlflow
from mlflow import MlflowClient
from prefect import flow, get_run_logger, task

from src.data.validate import validate_data
from src.models.train import (
    get_candidate_models,
    load_data,
    log_run,
    split_data,
    train_and_evaluate,
)

EXPERIMENT_NAME = "churn-prediction"
TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "churn-prediction-model"


@task(retries=2, retry_delay_seconds=5)
def load_data_task():
    logger = get_run_logger()
    df = load_data()
    logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


@task
def validate_data_task(df):
    logger = get_run_logger()
    validate_data(df)
    logger.info("Validación de datos OK")
    return df


@task
def split_data_task(df):
    return split_data(df)


@task
def train_candidate_task(name, config, X_train, y_train, X_test, y_test):
    logger = get_run_logger()
    pipeline, metrics = train_and_evaluate(config["estimator"], X_train, y_train, X_test, y_test)
    run_id = log_run(name, pipeline, config["params"], metrics, config["trusted_types"])
    logger.info(f"{name} -> ROC-AUC: {metrics['roc_auc']:.4f} | Recall: {metrics['recall']:.4f}")
    return {"name": name, "run_id": run_id, "metrics": metrics}


@task
def register_best_model_task(results, metric="roc_auc"):
    logger = get_run_logger()
    best = max(results, key=lambda r: r["metrics"][metric])
    logger.info(f"Mejor modelo: {best['name']} ({metric}={best['metrics'][metric]:.4f})")

    model_uri = f"runs:/{best['run_id']}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    client = MlflowClient()
    client.set_registered_model_alias(name=MODEL_NAME, alias="champion", version=registered.version)
    return best


@flow(name="churn-training-flow")
def training_flow():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data_task()
    df = validate_data_task(df)
    X_train, X_test, y_train, y_test = split_data_task(df)

    candidates = get_candidate_models(y_train)
    results = [
        train_candidate_task(name, config, X_train, y_train, X_test, y_test)
        for name, config in candidates.items()
    ]

    return register_best_model_task(results)


if __name__ == "__main__":
    training_flow()
