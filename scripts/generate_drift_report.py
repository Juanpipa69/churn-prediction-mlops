"""
Prueba de concepto de monitoreo de drift con Evidently AI.

Compara el split de entrenamiento (reference) contra el split de prueba
(current, simulando datos de "producción") y genera un reporte HTML.
"""
from pathlib import Path

from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report

from src.models.train import load_data, split_data

REPORT_PATH = Path("reports/drift_report.html")


def generate_drift_report():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    reference_data = X_train.copy()
    reference_data["Churn"] = y_train.values

    current_data = X_test.copy()
    current_data["Churn"] = y_test.values

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))
    print(f"Reporte de drift generado en {REPORT_PATH}")


if __name__ == "__main__":
    generate_drift_report()
