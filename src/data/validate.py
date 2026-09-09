"""Validación del esquema y calidad del dataset de churn antes de entrenar."""

import pandas as pd

EXPECTED_COLUMNS = {
    'Call  Failure', 'Complains', 'Subscription  Length', 'Charge  Amount',
    'Seconds of Use', 'Frequency of use', 'Frequency of SMS',
    'Distinct Called Numbers', 'Age Group', 'Tariff Plan', 'Status',
    'Age', 'Customer Value', 'Churn',
}
TARGET_COLUMN = "Churn"
MIN_ROWS = 100
MAX_NULL_RATIO = 0.05


class DataValidationError(Exception):
    """Se lanza cuando el dataset no cumple el esquema o la calidad mínima esperada."""


def validate_data(df: pd.DataFrame) -> None:
    """Valida columnas esperadas, tamaño mínimo, nulos y valores del target.

    Lanza DataValidationError con un mensaje claro si algo no cumple.
    """
    errors = []

    missing_columns = EXPECTED_COLUMNS - set(df.columns)
    if missing_columns:
        errors.append(f"Faltan columnas esperadas: {sorted(missing_columns)}")

    if len(df) < MIN_ROWS:
        errors.append(f"El dataset tiene muy pocas filas ({len(df)} < {MIN_ROWS})")

    if TARGET_COLUMN in df.columns:
        valores_invalidos = set(df[TARGET_COLUMN].unique()) - {0, 1}
        if valores_invalidos:
            errors.append(f"'{TARGET_COLUMN}' tiene valores fuera de {{0, 1}}: {valores_invalidos}")

    null_ratios = df.isnull().mean()
    columnas_con_muchos_nulos = null_ratios[null_ratios > MAX_NULL_RATIO]
    if not columnas_con_muchos_nulos.empty:
        errors.append(
            f"Columnas con más de {MAX_NULL_RATIO:.0%} de nulos: {columnas_con_muchos_nulos.to_dict()}"
        )

    if errors:
        raise DataValidationError("Validación de datos fallida:\n- " + "\n- ".join(errors))
