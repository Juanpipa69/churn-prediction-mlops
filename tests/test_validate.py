import pandas as pd
import pytest

from src.data.validate import EXPECTED_COLUMNS, DataValidationError, validate_data


def _valid_df(n_rows: int = 100) -> pd.DataFrame:
    data = {col: [i % 5 for i in range(n_rows)] for col in EXPECTED_COLUMNS}
    data["Churn"] = [i % 2 for i in range(n_rows)]
    return pd.DataFrame(data)


def test_validate_data_ok():
    df = _valid_df()
    validate_data(df)  # no debe lanzar excepcion


def test_validate_data_missing_column():
    df = _valid_df().drop(columns=["Age"])
    with pytest.raises(DataValidationError):
        validate_data(df)


def test_validate_data_too_few_rows():
    df = _valid_df(n_rows=10)
    with pytest.raises(DataValidationError):
        validate_data(df)


def test_validate_data_invalid_target():
    df = _valid_df()
    df["Churn"] = [0, 2] * (len(df) // 2)
    with pytest.raises(DataValidationError):
        validate_data(df)
