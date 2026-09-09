import pandas as pd

from src.features.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor


def test_build_preprocessor_returns_transformer():
    preprocessor = build_preprocessor()
    assert preprocessor is not None


def test_preprocessor_fit_transform_shape():
    n_rows = 20
    data = {col: list(range(n_rows)) for col in NUMERIC_FEATURES}
    for col in CATEGORICAL_FEATURES:
        data[col] = [0, 1] * (n_rows // 2)
    df = pd.DataFrame(data)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == n_rows
