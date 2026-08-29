from pathlib import Path
from ucimlrepo import fetch_ucirepo

RAW_DATA_PATH = Path("data/raw/churn.csv")

def fetch_and_save():
    dataset = fetch_ucirepo(id=563)
    X = dataset.data.features
    y = dataset.data.targets
    df = X.copy()
    df["Churn"] = y

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Guardado: {RAW_DATA_PATH} ({df.shape[0]} filas, {df.shape[1]} columnas)")

if __name__ == "__main__":
    fetch_and_save()