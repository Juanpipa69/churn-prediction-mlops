from src.api import main as api_main
from src.api.schemas import ChurnFeatures


class DummyModel:
    def predict(self, df):
        return [0]

    def predict_proba(self, df):
        return [[0.9, 0.1]]


def test_predict_endpoint_logic(monkeypatch):
    monkeypatch.setattr(api_main, "model", DummyModel())

    features = ChurnFeatures(
        **{
            "Call  Failure": 2,
            "Complains": 0,
            "Subscription  Length": 10,
            "Charge  Amount": 1,
            "Seconds of Use": 500,
            "Frequency of use": 20,
            "Frequency of SMS": 5,
            "Distinct Called Numbers": 15,
            "Age Group": 2,
            "Tariff Plan": 1,
            "Status": 1,
            "Age": 30,
            "Customer Value": 200.0,
        }
    )

    result = api_main.predict(features)

    assert result.churn_prediction == 0
    assert 0.0 <= result.churn_probability <= 1.0


def test_health_endpoint_reports_model_loaded(monkeypatch):
    monkeypatch.setattr(api_main, "model", DummyModel())
    result = api_main.health()
    assert result == {"status": "ok", "model_loaded": True}
