"""Esquemas de entrada/salida de la API de predicción de churn."""

from pydantic import BaseModel, Field


class ChurnFeatures(BaseModel):
    """Un cliente a evaluar. Los nombres coinciden con las columnas del dataset original."""

    call_failure: int = Field(..., alias="Call  Failure", ge=0)
    complains: int = Field(..., alias="Complains", ge=0, le=1)
    subscription_length: int = Field(..., alias="Subscription  Length", ge=0)
    charge_amount: int = Field(..., alias="Charge  Amount", ge=0)
    seconds_of_use: int = Field(..., alias="Seconds of Use", ge=0)
    frequency_of_use: int = Field(..., alias="Frequency of use", ge=0)
    frequency_of_sms: int = Field(..., alias="Frequency of SMS", ge=0)
    distinct_called_numbers: int = Field(..., alias="Distinct Called Numbers", ge=0)
    age_group: int = Field(..., alias="Age Group")
    tariff_plan: int = Field(..., alias="Tariff Plan")
    status: int = Field(..., alias="Status")
    age: int = Field(..., alias="Age", ge=0)
    customer_value: float = Field(..., alias="Customer Value", ge=0)

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "Call  Failure": 8,
                "Complains": 0,
                "Subscription  Length": 38,
                "Charge  Amount": 0,
                "Seconds of Use": 4370,
                "Frequency of use": 71,
                "Frequency of SMS": 5,
                "Distinct Called Numbers": 17,
                "Age Group": 3,
                "Tariff Plan": 1,
                "Status": 1,
                "Age": 30,
                "Customer Value": 197.64,
            }
        },
    }


class ChurnPrediction(BaseModel):
    """Respuesta de la API."""

    churn_prediction: int
    churn_probability: float