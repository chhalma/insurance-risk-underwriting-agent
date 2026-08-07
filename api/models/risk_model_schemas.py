from typing import Literal

from pydantic import BaseModel


class ApplicantRequest(BaseModel):
    age: int
    sex: Literal["male", "female"]
    bmi: float
    children: int
    smoker: Literal["yes", "no"]
    region: Literal["northeast", "northwest", "southeast", "southwest"]


class RiskFactor(BaseModel):
    feature: str
    impact: float


class PredictionResponse(BaseModel):
    predicted_annual_charge: float
    risk_category: Literal["Low", "Medium", "High"]
    key_risk_factors: list[RiskFactor]
