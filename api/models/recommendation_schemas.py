from pydantic import BaseModel

from api.models.policy_schemas import PolicySection
from api.models.risk_model_schemas import RiskFactor


class RecommendationResponse(BaseModel):
    predicted_annual_charge: float
    risk_category: str
    key_risk_factors: list[RiskFactor]
    explanation: str
    policy_sources: list[PolicySection]
