import pandas as pd

from api.models.risk_model_schemas import ApplicantRequest, PredictionResponse, RiskFactor
from api.repository.risk_model_repository import RiskModelRepository


class RiskService:
    """Business logic: turns applicant data into a scored, explained prediction."""

    def __init__(self, repository: RiskModelRepository):
        self.repository = repository

    def _categorize(self, predicted_charge: float) -> str:
        if predicted_charge < self.repository.low_max:
            return "Low"
        if predicted_charge < self.repository.medium_max:
            return "Medium"
        return "High"

    def score(self, applicant: ApplicantRequest) -> PredictionResponse:
        df = pd.DataFrame([applicant.model_dump()])
        transformed = self.repository.preprocessor.transform(df)

        predicted_charge = float(self.repository.model.predict(transformed)[0])
        risk_category = self._categorize(predicted_charge)

        shap_values = self.repository.explainer.shap_values(transformed)[0]
        top_factors = sorted(
            zip(self.repository.feature_names, shap_values),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:3]

        return PredictionResponse(
            predicted_annual_charge=round(predicted_charge, 2),
            risk_category=risk_category,
            key_risk_factors=[
                RiskFactor(feature=name, impact=round(float(value), 2))
                for name, value in top_factors
            ],
        )
