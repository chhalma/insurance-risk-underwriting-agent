from api.models.recommendation_schemas import RecommendationResponse
from api.models.risk_model_schemas import ApplicantRequest
from api.services.policy_service import PolicyService
from api.services.risk_service import RiskService


class RecommendationService:
    """Orchestrates the risk model and the policy RAG pipeline into one explainable recommendation."""

    def __init__(self, risk_service: RiskService, policy_service: PolicyService):
        self.risk_service = risk_service
        self.policy_service = policy_service

    def recommend(self, applicant: ApplicantRequest) -> RecommendationResponse:
        prediction = self.risk_service.score(applicant)
        query = self._build_policy_query(applicant, prediction.risk_category)
        policy_answer = self.policy_service.answer(query, k=3)

        return RecommendationResponse(
            predicted_annual_charge=prediction.predicted_annual_charge,
            risk_category=prediction.risk_category,
            key_risk_factors=prediction.key_risk_factors,
            explanation=policy_answer.answer,
            policy_sources=policy_answer.sources,
        )

    def _build_policy_query(self, applicant: ApplicantRequest, risk_category: str) -> str:
        smoker_clause = "a smoker" if applicant.smoker == "yes" else "a non-smoker"
        return (
            f"What does the policy say about coverage and cost considerations for "
            f"a {risk_category.lower()}-risk applicant, aged {applicant.age}, who is {smoker_clause}?"
        )
