from fastapi import APIRouter

from api.models.recommendation_schemas import RecommendationResponse
from api.models.risk_model_schemas import ApplicantRequest
from api.routes.policy_routes import _service as _policy_service
from api.routes.risk_routes import _service as _risk_service
from api.services.recommendation_service import RecommendationService

recommendation_router = APIRouter()
_recommendation_service = RecommendationService(_risk_service, _policy_service)


@recommendation_router.post("/recommendation", response_model=RecommendationResponse)
def get_recommendation(applicant: ApplicantRequest):
    return _recommendation_service.recommend(applicant)
