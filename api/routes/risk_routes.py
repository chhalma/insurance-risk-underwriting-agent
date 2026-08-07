from fastapi import APIRouter

from api.models.risk_model_schemas import ApplicantRequest, PredictionResponse
from api.repository.risk_model_repository import RiskModelRepository
from api.services.risk_service import RiskService

risk_router = APIRouter()
_repository = RiskModelRepository()
_service = RiskService(_repository)


@risk_router.get("/health")
def health():
    return {"status": "ok"}


@risk_router.post("/predict", response_model=PredictionResponse)
def predict(applicant: ApplicantRequest):
    return _service.score(applicant)
